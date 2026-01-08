import torch.nn as nn
import pytorch_lightning as pl
import torch
from typing import Any, Sequence


class LitDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, dataloader_cls, dataloader_args):
        super().__init__()
        self.train_ds, self.val_ds = train_ds, val_ds
        self.dataloader_cls = dataloader_cls

        self.batch_size = dataloader_args.pop("batch_size", 1)
        self.dataloader_args = dataloader_args

    def train_dataloader(self):
        return self.dataloader_cls(
            self.train_ds,
            shuffle=True,
            batch_size=self.batch_size,
            **self.dataloader_args,
        )

    def val_dataloader(self):
        return self.dataloader_cls(
            self.val_ds,
            shuffle=False,
            batch_size=self.batch_size,
            **self.dataloader_args,
        )


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model_class: nn.Module,
        loss_class: nn.Module,
        model_cfg: dict,
        loss_cfg: dict,
        trainer_cfg: dict,
        postprocessor=None,
        free_run_cfg: dict | None = None,
    ) -> None:
        """Args:

        model_class: Name of the model to use loss_class: Name of the loss function to
        use datasets: Dataset configuration dictionary model_cfg: Model configuration
        dictionary loss_cfg: Loss configuration dictionary trainer_cfg: Trainer
        configuration dictionary
        """
        super().__init__()
        try:
            self.save_hyperparameters()
        except TypeError:
            print("Could not save hyperparameters")
            pass

        # keep a reference to the trainer config (for scheduler, etc.)
        self.trainer_cfg = trainer_cfg
        self.postprocessor = postprocessor
        self.free_run_cfg = free_run_cfg
        # Collect test-time outputs to enable downstream metrics/plots
        self.test_predictions: list[torch.Tensor] = []
        self.test_targets: list[torch.Tensor] = []

        # Create model and loss instances
        self.model = model_class(**model_cfg)

        # compile the model if requested
        if trainer_cfg.get("compile", False):
            self.model = torch.compile(self.model)

        self.loss = loss_class(**loss_cfg)

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ):
        inputs, targets = batch
        outputs = self.model(inputs)

        logits = outputs
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]

        if self.postprocessor is not None:
            inputs, logits, targets = self.postprocessor(inputs, logits, targets)

        if isinstance(outputs, (tuple, list)):
            outputs = [logits] + [out for out in outputs[1:]]

        loss = self.loss(outputs, targets, model=self.model)

        metrics_for_stage: dict[str, torch.Tensor] = {}
        self.log(f"{stage}_loss", loss, prog_bar=True)
        for name, metric in self.loss.metrics.items():
            metric_val = metric(outputs, targets)
            metrics_for_stage[name] = metric_val
            self.log(f"{stage}_{name}", metric_val, prog_bar=True)

        # log learning rate
        lr = None
        if stage == "train":
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr)

        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Args: batch: Batch of data batch_idx: Batch index.

        Returns:     Loss value
        """
        return self._step(batch, batch_idx, stage="train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Args: batch: Batch of data batch_idx: Batch index.

        Returns:     Loss value
        """
        _ = self._step(batch, batch_idx, stage="val")

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        targets = None
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch, targets = batch

        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = batch[0]

        if isinstance(targets, tuple) or isinstance(targets, list):
            targets = targets[0]

        logits = self.model(batch)
        probs = torch.softmax(logits, dim=1)

        self.test_predictions.append(probs.cpu())

        if targets is not None:
            self.test_targets.append(targets.cpu())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.trainer_cfg["lr"],
            weight_decay=self.trainer_cfg["weight_decay"],
            betas=(0.9, 0.95),  # recommended by the Qwen team
        )

        raw_sched_cfg = self.trainer_cfg.get("lr_scheduler", None)
        # Work on a copy to avoid mutating hyperparameters
        sched_cfg = dict(raw_sched_cfg) if raw_sched_cfg else None
        if not sched_cfg:
            return optimizer

        # Extract PL-specific options
        interval = sched_cfg.pop("interval", "epoch")
        frequency = sched_cfg.pop("frequency", 1)
        monitor = sched_cfg.pop("monitor", None)
        class_name = sched_cfg.pop("class_name")

        # Build the scheduler from torch.optim.lr_scheduler
        try:
            scheduler_cls = getattr(torch.optim.lr_scheduler, class_name)
        except AttributeError as e:
            raise ValueError(
                f"Unknown lr_scheduler class '{class_name}' in torch.optim.lr_scheduler"
            ) from e

        # Remaining entries in sched_cfg are passed to the constructor
        scheduler = scheduler_cls(optimizer, **sched_cfg)

        # Compose the PL optimizer/scheduler config
        lr_sched_dict = {
            "scheduler": scheduler,
            "interval": interval,  # unit of the scheduler's step
            "frequency": frequency,  # how often to call
        }
        if monitor is not None:
            lr_sched_dict["monitor"] = monitor

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_sched_dict,
        }


class LitModelFreerun(LitModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.free_run_cfg = self._prepare_free_run_cfg(self.free_run_cfg)

    def _prepare_free_run_cfg(self, cfg: dict | None) -> dict[str, Any]:
        """Normalise optional K-step free-run settings."""
        base_cfg = {
            "enabled": False,
            "warmup_range": (0, 0),
            "rollout_range": (0, 0),
            "sample_strategy": "argmax",
            "temperature": 1.0,
            "log_lengths": False,
        }

        if not cfg or not cfg.get("enabled", False):
            return base_cfg

        warmup_range = self._parse_length_range(cfg.get("warmup_range"), "warmup_range")
        rollout_range = self._parse_length_range(
            cfg.get("rollout_range"), "rollout_range"
        )

        strategy = cfg.get("sample_strategy", "argmax").lower()
        if strategy not in {"argmax", "sample"}:
            raise ValueError(
                "sample_strategy must be either 'argmax' or 'sample', "
                f"got '{strategy}'"
            )

        temperature = float(cfg.get("temperature", 1.0))
        if temperature <= 0:
            raise ValueError("temperature must be strictly positive.")

        base_cfg.update(
            {
                "enabled": True,
                "warmup_range": warmup_range,
                "rollout_range": rollout_range,
                "sample_strategy": strategy,
                "temperature": temperature,
                "log_lengths": bool(cfg.get("log_lengths", False)),
            }
        )
        return base_cfg

    @staticmethod
    def _parse_length_range(
        value: int | Sequence[int] | None, name: str
    ) -> tuple[int, int]:
        """Parse integer or [min, max] length spec."""
        if value is None:
            raise ValueError(f"{name} must be provided when enabling free-run.")

        if isinstance(value, int):
            low = high = int(value)
        elif isinstance(value, Sequence) and len(value) == 2:
            low, high = int(value[0]), int(value[1])
        else:
            raise ValueError(
                f"{name} must be an int or a [min, max] sequence; got {value}"
            )

        if low <= 0 or high <= 0:
            raise ValueError(f"{name} elements must be > 0, got {value}")
        if high < low:
            raise ValueError(f"{name} upper bound must be >= lower bound, got {value}")
        return low, high

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ):
        """K-step free-run training pass.

        Returns None to fall back to teacher forcing.
        """
        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
            return None

        inputs, targets = batch
        input_tensor, input_path = self._extract_primary_tensor(inputs)
        target_tensor, target_path = self._extract_primary_tensor(targets)

        if input_tensor is None or target_tensor is None:
            return None

        if input_tensor.shape[-1] != target_tensor.shape[-1]:
            return None

        seq_len = int(input_tensor.shape[-1])
        lengths = self._determine_rollout_lengths(seq_len)
        if lengths is None:
            return None
        warmup, rollout = lengths

        start_idx = self._sample_start_index(seq_len, warmup, rollout)
        if start_idx is None:
            return None

        if self.free_run_cfg.get("log_lengths", False):
            self.log(
                "free_run_warmup",
                warmup,
                on_step=True,
                prog_bar=False,
                batch_size=input_tensor.shape[0],
            )
            self.log(
                "free_run_rollout",
                rollout,
                on_step=True,
                prog_bar=False,
                batch_size=input_tensor.shape[0],
            )

        device = input_tensor.device
        context_tokens = input_tensor[..., start_idx: start_idx + warmup].clone()

        logits_steps: list[torch.Tensor] = []
        target_steps: list[torch.Tensor] = []

        curr_start = start_idx
        logits_time_dim = None
        target_time_dim = None
        input_seq_len = seq_len
        target_seq_len = int(target_tensor.shape[-1])

        for _ in range(rollout):
            ctx_inputs = self._slice_time_axis(
                inputs, curr_start, curr_start + warmup, input_seq_len
            )
            ctx_inputs = self._set_by_path(ctx_inputs, input_path, context_tokens)

            ctx_targets = self._slice_time_axis(
                targets, curr_start, curr_start + warmup, target_seq_len
            )

            logits = self.model(ctx_inputs)
            if not torch.is_tensor(logits):
                return None

            ctx_targets_tensor = self._get_from_path(ctx_targets, target_path)
            if self.postprocessor is not None and torch.is_tensor(ctx_targets_tensor):
                ctx_inputs_tensor = self._get_from_path(ctx_inputs, input_path)
                (
                    ctx_inputs_tensor,
                    logits,
                    ctx_targets_tensor,
                ) = self.postprocessor(ctx_inputs_tensor, logits, ctx_targets_tensor)
                ctx_inputs = ctx_inputs_tensor

            if logits_time_dim is None:
                logits_time_dim = logits.dim() - 2
            if target_time_dim is None:
                target_time_dim = min(logits_time_dim, ctx_targets_tensor.dim() - 1)

            last_idx = logits.shape[logits_time_dim] - 1
            step_logits = logits.select(logits_time_dim, last_idx)
            logits_steps.append(step_logits.unsqueeze(logits_time_dim))

            last_target = ctx_targets_tensor.select(
                target_time_dim, ctx_targets_tensor.shape[target_time_dim] - 1
            )
            target_steps.append(last_target.unsqueeze(target_time_dim))

            next_token = self._sample_next_token(
                step_logits, dtype=context_tokens.dtype
            ).to(device)
            next_token = next_token.unsqueeze(-1)

            if context_tokens.shape[-1] == 1:
                context_tokens = next_token
            else:
                context_tokens = torch.cat(
                    [context_tokens[..., 1:], next_token], dim=-1
                )
            curr_start += 1

        if not logits_steps or logits_time_dim is None or target_time_dim is None:
            return None

        rollout_logits = torch.cat(logits_steps, dim=logits_time_dim)
        rollout_targets_tensor = torch.cat(target_steps, dim=target_time_dim)
        rollout_targets = self._set_by_path(
            targets, target_path, rollout_targets_tensor
        )

        loss = self.loss(rollout_logits, rollout_targets, model=self.model)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        metrics_for_stage: dict[str, torch.Tensor] = {}
        for name, metric in self.loss.metrics.items():
            metric_val = metric(rollout_logits, rollout_targets)
            metrics_for_stage[name] = metric_val
            self.log(
                f"{stage}_{name}",
                metric_val,
                prog_bar=True,
            )

        lr = None
        if stage == "train":
            lr = self.optimizers().param_groups[0]["lr"]
            self.log("lr", lr)
        return loss

    def _extract_primary_tensor(
        self, data: Any
    ) -> tuple[torch.Tensor | None, tuple[Any, ...]]:
        if torch.is_tensor(data):
            return data, tuple()
        path = self._find_tensor_path(data)
        if path is None:
            return None, tuple()
        tensor = self._get_from_path(data, path)
        if torch.is_tensor(tensor):
            return tensor, path
        return None, tuple()

    def _find_tensor_path(
        self, data: Any, prefix: tuple[Any, ...] = tuple()
    ) -> tuple[Any, ...] | None:
        if torch.is_tensor(data):
            return prefix
        if isinstance(data, (list, tuple)):
            for idx, item in enumerate(data):
                path = self._find_tensor_path(item, prefix + (idx,))
                if path is not None:
                    return path
        elif isinstance(data, dict):
            for key in data:
                path = self._find_tensor_path(data[key], prefix + (key,))
                if path is not None:
                    return path
        return None

    def _get_from_path(self, data: Any, path: tuple[Any, ...]) -> Any:
        if not path:
            return data
        key = path[0]
        rest = path[1:]
        if isinstance(data, (list, tuple)):
            return self._get_from_path(data[key], rest)
        if isinstance(data, dict):
            return self._get_from_path(data[key], rest)
        raise TypeError(f"Unsupported container type {type(data)} for key {key}")

    def _set_by_path(self, data: Any, path: tuple[Any, ...], value: Any) -> Any:
        if not path:
            return value
        key = path[0]
        rest = path[1:]
        if isinstance(data, tuple):
            data_list = list(data)
            data_list[key] = self._set_by_path(data_list[key], rest, value)
            return type(data)(data_list)
        if isinstance(data, list):
            data_list = list(data)
            data_list[key] = self._set_by_path(data_list[key], rest, value)
            return data_list
        if isinstance(data, dict):
            new_data = dict(data)
            new_data[key] = self._set_by_path(new_data[key], rest, value)
            return new_data
        raise TypeError(f"Unsupported container type {type(data)} for key {key}")

    def _slice_time_axis(self, data: Any, start: int, end: int, seq_len: int) -> Any:
        if torch.is_tensor(data):
            if data.shape[-1] == seq_len:
                return data[..., start:end]
            return data
        if isinstance(data, tuple):
            return tuple(
                self._slice_time_axis(item, start, end, seq_len) for item in data
            )
        if isinstance(data, list):
            return [self._slice_time_axis(item, start, end, seq_len) for item in data]
        if isinstance(data, dict):
            return {
                key: self._slice_time_axis(val, start, end, seq_len)
                for key, val in data.items()
            }
        return data

    def _determine_rollout_lengths(self, seq_len: int) -> tuple[int, int] | None:
        if seq_len < 2:
            return None
        full_len = seq_len + 1
        warmup = min(
            self._sample_length(self.free_run_cfg["warmup_range"]), full_len - 1
        )
        warmup = max(1, warmup)
        future_capacity = full_len - warmup
        if future_capacity <= 0:
            return None
        rollout = min(
            self._sample_length(self.free_run_cfg["rollout_range"]),
            future_capacity,
        )
        rollout = max(1, rollout)
        return warmup, rollout

    def _sample_length(self, bounds: tuple[int, int]) -> int:
        low, high = bounds
        if low == high:
            return low
        device = getattr(self, "device", torch.device("cpu"))
        return int(torch.randint(low, high + 1, (1,), device=device).item())

    def _sample_start_index(
        self, seq_len: int, warmup: int, rollout: int
    ) -> int | None:
        full_len = seq_len + 1
        max_start = full_len - (warmup + rollout)
        if max_start < 0:
            return None
        if max_start == 0:
            return 0
        device = getattr(self, "device", torch.device("cpu"))
        return int(torch.randint(0, max_start + 1, (1,), device=device).item())

    def _sample_next_token(
        self, logits: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        strategy = self.free_run_cfg.get("sample_strategy", "argmax")
        temperature = float(self.free_run_cfg.get("temperature", 1.0))

        with torch.no_grad():
            if strategy == "argmax":
                next_token = torch.argmax(logits, dim=-1)
            else:
                scaled = logits / temperature
                probs = torch.softmax(scaled, dim=-1)
                flat = probs.reshape(-1, probs.shape[-1])
                sampled = torch.multinomial(flat, num_samples=1).view(probs.shape[:-1])
                next_token = sampled
        return next_token.to(dtype=dtype)

    def _log_teacher_forced_batch(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> None:
        inputs, targets = batch
        with torch.no_grad():
            logits = self.model(inputs)
            if not torch.is_tensor(logits):
                return
        if self.postprocessor is not None:
            inputs, logits, targets = self.postprocessor(inputs, logits, targets)
        self._log_random_samples(inputs, logits, targets, stage)


class DatasetEpochCallback(pl.Callback):
    """Calls a dataset-provided epoch hook at train epoch boundaries.

    Supports datasets that expose either `set_epoch(int)` for cross-worker coordination
    via shared state, or a simpler `on_epoch_start(epoch)` method.
    """

    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # 0-based current epoch index in PL
        epoch = int(getattr(trainer, "current_epoch", 0))
        hook = getattr(self.dataset, "set_epoch", None)
        if callable(hook):
            hook(epoch)
        hook2 = getattr(self.dataset, "on_epoch_start", None)
        if callable(hook2):
            hook2(epoch)

        # Optional: print a fingerprint for debugging multi-worker consistency
        fp = getattr(self.dataset, "epoch_fingerprint", None)
        if callable(fp):
            try:
                print(f"[Dataset] epoch={epoch} fingerprint={fp()}")
            except Exception:
                pass
