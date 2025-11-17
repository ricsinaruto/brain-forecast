import torch.nn as nn
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from einops import rearrange

from ..utils.plotting import plot_psd


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model_class: nn.Module,
        loss_class: nn.Module,
        model_cfg: dict,
        loss_cfg: dict,
        trainer_cfg: dict,
        postprocessor=None,
    ) -> None:
        """
        Args:
            model_class: Name of the model to use
            loss_class: Name of the loss function to use
            datasets: Dataset configuration dictionary
            model_cfg: Model configuration dictionary
            loss_cfg: Loss configuration dictionary
            trainer_cfg: Trainer configuration dictionary
        """
        super().__init__()
        try:
            self.save_hyperparameters()
        except TypeError:
            print("Could not save hyperparameters")
            pass

        # keep a reference to the trainer config (for scheduler, etc.)
        self.trainer_cfg = trainer_cfg
        self.log_samples = trainer_cfg.get("log_samples", False)
        self._n_log_samples = trainer_cfg.get("n_log_samples", 3)
        self.log_psd = trainer_cfg.get("log_psd", False)
        self.sfreq = trainer_cfg.get("sfreq", None)
        self.postprocessor = postprocessor

        # Create model and loss instances
        self.model = model_class(**model_cfg)

        # compile the model if requested
        if trainer_cfg.get("compile", False):
            self.model = torch.compile(self.model)

        self.loss = loss_class(**loss_cfg)

        self.test_predictions = []
        self.test_targets = []

    def _log_random_samples(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor],
        outputs: torch.Tensor | tuple[torch.Tensor],
        targets: torch.Tensor | tuple[torch.Tensor],
        stage: str,
    ) -> None:
        """Log a few random (input, output, target) triplets to TensorBoard.

        Args:
            inputs: Batch input tensor.
            outputs: Corresponding model outputs (logits, predictions, etc.).
            targets: Ground-truth targets.
            stage: Either "train" or "val" – used for the log prefix.
        """

        # Additionally log time-series line plots as images
        def _make_ts_figure(ts: torch.Tensor, title: str):
            ts_np = ts.detach().cpu().numpy()
            # Ensure 2-D shape (channels, timesteps)
            if ts_np.ndim == 1:
                ts_np = ts_np[None, :]  # single channel
            elif ts_np.ndim == 3:
                ts_np = ts_np.squeeze(0)

            fig, ax = plt.subplots(figsize=(6, 2))
            for c in range(ts_np.shape[0]):
                ax.plot(ts_np[c], alpha=0.1, linewidth=0.2)
            ax.set_title(title)
            ax.set_xlabel("timestep")
            ax.set_ylabel("value")
            ax.set_xlim(0, ts_np.shape[-1] - 1)
            ax.grid(False)
            return fig

        if isinstance(inputs, tuple) or isinstance(inputs, list):
            inputs = inputs[0]
        if isinstance(outputs, tuple) or isinstance(outputs, list):
            outputs = outputs[0]
        if isinstance(targets, tuple) or isinstance(targets, list):
            targets = targets[0]

        if not torch.is_tensor(outputs):
            # Some models (e.g., flow-based) return dictionaries instead of raw tensors.
            # Skip logging in that case to avoid attribute errors.
            return

        if self.logger is None:  # pragma: no cover – logger can be disabled
            return

        writer = self.logger.experiment  # type: ignore[attr-defined]
        if writer is None:
            return

        if inputs.ndim == 4:
            inputs = rearrange(inputs, "b h w t -> b (h w) t")
            outputs = rearrange(outputs, "b h w t -> b (h w) t")
            targets = rearrange(targets, "b h w t -> b (h w) t")

        bsz = inputs.shape[0]
        n_samples = min(self._n_log_samples, bsz)
        # Randomly select indices
        idx = torch.randperm(bsz)[:n_samples]

        global_step = self.global_step  # use global_step for proper ordering

        for i, j in enumerate(idx):
            writer.add_histogram(
                f"{stage}/input_sample_{i}", inputs[j].detach().cpu(), global_step
            )
            writer.add_histogram(
                f"{stage}/output_sample_{i}", outputs[j].detach().cpu(), global_step
            )
            writer.add_histogram(
                f"{stage}/target_sample_{i}", targets[j].detach().cpu(), global_step
            )

            # Create and log figures
            fig_in = _make_ts_figure(inputs[j], title="Input")
            writer.add_figure(f"{stage}/input_sample_{i}_img", fig_in, global_step)
            plt.close(fig_in)

            fig_out = _make_ts_figure(outputs[j], title="Output")
            writer.add_figure(f"{stage}/output_sample_{i}_img", fig_out, global_step)
            plt.close(fig_out)

            fig_tgt = _make_ts_figure(targets[j], title="Target")
            writer.add_figure(f"{stage}/target_sample_{i}_img", fig_tgt, global_step)
            plt.close(fig_tgt)

    def _log_psd(
        self,
        inputs: torch.Tensor | tuple[torch.Tensor],
        stage: str,
    ) -> None:
        if not hasattr(self.model, "sample"):
            return

        if isinstance(inputs, tuple) or isinstance(inputs, list):
            inputs = inputs[0]

        # use plot_psd to plot input psd
        fig = plot_psd(inputs[0].detach().cpu().numpy(), self.sfreq, stage)
        self.logger.experiment.add_figure(f"{stage}/input_psd", fig, self.global_step)
        plt.close(fig)

        # use model.sample
        samples = self.model.sample(B=1, device=inputs.device)
        fig = plot_psd(samples[0].detach().cpu().numpy(), self.sfreq, stage)
        self.logger.experiment.add_figure(f"{stage}/sampled_psd", fig, self.global_step)
        plt.close(fig)

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, stage: str
    ):
        inputs, targets = batch
        logits = self.model(inputs)

        if self.postprocessor is not None and torch.is_tensor(logits):
            inputs, logits, targets = self.postprocessor(inputs, logits, targets)

        loss = self.loss(logits, targets, model=self.model)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        for name, metric in self.loss.metrics.items():
            self.log(f"{stage}_{name}", metric(logits, targets), prog_bar=True)

        # Log a few training samples once per epoch (first batch)
        if self.log_samples and batch_idx == 0:
            self._log_random_samples(inputs, logits, targets, stage=stage)

        if self.log_psd and batch_idx == 0:
            self._log_psd(inputs, stage=stage)

        # log learning rate
        self.log("lr", self.optimizers().param_groups[0]["lr"])

        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Loss value
        """
        return self._step(batch, batch_idx, stage="train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """
        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Loss value
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


class DatasetEpochCallback(pl.Callback):
    """Calls a dataset-provided epoch hook at train epoch boundaries.

    Supports datasets that expose either `set_epoch(int)` for cross-worker
    coordination via shared state, or a simpler `on_epoch_start(epoch)` method.
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
