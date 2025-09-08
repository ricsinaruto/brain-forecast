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
        datasets: dict,
        model_cfg: dict,
        loss_cfg: dict,
        trainer_cfg: dict,
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
            pass

        self.lr = trainer_cfg["lr"]
        self.weight_decay = trainer_cfg["weight_decay"]
        self.log_samples = trainer_cfg.get("log_samples", False)
        self.sfreq = trainer_cfg.get("sfreq", 200)
        self.log_psd = trainer_cfg.get("log_psd", False)
        self.dataset = datasets.train

        # Create model and loss instances
        self.model = model_class(**model_cfg)

        # compile the model if requested
        if trainer_cfg.get("compile", False):
            self.model = torch.compile(self.model)

        self.loss = loss_class(**loss_cfg)

        # Number of random samples to log
        self._n_log_samples = trainer_cfg.get("n_log_samples", 3)

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

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """
        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Loss value
        """
        inputs, targets = batch
        logits = self.model(inputs)

        if hasattr(self.dataset, "postprocess"):
            inputs, logits, targets = self.dataset.postprocess(inputs, logits, targets)

        loss = self.loss(logits, targets, model=self.model)
        self.log("train_loss", loss, prog_bar=True)
        for name, metric in self.loss.metrics.items():
            self.log(f"train_{name}", metric(logits, targets), prog_bar=True)

        # Log a few training samples once per epoch (first batch)
        if self.log_samples and batch_idx == 0:
            self._log_random_samples(inputs, logits, targets, stage="train")
        return loss

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
        inputs, targets = batch
        logits = self.model(inputs)

        if hasattr(self.dataset, "postprocess"):
            inputs, logits, targets = self.dataset.postprocess(inputs, logits, targets)

        loss = self.loss(logits, targets, model=self.model)
        self.log("val_loss", loss, prog_bar=True)
        for name, metric in self.loss.metrics.items():
            self.log(f"val_{name}", metric(logits, targets), prog_bar=True)

        # Log a few validation samples once per validation epoch (first batch)
        if self.log_samples and batch_idx == 0:
            self._log_random_samples(inputs, logits, targets, stage="val")

        if self.log_psd and batch_idx == 0:
            self._log_psd(inputs, stage="val")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
