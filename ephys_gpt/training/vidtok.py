from contextlib import contextmanager
from typing import Any, Dict, Tuple, Optional, Callable
from pathlib import Path
import torch
import pytorch_lightning as pl
from einops import rearrange
import torch.nn as nn
import matplotlib.pyplot as plt

from ephys_gpt.layers.vidtok.ema import LitEma
from ephys_gpt.layers.vidtok.util import default, get_obj_from_str
from ephys_gpt.layers.vidtok.model import EncoderCausal3DPadding, DecoderCausal3DPadding
from ephys_gpt.layers.vidtok.losses import GeneralLPIPSWithDiscriminator
from ephys_gpt.layers.vidtok.regularizers import FSQRegularizer


class Vidtok(nn.Module):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
    ):
        super().__init__()
        self.encoder = EncoderCausal3DPadding(**model_cfg["encoder"])
        self.decoder = DecoderCausal3DPadding(**model_cfg["encoder"])
        self.loss = GeneralLPIPSWithDiscriminator(**loss_cfg)
        self.regularization = FSQRegularizer(**model_cfg["regularizer"])

    def _empty_causal_cached(self, parent):
        for name, module in parent.named_modules():
            if hasattr(module, "causal_cache"):
                module.causal_cache = None

    def _set_first_chunk(self, is_first_chunk=True):
        for module in self.modules():
            if hasattr(module, "is_first_chunk"):
                module.is_first_chunk = is_first_chunk

    def encode(self, x: Any, return_reg_log: bool = False, global_step: int = 0) -> Any:
        self._empty_causal_cached(self.encoder)
        self._set_first_chunk(True)

        z = self.encoder(x)
        z, reg_log = self.regularization(z, n_steps=global_step // 2)

        if return_reg_log:
            return z, reg_log
        return z

    def indices_to_latent(self, token_indices: torch.Tensor) -> torch.Tensor:
        assert token_indices.dim() == 4, "token_indices should be of shape (b, t, h, w)"
        b, t, h, w = token_indices.shape
        token_indices = token_indices.unsqueeze(-1).reshape(b, -1, 1)
        codes = self.regularization.indices_to_codes(token_indices)
        codes = codes.permute(0, 2, 3, 1).reshape(b, codes.shape[2], -1)
        z = self.regularization.project_out(codes)
        return z.reshape(b, t, h, w, -1).permute(0, 4, 1, 2, 3)

    def decode(self, z: Any, decode_from_indices: bool = False) -> torch.Tensor:
        if decode_from_indices:
            z = self.indices_to_latent(z)
        self._empty_causal_cached(self.decoder)
        self._set_first_chunk(True)

        x = self.decoder(z)
        return x

    def forward(
        self, x: Any, global_step: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.encoder.fix_encoder:
            with torch.no_grad():
                z, reg_log = self.encode(
                    x, return_reg_log=True, global_step=global_step
                )
        else:
            z, reg_log = self.encode(x, return_reg_log=True, global_step=global_step)
        dec = self.decode(z)
        if dec.shape[2] != x.shape[2]:
            dec = dec[:, :, -x.shape[2] :, ...]
        return dec, z, reg_log

    def get_autoencoder_params(self) -> list:
        params = (
            list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
            + list(filter(lambda p: p.requires_grad, self.decoder.parameters()))
            + list(self.regularization.get_trainable_parameters())
            + list(self.loss.get_trainable_autoencoder_parameters())
        )
        return params


class Metrics:
    def __init__(self):
        self.metrics = {"pcc": self.get_pcc, "l1": self.compute_l1_loss}

    def get_input(self, batch: tuple) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        return batch

    def get_pcc(self, rec: torch.Tensor, raw: torch.Tensor):
        rec = self.get_input(rec)
        if rec.ndim == 2:
            rec = rec.unsqueeze(0)
        if raw.ndim == 2:
            raw = raw.unsqueeze(0)
        # B T C
        T = rec.shape[1]

        x = rearrange(rec, "B T C ->(B C) 1 T")
        y = rearrange(raw, "B T C -> (B C) 1 T")
        c = (
            (x - x.mean(dim=-1, keepdim=True))
            @ ((y - y.mean(dim=-1, keepdim=True)).transpose(1, 2))
            * (1.0 / (T - 1))
        ).squeeze()
        sigma = (torch.std(x, dim=-1) * torch.std(y, dim=-1)).squeeze() + 1e-6
        return (c / sigma).mean()

    def compute_l1_loss(self, rec, raw):
        rec = self.get_input(rec)
        l1_distance = torch.abs(rec - raw)
        return torch.mean(l1_distance)

    def __call__(self, rec: torch.Tensor, raw: torch.Tensor, model=None):
        return self.compute_l1_loss(rec, raw)


class VidtokLightning(pl.LightningModule):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        trainer_cfg: Dict[str, Any],
        postprocessor: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__()
        # Save hyperparameters for checkpoint loading
        # (ignore postprocessor as it's not serializable)
        self.save_hyperparameters(ignore=["postprocessor"])
        self.postprocessor = postprocessor
        self.loss = Metrics()

        self.model = Vidtok(model_cfg, loss_cfg)
        self.optimizer_config = default(
            trainer_cfg["optimizer"], {"target": "torch.optim.AdamW"}
        )
        self.lr_g_factor = trainer_cfg.get("lr_g_factor", 1.0)
        self.learning_rate = trainer_cfg["lr"]

        self.automatic_optimization = False
        self.ema_decay = trainer_cfg.get("ema_decay", None)
        self.use_ema = self.ema_decay is not None
        if self.use_ema:
            self.model_ema = LitEma(self, decay=self.ema_decay)

        if trainer_cfg.get("compile", False):
            self.model = torch.compile(self.model)

    def on_train_batch_end(self, *args, **kwargs):
        # for EMA computation
        if self.use_ema:
            self.model_ema(self)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())

    def instantiate_optimizer_from_config(self, params, lr, cfg):
        return get_obj_from_str(cfg["target"])(
            params, lr=lr, **cfg.get("params", dict())
        )

    def get_input(self, batch: tuple) -> torch.Tensor:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        return batch

    def get_autoencoder_params(self) -> list:
        return self.model.get_autoencoder_params()

    def get_discriminator_params(self) -> list:
        params = list(self.model.loss.get_trainable_parameters())
        return params

    def get_last_layer(self):
        return self.model.decoder.get_last_layer()

    def training_step(self, batch, batch_idx) -> Any:
        x = self.get_input(batch)

        if x.ndim == 4:
            x = x.unsqueeze(2)

        xrec, z, regularization_log = self.model(x, global_step=self.global_step)

        if x.ndim == 5 and xrec.ndim == 4:
            xrec = xrec.unsqueeze(2)

        opt_g, opt_d = self.optimizers()

        # autoencode loss
        self.toggle_optimizer(opt_g)
        aeloss, log_dict_ae = self.model.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_g.zero_grad()
        self.manual_backward(aeloss)

        # gradient clip
        torch.nn.utils.clip_grad_norm_(self.get_autoencoder_params(), 20.0)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # discriminator loss
        self.toggle_optimizer(opt_d)
        discloss, log_dict_disc = self.model.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        opt_d.zero_grad()
        self.manual_backward(discloss)
        torch.nn.utils.clip_grad_norm_(self.get_discriminator_params(), 20.0)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # logging
        # log_dict.update(log_dict_ae)
        # log_dict.update(log_dict_disc)
        # self.log("train/rec_loss", log_dict_ae["train/rec_loss"], prog_bar=True)

        x_pixel, xrec_pixel = self.postprocessor(x, xrec)
        x_gaussian, xrec_gaussian = self.postprocessor(x, xrec, gaussian=True)

        pcc = self.loss.get_pcc(x_pixel, xrec_pixel)
        pcc_gaussian = self.loss.get_pcc(x_gaussian, xrec_gaussian)
        l1 = self.loss.compute_l1_loss(x_pixel, xrec_pixel)

        self.log("train/pcc_gaussian", pcc_gaussian, prog_bar=True)
        self.log("train/pcc", pcc, prog_bar=True)
        self.log("train/l1", l1, prog_bar=True)
        self.log("train/loss", log_dict_ae["train/total_loss"], prog_bar=True)
        self.log("train/d_loss", log_dict_ae["train/d_loss"], prog_bar=True)
        self.log("train/nll_loss", log_dict_ae["train/nll_loss"], prog_bar=True)

    def validation_step(self, batch, batch_idx) -> Dict:
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, postfix="_ema")
            log_dict.update(log_dict_ema)

    def _validation_step(self, batch, batch_idx, postfix="") -> Dict:
        x = self.get_input(batch)

        if x.ndim == 4:
            x = x.unsqueeze(2)

        xrec, z, regularization_log = self.model(x, global_step=self.global_step)

        if x.ndim == 5 and xrec.ndim == 4:
            xrec = xrec.unsqueeze(2)

        aeloss, log_dict_ae = self.model.loss(
            regularization_log,
            x,
            xrec,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        discloss, log_dict_disc = self.model.loss(
            regularization_log,
            x,
            xrec,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val" + postfix,
        )

        x, xrec = self.postprocessor(x, xrec)
        pcc = self.loss.get_pcc(x, xrec)
        l1 = self.loss.compute_l1_loss(x, xrec)

        self.log("val_loss", log_dict_ae[f"val{postfix}/rec_loss"], prog_bar=True)
        self.log("val/pcc", pcc, prog_bar=True)
        self.log("val/l1", l1, prog_bar=True)

        log_dict_ae.update(log_dict_disc)
        return log_dict_ae

    def configure_optimizers(self) -> Any:
        ae_params = self.get_autoencoder_params()
        disc_params = self.get_discriminator_params()

        opt_ae = self.instantiate_optimizer_from_config(
            ae_params,
            default(self.lr_g_factor, 1.0) * self.learning_rate,
            self.optimizer_config,
        )
        opt_disc = self.instantiate_optimizer_from_config(
            disc_params, self.learning_rate, self.optimizer_config
        )

        return [opt_ae, opt_disc], []


class ImageSaverCallback(pl.Callback):
    """Saves input and reconstructed images to disk at the end of each epoch."""

    def __init__(self, save_dir: str, num_samples: int = 4) -> None:
        super().__init__()
        self.save_dir = Path(save_dir) / "reconstructions"
        self.num_samples = num_samples
        self._cached_batch: Optional[torch.Tensor] = None

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # Cache first batch of validation data for reconstruction visualization
        if batch_idx == 0 and self._cached_batch is None:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            # Store a few samples (detached, on CPU to save GPU memory)
            self._cached_batch = x[: self.num_samples].detach().clone()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: "VidtokLightning"
    ) -> None:
        if self._cached_batch is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)
        epoch = trainer.current_epoch

        # Move batch to model device and run reconstruction
        x = self._cached_batch.to(pl_module.device)
        if x.ndim == 4:
            x = x.unsqueeze(2)

        with torch.no_grad():
            xrec, _, _ = pl_module.model(x, global_step=pl_module.global_step)

        if x.ndim == 5 and xrec.ndim == 4:
            xrec = xrec.unsqueeze(2)

        # Handle temporal dim: take middle frame if 5D
        if x.ndim == 5:
            t_mid = x.shape[2] // 2
            x_img = x[:, :, t_mid, :, :]  # (B, C, H, W)
            xrec_img = xrec[:, :, t_mid, :, :]
        else:
            x_img = x
            xrec_img = xrec

        # Move to CPU and convert to numpy
        x_np = x_img.cpu().numpy()
        xrec_np = xrec_img.cpu().numpy()

        # Create figure with input/reconstruction pairs
        n = min(self.num_samples, x_np.shape[0])
        fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
        if n == 1:
            axes = axes.reshape(2, 1)

        for i in range(n):
            # Input image - handle single or multi-channel
            img_in = x_np[i]
            img_rec = xrec_np[i]

            # If single channel, squeeze; else take first channel or average
            if img_in.shape[0] == 1:
                img_in = img_in[0]
                img_rec = img_rec[0]
            else:
                # Average over channels for visualization
                img_in = img_in.mean(axis=0)
                img_rec = img_rec.mean(axis=0)

            axes[0, i].imshow(img_in, cmap="RdBu_r", aspect="auto")
            axes[0, i].set_title(f"Input {i}")
            axes[0, i].axis("off")

            axes[1, i].imshow(img_rec, cmap="RdBu_r", aspect="auto")
            axes[1, i].set_title(f"Reconstruction {i}")
            axes[1, i].axis("off")

        fig.suptitle(f"Epoch {epoch}")
        plt.tight_layout()

        save_path = self.save_dir / f"epoch_{epoch:04d}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Clear cached batch so we get fresh samples next epoch
        self._cached_batch = None
