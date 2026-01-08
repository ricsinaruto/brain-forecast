from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn
import torch

from ephys_gpt.models.tokenizers.ibq import IBQMEGTokenizer
from ephys_gpt.losses.ibq import VQLPIPSWithDiscriminator
from .lightning import LitModel


class IBQLightning(LitModel):
    """Lightning module for IBQ tokenizer with manual GAN optimization."""

    def __init__(
        self,
        *args,
        model_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        model_class: nn.Module,
        loss_class: nn.Module,
        trainer_cfg: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(
            model_class=IBQMEGTokenizer,
            loss_class=VQLPIPSWithDiscriminator,
            model_cfg=model_cfg,
            loss_cfg=loss_cfg,
            trainer_cfg={},
        )
        self.loss_fn = self.loss
        self.lr = trainer_cfg["lr"]
        self.betas = trainer_cfg["betas"]
        self.automatic_optimization = False

    def get_input(self, batch):
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        if batch.dim() == 4:
            batch = batch.unsqueeze(1)
        return batch.float()

    @staticmethod
    def _flatten_video(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            b, c, t, h, w = x.shape
            return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        return x

    def training_step(self, batch, batch_idx):
        x = self.get_input(batch)
        output = self.model(x)
        xrec = output.recon
        codebook_loss = output.codebook_loss

        opt_gen, opt_disc = self.optimizers()

        flat_x = self._flatten_video(x)
        flat_xrec = self._flatten_video(xrec)

        # discriminator step first
        disc_loss, log_disc = self.loss_fn(
            codebook_loss,
            flat_x.detach(),
            flat_xrec.detach(),
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=None,
            split="train",
        )
        if not disc_loss.requires_grad:
            disc_loss = disc_loss + 0.0 * sum(
                p.sum() for p in self.loss_fn.discriminator.parameters()
            )
        opt_disc.zero_grad()
        self.manual_backward(disc_loss)
        opt_disc.step()

        # generator step
        gen_loss, log_gen = self.loss_fn(
            codebook_loss,
            flat_x,
            flat_xrec,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.model.get_last_layer(),
            split="train",
        )
        if not gen_loss.requires_grad:
            gen_loss = gen_loss + 0.0 * sum(p.sum() for p in self.model.parameters())
        opt_gen.zero_grad()
        self.manual_backward(gen_loss)
        opt_gen.step()

        log_all = {}
        log_all.update({k: v for k, v in log_disc.items()})
        log_all.update({k: v for k, v in log_gen.items()})
        self.log_dict(log_all, prog_bar=True, on_step=True, on_epoch=False)
        return {"loss": gen_loss.detach()}

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch)
        output = self.model(x)
        xrec = output.recon
        codebook_loss = output.codebook_loss

        flat_x = self._flatten_video(x)
        flat_xrec = self._flatten_video(xrec)

        # Evaluate generator-side loss only (no discriminator update in val).
        gen_loss, log_gen = self.loss_fn(
            codebook_loss,
            flat_x,
            flat_xrec,
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.model.get_last_layer(),
            split="val",
        )

        self.log_dict(
            {f"val/{k.split('/', 1)[-1]}": v for k, v in log_gen.items()},
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("val_loss", gen_loss, prog_bar=True, on_step=False, on_epoch=True)
        return gen_loss

    def configure_optimizers(self):
        opt_gen = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.lr,
            betas=self.betas,
        )
        opt_disc = torch.optim.Adam(
            self.loss_fn.discriminator.parameters(),
            lr=self.lr,
            betas=self.betas,
        )
        return [opt_gen, opt_disc]
