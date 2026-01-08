from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ephys_gpt.layers.ibq.losses.lpips import LPIPS
except Exception:  # pragma: no cover - optional dependency
    LPIPS = None
from ephys_gpt.layers.ibq.discriminator.model import NLayerDiscriminator, weights_init


def adopt_weight(
    weight: float, global_step: int, threshold: int = 0, value: float = 0.0
) -> float:
    return weight if global_step >= threshold else value


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


def vanilla_d_loss(logits_real, logits_fake):
    return 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )


def _sigmoid_cross_entropy_with_logits(labels, logits):
    zeros = torch.zeros_like(logits, dtype=logits.dtype)
    condition = logits >= zeros
    relu_logits = torch.where(condition, logits, zeros)
    neg_abs_logits = torch.where(condition, -logits, logits)
    return relu_logits - logits * labels + torch.log1p(torch.exp(neg_abs_logits))


def non_saturate_gen_loss(logits_fake):
    B, _, _, _ = logits_fake.shape
    logits_fake = logits_fake.reshape(B, -1).mean(dim=-1)
    return torch.mean(
        _sigmoid_cross_entropy_with_logits(
            labels=torch.ones_like(logits_fake), logits=logits_fake
        )
    )


def non_saturate_discriminator_loss(logits_real, logits_fake):
    B, _, _, _ = logits_fake.shape
    logits_real = logits_real.reshape(B, -1).mean(dim=-1)
    logits_fake = logits_fake.reshape(B, -1).mean(dim=-1)
    real_loss = _sigmoid_cross_entropy_with_logits(
        labels=torch.ones_like(logits_real), logits=logits_real
    )
    fake_loss = _sigmoid_cross_entropy_with_logits(
        labels=torch.zeros_like(logits_fake), logits=logits_fake
    )
    return real_loss.mean() + fake_loss.mean()


class LeCAM_EMA:
    def __init__(self, init: float = 0.0, decay: float = 0.999):
        self.logits_real_ema = init
        self.logits_fake_ema = init
        self.decay = decay

    def update(self, logits_real, logits_fake):
        self.logits_real_ema = self.logits_real_ema * self.decay + torch.mean(
            logits_real
        ).item() * (1 - self.decay)
        self.logits_fake_ema = self.logits_fake_ema * self.decay + torch.mean(
            logits_fake
        ).item() * (1 - self.decay)


def lecam_reg(real_pred, fake_pred, lecam_ema: LeCAM_EMA):
    return torch.mean(
        F.relu(real_pred - lecam_ema.logits_fake_ema).pow(2)
    ) + torch.mean(F.relu(lecam_ema.logits_real_ema - fake_pred).pow(2))


class VQLPIPSWithDiscriminator(nn.Module):
    """Combines reconstruction, perceptual, codebook, and adversarial losses for IBQ."""

    def __init__(
        self,
        disc_start: int,
        codebook_weight: float = 1.0,
        pixelloss_weight: float = 1.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        use_actnorm: bool = False,
        disc_ndf: int = 64,
        disc_loss: str = "hinge",
        gen_loss_weight: float | None = None,
        lecam_loss_weight: float | None = None,
        quant_loss_weight: float = 1.0,
        entropy_loss_weight: float = 1.0,
    ):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        self.gen_loss_weight = gen_loss_weight
        self.lecam_loss_weight = lecam_loss_weight
        if self.lecam_loss_weight is not None:
            self.lecam_ema = LeCAM_EMA()

        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels,
            n_layers=disc_num_layers,
            use_actnorm=use_actnorm,
            ndf=disc_ndf,
        ).apply(weights_init)

        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight

        self.quant_loss_weight = quant_loss_weight
        self.entropy_loss_weight = entropy_loss_weight

        if perceptual_weight > 0:
            if LPIPS is None:
                raise ImportError("torchvision is required for LPIPS perceptual loss.")
            self.perceptual_loss = LPIPS().eval()
        else:
            self.perceptual_loss = None

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight * self.discriminator_weight

    def _maybe_expand_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Expand single-channel input to 3 channels for LPIPS/discriminator.

        Note: LPIPS uses ImageNet RGB normalization, so repeating grayscale channels is
        an approximation. This works reasonably in practice but is not strictly correct
        for perceptual similarity on grayscale images.
        """
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        return x

    def forward(
        self,
        codebook_loss,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train",
    ):
        inputs = self._maybe_expand_channels(inputs)
        reconstructions = self._maybe_expand_channels(reconstructions)

        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss.clone()
        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            nll_loss = nll_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0], device=inputs.device)

        nll_loss = torch.mean(nll_loss)

        use_entropy_loss = isinstance(codebook_loss, tuple)

        log = {}
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = non_saturate_gen_loss(logits_fake)
            if self.gen_loss_weight is None:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    d_weight = torch.tensor(0.0, device=nll_loss.device)
            else:
                d_weight = torch.tensor(self.gen_loss_weight, device=nll_loss.device)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )

            g_loss_scaled = d_weight * disc_factor * g_loss
            if not use_entropy_loss:
                codebook_loss_scaled = self.codebook_weight * codebook_loss
            else:
                quant_loss, sample_entropy_loss, avg_entropy_loss, entropy_loss = (
                    codebook_loss
                )
                codebook_loss_scaled = (
                    self.quant_loss_weight * quant_loss
                    + self.entropy_loss_weight * entropy_loss
                )

            loss = nll_loss + g_loss_scaled + codebook_loss_scaled

            log = {
                f"{split}/total_loss": loss.detach(),
                f"{split}/codebook_quant_loss": (
                    codebook_loss_scaled.detach()
                    if torch.is_tensor(codebook_loss_scaled)
                    else torch.tensor(codebook_loss_scaled, device=loss.device)
                ),
                f"{split}/reconstruct_loss": rec_loss.detach().mean(),
                f"{split}/perceptual_loss": p_loss.detach().mean(),
                f"{split}/d_weight": d_weight.detach(),
                f"{split}/disc_factor": torch.tensor(disc_factor, device=loss.device),
                f"{split}/g_loss": g_loss.detach() * disc_factor,
                f"{split}/unsacled_g_loss": g_loss.detach(),
            }

            if use_entropy_loss:
                log[f"{split}/sample_entropy_loss"] = sample_entropy_loss.detach()
                log[f"{split}/avg_entropy_loss"] = avg_entropy_loss.detach()
                log[f"{split}/entropy_loss"] = entropy_loss.detach()
                log[f"{split}/quant_loss"] = quant_loss.detach()
            return loss, log

        # discriminator update
        logits_real = self.discriminator(inputs.contiguous().detach())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())

        if self.perceptual_weight > 0 and self.perceptual_loss is not None:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous().detach()
            )
            p_loss = p_loss.detach()
        else:
            p_loss = torch.tensor([0.0], device=inputs.device)

        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.discriminator_iter_start
        )
        if disc_factor == 0.0:
            d_loss = torch.tensor(0.0, device=inputs.device)
        else:
            d_loss = self.disc_loss(logits_real, logits_fake)
            if self.lecam_loss_weight is not None:
                d_loss = d_loss + self.lecam_loss_weight * lecam_reg(
                    logits_real, logits_fake, self.lecam_ema
                )
                self.lecam_ema.update(logits_real.detach(), logits_fake.detach())

        loss = disc_factor * d_loss
        log = {
            f"{split}/disc_loss": loss.clone().detach(),
            f"{split}/disc_factor": torch.tensor(disc_factor, device=loss.device),
            f"{split}/disc_loss_raw": d_loss.detach(),
        }
        return loss, log


class IBQSimpleLoss(nn.Module):
    """Lightweight reconstruction + codebook loss (single-optimizer fallback)."""

    def __init__(self, recon_weight: float = 1.0, codebook_weight: float = 1.0):
        super().__init__()
        self.recon_weight = recon_weight
        self.codebook_weight = codebook_weight
        self.metrics: dict[str, callable] = {
            "recon": lambda outputs, targets, **_: self._recon(outputs, targets),
            "codebook": lambda outputs, targets, **_: self._codebook(outputs),
        }

    @staticmethod
    def _extract(output_obj):
        # Support both IBQOutput dataclass and tuple outputs
        if hasattr(output_obj, "recon"):
            return output_obj.recon, output_obj.codebook_loss
        if isinstance(output_obj, (tuple, list)) and len(output_obj) >= 2:
            return output_obj[0], output_obj[1]
        raise ValueError("Unsupported IBQ model output.")

    def _recon(self, outputs, targets):
        recon, _ = self._extract(outputs)
        return F.l1_loss(recon, targets)

    def _codebook(self, outputs):
        _, codebook_loss = self._extract(outputs)
        if isinstance(codebook_loss, tuple):
            quant_loss = codebook_loss[0]
        else:
            quant_loss = codebook_loss
        return quant_loss

    def forward(self, outputs, targets, **kwargs):
        recon, codebook_loss = self._extract(outputs)
        if isinstance(codebook_loss, tuple):
            quant_loss = codebook_loss[0]
        else:
            quant_loss = codebook_loss
        return (
            self.recon_weight * F.l1_loss(recon, targets)
            + self.codebook_weight * quant_loss
        )
