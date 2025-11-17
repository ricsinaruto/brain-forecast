import torch
from torch import Tensor
from torch import nn
from typing import Any
import torch.nn.functional as F
import math


class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        # keep consistent interface with CrossEntropy
        self.metrics: dict[str, Any] = {}

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> Tensor:
        """Mean-squared-error loss with a flexible signature.
        Accepts arbitrary additional keyword arguments (ignored) so that it can be
        called in the same way as :class:`CrossEntropy` within the training
        loop, which always forwards the current *model* instance.
        """
        return F.mse_loss(logits, targets).mean()


class NLL(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics: dict[str, Any] = {}

    def forward(self, losses: Tensor, target: Tensor, **kwargs) -> Tensor:
        nll, logdet = losses
        return nll + logdet


class ChronoFlowLoss(nn.Module):
    """Wrapper around ChronoFlowSSM outputs for Lightning training.

    Expects the model forward pass to return a dictionary with at least the
    key ``"nll"`` and optionally a ``"stats"`` sub-dictionary containing
    ``"bits_per_dim"`` and ``"avg_boundary"`` tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.metrics: dict[str, Any] = {
            "bits_per_dim": self._bits_per_dim,
            "avg_boundary": self._avg_boundary,
        }

    def forward(
        self,
        outputs: dict[str, Any],
        targets: Tensor | tuple | None,
        **kwargs,
    ) -> Tensor:
        if not isinstance(outputs, dict) or "nll" not in outputs:
            raise ValueError(
                "ChronoFlowLoss expects the model to return a dict with an 'nll' key."
            )
        return outputs["nll"]

    @staticmethod
    def _bits_per_dim(outputs: dict[str, Any], *_) -> Tensor:
        stats = outputs.get("stats", {})
        val = stats.get("bits_per_dim")
        if val is None:
            return torch.tensor(float("nan"), device=outputs["nll"].device)
        return val.detach() if isinstance(val, torch.Tensor) else torch.as_tensor(
            val, device=outputs["nll"].device
        )

    @staticmethod
    def _avg_boundary(outputs: dict[str, Any], *_) -> Tensor:
        stats = outputs.get("stats", {})
        val = stats.get("avg_boundary")
        if val is None:
            return torch.tensor(float("nan"), device=outputs["nll"].device)
        return val.detach() if isinstance(val, torch.Tensor) else torch.as_tensor(
            val, device=outputs["nll"].device
        )


class VQVAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics: dict[str, Any] = {}

    def forward(self, outputs: Tensor, target: Tensor, **kwargs) -> Tensor:
        x_recon, vq_output = outputs

        recon_loss = F.mse_loss(x_recon, target)
        loss = recon_loss + vq_output["commitment_loss"]
        return loss


class VQNSPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics: dict[str, Any] = {}

    def calculate_rec_loss(self, rec: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(rec, target)

    def forward(
        self,
        inputs: tuple[
            Tensor,
            Tensor,
            Tensor,
            Tensor,
            Tensor,
        ],
        targets: Tensor,
        **kwargs,
    ) -> Tensor:
        xrec, xrec_angle, amplitude, angle, emb_loss = inputs

        rec_loss = self.calculate_rec_loss(xrec, amplitude)
        rec_angle_loss = self.calculate_rec_loss(xrec_angle, angle)

        loss = emb_loss + rec_loss + rec_angle_loss

        return loss


class BrainTokenizerLoss(nn.Module):
    """
    Implements the compound loss used for BrainTokenizer training
    (Eq. 1–5 in Section 2.2, ‘Training the BrainTokenizer’).

    Args
    ----
    amp_phase_norm : bool
        Whether to z-score-normalise amplitude and phase spectra
        before the L1 comparison (often stabilises early training).
    eps : float
        Numerical epsilon to avoid divide-by-zero in PCC.
    """

    def __init__(self, amp_phase_norm: bool = True, eps: float = 1e-8) -> None:
        super().__init__()
        self.amp_phase_norm = amp_phase_norm
        self.eps = eps
        self.beta = 0.25
        self.metrics = {}

    def _fft_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        rFFT along the last dimension → amplitude & wrapped phase
        """
        dim_to_reduce = tuple(range(1, x.ndim))  # all except batch dim 0

        spec = torch.fft.rfft(x, dim=-1)
        amp = spec.abs()
        phase = torch.angle(spec)  # returns values in (-π, π]
        if self.amp_phase_norm:
            amp = (amp - amp.mean(dim=dim_to_reduce, keepdim=True)) / (
                amp.std(dim=dim_to_reduce, keepdim=True) + self.eps
            )
            phase = (phase - phase.mean(dim=dim_to_reduce, keepdim=True)) / (
                phase.std(dim=dim_to_reduce, keepdim=True) + self.eps
            )
        return amp, phase

    def _pcc(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Pearson correlation coefficient along the last dimension,
        averaged over batch & channel dims.
        """
        x_mu = x.mean(dim=-1, keepdim=True)
        y_mu = y.mean(dim=-1, keepdim=True)
        x_cent = x - x_mu
        y_cent = y - y_mu
        cov = (x_cent * y_cent).mean(dim=-1)
        x_std = x_cent.std(dim=-1, unbiased=False) + self.eps
        y_std = y_cent.std(dim=-1, unbiased=False) + self.eps
        pcc = cov / (x_std * y_std)
        return pcc.mean()  # scalar

    def _phase_l1(self, a: Tensor, b: Tensor) -> Tensor:
        diff = torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi
        return diff.abs().mean()

    def forward(
        self,
        inputs: tuple[
            torch.Tensor,
            list[torch.Tensor],
            list[torch.Tensor],
            torch.Tensor,
            torch.Tensor,
        ],
        x_orig: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        x_rec, rvq_residuals, rvq_nearest, _, _ = inputs

        # 1) Time-domain L1
        l_time = F.l1_loss(x_rec, x_orig)

        # 2) Frequency-domain L1 on amplitude & phase spectra
        # amp_o, phase_o = self._fft_features(x_orig)
        # amp_r, phase_r = self._fft_features(x_rec)
        # l_freq = F.l1_loss(amp_r, amp_o) + self._phase_l1(phase_r, phase_o)

        # 3) PCC consistency term
        pcc = self._pcc(x_orig, x_rec)
        l_pcc = torch.exp(-pcc)  # Eq. 3

        # 4) RVQ commitment loss (sum over codebook layers)
        if not (len(rvq_residuals) == len(rvq_nearest)):
            raise ValueError("rvq_residuals and rvq_nearest must be same length")

        l_rvq = 0
        for res, nq in zip(rvq_residuals, rvq_nearest):
            l_rvq += (
                self.beta * (res - nq.detach()).pow(2).mean()
                + (res.detach() - nq).pow(2).mean()
            )

        # Total
        # l_total = l_time + l_freq + l_pcc + l_rvq
        l_total = l_time + l_pcc + l_rvq
        return l_total
