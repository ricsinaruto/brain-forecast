from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CosmosTokenizerLoss(nn.Module):
    """Reconstruction + quantization loss for the Cosmos tokenizer."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        quant_weight: float = 1.0,
        codebook_size: int | None = None,
    ) -> None:
        super().__init__()
        self.recon_weight = float(recon_weight)
        self.quant_weight = float(quant_weight)
        self.codebook_size = codebook_size
        self.metrics = {
            "recon_mse": self._recon_metric,
            "quant_loss": self._quant_metric,
            "code_perplexity": self._code_perplexity,
        }

    def _recon_metric(
        self, outputs: Tensor | dict | tuple, targets: Tensor, **_
    ) -> Tensor:
        recon = outputs[0]
        return F.mse_loss(recon, targets)

    def _quant_metric(
        self, outputs: Tensor | dict | tuple, targets: Tensor, **_
    ) -> Tensor:
        q = outputs[1]["quant_loss"]
        if torch.is_tensor(q):
            return q.detach()
        device = targets.device if torch.is_tensor(targets) else None
        return torch.as_tensor(q, device=device)

    def _code_perplexity(
        self, outputs: Tensor | dict | tuple, targets: Tensor, **_
    ) -> Tensor:
        del targets
        indices = outputs[1]["indices"]

        flat = indices.reshape(-1).long()
        if flat.numel() == 0:
            return torch.tensor(float("nan"), device=indices.device)

        counts = torch.bincount(flat, minlength=self.codebook_size).float()
        total = counts.sum()
        if total <= 0:
            return torch.tensor(float("nan"), device=indices.device)

        probs = counts / total
        entropy = -(probs * torch.log(probs + 1e-6)).sum()
        return torch.exp(entropy).detach()

    def forward(
        self, outputs: Tensor | dict | tuple, targets: Tensor, **kwargs
    ) -> Tensor:
        del kwargs

        recon, quant_loss = outputs[0], outputs[1]["quant_loss"]

        recon_loss = F.mse_loss(recon, targets)
        total = self.recon_weight * recon_loss + self.quant_weight * quant_loss
        return total
