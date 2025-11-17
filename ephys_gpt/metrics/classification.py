import torch
from torchmetrics import F1Score
import torch.nn.functional as F

from ..utils.quantizers import mulaw_inv_torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    # Support soft targets by converting to hard indices via argmax
    if isinstance(targets, tuple) or isinstance(targets, list):
        targets = targets[0]

    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean()


def top_k_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """Compute top-k accuracy. Supports soft targets."""
    topk = logits.topk(k, dim=-1).indices
    if isinstance(targets, tuple) or isinstance(targets, list):
        targets = targets[0]

    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)
    correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1).float()
    return correct.mean()


def f1_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if isinstance(targets, tuple) or isinstance(targets, list):
        targets = targets[0]

    # reshape to 2D
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)

    f1_macro = F1Score(
        task="multiclass", average="macro", num_classes=logits.size(-1)
    ).to(logits.device)

    return f1_macro(logits, targets)


def mse_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    # Support soft targets by converting to hard indices via argmax
    if isinstance(targets, tuple) or isinstance(targets, list):
        targets = targets[0]

    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)

    # convert to continuous with inverse mulaw
    preds = mulaw_inv_torch(preds)
    targets = mulaw_inv_torch(targets)

    return F.mse_loss(preds, targets)