import torch
from torch import nn

from ..metrics import accuracy, top_k_accuracy, f1_score


class CrossEntropy(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, l1: float = 0.0, l2: float = 0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.l1 = l1
        self.l2 = l2

        self.metrics = {
            "acc": accuracy,
            "top5": top_k_accuracy,
            "f1": f1_score,
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """Cross entropy supporting soft targets and regularisation."""
        mask = None

        if targets.dim() == logits.dim():
            loss = (-targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        else:
            if mask is not None:
                mask = mask.reshape(-1, 1).expand(-1, targets.size(-1)).reshape(-1)
                logits = logits.view(-1, logits.size(-1))[mask]
                targets = targets.view(-1)[mask]

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                label_smoothing=self.label_smoothing,
            )

        if model is not None and self.l1 > 0:
            l1_pen = sum(p.abs().sum() for p in model.parameters())
            loss = loss + self.l1 * l1_pen

        return loss


class CrossEntropyWithCodes(CrossEntropy):
    """
    Cross entropy loss with codes returned by model instead of dataloader.
    """

    def __init__(self, label_smoothing: float = 0.0, l1: float = 0.0, l2: float = 0.0):
        super().__init__(label_smoothing, l1, l2)
        self.metrics = {
            "acc": lambda logits, targets: accuracy(logits[0], logits[1]),
            "top5": lambda logits, targets: top_k_accuracy(logits[0], logits[1]),
        }

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        return super().forward(logits[0], logits[1], model)
