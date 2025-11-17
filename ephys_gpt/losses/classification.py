import torch
from torch import nn

from ..metrics import accuracy, top_k_accuracy, f1_score, mse_loss


class CrossEntropy(nn.Module):
    def __init__(
        self,
        label_smoothing: float = 0.0,
        l1: float = 0.0,
        l2: float = 0.0,
        use_f1: bool = False,
        use_mse: bool = True,
        half_window: bool = False,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.l1 = l1
        self.l2 = l2
        self.half_window = half_window

        self.metrics = {"acc": accuracy}

        if use_f1:
            self.metrics["f1"] = f1_score

        if use_mse:
            self.metrics["mse"] = mse_loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Cross entropy supporting soft targets and regularisation."""
        mask = None

        if isinstance(targets, tuple) or isinstance(targets, list):
            targets = targets[0]

        if self.half_window:
            targets = targets[..., targets.shape[-1] // 2 :]
            logits = logits[..., logits.shape[-2] // 2 :, :]

        if targets.dim() == logits.dim():
            loss = (-targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        else:
            if mask is not None:
                mask = mask.reshape(-1, 1).expand(-1, targets.size(-1)).reshape(-1)
                logits = logits.view(-1, logits.size(-1))[mask]
                targets = targets.view(-1)[mask]

            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                weight=weight,
                label_smoothing=self.label_smoothing,
            )

        if model is not None and self.l1 > 0:
            l1_pen = sum(p.abs().sum() for p in model.parameters())
            loss = loss + self.l1 * l1_pen

        return loss


class CrossEntropyMasked(CrossEntropy):
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        targets, mask = targets

        new_logits = []
        new_targets = []
        for i in range(logits.shape[0]):
            new_logits.append(logits[i, mask[i]])
            new_targets.append(targets[i, mask[i]])

        logits = torch.cat(new_logits, dim=0)
        targets = torch.cat(new_targets, dim=0)
        return super().forward(logits, targets, model)


class CrossEntropyWeighted(CrossEntropy):
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        weight = None
        if isinstance(targets, tuple) or isinstance(targets, list):
            targets, weight = targets[0], targets[1][0]

        return super().forward(logits, targets, model, weight)


class CrossEntropyBalanced(CrossEntropy):
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        if isinstance(targets, tuple) or isinstance(targets, list):
            targets, weight = targets[0], targets[1][0]
            logits = logits + weight
        return super().forward(logits, targets, model)


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
