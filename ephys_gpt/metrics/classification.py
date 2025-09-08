import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    preds = logits.argmax(dim=-1)
    # Support soft targets by converting to hard indices via argmax
    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean()


def top_k_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> torch.Tensor:
    """Compute top-k accuracy. Supports soft targets."""
    topk = logits.topk(k, dim=-1).indices
    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)
    correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1).float()
    return correct.mean()


def f1_score(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute micro F1 score. Supports soft targets."""
    preds = logits.argmax(dim=-1)
    if targets.dim() == logits.dim():
        targets = targets.argmax(dim=-1)
    num_classes = logits.size(-1)
    tp = torch.zeros(1, device=logits.device)
    fp = torch.zeros(1, device=logits.device)
    fn = torch.zeros(1, device=logits.device)
    for c in range(num_classes):
        tp += ((preds == c) & (targets == c)).sum()
        fp += ((preds == c) & (targets != c)).sum()
        fn += ((preds != c) & (targets == c)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return 2 * precision * recall / (precision + recall + 1e-8)
