import torch
from torch import Tensor


@torch.inference_mode()
def sample(
    logits: Tensor,
    strategy: str = "argmax",
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> Tensor:
    """Sample from *logits* according to *strategy* (last dim = vocab).

    Args:
        logits: Tensor of shape (B, C, Q) containing the logits for the next token

    Returns:
        Tensor of shape (B, C) containing the sampled tokens
    """

    temperature = temperature if temperature > 0.0 else 1.0

    if strategy == "argmax":
        return logits.argmax(dim=-1)

    probs = torch.softmax(logits / temperature, dim=-1)

    if strategy == "roulette":
        flat = probs.view(-1, probs.size(-1))
        return torch.multinomial(flat, 1).view(logits.shape[:-1])

    if strategy == "top_k":
        k = min(top_k, probs.size(-1))
        vals, idx = torch.topk(probs, k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True)
        samp = torch.multinomial(vals.view(-1, k), 1).view(*logits.shape[:-1])
        return torch.gather(idx, -1, samp.unsqueeze(-1)).squeeze(-1)

    if strategy == "top_p":
        sorted_p, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        mask = cum > top_p
        mask[..., 0] = False  # keep at least one
        sorted_p[mask] = 0.0
        sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
        samp = torch.multinomial(sorted_p.view(-1, sorted_p.size(-1)), 1).view(
            *logits.shape[:-1]
        )
        return torch.gather(sorted_idx, -1, samp.unsqueeze(-1)).squeeze(-1)

    raise ValueError(f"Unknown sampling strategy '{strategy}'")
