import torch


def assert_future_grad_zero(x: torch.Tensor, t_split: int, atol: float = 1e-7):
    assert x.grad is not None, "input gradients not computed"
    future_grad = x.grad[..., t_split:]
    assert torch.allclose(future_grad, torch.zeros_like(future_grad), atol=atol)
