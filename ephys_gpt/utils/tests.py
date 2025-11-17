import torch
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np


def assert_future_grad_zero(x: torch.Tensor, t_split: int, atol: float = 1e-7):
    assert x.grad is not None, "input gradients not computed"
    future_grad = x.grad[..., t_split:]
    assert torch.allclose(future_grad, torch.zeros_like(future_grad), atol=atol)


def make_dummy_session(
    root: str,
    session: str,
    *,
    data: Optional[np.ndarray] = None,
    C: int = 272,
    T: int = 64,
    sfreq: float = 200,
    ch_names: Optional[Sequence[str]] = None,
    pos_2d: Optional[Sequence[Tuple[float, float]]] = None,
    ch_types: Optional[Sequence[Union[int, str]]] = None,
    chunk_idx: int = 0,
):
    os.makedirs(os.path.join(root, session), exist_ok=True)
    if data is None:
        data = np.random.randn(C, T).astype(np.float32)
    C = data.shape[0]
    if ch_names is None:
        ch_names = [f"ch{i:03d}" for i in range(C)]
    if pos_2d is None:
        pos_2d = [(float(i % 4), float(i // 4)) for i in range(C)]

    chunk = {
        "data": data.astype(np.float32, copy=False),
        "ch_names": list(ch_names),
        "pos_2d": np.asarray(pos_2d, dtype=np.float32),
        "sfreq": sfreq,
    }
    if ch_types is not None:
        chunk["ch_types"] = list(ch_types)

    np.save(os.path.join(root, session, f"{chunk_idx}.npy"), chunk)
