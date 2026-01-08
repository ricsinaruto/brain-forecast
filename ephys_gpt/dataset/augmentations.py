from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "RandomTimeWarp",
    "RandomTimeMask",
    "RandomTimeShift",
    "AdditiveNoise",
    "RandomSpatialWarp",
    "RandomNeighborChannelSwap",
]


class Augmentations:
    def __init__(self, cfg):
        self.augmentations = None
        if cfg is not None:
            self.augmentations = []
            names = cfg.keys()
            for name in names:
                self.augmentations.append(globals()[name](**cfg[name]))

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if self.augmentations is not None and training:
            for augmentation in self.augmentations:
                x = augmentation(x)
        return x


class RandomTimeWarp:
    """Random, monotonic time warping applied along the last dimension (timesteps).

    Works with inputs shaped (C, T) or (H, W, T). For (H, W, T) inputs, the warping is
    shared across all H×W channels for temporal consistency.

    Args:     num_anchors: Number of control points (≥2). Higher → smoother warps.
    strength: Variability of local speed around identity (0 → identity).     p:
    Probability of applying the augmentation.
    """

    def __init__(self, num_anchors: int = 6, strength: float = 0.2, p: float = 1.0):
        if num_anchors < 2:
            raise ValueError("num_anchors must be ≥ 2")
        self.num_anchors = num_anchors
        self.strength = float(strength)
        self.p = float(p)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        if x.ndim == 2:  # (C, T)
            channels, timesteps = x.shape
            x_flat = x
            H, W = 1, channels
        elif x.ndim == 3:  # (H, W, T)
            H, W, timesteps = x.shape
            x_flat = x.reshape(H * W, timesteps)
        else:
            return x

        device = x.device
        dtype = x.dtype

        # Anchor positions on original time axis
        K = self.num_anchors
        xk = torch.linspace(0, timesteps - 1, K, device=device, dtype=torch.float32)

        # Random positive segment lengths around identity, then normalise
        base = torch.ones(K - 1, device=device, dtype=torch.float32)
        noise = (torch.rand(K - 1, device=device) * 2 - 1) * self.strength
        deltas = (base + noise).clamp_min(1e-3)
        yk = torch.cat(
            [
                torch.zeros(1, device=device),
                torch.cumsum(deltas, dim=0),
            ]
        )
        yk = (yk / yk[-1]) * (timesteps - 1)

        # Piecewise-linear mapping t -> s(t)
        t_grid = torch.arange(timesteps, device=device, dtype=torch.float32)

        # segment index for each t (0..K-2)
        seg = torch.bucketize(t_grid, xk[1:-1])
        xk0 = xk[seg]
        xk1 = xk[seg + 1]
        yk0 = yk[seg]
        yk1 = yk[seg + 1]
        slope = (yk1 - yk0) / (xk1 - xk0 + 1e-8)
        s = yk0 + slope * (t_grid - xk0)

        # Linear interpolation on source indices
        s0 = torch.clamp(s.floor().long(), 0, timesteps - 1)
        s1 = torch.clamp(s0 + 1, 0, timesteps - 1)
        w = (s - s0.to(s.dtype)).unsqueeze(0)  # (1, T)

        x0 = x_flat[:, s0]
        x1 = x_flat[:, s1]
        x_warp = x0 * (1.0 - w) + x1 * w
        x_warp = x_warp.to(dtype)

        if x.ndim == 3:
            return x_warp.reshape(H, W, timesteps)
        return x_warp


class RandomTimeMask:
    """Randomly masks a fixed number of time blocks across all channels.

    Works with (C, T) or (H, W, T) inputs. Masking is applied along time.

    Args:     num_blocks: Number of blocks to mask.     block_len: Length of each masked
    block in samples.     value: Fill value for masked regions (default 0.0).     p:
    Probability of applying the augmentation.
    """

    def __init__(
        self,
        num_blocks: int = 2,
        block_len: int = 32,
        value: float = 0.0,
        p: float = 1.0,
    ) -> None:
        self.num_blocks = int(num_blocks)
        self.block_len = int(block_len)
        self.value = float(value)
        self.p = float(p)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x

        if x.ndim == 2:
            _, T = x.shape
        elif x.ndim == 3:
            _, _, T = x.shape
        else:
            return x

        if T <= 1 or self.block_len <= 0:
            return x

        length = min(self.block_len, T)
        for _ in range(self.num_blocks):
            s = int(torch.randint(0, max(T - length + 1, 1), (1,)).item())
            e = min(s + length, T)
            if x.ndim == 2:
                x[:, s:e] = self.value
            else:
                x[:, :, s:e] = self.value
        return x


class RandomTimeShift:
    """Randomly shifts the signal along the time axis by up to ``max_shift`` steps.

    Works with inputs shaped (C, T) or (H, W, T). For positive shifts the content is
    delayed (shifted to the right); negative shifts advance it. By default, vacated
    samples are filled with ``fill_value``.

    Args:     max_shift: Maximum absolute integer shift (in samples).     fill_value:
    Value used to fill newly exposed samples when ``wrap=False``.     wrap: If True,
    performs a circular shift via ``torch.roll`` instead.     p: Probability of applying
    the augmentation.
    """

    def __init__(
        self,
        max_shift: int = 6,
        fill_value: float = 0.0,
        wrap: bool = True,
        p: float = 1.0,
    ) -> None:
        self.max_shift = int(max_shift)
        self.fill_value = float(fill_value)
        self.wrap = bool(wrap)
        self.p = float(p)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p or self.max_shift <= 0:
            return x

        if x.ndim not in (2, 3):
            return x

        shift = (
            int(torch.randint(0, 2 * self.max_shift + 1, ()).item()) - self.max_shift
        )
        if shift == 0:
            return x

        if self.wrap:
            return torch.roll(x, shifts=shift, dims=-1)

        out = x.new_full(x.shape, self.fill_value)
        if shift > 0:
            out[..., shift:] = x[..., :-shift]
        else:
            k = -shift
            out[..., :-k] = x[..., k:]
        return out


class AdditiveNoise:
    """Adds zero-mean Gaussian noise to the signal.

    Args:     sigma: Noise scale. If relative=True, it's relative to per-channel std.
    relative: Whether to scale sigma by per-channel standard deviation.     p:
    Probability of applying the augmentation.
    """

    def __init__(self, sigma: float = 0.01, relative: bool = True, p: float = 1.0):
        self.sigma = float(sigma)
        self.relative = bool(relative)
        self.p = float(p)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p or self.sigma <= 0:
            return x

        if x.ndim == 2:  # (C, T)
            reduce_dims = -1
            shape = x.shape
        elif x.ndim == 3:  # (H, W, T)
            reduce_dims = -1
            shape = x.shape
        else:
            return x

        noise = torch.randn_like(x, dtype=torch.float32)
        if self.relative:
            std = x.to(torch.float32).std(dim=reduce_dims, keepdim=True)
            scale = self.sigma * (std + 1e-8)
        else:
            scale = torch.tensor(self.sigma, device=x.device, dtype=torch.float32)
        x_noisy = x.to(torch.float32) + noise * scale
        return x_noisy.to(x.dtype).reshape(shape)


class RandomSpatialWarp:
    """Random 2-D affine warp for inputs shaped (H, W, T). Applies the same spatial
    transform to all timesteps to preserve temporal coherence. No-op for (C, T).

    Args:     max_degrees: Maximum absolute rotation in degrees.     max_translate:
    Maximum translation as fraction of image size.     scale_range: Tuple (min_scale,
    max_scale).     p: Probability of applying the augmentation.     mode: Interpolation
    mode: "bilinear" or "nearest" (auto for integer types).
    """

    def __init__(
        self,
        max_degrees: float = 10.0,
        max_translate: float = 0.05,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 1.0,
        mode: str | None = None,
    ) -> None:
        self.max_degrees = float(max_degrees)
        self.max_translate = float(max_translate)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.p = float(p)
        self.mode = mode

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:  # only for (H, W, T)
            return x
        if torch.rand(1).item() > self.p:
            return x

        H, W, T = x.shape
        device = x.device
        dtype = x.dtype

        angle = (torch.rand(1, device=device) * 2 - 1) * self.max_degrees
        rad = angle * torch.pi / 180.0
        scale = torch.empty(1, device=device).uniform_(*self.scale_range)
        tx = (torch.rand(1, device=device) * 2 - 1) * self.max_translate
        ty = (torch.rand(1, device=device) * 2 - 1) * self.max_translate

        cos = torch.cos(rad) * scale
        sin = torch.sin(rad) * scale
        theta = torch.zeros(1, 2, 3, device=device, dtype=torch.float32)
        theta[0, 0, 0] = cos
        theta[0, 0, 1] = -sin
        theta[0, 0, 2] = tx
        theta[0, 1, 0] = sin
        theta[0, 1, 1] = cos
        theta[0, 1, 2] = ty

        # Prepare input as N×C×H×W with C = T
        x4 = x.permute(2, 0, 1).unsqueeze(0)  # (1, T, H, W)
        grid = F.affine_grid(theta, size=x4.size(), align_corners=False)
        mode = self.mode
        if mode is None:
            mode = "bilinear" if x.is_floating_point() else "nearest"
        warped = F.grid_sample(
            x4,
            grid,
            mode=mode,
            padding_mode="zeros",
            align_corners=False,
        )
        out = warped.squeeze(0).permute(1, 2, 0)  # (H, W, T)
        return out.to(dtype)


class RandomNeighborChannelSwap:
    """Randomly swaps neighbouring spatial channels for (H, W, T) inputs. No-op for (C,
    T) inputs.

    Args:     num_swaps: Number of neighbour swaps to perform.     p: Probability of
    applying the augmentation.
    """

    def __init__(self, num_swaps: int = 8, p: float = 1.0) -> None:
        self.num_swaps = int(num_swaps)
        self.p = float(p)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:  # only for (H, W, T)
            return x
        if torch.rand(1).item() > self.p:
            return x

        H, W, T = x.shape
        for _ in range(self.num_swaps):
            r = int(torch.randint(0, H, (1,)).item())
            c = int(torch.randint(0, W, (1,)).item())
            # choose neighbour: right or down with equal prob, fallback to left/up
            if torch.rand(1).item() < 0.5:
                rr, cc = r, c + 1 if c + 1 < W else c - 1 if c - 1 >= 0 else c
            else:
                rr, cc = r + 1 if r + 1 < H else r - 1 if r - 1 >= 0 else r, c

            if rr == r and cc == c:
                continue
            tmp = x[r, c, :].clone()
            x[r, c, :] = x[rr, cc, :]
            x[rr, cc, :] = tmp
        return x
