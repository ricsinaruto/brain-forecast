"""
TODO: this was partly generated/modified by ChatGPT, still untested.
Better to use HF implementation.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, pack, unpack

Tensor = torch.Tensor

_WAVELETS = {
    "haar": torch.tensor([0.7071067811865476, 0.7071067811865476]),
    "rearrange": torch.tensor([1.0, 1.0]),
}
_PERSISTENT = True


def _factor_to_strides(factor: int, depth: int) -> list[int]:
    """Decompose a compression factor into a list of stride-2 steps."""
    strides: list[int] = []
    remaining = max(1, int(factor))
    for _ in range(depth):
        if remaining >= 2:
            strides.append(2)
            remaining = math.ceil(remaining / 2)
        else:
            strides.append(1)
    return strides


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def time2batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    batch_size = x.shape[0]
    return rearrange(x, "b c t h w -> (b t) c h w"), batch_size


def batch2time(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    return rearrange(x, "(b t) c h w -> b c t h w", b=batch_size)


def replication_pad(x):
    return torch.cat([x[:, :, :1, ...], x], dim=2)


def nonlinearity(x):
    return x * torch.sigmoid(x)


def space2batch(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    batch_size, height = x.shape[0], x.shape[-2]
    return rearrange(x, "b c t h w -> (b h w) c t"), batch_size, height


def batch2space(x: torch.Tensor, batch_size: int, height: int) -> torch.Tensor:
    return rearrange(x, "(b h w) c t -> b c t h w", b=batch_size, h=height)


def divisible_by(num: int, den: int) -> bool:
    return (num % den) == 0


def is_odd(n: int) -> bool:
    return not divisible_by(n, 2)


def cast_tuple(t: Any, length: int = 1) -> Any:
    return t if isinstance(t, tuple) else ((t,) * length)


class CausalConv3d_(nn.Module):
    """3D convolution with causal padding along the time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] | int = 3,
        stride: Tuple[int, int, int] | int = 1,
        *,
        time_stride: int | None = None,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            k_t = k_h = k_w = kernel_size
        else:
            k_t, k_h, k_w = kernel_size

        if isinstance(stride, int):
            s_t = s_h = s_w = stride
        else:
            s_t, s_h, s_w = stride
        s_t = time_stride or s_t

        self.time_pad = (k_t - 1) * 1  # dilation fixed to 1
        self.spatial_pad = padding
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            (k_t, k_h, k_w),
            stride=(s_t, s_h, s_w),
        )

    def forward(self, x: Tensor) -> Tensor:
        # pad only on the left of the temporal axis to keep causality
        pad = (
            self.spatial_pad,
            self.spatial_pad,
            self.spatial_pad,
            self.spatial_pad,
            self.time_pad,
            0,
        )
        x = F.pad(x, pad)
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        out_channels: int | None = None,
        dropout: float = 0.0,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        out_ch = channels if out_channels is None else out_channels
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv1 = CausalConv3d(channels, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            CausalConv3d(channels, out_ch, kernel_size=1, padding=0)
            if channels != out_ch
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(_swish(self.norm1(x)))
        h = self.conv2(self.dropout(_swish(self.norm2(h))))
        return self.skip(x) + h


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float,
        *,
        t_stride: int = 2,
        s_stride: int = 2,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                )
                for i in range(num_res_blocks)
            ]
        )
        needs_down = t_stride > 1 or s_stride > 1
        self.downsample = (
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=(t_stride, s_stride, s_stride),
                padding=1,
            )
            if needs_down
            else None
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = x
        for block in self.blocks:
            h = block(h)
        skip = h
        if self.downsample is not None:
            h = self.downsample(h)
        return skip, h


class Upsample3d(nn.Module):
    def __init__(self, channels: int, t_scale: int = 1, s_scale: int = 2) -> None:
        super().__init__()
        self.t_scale = max(1, t_scale)
        self.s_scale = max(1, s_scale)
        self.conv = CausalConv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        scale = (self.t_scale, self.s_scale, self.s_scale)
        x = F.interpolate(x, scale_factor=scale, mode="nearest")
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float,
        *,
        t_scale: int = 2,
        s_scale: int = 2,
    ) -> None:
        super().__init__()
        self.upsample = (
            Upsample3d(in_channels, t_scale=t_scale, s_scale=s_scale)
            if (t_scale > 1 or s_scale > 1)
            else None
        )
        blocks: list[ResidualBlock] = []
        for i in range(num_res_blocks):
            in_ch = in_channels + skip_channels if i == 0 else out_channels
            blocks.append(
                ResidualBlock(
                    in_ch,
                    out_channels=out_channels,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        if self.upsample is not None:
            x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            x = block(x)
        return x


class FSQuantizer_(nn.Module):
    """Finite scalar quantizer adapted from the Cosmos tokenizer."""

    def __init__(self, levels: Iterable[int]) -> None:
        super().__init__()
        levels = list(int(level) for level in levels)
        if not levels:
            raise ValueError("`levels` must contain at least one entry.")
        self.dim = len(levels)

        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int32))
        half_width = (self.levels - 1) // 2
        self.register_buffer("half_width", half_width, persistent=False)
        basis = torch.cumprod(torch.tensor([1] + levels[:-1], dtype=torch.int64), dim=0)
        self.register_buffer("basis", basis, persistent=False)
        self.codebook_size = int(math.prod(levels))

    def _bound(self, z: Tensor) -> Tensor:
        half_l = self.half_width.to(z.dtype)
        offset = torch.where(
            (self.levels % 2) == 0,
            torch.tensor(0.5, device=z.device),
            torch.tensor(0.0, device=z.device),
        )
        shift = (offset / (half_l + 1e-8)).atanh()
        return (z + shift).tanh() * half_l - offset

    def _codes_to_indices(self, codes: Tensor) -> Tensor:
        shifted = codes + self.half_width
        shifted = shifted.to(torch.int64)
        return (shifted * self.basis).sum(dim=-1)

    def _indices_to_codes(self, indices: Tensor) -> Tensor:
        flat = indices.reshape(-1)
        codes = (flat.unsqueeze(-1) // self.basis) % self.levels
        codes = codes.to(torch.float32) - self.half_width
        return codes

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if z.shape[1] != self.dim:
            raise ValueError(f"Expected channel dimension {self.dim}, got {z.shape[1]}")

        orig_shape = z.shape  # (B, D, T, H, W)
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()
        flat = z_perm.view(-1, self.dim)

        bounded = self._bound(flat)
        discrete = torch.round(bounded)
        quantized = discrete / self.half_width.to(discrete.dtype)

        indices = self._codes_to_indices(discrete)
        indices = indices.view(
            orig_shape[0], orig_shape[2], orig_shape[3], orig_shape[4]
        )

        quantized = quantized.view_as(z_perm).permute(0, 4, 1, 2, 3)
        quantized = z + (quantized - z).detach()

        loss = F.mse_loss(quantized, z)
        return indices, quantized, loss

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        codes = self._indices_to_codes(indices)
        codes = codes.view(indices.shape + (self.dim,))
        codes = codes.permute(0, 4, 1, 2, 3)
        return codes


class ResidualFSQuantizer_(nn.Module):
    """Stacked FSQ layers with residual refinement."""

    def __init__(self, levels: Iterable[int], num_quantizers: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [FSQuantizer(levels) for _ in range(num_quantizers)]
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        residual = z
        quantized_sum = torch.zeros_like(z)
        losses = []
        indices: list[Tensor] = []
        for layer in self.layers:
            idx, quantized, loss = layer(residual)
            indices.append(idx)
            quantized_sum = quantized_sum + quantized
            losses.append(loss)
            residual = residual - quantized.detach()

        stacked_indices = torch.stack(indices, dim=1)
        quant_loss = torch.stack(losses).mean()
        return stacked_indices, quantized_sum, quant_loss

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        if indices.ndim < 2:
            raise ValueError("indices must include the quantizer dimension")
        decoded: list[Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            decoded.append(layer.indices_to_codes(indices[:, layer_idx]))
        return sum(decoded)


"""
NEW MODULES
"""


class Patcher(nn.Module):
    """A module to convert image tensors into patches using torch operations.

    The main difference from `class Patching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Patching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer(
            "wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT
        )
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._haar(x)
        elif self.patch_method == "rearrange":
            return self._arrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _dwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        if rescale:
            out = out / 2
        return out

    def _haar(self, x):
        for _ in self.range:
            x = self._dwt(x, rescale=True)
        return x

    def _arrange(self, x):
        x = rearrange(
            x,
            "b c (h p1) (w p2) -> b (c p1 p2) h w",
            p1=self.patch_size,
            p2=self.patch_size,
        ).contiguous()
        return x


class Patcher3D(Patcher):
    """A 3D discrete wavelet transform for video data,
    expects 5D tensor, i.e. a batch of videos."""

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__(patch_method=patch_method, patch_size=patch_size)
        self.register_buffer(
            "patch_size_buffer",
            patch_size * torch.ones([1], dtype=torch.int32),
            persistent=_PERSISTENT,
        )

    def _dwt(self, x, wavelet, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # Handles temporal axis.
        x = F.pad(
            x, pad=(max(0, n - 2), n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode
        ).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        # Handles spatial axes.
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            out = out / (2 * torch.sqrt(torch.tensor(2.0)))
        return out

    def _haar(self, x):
        xi, xv = torch.split(x, [1, x.shape[2] - 1], dim=2)
        x = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)
        for _ in self.range:
            x = self._dwt(x, "haar", rescale=True)
        return x

    def _arrange(self, x):
        xi, xv = torch.split(x, [1, x.shape[2] - 1], dim=2)
        x = torch.cat([xi.repeat_interleave(self.patch_size, dim=2), xv], dim=2)
        x = rearrange(
            x,
            "b c (t p1) (h p2) (w p3) -> b (c p1 p2 p3) t h w",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        ).contiguous()
        return x


class UnPatcher(nn.Module):
    """A module to convert patches into image tensorsusing torch operations.

    The main difference from `class Unpatching` is that this module implements
    all operations using torch, rather than python or numpy, for efficiency purpose.

    It's bit-wise identical to the Unpatching module outputs, with the added
    benefit of being torch.jit scriptable.
    """

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__()
        self.patch_size = patch_size
        self.patch_method = patch_method
        self.register_buffer(
            "wavelets", _WAVELETS[patch_method], persistent=_PERSISTENT
        )
        self.range = range(int(torch.log2(torch.tensor(self.patch_size)).item()))
        self.register_buffer(
            "_arange",
            torch.arange(_WAVELETS[patch_method].shape[0]),
            persistent=_PERSISTENT,
        )
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.patch_method == "haar":
            return self._ihaar(x)
        elif self.patch_method == "rearrange":
            return self._iarrange(x)
        else:
            raise ValueError("Unknown patch method: " + self.patch_method)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]

        g = x.shape[1] // 4
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)

        # Inverse transform.
        yl = torch.nn.functional.conv_transpose2d(
            xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yl += torch.nn.functional.conv_transpose2d(
            xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh = torch.nn.functional.conv_transpose2d(
            xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh += torch.nn.functional.conv_transpose2d(
            xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        y = torch.nn.functional.conv_transpose2d(
            yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )
        y += torch.nn.functional.conv_transpose2d(
            yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )

        if rescale:
            y = y * 2
        return y

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p1 p2) h w -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x


class UnPatcher3D(UnPatcher):
    """A 3D inverse discrete wavelet transform for video wavelet decompositions."""

    def __init__(self, patch_size=1, patch_method="haar"):
        super().__init__(patch_method=patch_method, patch_size=patch_size)

    def _idwt(self, x, wavelet="haar", mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        g = x.shape[1] // 8  # split into 8 spatio-temporal filtered tesnors.
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hl = hl.to(dtype=dtype)
        hh = hh.to(dtype=dtype)

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(x, 8, dim=1)

        # Height height transposed convolutions.
        xll = F.conv_transpose3d(
            xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xll += F.conv_transpose3d(
            xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xlh = F.conv_transpose3d(
            xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xlh += F.conv_transpose3d(
            xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xhl = F.conv_transpose3d(
            xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhl += F.conv_transpose3d(
            xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xhh = F.conv_transpose3d(
            xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhh += F.conv_transpose3d(
            xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        # Handles width transposed convolutions.
        xl = F.conv_transpose3d(
            xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xl += F.conv_transpose3d(
            xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xh = F.conv_transpose3d(
            xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xh += F.conv_transpose3d(
            xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )

        # Handles time axis transposed convolutions.
        x = F.conv_transpose3d(
            xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )
        x += F.conv_transpose3d(
            xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )

        if rescale:
            x = x * (2 * torch.sqrt(torch.tensor(2.0)))
        return x

    def _ihaar(self, x):
        for _ in self.range:
            x = self._idwt(x, "haar", rescale=True)
        x = x[:, :, self.patch_size - 1 :, ...]
        return x

    def _iarrange(self, x):
        x = rearrange(
            x,
            "b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3)",
            p1=self.patch_size,
            p2=self.patch_size,
            p3=self.patch_size,
        )
        x = x[:, :, self.patch_size - 1 :, ...]
        return x


class CausalNormalize(nn.Module):
    def __init__(self, in_channels, num_groups=1):
        super().__init__()
        self.norm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=in_channels,
            eps=1e-6,
            affine=True,
        )
        self.num_groups = num_groups

    def forward(self, x):
        # if num_groups !=1, we apply a spatio-temporal groupnorm
        # for backward compatibility purpose.
        # All new models should use num_groups=1, otherwise causality is not guaranteed.
        if self.num_groups == 1:
            x, batch_size = time2batch(x)
            return batch2time(self.norm(x), batch_size)
        return self.norm(x)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int = 1,
        chan_out: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        time_stride = kwargs.pop("time_stride", 1)
        time_dilation = kwargs.pop("time_dilation", 1)
        padding = kwargs.pop("padding", 1)

        self.pad_mode = pad_mode
        time_pad = time_dilation * (time_kernel_size - 1) + (1 - time_stride)
        self.time_pad = time_pad

        self.spatial_pad = (padding, padding, padding, padding)

        stride = (time_stride, stride, stride)
        dilation = (time_dilation, dilation, dilation)
        self.conv3d = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def _replication_pad(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = x[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
        x = torch.cat([x_prev, x], dim=2)
        padding = self.spatial_pad + (0, 0)
        return F.pad(x, padding, mode=self.pad_mode, value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._replication_pad(x)
        return self.conv3d(x)


class CausalUpsample3d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        time_factor = 1.0 + 1.0 * (x.shape[2] > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()
        x = x.repeat_interleave(int(time_factor), dim=2)
        x = self.conv(x)
        return x[..., int(time_factor - 1) :, :, :]


class CausalDownsample3d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            time_stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1, 0, 0)
        x = F.pad(x, pad, mode="constant", value=0)
        x = replication_pad(x)
        x = self.conv(x)
        return x


class CausalHybridUpsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_up: bool = True,
        temporal_up: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = (
            CausalConv3d(
                in_channels,
                in_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                time_stride=1,
                padding=0,
            )
            if temporal_up
            else nn.Identity()
        )
        self.conv2 = (
            CausalConv3d(
                in_channels,
                in_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                time_stride=1,
                padding=1,
            )
            if spatial_up
            else nn.Identity()
        )
        self.conv3 = (
            CausalConv3d(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                time_stride=1,
                padding=0,
            )
            if spatial_up or temporal_up
            else nn.Identity()
        )
        self.spatial_up = spatial_up
        self.temporal_up = temporal_up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.spatial_up and not self.temporal_up:
            return x

        # hybrid upsample temporally.
        if self.temporal_up:
            time_factor = 1.0 + 1.0 * (x.shape[2] > 1)
            if isinstance(time_factor, torch.Tensor):
                time_factor = time_factor.item()
            x = x.repeat_interleave(int(time_factor), dim=2)
            x = x[..., int(time_factor - 1) :, :, :]
            x = self.conv1(x) + x

        # hybrid upsample spatially.
        if self.spatial_up:
            x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
            x = self.conv2(x) + x

        # final 1x1x1 conv.
        x = self.conv3(x)
        return x


class CausalHybridDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_down: bool = True,
        temporal_down: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = (
            CausalConv3d(
                in_channels,
                in_channels,
                kernel_size=(1, 3, 3),
                stride=2,
                time_stride=1,
                padding=0,
            )
            if spatial_down
            else nn.Identity()
        )
        self.conv2 = (
            CausalConv3d(
                in_channels,
                in_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                time_stride=2,
                padding=0,
            )
            if temporal_down
            else nn.Identity()
        )
        self.conv3 = (
            CausalConv3d(
                in_channels,
                in_channels,
                kernel_size=1,
                stride=1,
                time_stride=1,
                padding=0,
            )
            if spatial_down or temporal_down
            else nn.Identity()
        )

        self.spatial_down = spatial_down
        self.temporal_down = temporal_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.spatial_down and not self.temporal_down:
            return x

        # hybrid downsample spatially.
        if self.spatial_down:
            pad = (0, 1, 0, 1, 0, 0)
            x = F.pad(x, pad, mode="constant", value=0)
            x1 = self.conv1(x)
            x2 = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            x = x1 + x2

        # hybrid downsample temporally.
        if self.temporal_down:
            x = replication_pad(x)
            x1 = self.conv2(x)
            x2 = F.avg_pool3d(x, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            x = x1 + x2

        # final 1x1x1 conv.
        x = self.conv3(x)
        return x


class CausalResnetBlock3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalNormalize(in_channels, num_groups=num_groups)
        self.conv1 = CausalConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        x = self.nin_shortcut(x)

        return x + h


class CausalResnetBlockFactorized3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalNormalize(in_channels, num_groups=1)
        self.conv1 = nn.Sequential(
            CausalConv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        x = self.nin_shortcut(x)

        return x + h


class CausalAttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()

        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        q, batch_size = time2batch(q)
        k, batch_size = time2batch(k)
        v, batch_size = time2batch(v)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = batch2time(h_, batch_size)
        h_ = self.proj_out(h_)
        return x + h_


class CausalTemporalAttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()

        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = CausalConv3d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        q, batch_size, height = space2batch(q)
        k, _, _ = space2batch(k)
        v, _, _ = space2batch(v)

        bhw, c, t = q.shape
        q = q.permute(0, 2, 1)  # (bhw, t, c)
        k = k.permute(0, 2, 1)  # (bhw, t, c)
        v = v.permute(0, 2, 1)  # (bhw, t, c)

        w_ = torch.bmm(q, k.permute(0, 2, 1))  # (bhw, t, t)
        w_ = w_ * (int(c) ** (-0.5))

        # Apply causal mask
        mask = torch.tril(torch.ones_like(w_))
        w_ = w_.masked_fill(mask == 0, float("-inf"))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        h_ = torch.bmm(w_, v)  # (bhw, t, c)
        h_ = h_.permute(0, 2, 1).reshape(bhw, c, t)  # (bhw, c, t)

        h_ = batch2space(h_, batch_size, height)
        h_ = self.proj_out(h_)
        return x + h_


class ResidualFSQuantizer(nn.Module):
    """Residual Finite Scalar Quantization

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, levels: list[int], num_quantizers: int, **ignore_kwargs):
        super().__init__()
        self.dtype = ignore_kwargs.get("dtype", torch.float32)
        self.layers = nn.ModuleList(
            [FSQuantizer(levels=levels) for _ in range(num_quantizers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices_stack = []
        residual = x
        quantized_out = 0
        loss_out = 0
        for i, layer in enumerate(self.layers):
            quant_indices, z, loss = layer(residual)
            indices_stack.append(quant_indices)
            residual = residual - z.detach()
            quantized_out = quantized_out + z
            loss_out = loss_out + loss
        self.residual = residual
        indices = torch.stack(indices_stack, dim=1)
        return indices, quantized_out.to(self.dtype), loss_out.to(self.dtype)

    def indices_to_codes(self, indices_stack: torch.Tensor) -> torch.Tensor:
        quantized_out = 0
        for layer, indices in zip(self.layers, indices_stack.transpose(0, 1)):
            quantized_out += layer.indices_to_codes(indices)
        return quantized_out


class FSQuantizer(nn.Module):
    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        **ignore_kwargs,
    ):
        super().__init__()
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels)

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = (
            keep_num_codebooks_dim
            if keep_num_codebooks_dim is not None
            else num_codebooks > 1
        )
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = dim if dim is not None else len(_levels) * num_codebooks

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer(
            "implicit_codebook", implicit_codebook, persistent=self.persistent
        )

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor, project_out=True) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")
            indices = unpack_one(indices, ps, "b * c")
            dummy_loss = torch.zeros_like(out.mean(dim=[1, 2, 3], keepdim=True))
        else:
            dummy_loss = torch.zeros_like(out.mean(dim=[1, 2], keepdim=True)).unsqueeze(
                1
            )

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return (indices, out.to(self.dtype), dummy_loss)


class CosmosEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int = 16,
        temporal_compression: int = 8,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Patcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.patcher3d = Patcher3D(
            patch_size, ignore_kwargs.get("patch_method", "rearrange")
        )
        in_channels = in_channels * patch_size * patch_size * patch_size

        # calculate the number of downsample operations
        self.num_spatial_downs = int(math.log2(spatial_compression)) - int(
            math.log2(patch_size)
        )
        assert (
            self.num_spatial_downs <= self.num_resolutions
        ), f"Spatially downsample {self.num_resolutions} times at most"

        self.num_temporal_downs = int(math.log2(temporal_compression)) - int(
            math.log2(patch_size)
        )
        assert (
            self.num_temporal_downs <= self.num_resolutions
        ), f"Temporally downsample {self.num_resolutions} times at most"

        # downsampling
        self.conv_in = nn.Sequential(
            CausalConv3d(
                in_channels,
                channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                channels, channels, kernel_size=(3, 1, 1), stride=1, padding=0
            ),
        )

        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    CausalResnetBlockFactorized3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=1,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        nn.Sequential(
                            CausalAttnBlock(block_in, num_groups=1),
                            CausalTemporalAttnBlock(block_in, num_groups=1),
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                spatial_down = i_level < self.num_spatial_downs
                temporal_down = i_level < self.num_temporal_downs
                down.downsample = CausalHybridDownsample3d(
                    block_in,
                    spatial_down=spatial_down,
                    temporal_down=temporal_down,
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.mid.attn_1 = nn.Sequential(
            CausalAttnBlock(block_in, num_groups=1),
            CausalTemporalAttnBlock(block_in, num_groups=1),
        )
        self.mid.block_2 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(
                block_in, z_channels, kernel_size=(1, 3, 3), stride=1, padding=1
            ),
            CausalConv3d(
                z_channels,
                z_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher3d(x)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class CosmosDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int = 16,
        temporal_compression: int = 8,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # UnPatcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher3d = UnPatcher3D(
            patch_size, ignore_kwargs.get("patch_method", "rearrange")
        )
        out_ch = out_channels * patch_size * patch_size * patch_size

        # calculate the number of upsample operations
        self.num_spatial_ups = int(math.log2(spatial_compression)) - int(
            math.log2(patch_size)
        )
        assert (
            self.num_spatial_ups <= self.num_resolutions
        ), f"Spatially upsample {self.num_resolutions} times at most"
        self.num_temporal_ups = int(math.log2(temporal_compression)) - int(
            math.log2(patch_size)
        )
        assert (
            self.num_temporal_ups <= self.num_resolutions
        ), f"Temporally upsample {self.num_resolutions} times at most"

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Sequential(
            CausalConv3d(
                z_channels, block_in, kernel_size=(1, 3, 3), stride=1, padding=1
            ),
            CausalConv3d(
                block_in, block_in, kernel_size=(3, 1, 1), stride=1, padding=0
            ),
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.mid.attn_1 = nn.Sequential(
            CausalAttnBlock(block_in, num_groups=1),
            CausalTemporalAttnBlock(block_in, num_groups=1),
        )
        self.mid.block_2 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )

        legacy_mode = ignore_kwargs.get("legacy_mode", False)
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    CausalResnetBlockFactorized3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=1,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        nn.Sequential(
                            CausalAttnBlock(block_in, num_groups=1),
                            CausalTemporalAttnBlock(block_in, num_groups=1),
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                # The layer index for temporal/spatial downsampling performed
                # in the encoder should correspond to the layer index in
                # reverse order where upsampling is performed in the decoder.
                # If you've a pre-trained model, you can simply finetune.
                i_level_reverse = self.num_resolutions - i_level - 1
                if legacy_mode:
                    temporal_up = i_level_reverse < self.num_temporal_ups
                else:
                    temporal_up = 0 < i_level_reverse < self.num_temporal_ups + 1
                spatial_up = temporal_up or (
                    i_level_reverse < self.num_spatial_ups
                    and self.num_spatial_ups > self.num_temporal_ups
                )
                up.upsample = CausalHybridUpsample3d(
                    block_in, spatial_up=spatial_up, temporal_up=temporal_up
                )
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(block_in, out_ch, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(out_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

    def forward(self, z):
        h = self.conv_in(z)

        # middle block.
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # decoder blocks.
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.unpatcher3d(h)
        return h


class CosmosTokenizer(nn.Module):
    """Lightweight reimplementation of the Cosmos discrete video tokenizer."""

    def init(
        self,
        *,
        in_channels: int = 1,
        base_channels: int = 128,
        channel_multiplier: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        spatial_compression: int = 8,
        temporal_compression: int = 4,
        embedding_dim: int = 6,
        levels: Tuple[int, ...] = (8, 8, 8, 5, 5, 5),
        num_quantizers: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if embedding_dim != len(levels):
            raise ValueError(
                "embedding_dim must match the length of `levels` for FSQ quantization. "
                f"Got embedding_dim={embedding_dim}, len(levels)={len(levels)}."
            )
        depth = len(channel_multiplier)
        t_strides = _factor_to_strides(temporal_compression, depth)
        s_strides = _factor_to_strides(spatial_compression, depth)

        self.input_conv = CausalConv3d(
            in_channels, base_channels, kernel_size=3, padding=1
        )

        down_blocks: list[DownBlock] = []
        in_ch = base_channels
        skip_channels: list[int] = []
        for mult, t_stride, s_stride in zip(channel_multiplier, t_strides, s_strides):
            out_ch = base_channels * mult
            down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    t_stride=t_stride,
                    s_stride=s_stride,
                )
            )
            skip_channels.append(out_ch)
            in_ch = out_ch

        self.down_blocks = nn.ModuleList(down_blocks)
        self.mid_block = ResidualBlock(in_ch, out_channels=in_ch, dropout=dropout)

        self.quant_conv = nn.Conv3d(in_ch, embedding_dim, kernel_size=1)
        self.quantizer = ResidualFSQuantizer(
            levels=levels, num_quantizers=num_quantizers
        )
        self.post_quant_conv = nn.Conv3d(embedding_dim, in_ch, kernel_size=1)

        up_blocks: list[UpBlock] = []
        for mult, skip_ch, t_stride, s_stride in zip(
            reversed(channel_multiplier),
            reversed(skip_channels),
            reversed(t_strides),
            reversed(s_strides),
        ):
            out_ch = base_channels * mult
            up_blocks.append(
                UpBlock(
                    in_ch,
                    skip_ch,
                    out_ch,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    t_scale=t_stride,
                    s_scale=s_stride,
                )
            )
            in_ch = out_ch
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_conv = CausalConv3d(in_ch, in_channels, kernel_size=3, padding=1)

    def _format_inputs(self, x: Tensor) -> tuple[Tensor, Tuple[int, int, int]]:
        if isinstance(x, (tuple, list)):
            x = x[0]

        if x.ndim == 3:
            x = x.unsqueeze(0)  # H, W, T -> 1, H, W, T
        if x.ndim == 4:
            # assume channels last: (B, H, W, T)
            b, h, w, t = x.shape
            x = x.permute(0, 3, 1, 2).unsqueeze(1)  # (B, 1, T, H, W)
        elif x.ndim == 5:
            if x.shape[1] != self.in_channels and x.shape[-1] == self.in_channels:
                # (B, T, H, W, C)
                x = x.permute(0, 4, 1, 2, 3)
            elif x.shape[1] != self.in_channels:
                raise ValueError(
                    f"Unexpected input shape {tuple(x.shape)} for CosmosTokenizer."
                )
        else:
            raise ValueError(f"Unsupported input rank {x.ndim} for CosmosTokenizer.")

        _, _, t, h, w = x.shape
        return x.float(), (t, h, w)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        resolution: int,
        z_channels: int,
        z_factor: int = 1,
        dropout: float = 0.0,
        embedding_dim: int = 6,
        levels: Tuple[int, ...] = (8, 8, 8, 5, 5, 5),
        num_quantizers: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = CosmosEncoder(z_channels=z_factor * z_channels)

        self.decoder = CosmosDecoder(z_channels=z_channels)

        self.quant_conv = CausalConv3d(
            z_factor * z_channels, embedding_dim, kernel_size=1, padding=0
        )
        self.post_quant_conv = CausalConv3d(
            embedding_dim, z_channels, kernel_size=1, padding=0
        )

        self.quantizer = ResidualFSQuantizer(
            levels=levels, num_quantizers=num_quantizers
        )

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.quantizer(h)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def decode_code(self, code_b):
        quant_b = self.quantizer.indices_to_codes(code_b)
        quant_b = self.post_quant_conv(quant_b)
        return self.decoder(quant_b)

    def forward(self, input):
        quant_info, quant_codes, quant_loss = self.encode(input)
        reconstructions = self.decode(quant_codes)

        return dict(
            reconstructions=reconstructions,
            quant_loss=quant_loss,
            quant_info=quant_info,
        )

    def forward_(self, inputs: Tensor | tuple[Tensor, ...]) -> dict[str, Tensor]:
        x, target_shape = self._format_inputs(inputs)
        skips: list[Tensor] = []

        h = self.input_conv(x)
        for block in self.down_blocks:
            skip, h = block(h)
            skips.append(skip)

        h = self.mid_block(h)
        h = self.quant_conv(h)
        indices, quantized, quant_loss = self.quantizer(h)

        h = self.post_quant_conv(quantized)
        for block, skip in zip(self.up_blocks, reversed(skips)):
            h = block(h, skip)

        recon = self.output_conv(h)
        if recon.shape[2:] != target_shape:
            recon = F.interpolate(
                recon,
                size=target_shape,
                mode="trilinear",
                align_corners=False,
            )

        recon = recon.squeeze(1)  # (B, T, H, W)
        recon = recon.permute(0, 2, 3, 1)  # (B, H, W, T)

        logs = {
            "quant_loss": quant_loss,
            "indices": indices,
        }

        return (recon, logs)
