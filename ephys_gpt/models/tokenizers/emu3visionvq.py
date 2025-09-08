# coding=utf-8
# Copyright 2024 The Emu team, BAAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Emu3VisionVQ model"""

import math
from typing import Optional, Tuple, Union

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Emu3VisionVQConfig:
    r"""
    This is the configuration class to store the configuration of a [`Emu3VisionVQ`].
    It is used to instantiate an video movq
    model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the
    defaults will yield a configuration to the VQ model presented in Emu3 paper.


    Args:
        codebook_size (`int`, *optional*, defaults to 32768):
            Codebook size of the VQ model.
        embed_dim (`int`, *optional*, defaults to 4):
            Dimension of the quantized vector in codebook.
        z_channels (`int`, *optional*, defaults to 4):
            Dimension of the output channel of encoder and the input channel of decoder
        double_z (`bool`, *optional*, defaults to False):
            Whether double the output dim of the encoder.
        in_channels (`int`, *optional*, defaults to 3):
            Input channel of encoder.
        out_channels (`int`, *optional*, defaults to 3):
            Output channel of decoder.
        temporal_downsample_factor (`int`, *optional*, defaults to 4):
            Temporal downsample factor.
        ch (`int`, *optional*, defaults to 256):
            Basic channel number of the intermediate blocks.
        ch_mult (`List[int]`, *optional*, defaults to `[1, 2, 2, 4]`):
            Channel scaling factor of the intermediate blocks.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Residual block number in each stage.
        attn_resolutions (`List[int]`, *optional*, defaults to 3):
            Stage indices to apply attention.
        dropout (`float`, *optional*, defaults to 0.0):
            Dropout probability.

    """

    codebook_size: int = 32768
    embed_dim: int = 4
    z_channels: int = 4
    double_z: bool = False
    in_channels: int = 3
    out_channels: int = 3
    temporal_downsample_factor: int = 4
    ch: int = 256
    ch_mult: Tuple[int, ...] = (1, 2, 2, 4)
    num_res_blocks: int = 2
    attn_resolutions: Tuple[int, ...] = (3,)
    dropout: float = 0.0


class Emu3VisionVQActivation(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor):
        return x * torch.sigmoid(x)


class Emu3VisionVQUpsample(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Emu3VisionVQDownsample(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Emu3VisionVQCausalConv3d(nn.Module):

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Union[int, Tuple[int, ...]] = (3, 1, 1),
        stride: Union[int, Tuple[int, ...]] = (1, 1, 1),
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        hw_pad = [k - s for k, s in zip(kernel_size[1:], stride[1:])]
        self.padding = tuple()
        for p in hw_pad[::-1]:
            self.padding += (p // 2 + p % 2, p // 2)
        # Causal padding along time: all padding on the left, none on the right.
        # Use (kernel_size - 1) for strict causality with dilation=1.
        t_pad = max(kernel_size[0] - 1, 0)
        self.padding += (t_pad, 0)

        self.conv = nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.padding)
        x = self.conv(x)
        return x


class Emu3VisionVQTemporalGroupNorm(nn.Module):

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 32,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()
        # Apply GroupNorm per frame to avoid temporal leakage
        self.norm = nn.GroupNorm(
            num_channels=num_channels, num_groups=num_groups, eps=eps, affine=affine
        )

    def forward(self, x: torch.Tensor):
        # x: (B, C, T, H, W) -> (B*T, C, H, W) GN -> (B, C, T, H, W)
        b, c, t, h, w = x.shape
        x_2d = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        x_2d = self.norm(x_2d)
        x = x_2d.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
        return x


class Emu3VisionVQResnetTemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        stride = (1, 1, 1)
        kernel_size = (3, 3, 3)

        self.norm1 = Emu3VisionVQTemporalGroupNorm(in_channels, num_groups=in_channels)
        self.conv1 = Emu3VisionVQCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.norm2 = Emu3VisionVQTemporalGroupNorm(
            out_channels, num_groups=out_channels
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Emu3VisionVQCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.act = Emu3VisionVQActivation()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Emu3VisionVQCausalConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            else:
                self.nin_shortcut = nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Emu3VisionVQSpatialNorm(nn.Module):

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        norm_layer: nn.Module = nn.GroupNorm,
        add_conv: bool = False,
        num_groups: int = 32,
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()
        self.norm_layer = norm_layer(
            num_channels=f_channels,
            num_groups=num_groups,
            eps=eps,
            affine=affine,
        )

        self.add_conv = add_conv
        if self.add_conv:
            self.conv = nn.Conv2d(
                zq_channels,
                zq_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )

        self.conv_y = nn.Conv2d(
            zq_channels,
            f_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.conv_b = nn.Conv2d(
            zq_channels,
            f_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor, zq: torch.Tensor):
        zq = F.interpolate(zq, size=x.shape[-2:], mode="nearest")

        if self.add_conv:
            zq = self.conv(zq)

        x = self.norm_layer(x)
        x = x * self.conv_y(zq) + self.conv_b(zq)
        return x


class Emu3VisionVQResnetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        zq_ch: Optional[int] = None,
        add_conv: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.zq_ch = zq_ch

        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
            self.norm1 = nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
            self.norm2 = nn.GroupNorm(num_channels=out_channels, **norm_kwargs)
        else:
            self.norm1 = Emu3VisionVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)
            self.norm2 = Emu3VisionVQSpatialNorm(out_channels, zq_ch, add_conv=add_conv)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.act = Emu3VisionVQActivation()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x: torch.Tensor, zq: Optional[torch.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)

        h = self.norm1(x, *norm_args)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h, *norm_args)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Emu3VisionVQAttnBlock(nn.Module):

    def __init__(
        self, in_channels: int, zq_ch: Optional[int] = None, add_conv: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.zq_ch = zq_ch

        if zq_ch is None:
            norm_kwargs = dict(num_groups=32, eps=1e-6, affine=True)
            self.norm = nn.GroupNorm(num_channels=in_channels, **norm_kwargs)
        else:
            self.norm = Emu3VisionVQSpatialNorm(in_channels, zq_ch, add_conv=add_conv)

        self.q = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.k = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.v = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.proj_out = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor, zq: Optional[torch.Tensor] = None):
        norm_args = tuple() if self.zq_ch is None else (zq,)

        nx = self.norm(x, *norm_args)
        q = self.q(nx)
        k = self.k(nx)
        v = self.v(nx)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        k = k.reshape(b, c, h * w)
        score = torch.bmm(q.permute(0, 2, 1), k)
        score = score / (c**0.5)
        score = F.softmax(score, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        v = torch.bmm(v, score.permute(0, 2, 1))
        v = v.reshape(b, c, h, w)

        v = self.proj_out(v)

        return x + v


class Emu3VisionVQTemporalUpsample(nn.Module):

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (3, 3, 3),
        stride: Tuple[int, ...] = (1, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = Emu3VisionVQCausalConv3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor):
        # Causal temporal upsample: duplicate each time step (no future info)
        x = torch.repeat_interleave(x, repeats=2, dim=2)
        x = self.conv(x)
        return x


class Emu3VisionVQTemporalDownsample(nn.Module):

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (4, 3, 3),
        stride: Tuple[int, ...] = (2, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.conv = Emu3VisionVQCausalConv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x


class Emu3VisionVQVectorQuantizer(nn.Module):
    def __init__(
        self,
        config: Emu3VisionVQConfig,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = config.codebook_size
        self.embed_dim = config.embed_dim
        self.embedding = nn.Embedding(self.codebook_size, self.embed_dim)
        self.embedding.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size
        )
        self.beta = beta
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        # EMA buffers
        self.register_buffer("N", torch.zeros(self.codebook_size))
        self.register_buffer("z_avg", self.embedding.weight.data.clone())

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W) latent features to quantize
        Returns:
            z_q:    (B, T, C, H, W) quantized latents (straight-through)
            qloss:  scalar commitment/codebook loss
            codes:  (B, T, H, W) integer indices
        """
        b, t, c, h, w = x.shape
        x_perm = x.permute(0, 1, 3, 4, 2).contiguous()  # (b,t,h,w,c)
        x_flat = x_perm.view(-1, c)
        E = self.embedding.weight  # (K, c)
        d = (
            torch.sum(x_flat**2, dim=1, keepdim=True)
            + torch.sum(E**2, dim=1)
            - 2.0 * torch.einsum("nc,kc->nk", x_flat, E)
        )
        codes = torch.argmin(d, dim=1)
        z_q = self.embedding(codes).view(b, t, h, w, c)
        z_q = z_q.permute(0, 1, 4, 2, 3).contiguous()  # (b,t,c,h,w)

        # VQ-VAE losses
        qloss = (z_q.detach() - x).pow(2).mean() + self.beta * (z_q - x.detach()).pow(
            2
        ).mean()
        z_q = x + (z_q - x).detach()
        codes = codes.view(b, t, h, w)

        # EMA codebook update
        if self.training:
            with torch.no_grad():
                K = self.codebook_size
                # counts per code
                counts = torch.bincount(codes.view(-1), minlength=K).to(x_flat.dtype)
                # sums per code
                sums = torch.zeros(K, c, device=x_flat.device, dtype=x_flat.dtype)
                sums.index_add_(0, codes.view(-1), x_flat)

                self.N.mul_(self.ema_decay).add_(counts, alpha=(1.0 - self.ema_decay))
                self.z_avg.mul_(self.ema_decay).add_(sums, alpha=(1.0 - self.ema_decay))

                denom = (self.N + self.ema_eps).unsqueeze(1)
                new_embed = self.z_avg / denom
                self.embedding.weight.data.copy_(new_embed)

                # reinit unused codes from random samples
                unused = (counts == 0).nonzero(as_tuple=False).view(-1)
                if unused.numel() > 0 and x_flat.numel() > 0:
                    rand_idx = torch.randint(
                        0, x_flat.shape[0], (unused.numel(),), device=x_flat.device
                    )
                    self.embedding.weight.data[unused] = x_flat[rand_idx]

        return z_q, qloss, codes


class Emu3VisionVQEncoder(nn.Module):

    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks
        self.in_channels = config.in_channels

        # downsampling
        self.conv_in = nn.Conv2d(
            self.in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        # removed unused in_ch_mult to avoid linter error
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            prev_mult = 1 if i_level == 0 else config.ch_mult[i_level - 1]
            block_in = config.ch * prev_mult
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    Emu3VisionVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VisionVQAttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Emu3VisionVQDownsample(block_in)

            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )
        self.mid.attn_1 = Emu3VisionVQAttnBlock(block_in)
        self.mid.block_2 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
        )

        # end
        self.norm_out = nn.GroupNorm(
            num_channels=block_in, num_groups=32, eps=1e-6, affine=True
        )

        out_z_channels = 2 * config.z_channels if config.double_z else config.z_channels
        self.conv_out = nn.Conv2d(
            block_in,
            out_z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        temporal_down_blocks = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.ModuleList()

        for i in range(temporal_down_blocks):
            conv = Emu3VisionVQTemporalDownsample(out_z_channels, out_z_channels)
            self.time_conv.append(conv)

        self.time_res_stack = nn.Sequential(
            *[
                Emu3VisionVQResnetTemporalBlock(
                    in_channels=out_z_channels,
                    out_channels=out_z_channels,
                    dropout=config.dropout,
                )
                for _ in range(self.num_res_blocks)
            ]
        )

        self.act = Emu3VisionVQActivation()

    def forward(self, x: torch.Tensor):
        t = x.shape[1]
        x = x.reshape(-1, *x.shape[2:])

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = self.act(h)

        h = self.conv_out(h)

        h = h.reshape(-1, t, *h.shape[1:])
        h = h.permute(0, 2, 1, 3, 4)

        for conv in self.time_conv:
            h = self.act(conv(h))

        h = self.time_res_stack(h)

        # print(h.shape)
        return h


class Emu3VisionVQDecoder(nn.Module):

    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.ch = config.ch
        self.num_resolutions = len(config.ch_mult)
        self.num_res_blocks = config.num_res_blocks

        # removed unused in_ch_mult to avoid linter error
        zq_ch = config.embed_dim

        block_in = config.ch * config.ch_mult[-1]
        self.time_res_stack = nn.Sequential(
            *[
                Emu3VisionVQResnetTemporalBlock(
                    in_channels=config.z_channels,
                    out_channels=config.z_channels,
                    dropout=config.dropout,
                )
                for _ in range(config.num_res_blocks)
            ]
        )

        tempo_upsample_block_num = int(math.log2(config.temporal_downsample_factor))
        self.time_conv = nn.ModuleList()
        for i in range(tempo_upsample_block_num):
            conv = Emu3VisionVQTemporalUpsample(config.z_channels, config.z_channels)
            self.time_conv.append(conv)

        self.conv_in = nn.Conv2d(
            config.z_channels,
            block_in,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
            zq_ch=zq_ch,
        )
        self.mid.attn_1 = Emu3VisionVQAttnBlock(block_in, zq_ch)
        self.mid.block_2 = Emu3VisionVQResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            dropout=config.dropout,
            zq_ch=zq_ch,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    Emu3VisionVQResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=config.dropout,
                        zq_ch=zq_ch,
                    )
                )
                block_in = block_out
                if i_level in config.attn_resolutions:
                    attn.append(Emu3VisionVQAttnBlock(block_in, zq_ch))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Emu3VisionVQUpsample(block_in)

            self.up.insert(0, up)

        self.act = Emu3VisionVQActivation()

        self.norm_out = Emu3VisionVQSpatialNorm(block_in, zq_ch)
        self.conv_out = nn.Conv2d(
            block_in,
            config.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z: torch.Tensor, zq: torch.Tensor):
        z_zq = torch.cat((z, zq), dim=0)
        # z_zq = z_zq.permute(0, 2, 1, 3, 4)
        z_zq = self.time_res_stack(z_zq)

        for conv in self.time_conv:
            z_zq = self.act(conv(z_zq))

        z_zq = z_zq.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W)

        h, zq = torch.chunk(z_zq, 2, dim=0)

        t = h.shape[1]
        h = h.reshape(-1, *h.shape[2:])
        zq = zq.reshape(-1, *zq.shape[2:])

        h = self.conv_in(h)

        # middle
        h = self.mid.block_1(h, zq)
        h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)

            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h, zq)
        h = self.act(h)
        h = self.conv_out(h)
        h = h.reshape(-1, t, *h.shape[1:])

        return h


class Emu3VisionVQ(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        config = Emu3VisionVQConfig(**kwargs)
        self.config = config

        self.encoder = Emu3VisionVQEncoder(config)
        self.decoder = Emu3VisionVQDecoder(config)
        self.quantize = Emu3VisionVQVectorQuantizer(config)

        self.quant_conv = Emu3VisionVQCausalConv3d(config.z_channels, config.embed_dim)
        self.post_quant_conv = Emu3VisionVQCausalConv3d(
            config.embed_dim, config.z_channels
        )

        self.spatial_scale_factor = 2 ** (len(config.ch_mult) - 1)

        # self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        # copied from the `reset_parameters` method of `class Linear(Module)` in `torch`
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def encode(self, x: torch.Tensor):
        x_in = x
        if x_in.ndim == 3:
            x_in = x_in.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        elif x_in.ndim == 4:
            if x_in.shape[1] in (1, 3):
                t = self.config.temporal_downsample_factor
                x_in = x_in.unsqueeze(1).repeat(1, t, 1, 1, 1)
            else:
                x_in = x_in.permute(0, 3, 1, 2).unsqueeze(2)
        if x_in.shape[2] != self.config.in_channels:
            x_in = x_in.repeat(1, 1, self.config.in_channels, 1, 1)

        h = self.encoder(x_in)
        h = self.quant_conv(h)
        h = h.permute(0, 2, 1, 3, 4)  # (B,T,C,H,W)
        z_q, qloss, codes = self.quantize(h)

        # reshape codes to (B, L)
        codes = codes.reshape(codes.shape[0], -1)

        return z_q, qloss, codes

    def decode(self, x: torch.Tensor):
        ndim = x.ndim
        if ndim == 2:  # (H', W') single code map → (B=1, T=1, H', W')
            x = x.unsqueeze(0).unsqueeze(0)
        elif ndim == 3:
            # Could be (T', H', W') from encode(img) or (B, H', W')
            # Assume (T', H', W') and add batch at front
            x = x.unsqueeze(0)
        # elif ndim == 4: already (B, T', H', W')

        b, t, h, w = x.shape
        quant = self.quantize.embedding(x.flatten())
        c = quant.shape[-1]
        quant = quant.view(b, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        quant2 = self.post_quant_conv(quant)

        # quant = quant.permute(0, 2, 1, 3, 4)
        # quant2 = quant2.permute(0, 2, 1, 3, 4)

        video = self.decoder(quant2, quant)
        video = video.reshape(
            b,
            t * self.config.temporal_downsample_factor,
            self.config.out_channels,
            h * self.spatial_scale_factor,
            w * self.spatial_scale_factor,
        )
        if ndim in (2, 3):
            # Return (B, H, W, T_out) for brain images
            video = video.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
            return video.squeeze(-1)
        return video

    def forward(self, x: torch.Tensor):
        quant, qloss, _ = self.encode(x)

        quant = quant.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,T,H,W)
        quant2 = self.post_quant_conv(quant)
        video = self.decoder(quant2, quant)
        video = video.permute(0, 3, 4, 1, 2)  # (B,H,W,T,C)
        rec = video[..., 0]

        # rec_loss = F.mse_loss(rec, target) / 0.06
        return (rec, {"commitment_loss": qloss})


class Emu3VisionVQGAN(Emu3VisionVQ):
    """
    Adversarial VQ-VAE training wrapper around Emu3VisionVQModel.
    Follows MOVQ/VQGAN style with LPIPS + GAN losses using a 2D discriminator
    operating on video frames concatenated along batch.
    """

    def __init__(
        self,
        disc_start: int = 1000,
        disc_in_channels: int = 1,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_num_layers: int = 3,
        disc_ndf: int = 64,
        disc_loss: str = "hinge",
        **kwargs
    ):
        super().__init__(**kwargs)
        from .movqgan.loss import VQLPIPSWithDiscriminator2

        # build loss module – expects inputs/recon as 4D images, so we’ll flatten time
        self.gan_loss = VQLPIPSWithDiscriminator2(
            disc_start=disc_start,
            codebook_weight=1.0,
            pixelloss_weight=1.0,
            disc_num_layers=disc_num_layers,
            disc_in_channels=disc_in_channels,
            disc_factor=disc_factor,
            disc_weight=disc_weight,
            perceptual_weight=perceptual_weight,
            disc_ndf=disc_ndf,
            disc_loss=disc_loss,
        )

    def _frames_from_video(self, video_bhwt: torch.Tensor) -> torch.Tensor:
        # (B,H,W,T) -> (B*T, 1, H, W)
        B, H, W, T = video_bhwt.shape
        return video_bhwt.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)

    def forward(self, x: torch.Tensor, optimizer_idx: int = 0, global_step: int = 0):
        # Base VQ-VAE forward for reconstruction and codebook loss
        vq_loss, rec_bhwt, logs = super().forward(x)

        # Prepare 2D frames for GAN loss
        target_frames = self._frames_from_video(
            rec_bhwt.detach() if isinstance(x, torch.Tensor) else rec_bhwt
        )
        # Reconstruct again without detach to keep grads on generator
        with torch.enable_grad():
            vq_loss2, rec_bhwt2, _ = super().forward(x)
        recon_frames = self._frames_from_video(rec_bhwt2)

        # Combine losses using MOVQ-style module
        loss, log = self.gan_loss(
            codebook_loss=logs["q_loss"],
            inputs=target_frames,
            reconstructions=recon_frames,
            optimizer_idx=optimizer_idx,
            global_step=global_step,
            last_layer=None,
            split="train",
        )
        total = vq_loss2 + loss
        return total, rec_bhwt2, {**logs, **log}

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
