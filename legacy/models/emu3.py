"""
TODO: this was partly generated/modified by ChatGPT, still untested.
Better to use HF implementation.
"""

import math
from typing import Optional, Tuple, Union

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

from ...layers.quantizers import RVQ


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
    quantizer_revive_warmup_steps: int = 1000
    quantizer_revive_interval: int = 500
    quantizer_revive_min_usage: float = 1
    quantizer_track_usage: bool = True
    diversity_loss_weight: float = 0.0


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
        self.stride = stride

        self.conv = nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor):
        # adjust temporal padding based on input parity
        # if input is even, need to reduce temporal padding by stride - 1
        pad = self.padding
        if x.shape[2] % 2 == 0:
            t_pad = self.padding[-2] - self.stride[0] + 1
            pad = (pad[0], pad[1], pad[2], pad[3], t_pad, pad[5])

        x = F.pad(x, pad)
        x = self.conv(x)
        return x


class Emu3VisionVQCausalConvTranspose3d(nn.Module):

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

        self.kernel_size = kernel_size
        self.stride = stride

        def _padding_for_same(k: int, s: int) -> Tuple[int, int]:
            """
            Choose padding/output_padding so ConvTranspose keeps spatial
            dims when possible.
            Falls back to minimal padding if exact parity is impossible.
            """
            for out_pad in range(s):
                if (k - s + out_pad) % 2 == 0:
                    pad = (k - s + out_pad) // 2
                    if pad >= 0:
                        return pad, out_pad
            return 0, 0

        # keep spatial dimensions centered; temporal padding zero for strict causality
        pad_h, out_pad_h = _padding_for_same(kernel_size[1], stride[1])
        pad_w, out_pad_w = _padding_for_same(kernel_size[2], stride[2])
        t_out_pad = stride[0] - kernel_size[0] if kernel_size[0] < stride[0] else 0

        self.padding = (0, pad_h, pad_w)
        self.output_padding = (t_out_pad, out_pad_h, out_pad_w)

        self.conv = nn.ConvTranspose3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )

    def forward(self, x: torch.Tensor):
        target_t = x.shape[2] * self.stride[0]
        target_h = x.shape[3] * self.stride[1]
        target_w = x.shape[4] * self.stride[2]

        x = self.conv(x)

        # crop any extra right-padding to enforce causality in time, keep shapes aligned
        if x.shape[2] > target_t:
            x = x[:, :, :target_t, :, :]
        if x.shape[3] > target_h:
            x = x[:, :, :, :target_h, :]
        if x.shape[4] > target_w:
            x = x[:, :, :, :, :target_w]
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
        kernel_size: Tuple[int, ...] = (2, 3, 3),
        stride: Tuple[int, ...] = (2, 1, 1),
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = Emu3VisionVQCausalConvTranspose3d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x


class Emu3VisionVQTemporalDownsample(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: Tuple[int, ...] = (2, 3, 3),
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
        beta: float = 0.1,
        ema_decay: float = 0.95,
        ema_eps: float = 1e-5,
        revive_warmup_steps: int = 0,
        revive_interval: int = 0,
        revive_min_usage: float = 1.0,
        track_codebook_usage: bool = True,
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
        self.revive_warmup_steps = int(revive_warmup_steps)
        self.revive_interval = int(revive_interval)
        self.revive_min_usage = float(revive_min_usage)
        self.track_codebook_usage = bool(track_codebook_usage)
        self.diversity_loss_weight = float(
            getattr(config, "diversity_loss_weight", 0.0)
        )
        # EMA buffers
        self.register_buffer("N", torch.zeros(self.codebook_size))
        self.register_buffer("z_avg", self.embedding.weight.data.clone())
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.last_usage: dict[str, torch.Tensor] = {}

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

        if self.diversity_loss_weight > 0:
            soft_assign = torch.softmax(-d, dim=1)  # (B*T*H*W, K)
            mean_prob = soft_assign.mean(dim=0)
            kl_uniform = (
                mean_prob
                * (
                    mean_prob.clamp_min(self.ema_eps).log()
                    + math.log(self.codebook_size)
                )
            ).sum()
            qloss = qloss + self.diversity_loss_weight * kl_uniform

        # EMA codebook update
        if self.training:
            with torch.no_grad():
                self.step += 1
                # step_int = int(self.step.item())
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

                revived = torch.tensor(0, device=x_flat.device)
                if (
                    self.revive_interval > 0
                    and self.step >= self.revive_warmup_steps
                    and self.step % self.revive_interval == 0
                    and x_flat.numel() > 0
                ):
                    revive_mask = self.N < self.revive_min_usage
                    revive_idx = revive_mask.nonzero(as_tuple=False).view(-1)
                    if revive_idx.numel() > 0:
                        # pick farthest batch latents to encourage spread, add noise
                        dist_to_embed = (
                            torch.sum(x_flat**2, dim=1, keepdim=True)
                            + torch.sum(self.embedding.weight.data**2, dim=1)
                            - 2.0
                            * torch.einsum(
                                "nc,kc->nk", x_flat, self.embedding.weight.data
                            )
                        )
                        dmin = dist_to_embed.min(dim=1).values
                        topk = torch.topk(
                            dmin, k=min(revive_idx.numel(), x_flat.shape[0])
                        ).indices
                        new_vals = x_flat[topk].to(self.embedding.weight.data.dtype)
                        if new_vals.shape[0] < revive_idx.numel():
                            pad = revive_idx.numel() - new_vals.shape[0]
                            extra = torch.randn(
                                pad,
                                c,
                                device=x_flat.device,
                                dtype=self.embedding.weight.data.dtype,
                            )
                            new_vals = torch.cat([new_vals, extra], dim=0)
                        noise = torch.randn_like(new_vals) * 0.01
                        new_vals = new_vals + noise

                        self.embedding.weight.data[revive_idx] = new_vals
                        self.N[revive_idx] = self.revive_min_usage
                        self.z_avg[revive_idx] = new_vals
                        revived = torch.tensor(revive_idx.numel(), device=x_flat.device)

                if self.track_codebook_usage:
                    usage = self.N.clone()
                    total = usage.sum().clamp_min(self.ema_eps)
                    prob = usage / total
                    entropy = -(prob * (prob + self.ema_eps).log()).sum()
                    perplexity = torch.exp(entropy)
                    unused = (usage < self.revive_min_usage).sum()
                    active = (usage >= self.revive_min_usage).sum()
                    self.last_usage = {
                        "codebook_perplexity": perplexity.detach(),
                        "unused_codes": unused.detach(),
                        "active_codes": active.detach(),
                        "revived_codes": revived.detach(),
                    }

        return z_q, qloss, codes
        # return x, 0, codes


class Emu3RVQ(nn.Module):
    def __init__(self, config: Emu3VisionVQConfig):
        super().__init__()
        self.config = config
        self.quantizer = RVQ(
            dim=config.embed_dim,
            codebook_dim=config.embed_dim,
            codebook_size=config.codebook_size,
            num_quantizers=config.num_quantizers,
        )

    def forward(self, x: torch.Tensor):
        b, t, c, h, w = x.shape
        x_perm = x.permute(0, 1, 3, 4, 2).contiguous()  # (b,t,h,w,c)
        x_flat = x_perm.view(-1, c)

        z_q, codes, loss = self.quantizer(x_flat)

        # reshape back to original shape
        z_q = z_q.view(b, t, h, w, c)
        z_q = z_q.permute(0, 1, 4, 2, 3).contiguous()  # (b,t,c,h,w)
        codes = codes.view(b, t, h, w)

        return z_q, loss, codes


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
        self.quantize = Emu3VisionVQVectorQuantizer(
            config,
            revive_warmup_steps=config.quantizer_revive_warmup_steps,
            revive_interval=config.quantizer_revive_interval,
            revive_min_usage=config.quantizer_revive_min_usage,
            track_codebook_usage=config.quantizer_track_usage,
        )

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
        if isinstance(x, (tuple, list)):
            x = x[0]

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
        b, t, h, w = codes.shape
        codes = codes.reshape(codes.shape[0], -1)

        return z_q, qloss, codes, (t, h, w)

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
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]  # kind of hacky assuming tensor is first element of tuple/list
        quant, qloss, _, _ = self.encode(x)

        quant = quant.permute(0, 2, 1, 3, 4).contiguous()  # (B,C,T,H,W)
        quant2 = self.post_quant_conv(quant)
        video = self.decoder(quant2, quant)
        video = video.permute(0, 3, 4, 1, 2)  # (B,H,W,T,C)
        rec = video[..., 0]

        logs = {"commitment_loss": qloss}
        if getattr(self.quantize, "last_usage", None):
            logs.update(self.quantize.last_usage)

        return (rec, logs)


class Emu3VisionVQLPIPS(Emu3VisionVQ):
    """
    Non-adversarial variant that adds an LPIPS perceptual loss on top of MSE +
    commitment loss, without instantiating or training a discriminator.
    """

    def __init__(
        self,
        perceptual_weight: float = 0.1,
        pixelloss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from .movqgan.lpips import LPIPS

        self.perceptual = LPIPS().eval()
        self.perceptual_weight = float(perceptual_weight)
        self.pixelloss_weight = float(pixelloss_weight)

    def _prepare_target_video(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input into (B, H, W, T) video for LPIPS/MSE losses.
        Mirrors the VQGAN helper but omits discriminator bits.
        """
        if x.ndim == 3:
            return x.unsqueeze(0)

        if x.ndim == 4 and x.shape[1] in (1, 3):
            video = x[:, :1].permute(0, 2, 3, 1)  # (B, H, W, 1)
            t = self.config.temporal_downsample_factor
            return video.repeat(1, 1, 1, t)

        return x

    def _frames_for_losses(
        self, video_bhwt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (1ch frames, 3ch frames) with shape (B*T, C, H, W).
        LPIPS expects 3-channel input; replicate grayscale frames if needed.
        """
        B, H, W, T = video_bhwt.shape
        frames_1ch = video_bhwt.permute(0, 3, 1, 2).reshape(B * T, 1, H, W)
        if frames_1ch.shape[1] == 1:
            frames_lpips = frames_1ch.repeat(1, 3, 1, 1)
        else:
            frames_lpips = frames_1ch
        return frames_1ch, frames_lpips

    def forward(self, x: torch.Tensor, optimizer_idx: int = 0, global_step: int = 0):
        del optimizer_idx, global_step  # unused in the non-adversarial path

        if isinstance(x, (tuple, list)):
            x = x[0]

        target_video = self._prepare_target_video(x)

        quant, qloss, _, _ = self.encode(x)
        quant = quant.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)

        quant2 = self.post_quant_conv(quant)
        video = self.decoder(quant2, quant)
        video = video.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
        rec_bhwt = video[..., 0]

        target_frames_1ch, target_frames_lpips = self._frames_for_losses(target_video)
        recon_frames_1ch, recon_frames_lpips = self._frames_for_losses(rec_bhwt)

        mse = F.mse_loss(recon_frames_1ch, target_frames_1ch)
        lpips_loss = self.perceptual(
            recon_frames_lpips.contiguous(), target_frames_lpips.contiguous()
        ).mean()

        total_loss = (
            self.pixelloss_weight * mse + self.perceptual_weight * lpips_loss + qloss
        )

        logs = {
            "commitment_loss": qloss,
            "total_loss": total_loss,
            "recon_mse": mse.detach(),
            "lpips": lpips_loss.detach(),
        }
        if getattr(self.quantize, "last_usage", None):
            logs.update(self.quantize.last_usage)

        return rec_bhwt, logs


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
        **kwargs,
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

    def _prepare_target_video(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input into (B, H, W, T) video for GAN/LPIPS losses.
        - If input is channel-first, repeat single channel across the expected
          temporal dimension so it matches the reconstruction length.
        - Otherwise assume input is already (B, H, W, T).
        """
        if x.ndim == 3:
            return x.unsqueeze(0)

        if x.ndim == 4 and x.shape[1] in (1, 3):
            video = x[:, :1].permute(0, 2, 3, 1)  # (B, H, W, 1)
            t = self.config.temporal_downsample_factor
            return video.repeat(1, 1, 1, t)

        return x

    def forward(self, x: torch.Tensor, optimizer_idx: int = 0, global_step: int = 0):
        if isinstance(x, (tuple, list)):
            x = x[0]

        target_video = self._prepare_target_video(x)

        # Encode once to get quantized latents and commitment loss
        quant, qloss, _, _ = self.encode(x)
        quant = quant.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)

        # Decode reconstruction
        quant2 = self.post_quant_conv(quant)
        video = self.decoder(quant2, quant)
        video = video.permute(0, 3, 4, 1, 2)  # (B, H, W, T, C)
        rec_bhwt = video[..., 0]

        # Prepare 2D frames for GAN / LPIPS loss
        target_frames = self._frames_from_video(target_video)
        recon_frames = self._frames_from_video(rec_bhwt)

        loss, gan_log = self.gan_loss(
            codebook_loss=qloss,
            inputs=target_frames,
            reconstructions=recon_frames,
            optimizer_idx=optimizer_idx,
            global_step=global_step,
            last_layer=None,
            split="train",
        )

        logs = {
            "commitment_loss": qloss,
            "total_loss": loss,
            "recon_mse": F.mse_loss(rec_bhwt, target_video).detach(),
        }
        logs.update(gan_log)
        if getattr(self.quantize, "last_usage", None):
            logs.update(self.quantize.last_usage)

        return rec_bhwt, logs

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
