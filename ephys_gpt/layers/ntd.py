import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from scipy.linalg import cholesky_banded, solve_banded

import logging

log = logging.getLogger(__name__)


def _off_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (np.exp(-(1 / ell))) / (np.exp(-(2 / ell)) - 1.0)


def _corner_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (1.0 / (1.0 - np.exp(-(2 / ell))))


def _mid_diag(ell, sigma_squared):
    """Helper function of banded OU precision matrix."""
    return (1.0 / sigma_squared) * (
        (1.0 + np.exp(-(2 / ell))) / (1.0 - np.exp(-(2 / ell)))
    )


def get_in_mask(
    signal_channel,
    hidden_channel,
    cond_channel=0,
):
    """
    Returns the input mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.
        time_channel: Number of diffusion time embedding channels.
        cond_channel: Number of conditioning channels.
    Returns:
        Input mask as torch tensor.
    """
    np_mask = np.concatenate(
        (
            get_restricted(signal_channel, 1, hidden_channel),
            get_full(cond_channel, signal_channel * hidden_channel),
        ),
        axis=1,
    )
    return torch.from_numpy(np.float32(np_mask))


def get_mid_mask(signal_channel, hidden_channel, off_diag, num_heads=1):
    """
    Returns the hidden mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.
        off_diag: Number of off-diagonal interactions.
        num_heads: Number of heads.

    Returns:
        Mid mask as torch tensor.
    """
    np_mask = np.maximum(
        get_restricted(signal_channel, hidden_channel, hidden_channel),
        get_sub_interaction(signal_channel, hidden_channel, off_diag),
    )

    return torch.from_numpy(np.float32(np.repeat(np_mask, num_heads, axis=1)))


def get_out_mask(signal_channel, hidden_channel):
    """
    Returns the output mask for the specified mode.

    Args:
        signal_channel: Number of signal channels.
        hidden_channel: Number of hidden channels.

    Returns:
        Output mask as torch tensor.
    """
    np_mask = get_restricted(signal_channel, hidden_channel, 1)
    return torch.from_numpy(np.float32(np_mask))


def get_full(num_in, num_out):
    """Get full mask containing all ones."""
    return np.ones((num_out, num_in))


def get_restricted(num_signal, num_in, num_out):
    """Get mask with ones only on the block diagonal."""
    return np.repeat(np.repeat(np.eye(num_signal), num_out, axis=0), num_in, axis=1)


def get_sub_interaction(num_signal, size_hidden, num_sub_interaction):
    """Get off-diagonal interactions"""
    sub_interaction = np.zeros((size_hidden, size_hidden))
    sub_interaction[:num_sub_interaction, :num_sub_interaction] = 1.0
    return np.tile(sub_interaction, (num_signal, num_signal))


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class WhiteNoiseProcess(nn.Module):
    """
    White noise process.
    Provides a sample method and a Mahalabonis distance method.
    In the case of white noise, this is just the (scaled) L2 distance.

    Args:
        sigma_squared: Variance of the white noise.
        signal_length: Length of the signal to sample and compute the distance on.
    """

    def __init__(self, sigma_squared, signal_length):
        super().__init__()
        self.sigma_squared = sigma_squared
        self.signal_length = signal_length  # needs to be implemented
        self.device = "cpu"

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        return self

    # Expects and returns tensor with shape (B, C, L).
    def sample(self, sample_shape, device="cpu"):
        return np.sqrt(self.sigma_squared) * torch.randn(*sample_shape, device=device)

    def sqrt_mal(self, train_batch):
        return (1 / self.sigma_squared) * train_batch


class OUProcess(nn.Module):
    """
    Ornstein-Uhlenbeck process.
    Provides a sample method and a Mahalabonis distance method.
    Supports linear time operations.

    Args:
        sigma_squared: Variance of the process.
        ell: Length scale of the process.
        signal_length: Length of the signal to sample and compute the distance on.
    """

    def __init__(self, sigma_squared, ell, signal_length):
        super().__init__()
        self.sigma_squared = sigma_squared
        self.ell = ell
        self.signal_length = signal_length
        self.device = "cpu"  # default

        # Build banded precision (only diag and lower diag) because of symmetry.
        lower_banded = np.zeros((2, signal_length))
        lower_banded[0, 1:-1] = _mid_diag(ell, sigma_squared)
        lower_banded[0, 0] = _corner_diag(ell, sigma_squared)
        lower_banded[0, -1] = _corner_diag(ell, sigma_squared)
        lower_banded[1, :-1] = _off_diag(ell, sigma_squared)

        banded_lower_prec_numpy = cholesky_banded(lower_banded, lower=True)
        # Transpose as needed, matrix now in upper notation as a result.
        self.banded_upper_prec_numpy = np.zeros((2, signal_length))
        self.banded_upper_prec_numpy[0, 1:] = banded_lower_prec_numpy[1, :-1]
        self.banded_upper_prec_numpy[1, :] = banded_lower_prec_numpy[0, :]

        # Convert to torch tensor
        self.register_buffer(
            "banded_upper_prec",
            torch.from_numpy(np.float32(self.banded_upper_prec_numpy)),
        )

        self.register_buffer(
            "dense_upper_matrix",
            torch.diag(self.banded_upper_prec[0, 1:], diagonal=1)
            + torch.diag(self.banded_upper_prec[1, :], diagonal=0),
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = args[0]
        return self

    def sqrt_mal(self, train_batch):  # (B, C, L)
        assert self.signal_length == train_batch.shape[2]
        # Use lower-triangular application for temporal causality:
        # y_t depends on x_{<=t}
        superdiag_as_subdiag = self.banded_upper_prec[0]  # length L, 0 at index 0
        main_diag = self.banded_upper_prec[1]
        sub_mult = torch.einsum("l, bcl -> bcl", superdiag_as_subdiag, train_batch)
        main_mult = torch.einsum("l, bcl -> bcl", main_diag, train_batch)
        # add sub-diagonal contribution: shift left by one (from t-1 to t)
        main_mult[:, :, 1:] += sub_mult[:, :, :-1]
        return main_mult

    def sample(self, sample_shape, device="cpu"):
        return self.sample_gpu_dense(sample_shape, device)

    # O(n)
    def sample_numpy_banded(self, sample_shape):
        normal_samples = np.random.randn(*sample_shape)
        ou_samples = solve_banded(
            (0, 1),  # Upper triangular matrix.
            self.banded_upper_prec_numpy,
            np.transpose(normal_samples, (2, 1, 0)),
        )
        return torch.from_numpy(np.float32(np.transpose(ou_samples, (2, 1, 0)))).to(
            self.device
        )

    # This is not O(n), but for shorter sequences,
    # the theoretical advantage is dwarfed by GPU acceleration.
    def sample_gpu_dense(self, sample_shape, device="cpu"):
        """
        Draw samples from the OU process using a dense (upper triangular) precision
        matrix.  This path leverages GPU acceleration for speed but falls back to
        a CPU implementation on Apple M-series GPUs (``mps`` device) because
        ``torch.linalg.solve_triangular`` is known to segfault on that backend
        for certain input sizes (observed on M4 MacBook, see
        https://github.com/pytorch/pytorch/issues/98292).

        Parameters
        ----------
        sample_shape : Tuple[int, ...]
            Desired output shape ``(B, C, L)`` matching the call signature of
            ``torch.randn``.
        """

        # draw standard normal noise on the *target* device so the final output
        # stays there regardless of which backend performs the triangular solve
        normal_samples = torch.randn(*sample_shape, device=device)

        res = torch.linalg.solve_triangular(
            self.dense_upper_matrix,
            torch.transpose(normal_samples, 1, 2),
            upper=True,
        )

        return torch.transpose(res, 1, 2)


class MaskedConv1d(nn.Module):
    """
    1D Convolutional layer with masking.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        assert (out_channels, in_channels) == mask.shape

        # Enforce causal (left-only) zero padding regardless of padding_mode
        # so no future timestep influences the present.
        self.padding_mode = "constant"
        total_padding = kernel_size - 1
        self.pad = [total_padding, 0]

        init_k = np.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = nn.Parameter(
            data=torch.FloatTensor(out_channels, in_channels, kernel_size).uniform_(
                -init_k, init_k
            ),
            requires_grad=True,
        )
        self.register_buffer("mask", mask)
        self.bias = (
            nn.Parameter(
                data=torch.FloatTensor(out_channels).uniform_(-init_k, init_k),
                requires_grad=True,
            )
            if bias
            else None
        )

    def forward(self, x):
        return F.conv1d(
            F.pad(x, self.pad, mode=self.padding_mode),
            self.weight * self.mask.unsqueeze(-1),
            self.bias,
        )


class EfficientMaskedConv1d(nn.Module):
    """
    1D Convolutional layer with masking.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mask=None,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()

        if mask is None:
            # Build a standard conv without internal padding; we will apply
            # left-only zero padding in forward to ensure causality.
            self.layer = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                bias=bias,
                padding=0,
                padding_mode="zeros",
            )
            self.kernel_size = kernel_size
        else:
            self.layer = MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                mask,
                bias=bias,
                padding_mode=padding_mode,
            )

    def forward(self, x):
        if isinstance(self.layer, nn.Conv1d):
            left_pad = self.kernel_size - 1
            x = F.pad(x, [left_pad, 0], mode="constant")
            return self.layer(x)
        return self.layer.forward(x)


class GeneralEmbedder(nn.Module):
    def __init__(self, cond_channel, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_channel, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, cond):
        cond = rearrange(cond, "b c l -> b l c")
        cond = self.mlp(cond)
        return rearrange(cond, "b l c -> b c l")


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SLConv(nn.Module):
    """
    Structured Long Convolutional layer.
    Adapted from https://github.com/ctlllll/SGConv

    Args:
        kernel_size: Kernel size used to build convolution.
        num_channels: Number of channels.
        num_scales: Number of scales.
            Overall length will be: kernel_size * (2 ** (num_scales - 1))
        decay_min: Minimum decay. Advanced option.
        decay_max: Maximum decay. Advanced option.
        heads: Number of heads.
        padding_mode: Padding mode. Either "zeros" or "circular".
        use_fft_conv: Whether to use FFT convolution.
        interpolate_mode: Interpolation mode. Either "nearest" or "linear".
    """

    def __init__(
        self,
        kernel_size,
        num_channels,
        num_scales,
        decay_min=2.0,
        decay_max=2.0,
        heads=1,
        padding_mode="zeros",
        use_fft_conv=False,
        interpolate_mode="nearest",
    ):
        super().__init__()
        assert decay_min <= decay_max

        self.h = num_channels
        self.num_scales = num_scales
        self.kernel_length = kernel_size * (2 ** (num_scales - 1))

        self.heads = heads

        # Causal semantics are enforced irrespective of the padding_mode argument.
        # We preserve the attribute but always perform left-only zero padding.
        self.padding_mode = "constant"
        self.use_fft_conv = use_fft_conv
        self.interpolate_mode = interpolate_mode

        self.D = nn.Parameter(torch.randn(self.heads, self.h))

        total_padding = self.kernel_length - 1
        # Left-only padding for causality
        self.pad = [total_padding, 0]

        # Init of conv kernels. There are more options here.
        # Full kernel is always normalized by initial kernel norm.
        self.kernel_list = nn.ParameterList()
        for _ in range(self.num_scales):
            kernel = nn.Parameter(torch.randn(self.heads, self.h, kernel_size))
            self.kernel_list.append(kernel)

        # Support multiple scales. Only makes sense in non-sparse setting.
        self.register_buffer(
            "multiplier",
            torch.linspace(decay_min, decay_max, self.h).view(1, -1, 1),
        )
        self.register_buffer("kernel_norm", torch.ones(self.heads, self.h, 1))
        self.register_buffer(
            "kernel_norm_initialized", torch.tensor(0, dtype=torch.bool)
        )

    def forward(self, x):
        signal_length = x.size(-1)

        kernel_list = []
        for i in range(self.num_scales):
            kernel = F.interpolate(
                self.kernel_list[i],
                scale_factor=2 ** (max(0, i - 1)),
                mode=self.interpolate_mode,
            ) * self.multiplier ** (self.num_scales - i - 1)
            kernel_list.append(kernel)
        k = torch.cat(kernel_list, dim=-1)

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device
            )
            log.debug(f"Kernel norm: {self.kernel_norm.mean()}")
            log.debug(f"Kernel size: {k.size()}")

        assert k.size(-1) < signal_length
        if self.use_fft_conv:
            k = F.pad(k, (0, signal_length - k.size(-1)))

        k = k / self.kernel_norm

        # Convolution
        if self.use_fft_conv:
            if self.padding_mode == "constant":
                factor = 2
            elif self.padding_mode == "circular":
                factor = 1

            k_f = torch.fft.rfft(k, n=factor * signal_length)  # (C H L)
            u_f = torch.fft.rfft(x, n=factor * signal_length)  # (B H L)
            y_f = torch.einsum("bhl,chl->bchl", u_f, k_f)
            slice_start = self.kernel_length // 2
            y = torch.fft.irfft(y_f, n=factor * signal_length)

            if self.padding_mode == "constant":
                y = y[..., slice_start : slice_start + signal_length]  # (B C H L)
            elif self.padding_mode == "circular":
                y = torch.roll(y, -slice_start, dims=-1)
            y = rearrange(y, "b c h l -> b (h c) l")
        else:
            # Pytorch implements convolutions as cross-correlations! flip necessary
            padded = F.pad(x, self.pad, mode=self.padding_mode)
            k_flip = rearrange(k.flip(-1), "c h l -> (h c) 1 l")
            y = F.conv1d(padded, k_flip, groups=self.h)

        # Compute D term in state space equation - essentially a skip connection
        y = y + rearrange(
            torch.einsum("bhl,ch->bchl", x, self.D),
            "b c h l -> b (h c) l",
        )

        return y


class ChannelLayerNorm1d(nn.Module):
    """
    LayerNorm applied across channels per time step for inputs of shape (B, C, L).
    This avoids mixing statistics across the temporal dimension, preserving causality.
    """

    def __init__(self, num_channels: int, affine: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(
            normalized_shape=num_channels, elementwise_affine=affine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        x_perm = rearrange(x, "b c l -> b l c")
        x_norm = self.norm(x_perm)
        return rearrange(x_norm, "b l c -> b c l")


class AdaConvBlock(nn.Module):
    def __init__(
        self,
        kernel_size,
        channel,
        num_scales,
        signal_length=1200,
        mid_mask=None,
        padding_mode="circular",
        use_fft_conv=False,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_scales = num_scales
        self.mid_mask = mid_mask

        self.conv = SLConv(
            self.kernel_size,
            channel,
            num_scales=self.num_scales,
            padding_mode=padding_mode,
            use_fft_conv=use_fft_conv,
        )
        self.mlp = nn.Sequential(
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
            nn.GELU(),
            EfficientMaskedConv1d(channel, channel, 1, mask=self.mid_mask),
        )

        # Channel-only normalization per time step to maintain temporal causality
        self.norm1 = ChannelLayerNorm1d(channel, affine=False)
        self.norm2 = ChannelLayerNorm1d(channel, affine=False)

        self.ada_ln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channel // 3, channel * 6, bias=True),
        )

        self.ada_ln[-1].weight.data.zero_()
        self.ada_ln[-1].bias.data.zero_()

    def forward(self, x, t_cond):
        y = x
        y = self.norm1(y)
        temp = self.ada_ln(rearrange(t_cond, "b c l -> b l c"))
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = rearrange(
            temp, "b l c -> b c l"
        ).chunk(6, dim=1)
        y = modulate(y, shift_tm, scale_tm)
        y = self.conv(y)
        y = x + gate_tm * y

        x = y
        y = self.norm2(y)
        y = modulate(y, shift_cm, scale_cm)
        y = x + gate_cm * self.mlp(y)
        return y


class AdaConv(nn.Module):
    def __init__(
        self,
        signal_length,
        signal_channel,
        cond_dim=0,
        hidden_channel=8,
        in_kernel_size=1,
        out_kernel_size=1,
        slconv_kernel_size=17,
        num_scales=5,
        num_blocks=3,
        num_off_diag=8,
        use_pos_emb=False,
        padding_mode="circular",
        use_fft_conv=False,
        mask_channel=0,
    ):
        super().__init__()
        self.signal_channel = signal_channel
        signal_channel += mask_channel

        self.signal_length = signal_length
        self.use_pos_emb = use_pos_emb

        hidden_channel_full = signal_channel * hidden_channel

        in_mask = get_in_mask(signal_channel, hidden_channel, 0)
        mid_mask = (
            None
            if num_off_diag == hidden_channel
            else get_mid_mask(signal_channel, hidden_channel, num_off_diag, 1)
        )

        self.conv_in = EfficientMaskedConv1d(
            signal_channel,
            hidden_channel_full,
            in_kernel_size,
            in_mask,
            padding_mode=padding_mode,
        )
        self.blocks = nn.ModuleList(
            [
                AdaConvBlock(
                    slconv_kernel_size,
                    hidden_channel_full,
                    num_scales,
                    signal_length=signal_length,
                    mid_mask=mid_mask,
                    padding_mode=padding_mode,
                    use_fft_conv=use_fft_conv,
                )
                for _ in range(num_blocks)
            ]
        )

        out_mask = get_out_mask(signal_channel, hidden_channel)
        out_mask = out_mask[: signal_channel - mask_channel]

        self.conv_out = EfficientMaskedConv1d(
            hidden_channel_full,
            signal_channel - mask_channel,
            out_kernel_size,
            out_mask,
            padding_mode=padding_mode,
        )

        self.t_emb = TimestepEmbedder(hidden_channel_full // 3)
        if self.use_pos_emb:
            self.pos_emb = TimestepEmbedder(hidden_channel_full // 3)
        if cond_dim > 0:
            self.cond_emb = GeneralEmbedder(cond_dim, hidden_channel_full // 3)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        TODO: add channel embedding

        Args:
            x: (B, C, L)
            t: (B,)
            cond: (B, C_cond, L)

        Returns:
            (B, C, L)
        """
        x = self.conv_in(x)

        t_emb = self.t_emb(t.to(x.device))
        t_emb = repeat(t_emb, "b c -> b c l", l=self.signal_length)

        pos_emb = 0
        if self.use_pos_emb:
            pos_emb = self.pos_emb(torch.arange(self.signal_length).to(x.device))
            pos_emb = repeat(pos_emb, "l c -> b c l", b=x.shape[0])

        cond_emb = 0
        if cond is not None:
            cond_emb = self.cond_emb(cond)

        emb = t_emb + pos_emb + cond_emb

        for block in self.blocks:
            x = block(x, emb)

        x = self.conv_out(x)
        return x
