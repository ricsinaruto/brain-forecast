from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


def make_seanet(
    in_ch: int, base_channels: int = 64, depth: int = 4, stride: int = 2
) -> nn.Sequential:
    """Returns a down‑sampling SEANet encoder.

    Args:
        in_ch: Number of input channels
        base_channels: Number of base channels
        depth: Depth of the encoder
        stride: Stride of the encoder

    Returns:
        Tuple[nn.Sequential, int]: Tuple containing
            the encoder and the output channel dimension
    """
    layers: List[nn.Module] = []
    ch = in_ch
    for i in range(depth):
        out = base_channels * 2**i
        layers.append(ResConvBlock(ch, out, stride=stride))
        ch = out

    # final output channel dim
    out_channels = base_channels * 2 ** (depth - 1)
    return nn.Sequential(*layers), out_channels


def make_seanet_decoder(
    out_ch: int, base_channels: int = 64, depth: int = 4, stride: int = 2
) -> nn.Sequential:
    """
    Returns a up-sampling SEANet decoder.

    Args:
        out_ch: Number of output channels
        base_channels: Number of base channels
        depth: Depth of the decoder
        stride: Stride of the decoder

    Returns:
        nn.Sequential: The decoder
    """
    layers: List[nn.Module] = []
    ch = base_channels * 2 ** (depth - 1)
    for i in reversed(range(depth)):
        layers.append(nn.ConvTranspose1d(ch, base_channels * 2**i, stride, stride))
        layers.append(nn.GELU())
        ch = base_channels * 2**i
    layers.append(nn.Conv1d(ch, out_ch, 3, padding=1))
    return nn.Sequential(*layers)


class SensorEncoder(nn.Module):
    """Encodes sensor physical metadata into dense per‑channel embeddings.

    Args
    ----
    pos_dim:    Output dimension of the learned position/orientation MLP.
    type_dim:   Size of sensor‑type embedding table (embedding_dim == pos_dim).
    hidden_dim: Hidden size for the two‑layer position MLP.
    num_types: number of sensor types
    num_positions: number of position dimensions
    """

    def __init__(
        self,
        num_types: int = 3,
        num_positions: int = 6,
        hidden_dim: int = 128,
        final_dim: int = 512,
    ) -> None:
        """
        Args:
            num_types: Number of sensor types
            num_positions: Number of position dimensions
            hidden_dim: Hidden dimension of the MLP
            final_dim: Final dimension of the output
        """
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(num_positions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, final_dim),
        )
        self.type_emb = nn.Embedding(num_types, final_dim)

        self.mlp = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, final_dim),
        )
        self.norm = nn.RMSNorm(final_dim)

    def forward(self, pos_ori: torch.Tensor, sensor_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_ori: (B, C, 6) - xyz position **and** xyz orientation
            sensor_type: (B, C) - integer codes in {0,1,2}

        Returns:
            (B, C, D) sensor embeddings
        """
        pos_emb = self.pos_mlp(pos_ori)  # (B, C, D)
        type_emb = self.type_emb(sensor_type)  # (B, C, D)
        common_emb = pos_emb + type_emb

        common_emb = self.mlp(common_emb) + common_emb
        return self.norm(common_emb)


class ResConvBlock(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, kernel: int = 5, stride: int = 1, groups: int = 1
    ):
        super().__init__()
        # pad = (kernel - 1) // 2
        # causal left padding
        pad = kernel - stride
        self.pad = nn.ConstantPad1d((pad, 0), 0)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride, groups=groups, bias=False)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

        self.max_pool = nn.MaxPool1d(stride)
        self.conv_out = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        x = x + self.max_pool(self.conv_out(residual))
        return x


class CrossAttentionCompressor(nn.Module):
    """Aggregates (C, T, D) into (C_out, T, D) using learnable query vectors."""

    def __init__(self, dim: int, C_out: int, num_heads: int = 4):
        super().__init__()
        self.q = nn.Parameter(torch.randn(C_out, dim))  # (C', D)
        self.attn = MultiHeadAttention(dim, num_heads)

    def forward(self, value: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """value/key: (B, C, T, D) – returns (B, C_out, T, D)"""
        B, C, T, D = value.shape
        v = (
            value.permute(0, 2, 1, 3)  # (B, T, C, D)
            .contiguous()
            .view(B * T, C, D)  # (B*T, C, D)
        )
        k = (
            key.permute(0, 2, 1, 3)  # (B, T, C, D)
            .contiguous()
            .view(B * T, C, D)  # (B*T, C, D)
        )
        q = self.q.unsqueeze(0).expand(B * T, -1, -1)  # (B*T, C', D)
        out = self.attn(q, k, v)  # (B*T, C', D)
        out = out.view(B, T, -1, D).permute(0, 2, 1, 3).contiguous()
        return out  # (B, C', T, D)


class CrossAttentionExpander(nn.Module):
    """Aggregates (C, T, D) into (C_out, T, D) using learnable query vectors.
    Used in the original paper, but can't be used for forecasting."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """q/k/v: (B, C_latent, T, D) – returns (B, C, T, D)"""
        B, C, T, D = k.shape
        v = (
            v.permute(0, 2, 1, 3)  # (B, T, C_latent, D)
            .contiguous()
            .view(B * T, C, D)  # (B*T, C_latent, D)
        )
        k = (
            k.permute(0, 2, 1, 3)  # (B, T, C_latent, D)
            .contiguous()
            .view(B * T, C, D)  # (B*T, C_latent, D)
        )
        q = (
            q.permute(0, 2, 1, 3)  # (B, T, C, D)
            .contiguous()
            .view(B * T, -1, D)  # (B*T, C, D)
        )

        out = self.attn(q, k, v)  # (B*T, C, D)
        out = out.view(B, T, -1, D).permute(0, 2, 1, 3).contiguous()
        return out  # (B, C, T, D)
