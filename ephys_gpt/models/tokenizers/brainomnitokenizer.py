from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:  # Optional import; not required for forward usage
    from ...training.lightning import LitModel
except Exception:  # pragma: no cover - allow importing without lightning deps
    LitModel = None  # type: ignore
from ...layers.brainomni import (
    make_seanet,
    make_seanet_decoder,
    SensorEncoder,
    CrossAttentionCompressor,
)
from ...layers.quantizers import ResidualVectorQuantizer


class BrainOmniTokenizer(nn.Module):
    """Quantises multi-channel EEG/MEG data into discrete latent tokens."""

    def __init__(
        self,
        in_channels: int,
        latent_channels: int = 16,
        base_channels: int = 64,
        encoder_depth: int = 4,
        codebook_size: int = 1024,
        num_quantizers: int = 4,
        emb_dim: int = 128,
        num_types: int = 3,
        num_positions: int = 6,
        dropout: float = 0.25,
    ) -> None:
        """
        Args:
            in_channels: number of input channels
            latent_channels: number of latent channels
            base_channels: number of base channels
            encoder_depth: depth of the encoder
            codebook_size: size of the codebook
            num_quantizers: number of quantizers
            emb_dim: dimension of the embedding
            num_types: number of sensor types
            num_positions: number of position dimensions
        """
        super().__init__()
        self.in_channels = in_channels
        self.time_encoder, out_channels = make_seanet(1, base_channels, encoder_depth)
        assert out_channels == emb_dim, "out_channels must be equal to emb_dim"

        self.receptive_field = 2**encoder_depth

        self.sensor_encoder = SensorEncoder(
            num_types=num_types,
            num_positions=num_positions,
            hidden_dim=emb_dim // 2,
            final_dim=emb_dim,
        )
        self.compress = CrossAttentionCompressor(emb_dim, latent_channels)
        self.rvq = ResidualVectorQuantizer(emb_dim, num_quantizers, codebook_size)
        # for reconstruction (optional)
        self.expand = CrossAttentionCompressor(emb_dim, in_channels)
        self.time_decoder = make_seanet_decoder(1, base_channels, encoder_depth)

        self.drop_channels = nn.Dropout1d(dropout)

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        return_reconstruction: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: input tensor of shape (B, C, T)
            sensor_pos_ori: tensor of shape (B, C, 6)
            sensor_type: tensor of shape (B, C)
            return_reconstruction: whether to return the reconstruction

        Returns:
            quantised:   (B, C_latent, T', D)
            codes:       (B, C_latent, T', num_q)  – integer indices
            reconstruction: (B, C, T') if return_reconstruction is True, otherwise None
        """
        x0, sensor_pos_ori, sensor_type = inputs
        B, C, T = x0.shape
        x = self.drop_channels(x0)
        x = x.reshape(B * C, 1, T)  # B*C, 1, T

        z = self.time_encoder(x)  # (B*C, D, W)
        z = z.reshape(B, C, -1, z.shape[-1]).transpose(2, 3)  # (B, C, W, D)

        # Sensor embeddings
        # Ensure correct shapes for sensor metadata
        if sensor_pos_ori.dim() == 2:
            sensor_pos_ori = sensor_pos_ori.unsqueeze(0).expand(B, -1, -1)
        if sensor_type.dim() == 1:
            sensor_type = sensor_type.unsqueeze(0).expand(B, -1)

        s = self.sensor_encoder(sensor_pos_ori, sensor_type)  # (B, C, D)
        # Pad sensor emb along time dim to match T': repeat
        s = s.unsqueeze(3).expand(-1, -1, -1, z.shape[-2])  # (B, C, D, W)
        s = s.permute(0, 1, 3, 2)  # (B, C, W, D)

        k = z + s  # Key
        latent = self.compress(z, k)  # (B, C_latent, W, D)

        # q: (B, C_latent, W, D), codes: (B, C_latent, W, num_q)
        z_latent, codes, residuals, nearest = self.rvq(latent)
        if not return_reconstruction:
            return codes

        # Optional reconstruction path – for auto‑encoder training
        x_hat = self._reconstruct(z_latent)
        return x_hat, residuals, nearest, z_latent, codes

    def _reconstruct(self, z_latent: torch.Tensor) -> torch.Tensor:
        B = z_latent.shape[0]

        z_hat = self.expand(z_latent, z_latent)  # (B, C, W, D)
        z_hat = z_hat.reshape(-1, z_hat.shape[-2], z_hat.shape[-1]).transpose(2, 1)
        x_hat = self.time_decoder(z_hat)
        x_hat = x_hat.reshape(B, self.in_channels, -1)
        return x_hat

    def reconstruct(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct the original signal from the latent tokens."""
        # Accept codes either as (B, C_latent, T, num_q) or (N, num_q)
        if codes.dim() == 4:
            B, C_latent, T, Nq = codes.shape
            z_list = []
            for t in range(T):
                ct = codes[:, :, t, :].reshape(B * C_latent, Nq)
                zt = self.rvq.decode(ct)  # (B*C_latent, D)
                zt = zt.view(B, C_latent, -1)  # (B, C_latent, D)
                z_list.append(zt)
            z_latent = torch.stack(z_list, dim=2)  # (B, C_latent, T, D)
        else:
            z_latent = self.rvq.decode(codes)
        return self._reconstruct(z_latent)
