from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from ...layers.ibq.diffusionmodules import Encoder, Decoder
from ...layers.ibq.vqvae import IndexPropagationQuantize


@dataclass
class IBQOutput:
    recon: torch.Tensor
    codebook_loss: torch.Tensor
    info: (
        Tuple[torch.Tensor, torch.Tensor]
        | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple
    )


class IBQ(nn.Module):
    """Image tokenizer with index propagation quantization (Emu3.5)."""

    def __init__(
        self,
        ddconfig,
        n_embed,
        embed_dim,
        beta=0.25,
        use_entropy_loss=False,
        cosine_similarity=False,
        entropy_temperature=0.01,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
        **kwargs,
    ):
        super().__init__()

        encoder_cfg = dict(ddconfig)
        decoder_cfg = dict(ddconfig)
        decoder_cfg.pop("double_z", None)
        decoder_cfg.pop("use_linear_attn", None)

        self.encoder = Encoder(**encoder_cfg)
        self.decoder = Decoder(**decoder_cfg)

        self.quantize = IndexPropagationQuantize(
            n_embed,
            embed_dim,
            beta,
            use_entropy_loss,
            cosine_similarity=cosine_similarity,
            entropy_temperature=entropy_temperature,
            sample_minimization_weight=sample_minimization_weight,
            batch_maximization_weight=batch_maximization_weight,
        )

        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant, return_intermediate_feature=False):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(
            quant, return_intermediate_feature=return_intermediate_feature
        )
        return dec

    def decode_code(self, code_b, shape=None):
        # shape specifying (batch, height, width, channel)
        quant_b = self.quantize.get_codebook_entry(code_b, shape=shape)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_intermediate_feature=False):
        quant, diff, _ = self.encode(input)
        dec = self.decode(
            quant, return_intermediate_feature=return_intermediate_feature
        )
        return dec, diff


class IBQMEGTokenizer(nn.Module):
    """IBQ tokenizer adapted to MEG interpolated images (video-style tensor).

    Expects input shaped (B, 1, T, H, W) and applies the 2D IBQ encoder/decoder frame-
    wise, keeping the temporal dimension intact.
    """

    def __init__(self, ddconfig, n_embed: int, embed_dim: int, **kwargs) -> None:
        super().__init__()
        self.ibq = IBQ(
            ddconfig=ddconfig, n_embed=n_embed, embed_dim=embed_dim, **kwargs
        )

    def _flatten_time(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int, int]]:
        """Reshape (B, C, T, H, W) -> (B*T, C, H, W) and return original shape for
        unflat."""
        if x.dim() == 4:  # (B, T, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected input with 5 dims (B,C,T,H,W), got {x.shape}")
        b, c, t, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (b, c, t, h, w)

    def _unflatten_time(
        self, x: torch.Tensor, shape: Tuple[int, int, int, int, int]
    ) -> torch.Tensor:
        b, c, t, h, w = shape
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    def encode(self, x: torch.Tensor):
        flat, orig_shape = self._flatten_time(x)
        quant, codebook_loss, info = self.ibq.encode(flat)
        # stash orig shape for decode
        return quant, codebook_loss, info, orig_shape

    def decode(self, quant: torch.Tensor, orig_shape: Tuple[int, int, int, int, int]):
        b, _, t, _, _ = orig_shape
        dec_flat = self.ibq.decode(quant)
        return self._unflatten_time(
            dec_flat, (b, dec_flat.shape[1], t, dec_flat.shape[2], dec_flat.shape[3])
        )

    def forward(self, x: torch.Tensor) -> IBQOutput:
        quant, codebook_loss, info, orig_shape = self.encode(x)
        dec = self.decode(quant, orig_shape)
        return IBQOutput(recon=dec, codebook_loss=codebook_loss, info=info)

    def get_last_layer(self):
        return self.ibq.decoder.conv_out.weight
