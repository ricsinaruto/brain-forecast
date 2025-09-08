"""
TODO: UNDER CONSTRUCTION

EEGFormer: PyTorch implementation
=================================
A faithful, modular re‑implementation of the model architecture from the paper
"EEGFormer: Towards Transferable and Interpretable Large‑Scale EEG Foundation Model"
(Chen *et al.*, 2024).

This file contains **only** the model components – no training loops, data‐loading,
or feature‑engineering utilities – so it can be dropped into an existing pipeline
with minimal friction.

Main public classes
-------------------
* ``PatchEmbed``   Slice a 1‑D signal into (possibly overlapping) patches and project
                    them to the Transformer embedding space with learnable positional
                    embeddings.
* ``TransformerEncoder`` / ``TransformerDecoder``   Thin wrappers around PyTorch
                    ``nn.TransformerEncoder``/``nn.TransformerDecoder`` with sensible
                    defaults for EEG.
* ``VectorQuantizer``   Straight‑through VQ‑VAE‑style vector quantiser (with optional
                    EMA updates) that converts continuous hidden states to discrete
                    codebook indices + commitment loss.
* ``EEGFormer``   End‑to‑end encoder‑quantiser‑decoder stack.  Calling
                    ``model(x)`` returns a dict containing the reconstruction, VQ loss,
                    and codebook indices so that downstream code can assemble
                    whichever training objective it needs.

Notation & shapes
-----------------
``x``: tensor of shape ``(B, C, L)`` where
    ``B`` = batch size, ``C`` = number of EEG channels (aka “variates”),
    ``L`` = signal length (samples in the frequency domain).

After patching:
    ``N = floor((L - P) / S) + 2`` is the number of patches per channel,
    where ``P`` is ``patch_size`` and ``S`` is ``stride``.

The code follows the paper by treating **each channel independently** inside the
encoder.  Internally we merge the channel axis into the batch dimension so we
can call the vanilla PyTorch Transformer blocks.

Every module is type‑annotated and tries to be as self‑contained as possible –
pass Hyper‑Parameters in the constructor, get deterministic behaviour out.  You
may of course replace or extend anything (e.g. sinusoidal positions, Flash‑Attn
blocks, EMA codebook updates, etc.) if desired.

Author: OpenAI ChatGPT
License: MIT
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ...layers.quantizers import VectorQuantizer


class PatchEmbed(nn.Module):
    """Slice a 1‑D sequence into patches and embed each patch.

    Parameters
    ----------
    in_channels : int
        Number of EEG channels (C).
    patch_size  : int
        Length of each patch (P).
    stride      : int
        Stride between the *starts* of consecutive patches (S).  ``stride < patch_size``
        enables overlap.
    embed_dim   : int
        Transformer embedding dimension (D).
    max_len     : int, optional
        Maximum *number of patches per channel* expected at *model build time*.
        Positional embeddings are initialised to this length but can be resized
        later via ``resize_positional_embedding``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        patch_size: int,
        stride: int,
        embed_dim: int,
        max_len: int = 1024,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.stride = stride
        self.embed_dim = embed_dim

        # Linear projection shared across channels
        self.proj = nn.Linear(patch_size, embed_dim)

        # Learnable absolute positional embedding (shared across channels)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Parameters
        -----------
        x : Tensor, shape (B, C, L)
            Frequency‑domain EEG.

        Returns
        -------
        seq : Tensor, shape (B * C, N, D)
            Embedded patch sequence stacked over channels.
        N   : int
            Number of patches per channel in this forward call.
        """
        B, C, L = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}.")

        # Unfold creates *overlapping* patches of shape (B, C, patch_size, N)
        patches = x.unfold(
            dimension=-1, size=self.patch_size, step=self.stride
        )  # (B, C, P, N)
        P, N = self.patch_size, patches.shape[-1]
        patches = patches.contiguous().view(B * C, P, N).transpose(1, 2)  # (B*C, N, P)

        # Linear projection + position
        seq = self.proj(patches)  # (B*C, N, D)
        seq = seq + self.pos_embed[:, :N, :]
        return seq, N

    # ---------------------------------------------------------------------
    @torch.no_grad()
    def resize_positional_embedding(self, new_max_len: int) -> None:
        """Resize ``pos_embed`` if you need longer sequences after initialisation."""
        if new_max_len <= self.pos_embed.size(1):
            return  # already long enough
        emb_old = self.pos_embed.data
        emb_new = torch.zeros(
            1, new_max_len, self.embed_dim, device=emb_old.device, dtype=emb_old.dtype
        )
        emb_new[:, : emb_old.size(1)] = emb_old
        nn.init.trunc_normal_(emb_new[:, emb_old.size(1) :], std=0.02)
        self.pos_embed = nn.Parameter(emb_new)


def _build_transformer_layers(
    *,
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    attn_dropout: float = 0.1,
    norm_first: bool = True,
) -> nn.TransformerEncoder:
    """Factory helper for a vanilla ``nn.TransformerEncoder`` stack."""
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=norm_first,
    )
    # stash construction kwargs so ``_get_clones`` can replicate layers later
    encoder_layer.__dict__["_init_params"] = dict(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=norm_first,
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


def _build_transformer_decoder(
    *,
    num_layers: int,
    embed_dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    attn_dropout: float = 0.1,
    norm_first: bool = True,
) -> nn.TransformerDecoder:
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=norm_first,
    )
    decoder_layer.__dict__["_init_params"] = dict(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=int(embed_dim * mlp_ratio),
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=norm_first,
    )
    return nn.TransformerDecoder(decoder_layer, num_layers=num_layers)


class EEGFormer(nn.Module):
    """Full EEGFormer architecture (encoder → VQ → decoder).

    Parameters
    ----------
    in_channels         : int
        Number of EEG channels.
    seq_len             : int
        Input *signal length (L)* at **model‑build time**.  This is used to derive the
        maximum number of patches for positional embeddings; the model still
        works with shorter sequences at inference (they simply use a prefix of
        the embedding table).
    patch_size          : int
        Patch length ``P``.
    stride              : int
        Patch stride ``S``.
    embed_dim           : int
        Transformer hidden size ``D``.
    num_heads           : int
        Attention heads per layer.
    num_encoder_layers  : int
    num_decoder_layers  : int
    codebook_size       : int
        Number of discrete tokens ``K``.
    beta                : float, default 0.25
        Commitment loss multiplier for VQ.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        seq_len: int,
        patch_size: int = 64,
        stride: int = 32,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 3,
        codebook_size: int = 1024,
        beta: float = 0.25,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert (
            patch_size > 0 and stride > 0 and patch_size >= stride
        ), "Invalid patch hyper‑parameters"

        # ---- Patch embedding ------------------------------------------------
        # Maximum number of patches per channel at *build* time
        max_patches = math.floor((seq_len - patch_size) / stride) + 2
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_size=patch_size,
            stride=stride,
            embed_dim=embed_dim,
            max_len=max_patches,
        )

        # ---- Encoder --------------------------------------------------------
        self.encoder = _build_transformer_layers(
            num_layers=num_encoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # ---- Vector Quantiser ----------------------------------------------
        self.quantiser = VectorQuantizer(
            num_embeddings=codebook_size, embedding_dim=embed_dim, beta=beta
        )

        # ---- Decoder --------------------------------------------------------
        self.decoder = _build_transformer_decoder(
            num_layers=num_decoder_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.output_proj = nn.Linear(embed_dim, patch_size)

        # Keep hyper‑parameters for easy config export
        self.config = {
            "in_channels": in_channels,
            "seq_len": seq_len,
            "patch_size": patch_size,
            "stride": stride,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "codebook_size": codebook_size,
            "beta": beta,
            "mlp_ratio": mlp_ratio,
            "dropout": dropout,
        }

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run full reconstruction pass.

        Parameters
        ----------
        x : Tensor, shape (B, C, L)
            Pre‑processed (frequency‑domain) EEG batch.

        Returns
        -------
        out : dict
            * ``reconstruction`` : Tensor, shape (B, C, L_recon)
            * ``vq_loss``        : Tensor, scalar
            * ``code_indices``   : LongTensor, shape (B*C, N)
        """
        # 1) Patch embed (channel‑wise).  seq -> (B*C, N, D)
        seq, N = self.patch_embed(x)

        # 2) Encoder (shared weights across channels by virtue of stacking batch)
        h = self.encoder(seq)  # (B*C, N, D)

        # 3) Vector quantisation
        h_q, vq_loss, indices = self.quantiser(h)

        # 4) Decoder
        #    We do not use a separate *target* sequence; feed quantised tokens only.
        h_dec = self.decoder(tgt=h_q, memory=h_q)
        patches_out = self.output_proj(h_dec)  # (B*C, N, P)

        # 5) Re‑assemble to the original signal shape (B, C, L_recon)
        B = x.size(0)
        C = self.patch_embed.in_channels
        P = self.patch_embed.patch_size
        patches_out = patches_out.transpose(1, 2).view(B, C, P, N)  # (B, C, P, N)

        # Overlap‑add reconstruction (fold).  We build an overlap‑add kernel on the fly.
        fold = torch.nn.Fold(
            output_size=(1, self.config["seq_len"]),
            kernel_size=(1, P),
            stride=(1, self.config["stride"]),
        )
        _ = torch.nn.Unfold(kernel_size=(1, P), stride=(1, self.config["stride"]))
        # Stick to 2D API (N patches = width dim)
        patches_out_2d = patches_out.view(B, C * P, N)  # (B, C*P, N)
        patches_out_2d = patches_out_2d.unsqueeze(-2)  # (B, C*P, 1, N)

        recon = fold(patches_out_2d)  # (B, C*P, 1, L)
        recon = recon.view(B, C, P, -1)
        recon = recon.flatten(start_dim=2)  # (B, C, L_recon)

        return {
            "reconstruction": recon,
            "vq_loss": vq_loss,
            "code_indices": indices,
        }

    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return *continuous* latent representations before VQ (B*C, N, D)."""
        seq, _ = self.patch_embed(x)
        return self.encoder(seq)

    def quantise(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vector‑quantise a hidden representation.

        Returns (quantised, code_indices).
        """
        q, _, idx = self.quantiser(h)
        return q, idx

    def decode(
        self,
        q: torch.Tensor,
        *,
        N: Optional[int] = None,
        B: Optional[int] = None,
        C: Optional[int] = None,
    ) -> torch.Tensor:
        """Decode quantised tokens back to signal space.  If *N/B/C* are missing
        they are derived from ``q`` assuming flattened (B*C, N, D).
        """
        h_dec = self.decoder(tgt=q, memory=q)
        patches_out = self.output_proj(h_dec)  # (B*C, N, P)

        # Infer dims if not provided
        BC, Np, P = patches_out.shape[0], patches_out.shape[1], patches_out.shape[2]
        if B is None or C is None:
            if B is not None and C is None:
                C = BC // B
            elif B is None and C is not None:
                B = BC // C
            else:
                raise ValueError("Cannot infer batch/channel counts; provide B or C.")
        if N is None:
            N = Np

        patches_out = patches_out.transpose(1, 2).view(B, C, P, N)
        fold = torch.nn.Fold(
            output_size=(1, self.config["seq_len"]),
            kernel_size=(1, P),
            stride=(1, self.config["stride"]),
        )
        patches_out_2d = patches_out.view(B, C * P, N).unsqueeze(-2)
        recon = fold(patches_out_2d).view(B, C, P, -1).flatten(start_dim=2)
        return recon
