"""brain_models.py

PyTorch implementation of **BrainTokenizer** and **BrainOmni** (autoregressive
forecasting variant) as described in *BrainOmni: A Brain Foundation Model for
Unified EEG and MEG Signals* (Xiao *et al.* 2025).

This file purposefully focuses *only* on model definitions – no data‑loading,
training, or evaluation utilities are included.  Hyper‑parameters follow the
paper when explicit, otherwise sensible defaults are provided and can be tuned
externally.

Both models are **device‑agnostic** and make minimal assumptions about input
shape.

Key architectural adaptations versus the original paper
------------------------------------------------------
* **BrainOmniForecast** – replaces masked‑token prediction with causal
  *next‑token* forecasting (GPT‑style) over the discrete token stream emitted
  by BrainTokenizer.
* The Criss‑Cross blocks are kept, but temporal heads use a *causal* mask so
  future timesteps are hidden while spatial heads remain full‑context.
* The residual‑vector‑quantiser (RVQ) follows the EMA update & commitment‑loss
  formulation (§2.2) but exposes a *forward* pass suitable for inference (no
  EMA).
"""

from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn

try:  # Optional import; not required for forward usage
    from ..training.lightning import LitModel
except Exception:  # pragma: no cover - allow importing without lightning deps
    LitModel = None  # type: ignore
from ..layers.st_blocks import (
    CrissCrossBlock,
)
from .tokenizers.brainomnitokenizer import BrainOmniTokenizer


class BrainOmniForecast(nn.Module):
    """Autoregressive predictor over RVQ stage‑wise tokens.

    Parameters
    ----------
    codebook_size : int
        Size *K* of each RVQ codebook.
    num_quantizers : int
        Number of residual VQ stages *Nq*.
    latent_channels : int, default 16
        Latent source variables (C′ in the paper).
    emb_dim : int, default 512
        Transformer width.
    depth : int, default 12
        Number of Criss‑Cross blocks.
    num_heads : int, default 8
        Total attention heads (split half/half spatial‑temporal inside the
        block).
    dropout : float, default 0.1
    max_time_steps : int, default 1024
        Controls rotary/positional encodings.
    rotary_base : int, default 10000
    """

    def __init__(
        self,
        attn_args: dict,
        codebook_size: int = 1024,
        num_quantizers: int = 4,
        latent_channels: int = 16,
        emb_dim: int = 512,
        depth: int = 12,
        dropout: float = 0.1,
        max_time_steps: int = 1024,
        rotary_base: int = 10000,
        attn_type: str = "standard",
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size

        # ── input embeddings ────────────────────────────────────────────
        self.stage_embeds = nn.ModuleList(
            [nn.Embedding(codebook_size, emb_dim) for _ in range(num_quantizers)]
        )
        self.channel_emb = nn.Embedding(latent_channels, emb_dim)
        self.depth_emb = nn.Embedding(num_quantizers, emb_dim)

        # Pre‑compute rotary positional encodings (sin/cos interleaved)
        rope = self._build_rotary_emb(emb_dim, max_time_steps, rotary_base)
        # shape: (max_T, d_model) -> register for easy slicing and device move
        self.register_buffer("rope", rope, persistent=False)

        # Transformer backbone – Criss‑Cross keeps spatial/temporal heads
        self.layers = nn.ModuleList(
            [
                CrissCrossBlock(
                    attn_args=attn_args,
                    dropout=dropout,
                    attn_type=attn_type,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.RMSNorm(emb_dim)

        # ── output heads – one per stage ────────────────────────────────
        self.stage_heads = nn.ModuleList(
            [
                nn.Linear(emb_dim, codebook_size, bias=False)
                for _ in range(num_quantizers)
            ]
        )

    @staticmethod
    def _build_rotary_emb(dim: int, max_len: int, base: int = 10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len, dtype=torch.float)
        freqs = torch.einsum("t,d->td", t, inv_freq)  # outer product
        return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # (T, dim)

    def forward(self, codes: torch.Tensor, embeds: torch.Tensor = None) -> torch.Tensor:
        """Predict logits for the *next* time‑step.

        Parameters
        ----------
        codes : (B, C_latent, Nq, T) int64
            Tokens **up‑to** the current time ``t``.  During training usual
            teacher‑forcing (shift‑by‑one) should be applied externally so
            that ``codes[..., -1]`` is the *input* and the model is trained
            to predict the gold tokens at that position.

        Returns
        -------
        logits : (B, C_latent, Nq, T, K) float32
            Stage‑wise distribution over the codebook for each latent
            channel & time‑step.
        """
        B, C_latent, Nq, T = codes.shape
        assert C_latent == self.latent_channels, "latent channel mismatch"
        assert Nq == self.num_quantizers, "quantizer depth mismatch"
        device = codes.device

        # ── embed & sum residual‑stage vectors ─────────────────────────
        #   RVQ decodes by *summing* stage embeddings; we mirror that.
        if embeds is None:
            embeds = 0.0
            for i in range(Nq):
                emb_i = self.stage_embeds[i](codes[:, :, i, :])  # (B, C, T, D)
                depth_bias = self.depth_emb.weight[i]  # (D,)
                embeds = embeds + (emb_i + depth_bias)  # broadcast (B,C,T,D)

        # add per‑channel embedding
        ch = self.channel_emb(torch.arange(C_latent, device=device))  # (C, D)
        emb = embeds + ch[None, :, None, :]

        # add rotary positional encodings
        rope_slice = self.rope[:T]  # (T,d)
        emb = emb + rope_slice.view(1, 1, T, emb.shape[-1])

        # flatten to sequence  (B, S, D) where S = C_latent * T
        tok = emb.view(B, C_latent * T, -1)

        # ── Transformer ────────────────────────────────────────────────
        x = tok
        for blk in self.layers:
            x = blk(x, T)
        x = self.norm(x)

        # ── project to per‑stage logits ────────────────────────────────
        logits_per_stage: List[torch.Tensor] = []
        for head in self.stage_heads:
            logit = head(x)  # (B, S, K)
            logit = logit.view(B, C_latent, T, self.codebook_size)  # (B, C, T, K)
            logits_per_stage.append(logit)
        # stack along stage dimension – (B, C, Nq, T, K)
        logits = torch.stack(logits_per_stage, dim=2)
        return logits.contiguous()


class BrainOmniSystem(nn.Module):
    """Tokenizer + Forecast wrapper with the new factorised forecast model."""

    def __init__(
        self,
        codebook_size: int,
        num_quantizers: int,
        latent_channels: int,
        tokenizer: dict,
        forecaster: dict,
        tokenizer_path: str = None,
        train_tokenizer: bool = False,
    ) -> None:
        super().__init__()

        self.train_tokenizer = train_tokenizer

        if tokenizer_path is not None:
            lit = LitModel.load_from_checkpoint(tokenizer_path, strict=False)
            self.tokenizer = lit.model

            # check if model is compiled
            if hasattr(self.tokenizer, "_orig_mod"):
                self.tokenizer = self.tokenizer._orig_mod

        else:
            self.tokenizer = BrainOmniTokenizer(
                latent_channels=latent_channels,
                codebook_size=codebook_size,
                num_quantizers=num_quantizers,
                **tokenizer,
            )

        # freeze tokenizer during autoregressive training (optional)

        if not train_tokenizer:
            for p in self.tokenizer.parameters():
                p.requires_grad_(False)

        self.forecaster = BrainOmniForecast(
            codebook_size=codebook_size,
            num_quantizers=num_quantizers,
            latent_channels=latent_channels,
            **forecaster,
        )

    def forward_train(
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                logits: (B, C_latent, Nq, T, K)
                codes: (B, C_latent, Nq, T') - these are the tokens to predict
        """
        # Tokenise, TODO: extra loss for reconstruction
        # (B, C', W, Nq)
        x_hat, residuals, nearest, embeds, codes = self.tokenizer(
            inputs, return_reconstruction=True
        )
        codes = codes.permute(0, 1, 3, 2).contiguous()  # (B, C', Nq, T')

        # Forecast
        logits = self.forecaster(codes[:, :, :, :-1], embeds[:, :, :-1, :])
        return logits, codes[:, :, :, 1:]

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                logits: (B, C_latent, Nq, T, K)
                codes: (B, C_latent, Nq, T') - these are the tokens to predict
        """
        if self.train_tokenizer:
            return self.forward_train(inputs)

        self.tokenizer.eval()

        # Tokenise
        with torch.no_grad():
            # (B, C', W, Nq)
            codes = self.tokenizer(inputs, return_reconstruction=False)
        codes = codes.permute(0, 1, 3, 2).contiguous()  # (B, C', Nq, T')

        # Forecast
        logits = self.forecaster(codes[:, :, :, :-1])
        return logits, codes[:, :, :, 1:]
