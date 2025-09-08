from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

from ..layers.megformer import GMMHead, PatchSignal, JetNVP
from ..layers.transformer_blocks import TransformerBlock


class MEGFormer(nn.Module):
    """JetFormer-style AR model for MEG forecasting with PatchPCA + Jet NVP."""

    def __init__(
        self,
        input_size: int,
        patch_size: int,
        latent_dim: int,
        flow_layers: int,
        d_model: int,
        n_layers: int,
        n_mixtures: int,
        context_length: int,
        forecast_steps: int,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "standard",
    ):
        super().__init__()
        self.patch = PatchSignal(input_size, patch_size)

        # Derive the actual token (patch) dimension from patch size and use it
        # consistently for the flow and the AR head to avoid shape mismatches.
        # derive patch (token) dim from patch_size
        patch_dim = patch_size * patch_size
        if latent_dim is not None and latent_dim != patch_dim:
            # be explicit: we ignore provided latent_dim but warn via exception
            # to avoid silent bugs
            raise ValueError(
                f"latent_dim ({latent_dim}) must equal patch_size**2 ({patch_dim})."
            )
        self.latent_dim = patch_dim
        self.flow = JetNVP(self.latent_dim, flow_layers)

        self.context_length = context_length
        self.forecast_steps = forecast_steps
        self.n_patches = (input_size // patch_size) ** 2

        # Token projection and time embeddings
        self.token_proj = nn.Linear(self.latent_dim, d_model)

        seq_len = context_length + forecast_steps
        self.time_pos = nn.Embedding(seq_len, d_model)

        self.space_pos = nn.Embedding(self.n_patches, d_model)

        attn_args["d_model"] = d_model
        mlp_args["d_model"] = d_model
        self.decoder = nn.ModuleList(
            [
                TransformerBlock(attn_args, mlp_args, attn_type, mlp_type)
                for _ in range(n_layers)
            ]
        )
        self.head = GMMHead(d_model, self.latent_dim, n_mixtures)

    @staticmethod
    def _causal_mask(S: int, device: torch.device):
        m = torch.full((S, S), float("-inf"), device=device)
        m = torch.triu(m, diagonal=1)  # (i,j)= -inf for j>i
        return m

    def _encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Flow‑encode sensor images into latents.
        Returns:
          z_tokens: (B*T, L, C)
          logdet_per_t: (B, T)   sum over patches and channels for each time step
        """
        B, H, W, T = x.shape
        z_tokens = self.patch(x)  # (B*T, L, C) with L=(H/p)*(W/p)
        L = z_tokens.shape[1]
        # reshape to 4D to feed flow (acts along last dim independently per token)
        h = w = int(L**0.5)
        z_grid = rearrange(z_tokens, "(b t) (h w) c -> (b t) h w c", b=B, t=T, h=h, w=w)
        z_flow, logdet_map = self.flow(z_grid, reverse=False)  # logdet_map: (B*T, h, w)
        # back to (B*T, L, C)
        z_tokens = rearrange(z_flow, "(b t) h w c -> (b t) (h w) c", b=B, t=T, h=h, w=w)
        # collect logdet per (B, T)
        logdet_per_bt = logdet_map.view(B, T, -1).sum(dim=-1)  # (B, T)
        return z_tokens, logdet_per_bt

    def forward(self, x: Tensor, reduce: str = "mean") -> Tensor:
        B, H, W, T = x.shape

        z, logdet = self._encode(x)
        n_patches = z.shape[1]

        # put time dimension into sequence
        z = rearrange(z, "(b t) l c -> b (t l) c", t=T)

        # AR decoder (teacher forcing)
        tok = self.token_proj(z)

        patch_ids = torch.arange(n_patches, device=x.device).repeat(T)  # [L]
        patch_ids = patch_ids.unsqueeze(0).expand(B, -1)  # [B, L]

        # time_id: 0…T_ctx-1 repeating *interleaved* with patches
        time_ids = torch.arange(T, device=x.device).repeat_interleave(n_patches)  # [L]
        time_ids = time_ids.unsqueeze(0).expand(B, -1)  # [B, L]

        tok = tok + self.space_pos(patch_ids) + self.time_pos(time_ids)

        for layer in self.decoder:
            tok = layer(tok, causal=True)

        nll_tok = self.head.nll(tok[:, :-1, :], z[:, 1:, :], reduce=reduce)
        return nll_tok, logdet.mean()

    @torch.no_grad()
    def forecast(self, x_context: Tensor, steps: Optional[int] = None) -> Tensor:
        """
        Autoregressively forecast `steps` future frames.
        Args:
            x_context: (B, H, W, T_ctx)
            steps: int (defaults to self.forecast_steps)
        Returns:
            (B, H, W, T_ctx + steps)
        """
        self.eval()
        steps = steps or self.forecast_steps
        B, H, W, T_ctx = x_context.shape

        # encode context
        z_ctx, _ = self._encode(x_context)  # (B*T_ctx, L, C)
        L = z_ctx.shape[1]
        seq = rearrange(z_ctx, "(b t) l c -> b (t l) c", b=B, t=T_ctx)  # (B, S_ctx, C)

        # generate tokens one-by-one; need steps*L tokens to form `steps` full frames
        for _ in range(steps * L):
            S = seq.shape[1]
            tok = self.token_proj(seq)

            # positions up to the current S
            patch_ids = (
                (torch.arange(S, device=seq.device) % L).unsqueeze(0).expand(B, -1)
            )
            time_ids = (
                (torch.arange(S, device=seq.device) // L).unsqueeze(0).expand(B, -1)
            )
            tok = (
                tok
                + self.space_pos(patch_ids)
                + self.time_pos(time_ids.clamp_max(self.time_pos.num_embeddings - 1))
            )

            for layer in self.decoder:
                tok = layer(tok, causal=True)

            # sample next latent token z_{S}
            logpi, mu, log_sigma = self.head(tok[:, -1:, :])  # last position only
            pi = torch.softmax(logpi.squeeze(1), dim=-1)  # (B, n_mix)
            mix_idx = torch.multinomial(pi, 1).squeeze(-1)  # (B,)
            mu = mu.squeeze(1)  # (B, n_mix, C)
            sigma = log_sigma.squeeze(1).exp()  # (B, n_mix, C)
            z_next = mu[torch.arange(B, device=seq.device), mix_idx]
            z_next = z_next + sigma[
                torch.arange(B, device=seq.device), mix_idx
            ] * torch.randn_like(z_next)
            seq = torch.cat([seq, z_next.unsqueeze(1)], dim=1)

        total_T = T_ctx + steps
        z_full = rearrange(seq, "b (t l) c -> (b t) l c", l=L)
        # reshape to grid and invert flow
        h = w = int(L**0.5)
        z_grid = rearrange(z_full, "(bt) (h w) c -> bt h w c", bt=B * total_T, h=h, w=w)
        x_grid, _ = self.flow(z_grid, reverse=True)
        x_tokens = rearrange(x_grid, "bt h w c -> bt (h w) c")
        x_rec = self.patch.unpatch(x_tokens, total_T)
        return x_rec
