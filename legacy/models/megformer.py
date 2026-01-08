from __future__ import annotations
from typing import Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange

import math
import torch.nn.functional as F

from ..layers.transformer_blocks import TransformerBlock


class GMMHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        n_mix: int,
        min_log_sigma: float = -6.0,
        max_log_sigma: float = 6.0,
        var_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mix = n_mix
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        self.var_reg_weight = var_reg_weight
        self.proj = nn.Linear(d_model, n_mix * (2 * latent_dim + 1))

    def forward(self, h: torch.Tensor):
        out = self.proj(h)  # [..., n_mix*(2D+1)]
        out = out.view(*h.shape[:-1], self.n_mix, 2 * self.latent_dim + 1)

        logit_pi = out[..., 0]  # [..., n_mix]
        mu = out[..., 1 : 1 + self.latent_dim]  # [..., n_mix, D]
        log_sigma = out[..., 1 + self.latent_dim :]  # [..., n_mix, D]

        # clamp
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)

        log_pi = F.log_softmax(logit_pi, dim=-1)  # normalised log-π
        return log_pi, mu, log_sigma

    def nll(
        self,
        h: torch.Tensor,
        target: torch.Tensor,
        reduce: str = "mean",
        mask: torch.Tensor | None = None,
    ):
        log_pi, mu, log_sigma = self(h)  # shapes as above

        target = target.unsqueeze(-2)  # broadcast to mix dim
        inv_sigma2 = torch.exp(-2.0 * log_sigma)  # 1/σ² for stability

        # log N(x | μ, σ) under diagonal covariance
        log_prob = -0.5 * (
            ((target - mu) ** 2) * inv_sigma2 + 2 * log_sigma + math.log(2 * math.pi)
        )
        log_prob = log_prob.sum(-1)  # sum over D
        log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # mix over components

        nll = -log_mix  # [...,]

        # variance regularizer to discourage vanishing σ
        if self.var_reg_weight > 0.0:
            reg = torch.clamp(-log_sigma, min=0.0).mean(dim=(-1, -2))  # [...,]
            nll = nll + self.var_reg_weight * reg

        if mask is not None:
            nll = nll * mask
            denom = mask.sum()
        else:
            denom = nll.numel()

        if reduce == "mean":
            return nll.sum() / denom.clamp(min=1.0)
        if reduce == "sum":
            return nll.sum()
        return nll


class PatchSignal(nn.Module):
    def __init__(
        self,
        input_size: int,
        patch_size: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Take in a sensor space image of h x w x t and flatten it into a
        sequence of patches. t is a couple of timesteps.
        TODO: t will have to be treated as 3D conv later
        x: [B, H, W, T]
        """
        x = rearrange(
            x,
            "b (h p) (w q) t -> (b t) (h w) (p q)",
            p=self.patch_size,
            q=self.patch_size,
        )

        return x

    def unpatch(self, x: Tensor, T: int) -> Tensor:
        """Reconstruct sensor images from patch sequence.

        Args:
            x: ``(B*T, n_patches, patch_dim)`` tensor.
            T: total number of timesteps.

        Returns:
            Tensor of shape ``(B, input_size, input_size, T)``.
        """
        h = w = self.input_size // self.patch_size
        x = rearrange(
            x,
            "(b t) (h w) (p q) -> b (h p) (w q) t",
            t=T,
            h=h,
            w=w,
            p=self.patch_size,
            q=self.patch_size,
        )
        return x


class MLPCoupling(nn.Module):
    """
    Affine coupling y2 = x2 * exp(s(x1)) + t(x1), operating along the last dim.
    Acts independently across all leading dimensions (tokens, time, batch).
    """

    def __init__(self, dim: int, width: int = 256):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Coupling layer expects even dim, got {dim}.")
        self.net = nn.Sequential(
            nn.Linear(dim // 2, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, dim),  # outputs [s, t] packed
        )

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        x1, x2 = x.chunk(2, dim=-1)
        st = self.net(x1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * math.log(2)  # stabilise

        if reverse:
            y2 = (x2 - t) * torch.exp(-s)
            logdet = -s.sum(dim=-1)
        else:
            y2 = x2 * torch.exp(s) + t
            logdet = s.sum(dim=-1)

        y = torch.cat([x1, y2], dim=-1)
        return y, logdet


class JetNVP(nn.Module):
    """Stack of *L* MLP coupling layers + random permutations."""

    def __init__(
        self, dim: int, num_layers: int = 8, hidden_width: int = 128, ps: int = 1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPCoupling(dim, width=hidden_width) for _ in range(num_layers)]
        )
        perm = torch.randperm(dim)
        self.register_buffer("perm_fwd", perm, persistent=False)
        self.register_buffer("perm_inv", perm.argsort(), persistent=False)

    def _perm(self, x: Tensor, reverse: bool):
        return x[..., self.perm_inv if reverse else self.perm_fwd]

    def forward(self, x: Tensor, *, reverse: bool = False):
        # x: (BT, H, W, C)
        BT, H, W, C = x.shape
        z = rearrange(x, "bt h w c -> bt (h w) c")

        logdet = torch.zeros(z.shape[:-1], device=x.device)
        layers = self.layers[::-1] if reverse else self.layers
        for layer in layers:
            z = self._perm(z, reverse)
            z, ld = layer(z, reverse)
            logdet = logdet + ld

        # reshape back to grid for downstream code
        z_grid = rearrange(z, "bt (h w) c -> bt h w c", h=H, w=W)
        logdet_map = logdet.view(BT, H, W)
        return z_grid, logdet_map


class ViTCoupling(nn.Module):
    """Affine coupling *y₂ = x₂·exp(s) + t* with a tiny ViT predicting *(s,t).*"""

    def __init__(self, dim: int, depth: int = 2, width: int = 128, n_heads: int = 4):
        super().__init__()
        attn_args = {
            "d_model": width,
            "nheads": n_heads,
        }
        mlp_args = {
            "d_model": width,
            "d_ff": 4 * width,
        }
        self.encoder = torch.nn.ModuleList(
            [TransformerBlock(attn_args, mlp_args) for _ in range(depth)]
        )

        self.proj_in = nn.Linear(dim // 2, width)
        self.proj_out = nn.Linear(width, dim // 2 * 2)

    def forward(self, x: Tensor, reverse: bool = False) -> Tuple[Tensor, Tensor]:
        x1, x2 = x.chunk(2, dim=-1)
        h = self.proj_in(x1)
        h = self.encoder(h)
        s, t = self.proj_out(h).chunk(2, dim=-1)
        s = torch.tanh(s) * math.log(2)  # scale stabiliser

        if reverse:
            y2 = (x2 - t) * torch.exp(-s)
            logdet = -s.sum(-1)
        else:
            y2 = x2 * torch.exp(s) + t
            logdet = s.sum(-1)
        y = torch.cat([x1, y2], dim=-1)
        return y, logdet


class ViTCouplingChatgpt(nn.Module):
    def __init__(
        self,
        dim: int,
        n_patches: int,
        d_model: int = 256,
        n_heads: int = 8,
        depth: int = 4,
    ):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.n_patches = n_patches

        self.in_proj = nn.Linear(dim // 2, d_model)
        self.pos = nn.Embedding(n_patches, d_model)  # spatial pos within a frame

        attn_args = {
            "d_model": d_model,
            "nheads": n_heads,
        }
        mlp_args = {
            "d_model": d_model,
            "d_ff": 4 * d_model,
        }
        self.encoder = torch.nn.ModuleList(
            [TransformerBlock(attn_args, mlp_args) for _ in range(depth)]
        )
        self.out_proj = nn.Linear(d_model, dim)  # predicts [s, t] for the second half

    def forward(self, x, reverse: bool = False):
        # x: (N, L, C)   (flattened per-frame tokens)
        x1, x2 = x.chunk(2, dim=-1)  # split along channels
        L = x1.shape[1]
        h = self.in_proj(x1) + self.pos(torch.arange(L, device=x.device))[None, :, :]
        h = self.encoder(h)  # attend across patches (within the frame)
        st = self.out_proj(h)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * math.log(2)

        if reverse:
            y2 = (x2 - t) * torch.exp(-s)
            logdet = -s.sum(dim=(-1, -2))  # sum over patches and channels/2
        else:
            y2 = x2 * torch.exp(s) + t
            logdet = s.sum(dim=(-1, -2))
        y = torch.cat([x1, y2], dim=-1)
        return y, logdet


class JetViTFlow(nn.Module):
    def __init__(
        self,
        dim: int,
        n_patches: int,
        num_layers: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        depth: int = 4,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ViTCoupling(dim, depth=depth, width=d_model, n_heads=n_heads)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, reverse: bool = False):
        # x: (BT, H, W, C)  -> flatten to (BT, L, C)
        BT, H, W, C = x.shape
        L = H * W
        z = x.view(BT, L, C)
        logdet = torch.zeros(BT, device=x.device)
        layers = reversed(self.layers) if reverse else self.layers
        for layer in layers:
            z, ld = layer(z, reverse=reverse)
            logdet = logdet + ld
        z = z.view(BT, H, W, C)
        return z, logdet.view(BT)


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
        min_log_sigma: float = -6.0,
        max_log_sigma: float = 6.0,
        var_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.patch = PatchSignal(input_size, patch_size)

        # Derive the actual token (patch) dimension from patch size and use it
        # consistently for the flow and the AR head to avoid shape mismatches.
        # derive patch (token) dim from patch_size
        patch_dim = patch_size * patch_size
        if patch_dim % 2 != 0:
            # RealNVP-style couplings split the channel dim in half; require even
            raise ValueError(f"patch_size**2 must be even, got {patch_dim}.")
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
        self.head = GMMHead(
            d_model,
            self.latent_dim,
            n_mixtures,
            min_log_sigma=min_log_sigma,
            max_log_sigma=max_log_sigma,
            var_reg_weight=var_reg_weight,
        )

        # patch mask for sparse layouts; lazily initialized on first forward
        self.register_buffer("patch_mask", None, persistent=False)
        self._patch_mask_ready = False

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
          logdet_per_token: (B, T, L) per‑patch |det J|
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
        # per-token logdet mapped back to (B, T, L)
        logdet_per_token = logdet_map.view(B, T, h * w)
        return z_tokens, logdet_per_token

    def _init_patch_mask(
        self,
        mask_info: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if self._patch_mask_ready:
            return

        if mask_info is not None and "row_idx" in mask_info and "col_idx" in mask_info:
            row_idx = mask_info["row_idx"][0]
            col_idx = mask_info["col_idx"][0]
            w = self.input_size // self.patch_size
            patch_ids = (row_idx // self.patch_size) * w + (col_idx // self.patch_size)
            pm = torch.zeros(self.n_patches, dtype=torch.float32, device=device)
            pm[patch_ids.long()] = 1.0
        else:
            pm = torch.ones(self.n_patches, dtype=torch.float32, device=device)

        self.patch_mask = pm
        self._patch_mask_ready = True

    def forward(self, x, reduce: str = "mean") -> Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]

        B, H, W, T = x.shape

        # self._init_patch_mask(input_dict, device=x.device)

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

        nll_tok = self.head.nll(tok[:, :-1, :], z[:, 1:, :], reduce="none")  # (B,S-1)
        logdet_tok = rearrange(logdet, "b t l -> b (t l)")[:, 1:]  # (B,S-1)

        # token-wise GMM nll; align flow logdet per-patch to the same tokens
        if self._patch_mask_ready:
            mask_seq = self.patch_mask.repeat(T)  # (S_ctx)
            mask_pred = mask_seq[1:].to(x.device).unsqueeze(0)  # (1,S-1)
        else:
            mask_pred = torch.ones(*nll_tok.shape, device=x.device)

        # apply patch mask to ignore blank patches
        nll_tok = nll_tok * mask_pred
        logdet_tok = logdet_tok * mask_pred

        if reduce == "mean":
            denom = mask_pred.sum().clamp(min=1.0)
            return nll_tok.sum() / denom, -logdet_tok.sum() / denom
        if reduce == "sum":
            return nll_tok.sum(), -logdet_tok.sum()
        return nll_tok, -logdet_tok

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
        global_seq = rearrange(
            z_ctx, "(b t) l c -> b (t l) c", b=B, t=T_ctx
        )  # (B, S_ctx, C)

        # maintain a sliding window so decoder input never exceeds available time ids
        max_time_ids = self.time_pos.num_embeddings
        max_tokens = max_time_ids * L
        window_seq = global_seq
        window_start = 0  # index of the first token currently in the window
        if window_seq.shape[1] > max_tokens:
            tokens_to_drop = window_seq.shape[1] - max_tokens
            window_seq = window_seq[:, tokens_to_drop:, :]
            window_start += tokens_to_drop

        # precompute static patch embeddings once
        patch_positions = torch.arange(L, device=window_seq.device, dtype=torch.long)
        patch_emb = self.space_pos(patch_positions)  # (L, d_model)

        # generate tokens one-by-one; need steps*L tokens to form `steps` full frames
        for _ in tqdm(range(steps * L), desc="Forecasting "):
            S = window_seq.shape[1]
            tok = self.token_proj(window_seq)

            # positions up to the current S (relative to the global sequence)
            token_positions = torch.arange(
                window_start,
                window_start + S,
                device=window_seq.device,
                dtype=torch.long,
            )
            patch_ids = token_positions % L
            time_ids_full = token_positions // L
            time_offset = window_start // L  # min of time_ids_full
            time_ids = (time_ids_full - time_offset).unsqueeze(0).expand(B, -1)

            tok = tok + patch_emb[patch_ids].unsqueeze(0) + self.time_pos(time_ids)

            for layer in self.decoder:
                tok = layer(tok, causal=True)

            # sample next latent token z_{S}
            logpi, mu, log_sigma = self.head(tok[:, -1:, :])  # last position only
            pi = torch.softmax(logpi.squeeze(1), dim=-1)  # (B, n_mix)
            mix_idx = torch.multinomial(pi, 1).squeeze(-1)  # (B,)
            mu = mu.squeeze(1)  # (B, n_mix, C)
            sigma = log_sigma.squeeze(1).exp()  # (B, n_mix, C)
            z_next = mu[torch.arange(B, device=window_seq.device), mix_idx]
            z_next = z_next + sigma[
                torch.arange(B, device=window_seq.device), mix_idx
            ] * torch.randn_like(z_next)
            global_seq = torch.cat([global_seq, z_next.unsqueeze(1)], dim=1)
            window_seq = torch.cat([window_seq, z_next.unsqueeze(1)], dim=1)
            if window_seq.shape[1] > max_tokens:
                tokens_to_drop = L
                window_seq = window_seq[:, tokens_to_drop:, :]
                window_start += tokens_to_drop

        total_T = T_ctx + steps
        z_full = rearrange(global_seq, "b (t l) c -> (b t) l c", l=L)
        # reshape to grid and invert flow
        h = w = int(L**0.5)
        z_grid = rearrange(z_full, "(bt) (h w) c -> bt h w c", bt=B * total_T, h=h, w=w)
        x_grid, _ = self.flow(z_grid, reverse=True)
        x_tokens = rearrange(x_grid, "bt h w c -> bt (h w) c")
        x_rec = self.patch.unpatch(x_tokens, total_T)
        return x_rec
