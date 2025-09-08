from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F

from .transformer_blocks import TransformerBlock


class GMMHead(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, n_mix: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mix = n_mix
        self.proj = nn.Linear(d_model, n_mix * (2 * latent_dim + 1))

    def forward(self, h: torch.Tensor):
        out = self.proj(h)  # [..., n_mix*(2D+1)]
        out = out.view(*h.shape[:-1], self.n_mix, 2 * self.latent_dim + 1)

        logit_pi = out[..., 0]  # [..., n_mix]
        mu = out[..., 1 : 1 + self.latent_dim]  # [..., n_mix, D]
        log_sigma = out[..., 1 + self.latent_dim :]  # [..., n_mix, D]

        # clamp
        log_sigma = torch.clamp(log_sigma, -6, 6)

        log_pi = F.log_softmax(logit_pi, dim=-1)  # normalised log-π
        return log_pi, mu, log_sigma

    def nll(self, h: torch.Tensor, target: torch.Tensor, reduce="mean"):
        log_pi, mu, log_sigma = self(h)  # shapes as above

        target = target.unsqueeze(-2)  # broadcast to mix dim
        inv_sigma2 = torch.exp(-2.0 * log_sigma)  # 1/σ² for stability

        # log N(x | μ, σ) under diagonal covariance
        log_prob = -0.5 * (
            ((target - mu) ** 2) * inv_sigma2 + 2 * log_sigma + math.log(2 * math.pi)
        )
        log_prob = log_prob.sum(-1)  # sum over D
        log_mix = torch.logsumexp(log_pi + log_prob, dim=-1)  # mix over components

        nll = -log_mix
        return nll.mean() if reduce == "mean" else nll


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
                ViTCoupling(dim, n_patches, d_model, n_heads, depth)
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
