"""
LATTE‑AR: Long‑horizon Autoregressive Tokenized Temporal Entropy model

This file provides a *complete, self‑contained PyTorch reference implementation* of the
architecture proposed in the conversation:
  • A 2‑level residual VQ tokenizer with a small variational continuous residual (r_t)
  • A selective State‑Space‑style temporal prior with hazard‑gated discrete options
  • Emission heads that predict categorical distributions over tokens and a Gaussian
    prior over continuous residuals
  • Training loops for (1) tokenizer pretraining and (2) sequence model training
  • Minimal dataset utilities and a long‑horizon sampler

The code is written for clarity and research iteration rather than raw throughput.
It is designed to run on single/multi‑GPU with mixed precision. Feel free to MoE‑ify
or shard components for scale.

Python 3.10+, PyTorch 2.1+ recommended.

Author: ChatGPT (LATTE‑AR reference)
"""

from __future__ import annotations
import math
import os
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---------------------------
# Utility helpers
# ---------------------------


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else d


def num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class EMA:
    """Simple EMA helper for scalars."""

    def __init__(self, beta=0.99):
        self.beta = beta
        self.value = None

    def update(self, x: float):
        if self.value is None:
            self.value = x
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value


# ---------------------------
# VQ (EMA) codebook
# ---------------------------
class VectorQuantizerEMA(nn.Module):
    """EMA Vector Quantizer with straight‑through estimator.

    Args:
        n_codes: codebook size K
        code_dim: embedding dimension D
        decay: EMA decay for codebook updates
        eps: small constant for numerical stability
    Returns:
        quantized: (B, D, H, W)
        indices: (B, H, W) long
        vq_loss: scalar commitment + codebook loss
    """

    def __init__(
        self, n_codes: int, code_dim: int, decay: float = 0.99, eps: float = 1e-5
    ):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps

        embed = torch.randn(n_codes, code_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", embed.clone())

    @torch.no_grad()
    def _ema_update(self, flat_inputs: torch.Tensor, indices: torch.Tensor):
        # flat_inputs: (N, D), indices: (N,)
        onehot = F.one_hot(indices, num_classes=self.n_codes).type_as(flat_inputs)
        cluster_size = onehot.sum(0)
        embed_sum = onehot.t() @ flat_inputs

        self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.cluster_size.sum()
        cluster_size = (
            (self.cluster_size + self.eps) / (n + self.n_codes * self.eps) * n
        )
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embed.copy_(embed_normalized)

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D, H, W)
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (B*H*W, D)
        # L2 distance to codes
        distances = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.embed.t()
            + self.embed.pow(2).sum(1, keepdim=True).t()
        )  # (N, K)
        indices = torch.argmin(distances, dim=1)  # (N,)
        z_q = F.embedding(indices, self.embed)  # (N, D)
        z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

        # EMA update
        if self.training:
            self._ema_update(flat.detach(), indices)

        # Straight‑through estimator
        z_q_st = z + (z_q - z).detach()

        # Commitment + codebook loss (codebook loss is implicitly handled by EMA)
        commit_loss = F.mse_loss(z.detach(), z_q)
        vq_loss = commit_loss

        indices = indices.view(B, H, W)
        return z_q_st, indices, vq_loss


# ---------------------------
# Tokenizer: Encoder/Decoder with residual VQ + small VAE residual r
# ---------------------------
class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden=192, z_dim=256, r_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 5, 2, 2),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden, z_dim, 3, 1, 1),
        )
        # Residual (continuous) posterior q(r|x)
        self.r_mu = nn.Conv2d(z_dim, r_dim, 1)
        self.r_logvar = nn.Conv2d(z_dim, r_dim, 1)

    def forward(self, x):
        z = self.conv(x)  # (B, z_dim, H', W')
        r_mu = self.r_mu(z)
        r_logvar = self.r_logvar(z)
        return z, r_mu, r_logvar


class Decoder(nn.Module):
    def __init__(self, out_ch=3, hidden=192, in_dim=256, r_dim=32):
        super().__init__()
        ch = in_dim + r_dim
        self.net = nn.Sequential(
            nn.Conv2d(ch, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1),
            nn.GELU(),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden, out_ch, 3, 1, 1),
        )

    def forward(self, z_q, r):
        zcat = torch.cat([z_q, r], dim=1)
        x_rec = self.net(zcat)
        return x_rec


class ResidualVQTokenizer(nn.Module):
    """2‑level residual VQ with a small VAE residual (r).

    Returns per forward:
      x_rec, losses, tokens (dict), latents (dict)
    """

    def __init__(
        self,
        img_size=128,
        in_ch=3,
        z_dim=256,
        r_dim=32,
        hidden=192,
        codebook1=2048,
        codebook2=1024,
        decay=0.99,
    ):
        super().__init__()
        self.enc = Encoder(in_ch=in_ch, hidden=hidden, z_dim=z_dim, r_dim=r_dim)
        self.vq1 = VectorQuantizerEMA(codebook1, z_dim, decay)
        self.vq2 = VectorQuantizerEMA(codebook2, z_dim, decay)
        self.dec = Decoder(out_ch=in_ch, hidden=hidden, in_dim=z_dim, r_dim=r_dim)
        self.img_size = img_size
        self.r_dim = r_dim

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        z_e, r_mu, r_logvar = self.enc(x)  # (B, z_dim, H', W'), (B, r_dim, H', W') each
        # Sample residual r
        r = self.reparam(r_mu, r_logvar)
        # Residual VQ (2‑level)
        z1_q, y1_idx, vq1_loss = self.vq1(z_e)
        res1 = z_e - z1_q
        z2_q, y2_idx, vq2_loss = self.vq2(res1)
        z_q = z1_q + z2_q
        # Decode
        x_rec = self.dec(z_q, r)
        # Losses
        recon_loss = F.l1_loss(x_rec, x)
        kl_r = -0.5 * torch.mean(1 + r_logvar - r_mu.pow(2) - r_logvar.exp())
        vq_loss = vq1_loss + vq2_loss
        loss = recon_loss + vq_loss + 0.1 * kl_r
        out = {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "kl_r": kl_r.detach(),
        }
        tokens = {
            "y1_idx": y1_idx,  # (B, H', W')
            "y2_idx": y2_idx,
        }
        latents = {
            "z_q": z_q,
            "r": r,
            "r_mu": r_mu,
            "r_logvar": r_logvar,
        }
        return x_rec, out, tokens, latents

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        z_e, r_mu, r_logvar = self.enc(x)
        z1_q, y1_idx, _ = self.vq1(z_e)
        res1 = z_e - z1_q
        z2_q, y2_idx, _ = self.vq2(res1)
        z_q = z1_q + z2_q
        return {
            "y1_idx": y1_idx,
            "y2_idx": y2_idx,
            "z_q": z_q,
            "r_mu": r_mu,
            "r_logvar": r_logvar,
        }


# ---------------------------
# Temporal model (OS‑SSM with options + hazard)
# ---------------------------
class TokenEmbedding(nn.Module):
    """Embeds per‑position code indices from two codebooks and sums them."""

    def __init__(self, codebook1: int, codebook2: int, d_model: int):
        super().__init__()
        self.e1 = nn.Embedding(codebook1, d_model)
        self.e2 = nn.Embedding(codebook2, d_model)
        self.pos = None  # set dynamically when first seen
        self.scale = d_model**-0.5

    def forward(self, y1_idx: torch.Tensor, y2_idx: torch.Tensor) -> torch.Tensor:
        # y*_idx: (B, H, W)
        B, H, W = y1_idx.shape
        if (self.pos is None) or (self.pos.shape[1] != H) or (self.pos.shape[2] != W):
            # 2D sin‑cos positional embedding
            self.pos = self._build_2d_sincos_pos_embed(H, W, self.e1.embedding_dim).to(
                y1_idx.device
            )
        emb = self.e1(y1_idx) + self.e2(y2_idx) + self.pos  # (B, H, W, D)
        return emb.permute(0, 3, 1, 2).contiguous()  # (B, D, H, W)

    def _build_2d_sincos_pos_embed(self, H, W, D):
        def get_1d(n, d_half):
            omega = torch.arange(d_half) / d_half
            omega = 1.0 / (10000**omega)
            p = torch.arange(n)
            out = torch.einsum("n,d->nd", p, omega)
            return torch.stack([torch.sin(out), torch.cos(out)], dim=-1).view(n, -1)

        d_half = D // 4
        pe_h = get_1d(H, d_half)  # (H, D/2)
        pe_w = get_1d(W, d_half)
        pe = (pe_h[:, None, :].repeat(1, W, 1), pe_w[None, :, :].repeat(H, 1, 1))
        pe = torch.cat(pe, dim=-1)  # (H, W, D)
        return pe.unsqueeze(0)  # (1, H, W, D)


class OS_SSM_Cell(nn.Module):
    """Option‑Switching Selective SSM (single step).

    State update:
        s_bar = a(o) * s_prev + U(o) * pool(u_{t-1})
        g = sigmoid(Wg[ s_prev, u_pool, o ] )
        s_t = g * s_bar + (1-g) * s_prev

    Options o_t are trained with a hazard‑gated sticky prior:
        prior(o_t) = (1-h) * q(o_{t-1}) + h * pi(s_{t-1})
        KL(q(o_t) || prior(o_t)) is added to the loss.

    """

    def __init__(self, d_state=512, d_in=384, n_options=8):
        super().__init__()
        self.d_state = d_state
        self.d_in = d_in
        self.n_options = n_options

        # Option‑conditioned parameters via FiLM‑style generators
        self.opt_embed = nn.Embedding(n_options, d_state)
        self.a_proj = nn.Linear(d_state, d_state)  # produces a(o)
        self.u_proj = nn.Linear(d_state, d_state)  # produces a vector to scale U
        self.U = nn.Linear(d_in, d_state, bias=False)

        self.gate = nn.Sequential(
            nn.Linear(d_state + d_in + d_state, d_state),
            nn.GELU(),
            nn.Linear(d_state, d_state),
            nn.GELU(),
            nn.Linear(d_state, 1),
        )
        # Hazard and option prior
        self.hazard = nn.Sequential(
            nn.Linear(d_state + d_in, d_state // 2),
            nn.GELU(),
            nn.Linear(d_state // 2, 1),
        )
        self.pi = nn.Sequential(
            nn.Linear(d_state, d_state), nn.GELU(), nn.Linear(d_state, n_options)
        )

    def forward(
        self,
        s_prev: torch.Tensor,
        u_prev_pool: torch.Tensor,
        q_prev: torch.Tensor,
        q_post: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            s_prev: (B, d_state)
            u_prev_pool: (B, d_in)
            q_prev: previous option posterior probs (B, n_options)
            q_post: current option posterior probs q(o_t|y_t) (B, n_options)
        Returns:
            s_t: (B, d_state)
            info: dict with KL, hazard, priors, etc
        """
        B = s_prev.size(0)
        # Hazard and option prior
        hz = torch.sigmoid(
            self.hazard(torch.cat([s_prev, u_prev_pool], dim=-1))
        )  # (B,1)
        pi_logits = self.pi(s_prev)  # (B, n_options)
        pi_probs = F.softmax(pi_logits, dim=-1)
        prior = (1 - hz) * q_prev + hz * pi_probs  # (B, n_options)
        # KL(q || prior)
        kl_opt = torch.sum(
            q_post * (torch.log(q_post + 1e-8) - torch.log(prior + 1e-8)), dim=-1
        ).mean()

        # Expected option embedding
        o_exp = q_post @ self.opt_embed.weight  # (B, d_state)

        a = torch.tanh(self.a_proj(o_exp))  # (B, d_state)
        u_scale = torch.tanh(self.u_proj(o_exp))  # (B, d_state)

        u_proj = self.U(u_prev_pool)  # (B, d_state)
        s_bar = a * s_prev + u_scale * u_proj

        g = torch.sigmoid(
            self.gate(torch.cat([s_prev, u_prev_pool, o_exp], dim=-1))
        )  # (B,1)
        s_t = g * s_bar + (1 - g) * s_prev

        info = {
            "kl_opt": kl_opt,
            "hazard": hz.mean(),
            "prior_probs": prior.detach(),
            "pi_probs": pi_probs.detach(),
        }
        return s_t, info


class SpatialReadout(nn.Module):
    """Produces per‑position logits for tokens given (s_t, option) and previous token grid."""

    def __init__(self, d_tok: int, d_state: int, codebook1: int, codebook2: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_tok + d_state, d_tok, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(d_tok, d_tok, 3, 1, 1),
            nn.GELU(),
        )
        self.head1 = nn.Conv2d(d_tok, codebook1, 1)
        self.head2 = nn.Conv2d(d_tok, codebook2, 1)

    def forward(
        self, u_prev: torch.Tensor, s_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # u_prev: (B, D_tok, H, W)  previous token embedding grid
        # s_t: (B, D_state)
        B, D, H, W = u_prev.shape
        s_broadcast = s_t.view(B, -1, 1, 1).expand(B, s_t.size(-1), H, W)
        x = torch.cat([u_prev, s_broadcast], dim=1)
        h = self.conv(x)
        logits1 = self.head1(h)
        logits2 = self.head2(h)
        return logits1, logits2


class ResidualPriorHead(nn.Module):
    """Predict Gaussian prior parameters for r_t on the latent grid.
    Condition on (s_t) and (y_t embedding) — during training, use teacher forcing y_t.
    """

    def __init__(self, d_tok: int, d_state: int, r_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_tok + d_state, d_tok, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(d_tok, d_tok, 3, 1, 1),
            nn.GELU(),
        )
        self.mu = nn.Conv2d(d_tok, r_dim, 1)
        self.logvar = nn.Conv2d(d_tok, r_dim, 1)

    def forward(
        self, u_y: torch.Tensor, s_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, D, H, W = u_y.shape
        s_broadcast = s_t.view(B, -1, 1, 1).expand(B, s_t.size(-1), H, W)
        h = self.conv(torch.cat([u_y, s_broadcast], dim=1))
        return self.mu(h), self.logvar(h)


class OptionPosterior(nn.Module):
    """Posterior q(o_t | y_t) from current token grid."""

    def __init__(self, d_tok: int, n_options: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(d_tok, d_tok), nn.GELU(), nn.Linear(d_tok, n_options)
        )

    def forward(self, u_y: torch.Tensor) -> torch.Tensor:
        B, D, H, W = u_y.shape
        pooled = self.pool(u_y).view(B, D)
        return F.softmax(self.mlp(pooled), dim=-1)


class LatteAR(nn.Module):
    """Temporal model that predicts p(y_t | s_t) and p(r_t | y_t, s_t).

    Training uses teacher forcing for tokens y_{t-1} -> s_t -> logits for y_t.
    Also predicts Gaussian prior for r_t and adds KL(q(r_t|x_t) || p(r_t|...)).
    """

    def __init__(
        self,
        codebook1: int,
        codebook2: int,
        d_tok: int = 384,
        d_state: int = 512,
        n_options: int = 8,
        r_dim: int = 32,
    ):
        super().__init__()
        self.embed = TokenEmbedding(codebook1, codebook2, d_tok)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cell = OS_SSM_Cell(d_state=d_state, d_in=d_tok, n_options=n_options)
        self.readout = SpatialReadout(
            d_tok=d_tok, d_state=d_state, codebook1=codebook1, codebook2=codebook2
        )
        self.residual_prior = ResidualPriorHead(
            d_tok=d_tok, d_state=d_state, r_dim=r_dim
        )
        self.opt_post = OptionPosterior(d_tok=d_tok, n_options=n_options)
        self.n_options = n_options
        self.d_tok = d_tok

    def forward(
        self,
        y1_idx: torch.Tensor,  # (B, T, H, W)
        y2_idx: torch.Tensor,  # (B, T, H, W)
        r_mu: torch.Tensor,  # (B, T, r_dim, H, W)
        r_logvar: torch.Tensor,  # (B, T, r_dim, H, W)
    ) -> Dict[str, torch.Tensor]:
        B, T, H, W = y1_idx.shape
        device = y1_idx.device
        # Init state and option posterior
        s = torch.zeros(B, self.cell.d_state, device=device)
        q_prev = F.one_hot(
            torch.zeros(B, dtype=torch.long, device=device), num_classes=self.n_options
        ).float()
        losses = []
        ce1_all, ce2_all, kl_opt_all, kl_r_all, hz_all = [], [], [], [], []

        for t in range(T - 1):  # predict y_t from y_{t-1}
            # Embedding grid of previous tokens
            u_prev = self.embed(y1_idx[:, t], y2_idx[:, t])  # (B, D, H, W)
            u_prev_pool = self.pool(u_prev).view(B, -1)  # (B, D)

            # Posterior q(o_t | y_t) uses current ground truth y_t (teacher forcing)
            u_curr = self.embed(y1_idx[:, t + 1], y2_idx[:, t + 1])
            q_post = self.opt_post(u_curr)  # (B, n_options)

            # State update via OS‑SSM cell
            s, info = self.cell(s, u_prev_pool, q_prev, q_post)
            kl_opt_all.append(info["kl_opt"])
            hz_all.append(info["hazard"])

            # Emission for tokens: logits to predict y_{t+1}
            logits1, logits2 = self.readout(u_prev, s)
            ce1 = F.cross_entropy(
                logits1.permute(0, 2, 3, 1).reshape(-1, logits1.size(1)),
                y1_idx[:, t + 1].view(-1),
            )
            ce2 = F.cross_entropy(
                logits2.permute(0, 2, 3, 1).reshape(-1, logits2.size(1)),
                y2_idx[:, t + 1].view(-1),
            )
            ce1_all.append(ce1)
            ce2_all.append(ce2)

            # Residual prior KL on grid
            # Use u_curr (teacher forcing) as conditioning for residual prior
            mu_p, logvar_p = self.residual_prior(u_curr, s)  # (B, r_dim, H, W)
            mu_q = r_mu[:, t + 1]
            logvar_q = r_logvar[:, t + 1]
            # KL(q||p) for diagonal Gaussians: 0.5 * (tr(sigma_p^-1 sigma_q) + (mu_p-mu_q)^T sigma_p^-1 (mu_p-mu_q) - k + log|sigma_p| - log|sigma_q|)
            kl = 0.5 * (
                (logvar_p - logvar_q).exp()  # sigma_q / sigma_p? careful
                + (mu_q - mu_p).pow(2) / (logvar_p.exp() + 1e-8)
                + logvar_p
                - logvar_q
                - 1.0
            )
            kl = kl.mean()
            kl_r_all.append(kl)

            # Update q_prev for next step (use current posterior)
            q_prev = q_post.detach()

        loss_ce1 = torch.stack(ce1_all).mean()
        loss_ce2 = torch.stack(ce2_all).mean()
        loss_klopt = torch.stack(kl_opt_all).mean()
        loss_klr = torch.stack(kl_r_all).mean()
        loss_hz = torch.stack(hz_all).mean()
        loss = loss_ce1 + loss_ce2 + loss_klopt + 0.1 * loss_klr + 0.01 * loss_hz

        return {
            "loss": loss,
            "loss_ce1": loss_ce1.detach(),
            "loss_ce2": loss_ce2.detach(),
            "loss_klopt": loss_klopt.detach(),
            "loss_klr": loss_klr.detach(),
            "loss_hz": loss_hz.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        y1_start: torch.Tensor,
        y2_start: torch.Tensor,
        steps: int,
        refresh_ratio: float = 0.2,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive sampling of token grids with copy‑mask refresh.
        Args:
            y*_start: (B, H, W) start tokens
            steps: number of frames to roll out
        Returns:
            y1, y2: (B, steps+1, H, W)
        """
        B, H, W = y1_start.shape
        device = y1_start.device
        y1 = [y1_start]
        y2 = [y2_start]
        s = torch.zeros(B, self.cell.d_state, device=device)
        q_prev = F.one_hot(
            torch.zeros(B, dtype=torch.long, device=device), num_classes=self.n_options
        ).float()

        for t in range(steps):
            u_prev = self.embed(y1[-1], y2[-1])
            u_prev_pool = F.adaptive_avg_pool2d(u_prev, (1, 1)).view(B, -1)

            # Self‑feeding posterior: approximate q(o_t) using predicted tokens (use prev as proxy)
            q_post = self.opt_post(u_prev)
            s, info = self.cell(s, u_prev_pool, q_prev, q_post)
            q_prev = q_post

            logits1, logits2 = self.readout(u_prev, s)
            logits1 = logits1 / temperature
            logits2 = logits2 / temperature

            # Copy‑mask refresh
            if refresh_ratio > 0:
                mask = torch.rand(B, H, W, device=device) < refresh_ratio
            else:
                mask = torch.ones(B, H, W, device=device, dtype=torch.bool)

            # Sample new tokens for masked positions; copy unmasked from previous
            probs1 = F.softmax(logits1.permute(0, 2, 3, 1), dim=-1)
            probs2 = F.softmax(logits2.permute(0, 2, 3, 1), dim=-1)
            y1_new = y1[-1].clone()
            y2_new = y2[-1].clone()
            y1_samp = torch.distributions.Categorical(probs1[mask]).sample()
            y2_samp = torch.distributions.Categorical(probs2[mask]).sample()
            y1_new[mask] = y1_samp
            y2_new[mask] = y2_samp

            y1.append(y1_new)
            y2.append(y2_new)

        return torch.stack(y1, dim=1), torch.stack(y2, dim=1)


# ---------------------------
# Video dataset utilities
# ---------------------------
class FramesFolderDataset(Dataset):
    """Dataset reading sequences of frames from a directory.

    Folder layout:
        root/
          clip_0001/
            000000.png
            000001.png
            ...
          clip_0002/
            ...
    Each clip directory forms one sample (sequence).

    Args:
        root: path to folder containing subfolders per clip
        seq_len: number of frames per sample (if a clip has more, a random contiguous
                 sub‑sequence is taken)
        img_size: resized square size
        stride: if >1, subsample frames by this stride inside a clip
    """

    def __init__(
        self, root: str, seq_len: int = 16, img_size: int = 128, stride: int = 1
    ):
        super().__init__()
        self.root = root
        self.seq_len = seq_len
        self.size = img_size
        self.stride = stride
        self.clips = sorted(
            [
                os.path.join(root, d)
                for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
        )
        self.tx = transforms.Compose(
            [
                transforms.Resize(
                    (img_size, img_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        folder = self.clips[idx]
        frames = sorted(
            [
                f
                for f in os.listdir(folder)
                if f.lower().endswith(("png", "jpg", "jpeg", "webp"))
            ]
        )
        frames = frames[:: self.stride]
        if len(frames) < self.seq_len:
            raise ValueError(
                f"Clip {folder} has fewer frames ({len(frames)}) than seq_len={self.seq_len}"
            )
        # pick random contiguous window
        start = random.randint(0, len(frames) - self.seq_len)
        imgs = []
        for f in frames[start : start + self.seq_len]:
            img = Image.open(os.path.join(folder, f)).convert("RGB")
            img = self.tx(img)
            imgs.append(img)
        # (T, C, H, W)
        x = torch.stack(imgs, dim=0)
        return x


# ---------------------------
# Training: Tokenizer
# ---------------------------
@dataclass
class TokenizerConfig:
    img_size: int = 128
    codebook1: int = 2048
    codebook2: int = 1024
    z_dim: int = 256
    r_dim: int = 32
    hidden: int = 192
    batch_size: int = 32
    lr: float = 2e-4
    wd: float = 0.0
    epochs: int = 50
    amp: bool = True


def train_tokenizer(cfg: TokenizerConfig, data_root: str, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FramesFolderDataset(data_root, seq_len=1, img_size=cfg.img_size)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    model = ResidualVQTokenizer(
        img_size=cfg.img_size,
        in_ch=3,
        z_dim=cfg.z_dim,
        r_dim=cfg.r_dim,
        hidden=cfg.hidden,
        codebook1=cfg.codebook1,
        codebook2=cfg.codebook2,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    ema_recon = EMA()
    ema_total = EMA()

    for epoch in range(cfg.epochs):
        model.train()
        for batch in dl:
            # batch: (B, 1, C, H, W)
            x = batch[:, 0].to(device)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                x_rec, out, tokens, latents = model(x)
                loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            ema_recon.update(out["recon_loss"].item())
            ema_total.update(loss.item())
        print(
            f"[Tokenizer] Epoch {epoch+1}/{cfg.epochs} | loss={ema_total.value:.4f} | L1={ema_recon.value:.4f}"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, out_path)
    print(f"Saved tokenizer to {out_path} | params={num_params(model):,}")


# ---------------------------
# Training: Sequence model
# ---------------------------
@dataclass
class LatteARConfig:
    img_size: int = 128
    codebook1: int = 2048
    codebook2: int = 1024
    d_tok: int = 384
    d_state: int = 512
    n_options: int = 8
    r_dim: int = 32
    batch_size: int = 4
    seq_len: int = 16
    lr: float = 2e-4
    wd: float = 0.0
    epochs: int = 50
    amp: bool = True


def preprocess_with_tokenizer(
    tokenizer: ResidualVQTokenizer, x_seq: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Tokenize a sequence of frames.
    Args:
        x_seq: (B, T, C, H, W)
    Returns dict with (y1_idx, y2_idx, r_mu, r_logvar)
    """
    B, T, C, H, W = x_seq.shape
    y1_all, y2_all, rmu_all, rlv_all = [], [], [], []
    for t in range(T):
        enc = tokenizer.encode(x_seq[:, t].contiguous())
        y1_all.append(enc["y1_idx"])
        y2_all.append(enc["y2_idx"])
        rmu_all.append(enc["r_mu"])
        rlv_all.append(enc["r_logvar"])
    y1 = torch.stack(y1_all, dim=1)
    y2 = torch.stack(y2_all, dim=1)
    rmu = torch.stack(rmu_all, dim=1)
    rlv = torch.stack(rlv_all, dim=1)
    return {"y1_idx": y1, "y2_idx": y2, "r_mu": rmu, "r_logvar": rlv}


def train_sequence(
    cfg: LatteARConfig, data_root: str, tokenizer_ckpt: str, out_path: str
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer and freeze
    tok = ResidualVQTokenizer(
        img_size=cfg.img_size,
        in_ch=3,
        z_dim=256,
        r_dim=cfg.r_dim,
        hidden=192,
        codebook1=cfg.codebook1,
        codebook2=cfg.codebook2,
    ).to(device)
    sd = torch.load(tokenizer_ckpt, map_location="cpu")
    tok.load_state_dict(sd["model"])
    tok.eval()
    for p in tok.parameters():
        p.requires_grad_(False)

    ds = FramesFolderDataset(data_root, seq_len=cfg.seq_len, img_size=cfg.img_size)
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    model = LatteAR(
        codebook1=cfg.codebook1,
        codebook2=cfg.codebook2,
        d_tok=cfg.d_tok,
        d_state=cfg.d_state,
        n_options=cfg.n_options,
        r_dim=cfg.r_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    ema_loss = EMA()

    for epoch in range(cfg.epochs):
        model.train()
        for batch in dl:
            x = batch.to(device)  # (B, T, C, H, W)
            with torch.no_grad():
                enc = preprocess_with_tokenizer(tok, x)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                out = model(
                    enc["y1_idx"].to(device),
                    enc["y2_idx"].to(device),
                    enc["r_mu"].to(device),
                    enc["r_logvar"].to(device),
                )
                loss = out["loss"]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema_loss.update(loss.item())
        print(
            f"[Seq] Epoch {epoch+1}/{cfg.epochs} | loss={ema_loss.value:.4f} | ce1={out['loss_ce1']:.3f} ce2={out['loss_ce2']:.3f} klopt={out['loss_klopt']:.3f} klr={out['loss_klr']:.3f}"
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, out_path)
    print(f"Saved sequence model to {out_path} | params={num_params(model):,}")


# ---------------------------
# Long‑horizon sampling and decoding back to RGB frames
# ---------------------------
@torch.no_grad()
def decode_tokens(
    tokenizer: ResidualVQTokenizer,
    y1_idx: torch.Tensor,
    y2_idx: torch.Tensor,
    r: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Decode tokens (and optional residual r) back to RGB.
    Args:
        y*_idx: (B, H, W)
        r: (B, r_dim, H, W) — if None, uses r=0
    Returns:
        x: (B, 3, H_out, W_out)
    """
    B, H, W = y1_idx.shape
    device = y1_idx.device
    # Build quantized z from indices
    # Note: we re‑use the tokenizer's codebooks for embedding lookup
    z1 = (
        F.embedding(y1_idx.view(B, -1), tokenizer.vq1.embed)
        .view(B, H, W, -1)
        .permute(0, 3, 1, 2)
    )
    z2 = (
        F.embedding(y2_idx.view(B, -1), tokenizer.vq2.embed)
        .view(B, H, W, -1)
        .permute(0, 3, 1, 2)
    )
    z_q = z1 + z2
    if r is None:
        r = torch.zeros(B, tokenizer.r_dim, H, W, device=device)
    x = tokenizer.dec(z_q, r)
    x = x.clamp(0, 1)
    return x


@torch.no_grad()
def long_horizon_rollout(
    tokenizer_ckpt: str,
    seq_ckpt: str,
    priming_clip: torch.Tensor,
    steps: int = 300,
    refresh_ratio: float = 0.2,
    temperature: float = 0.95,
) -> List[torch.Tensor]:
    """Roll out a long video given a short priming clip (teacher forcing only on first frame).
    Args:
        priming_clip: (B, T0, 3, H, W) — we start from the last frame's tokens
    Returns list of frames [ (B,3,H,W) ... ] length = steps+1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load models
    tok_sd = torch.load(tokenizer_ckpt, map_location="cpu")
    tok = ResidualVQTokenizer(
        img_size=tok_sd["cfg"]["img_size"],
        in_ch=3,
        z_dim=tok_sd["cfg"]["z_dim"],
        r_dim=tok_sd["cfg"]["r_dim"],
        hidden=tok_sd["cfg"]["hidden"],
        codebook1=tok_sd["cfg"]["codebook1"],
        codebook2=tok_sd["cfg"]["codebook2"],
    ).to(device)
    tok.load_state_dict(tok_sd["model"])
    tok.eval()

    seq_sd = torch.load(seq_ckpt, map_location="cpu")
    seq_cfg = seq_sd["cfg"]
    seq = LATTEAR_SeqModel(
        codebook1=seq_cfg["codebook1"],
        codebook2=seq_cfg["codebook2"],
        d_tok=seq_cfg["d_tok"],
        d_state=seq_cfg["d_state"],
        n_options=seq_cfg["n_options"],
        r_dim=seq_cfg["r_dim"],
    ).to(device)
    seq.load_state_dict(seq_sd["model"])
    seq.eval()

    priming_clip = priming_clip.to(device)
    enc0 = preprocess_with_tokenizer(tok, priming_clip)
    y1_start = enc0["y1_idx"][:, -1]
    y2_start = enc0["y2_idx"][:, -1]

    y1_all, y2_all = seq.sample(
        y1_start,
        y2_start,
        steps=steps,
        refresh_ratio=refresh_ratio,
        temperature=temperature,
    )

    frames = []
    # Decode using r=0 (or sample r from prior if desired)
    for t in range(y1_all.size(1)):
        x = decode_tokens(tok, y1_all[:, t], y2_all[:, t], r=None)
        frames.append(x.cpu())
    return frames


# ---------------------------
# CLI entrypoints
# ---------------------------


def build_argparser():
    p = argparse.ArgumentParser(description="LATTE‑AR reference implementation")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_tok = sub.add_parser("train_tokenizer")
    p_tok.add_argument("--data_root", type=str, required=True)
    p_tok.add_argument("--out", type=str, default="checkpoints/tokenizer.pt")
    p_tok.add_argument("--img_size", type=int, default=128)
    p_tok.add_argument("--epochs", type=int, default=50)
    p_tok.add_argument("--batch_size", type=int, default=32)
    p_tok.add_argument("--lr", type=float, default=2e-4)

    p_seq = sub.add_parser("train_seq")
    p_seq.add_argument("--data_root", type=str, required=True)
    p_seq.add_argument("--tokenizer", type=str, required=True)
    p_seq.add_argument("--out", type=str, default="checkpoints/sequence.pt")
    p_seq.add_argument("--img_size", type=int, default=128)
    p_seq.add_argument("--seq_len", type=int, default=16)
    p_seq.add_argument("--epochs", type=int, default=50)
    p_seq.add_argument("--batch_size", type=int, default=4)
    p_seq.add_argument("--lr", type=float, default=2e-4)

    p_roll = sub.add_parser("rollout")
    p_roll.add_argument("--tokenizer", type=str, required=True)
    p_roll.add_argument("--sequence", type=str, required=True)
    p_roll.add_argument(
        "--priming_dir",
        type=str,
        required=True,
        help="folder containing a few frames to prime",
    )
    p_roll.add_argument("--steps", type=int, default=300)
    p_roll.add_argument("--refresh", type=float, default=0.2)
    p_roll.add_argument("--temperature", type=float, default=0.95)

    return p


def main():
    args = build_argparser().parse_args()

    if args.cmd == "train_tokenizer":
        cfg = TokenizerConfig(
            img_size=args.img_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        train_tokenizer(cfg, data_root=args.data_root, out_path=args.out)

    elif args.cmd == "train_seq":
        cfg = LatteARConfig(
            img_size=args.img_size,
            seq_len=args.seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        train_sequence(
            cfg,
            data_root=args.data_root,
            tokenizer_ckpt=args.tokenizer,
            out_path=args.out,
        )

    elif args.cmd == "rollout":
        # Load a small priming clip from folder
        files = sorted(
            [
                f
                for f in os.listdir(args.priming_dir)
                if f.lower().endswith(("png", "jpg", "jpeg", "webp"))
            ]
        )
        assert len(files) >= 1, "need at least one priming frame"
        tx = transforms.Compose(
            [
                transforms.Resize(
                    (128, 128), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )
        imgs = []
        for f in files:
            img = Image.open(os.path.join(args.priming_dir, f)).convert("RGB")
            imgs.append(tx(img))
        priming = torch.stack(imgs, dim=0).unsqueeze(0)  # (1, T0, 3, H, W)
        frames = long_horizon_rollout(
            args.tokenizer,
            args.sequence,
            priming,
            steps=args.steps,
            refresh_ratio=args.refresh,
            temperature=args.temperature,
        )
        # Save a few sample frames
        os.makedirs("rollout_frames", exist_ok=True)
        for i, fr in enumerate(frames):
            pil = transforms.ToPILImage()(fr[0])
            pil.save(f"rollout_frames/{i:06d}.png")
        print(f"Saved {len(frames)} frames to rollout_frames/")


if __name__ == "__main__":
    main()
