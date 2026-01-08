# litrA_video_ar.py
# Minimal LiTrA: causal-in-time, non-causal-in-space autoregressive video model
# Outputs per-pixel categorical distributions over 256 grayscale values.
import random
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..layers.attention import sdpa


# -----------------------------
# Utilities
# -----------------------------


def block_causal_mask(T: int, block: int, device=None) -> torch.Tensor:
    """
    Build a block-lower-triangular causal mask of shape [T*block, T*block].
    For the HT plane, 'block' is H; for the WT plane, 'block' is W.
    """
    base = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    mask = base.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
    return mask.logical_not()  # False = mask out (future)


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    D = x.shape[-1]
    assert D % n_heads == 0, "d_model must be divisible by n_heads"
    d_head = D // n_heads
    return x.view(*x.shape[:-1], n_heads, d_head)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(-2)


def block_prefix_mask_indices(T: int, block: int, t: int) -> slice:
    """
    For HT plane (block=H) or WT plane (block=W):
    Returns the index slice [0 : (t+1)*block] in the flattened axis (N = T*block)
    corresponding to keys at times <= t.
    """
    end = (t + 1) * block
    return slice(0, end)


# -----------------------------
# Positional embeddings (learned per-axis)
# -----------------------------


class AxisPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_T: int, max_H: int, max_W: int):
        super().__init__()
        self.t = nn.Embedding(max_T, d_model)
        self.h = nn.Embedding(max_H, d_model)
        self.w = nn.Embedding(max_W, d_model)

    def forward(self, B, T, H, W, device):
        t_ids = torch.arange(T, device=device)
        h_ids = torch.arange(H, device=device)
        w_ids = torch.arange(W, device=device)
        t = self.t(t_ids).view(1, T, 1, 1, -1)
        h = self.h(h_ids).view(1, 1, H, 1, -1)
        w = self.w(w_ids).view(1, 1, 1, W, -1)
        return t + h + w  # broadcasted to [1,T,H,W,D]


# -----------------------------
# LiTrA block: three planar attentions + gated fusion
# -----------------------------


class MLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expansion * d_model)
        self.fc2 = nn.Linear(expansion * d_model, d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class LiTrALayer(nn.Module):
    """
    One LiTrA layer:
      - A^{HW}: intra-frame attention (non-causal in H/W) at each time t
      - A^{HT}: row-time attention (fixed w), block-causal in T
      - A^{WT}: col-time attention (fixed h), block-causal in T
      - gated fusion across the three planes
      - pre-norm + MLP
    Input/Output shape: [B, T, H, W, D]
    """

    def __init__(self, d_model: int, n_heads: int, drop: float = 0.0):
        super().__init__()
        D = d_model
        h = n_heads
        assert D % h == 0, "d_model must be divisible by n_heads"
        # Projections per plane
        self.q_hw = nn.Linear(D, D)
        self.k_hw = nn.Linear(D, D)
        self.v_hw = nn.Linear(D, D)
        self.q_ht = nn.Linear(D, D)
        self.k_ht = nn.Linear(D, D)
        self.v_ht = nn.Linear(D, D)
        self.q_wt = nn.Linear(D, D)
        self.k_wt = nn.Linear(D, D)
        self.v_wt = nn.Linear(D, D)
        # Output and gating
        self.out = nn.Linear(D, D)
        self.drop = nn.Dropout(drop)
        self.gate = nn.Sequential(
            nn.Linear(D, D), nn.SiLU(), nn.Linear(D, 3 * h)  # per-head triplet of gates
        )
        # Pre-norm + MLP
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.mlp = MLP(D, expansion=4, drop=drop)
        self.n_heads = n_heads
        self.d_model = d_model

    # --- planar attentions ---

    def planar_attn_hw(self, x):
        B, T, H, W, D = x.shape
        h = self.n_heads
        dh = D // h
        # project + reshape
        q = (
            split_heads(self.q_hw(x), h).view(B * T, H * W, h, dh).transpose(1, 2)
        )  # [B*T, h, HW, dh]
        k = split_heads(self.k_hw(x), h).view(B * T, H * W, h, dh).transpose(1, 2)
        v = split_heads(self.v_hw(x), h).view(B * T, H * W, h, dh).transpose(1, 2)
        out = sdpa(q, k, v)
        out = out.transpose(1, 2).reshape(B, T, H, W, h, dh)  # [B, T, H, W, h, dh]
        return out

    def planar_attn_ht(self, x):
        # Fix w; attend over (T, H); causal in T (block-lower-triangular)
        B, T, H, W, D = x.shape
        h = self.n_heads
        dh = D // h
        q = split_heads(self.q_ht(x), h).permute(
            0, 3, 1, 2, 4, 5
        )  # [B, W, T, H, h, dh]
        k = split_heads(self.k_ht(x), h).permute(0, 3, 1, 2, 4, 5)
        v = split_heads(self.v_ht(x), h).permute(0, 3, 1, 2, 4, 5)
        qf = q.reshape(B * W, T * H, h, dh).transpose(1, 2)  # [B*W, h, TH, dh]
        kf = k.reshape(B * W, T * H, h, dh).transpose(1, 2)
        vf = v.reshape(B * W, T * H, h, dh).transpose(1, 2)

        mask = block_causal_mask(T, H, device=x.device)  # [TH, TH] True=future
        mask = mask.unsqueeze(0).unsqueeze(0)
        out = sdpa(qf, kf, vf, attn_mask=mask)
        out = (
            out.transpose(1, 2).reshape(B, W, T, H, h, dh).permute(0, 2, 3, 1, 4, 5)
        )  # [B, T, H, W, h, dh]
        return out

    def planar_attn_wt(self, x):
        # Fix h; attend over (T, W); causal in T (block-lower-triangular)
        B, T, H, W, D = x.shape
        h = self.n_heads
        dh = D // h
        q = split_heads(self.q_wt(x), h).permute(
            0, 2, 1, 3, 4, 5
        )  # [B, H, T, W, h, dh]
        k = split_heads(self.k_wt(x), h).permute(0, 2, 1, 3, 4, 5)
        v = split_heads(self.v_wt(x), h).permute(0, 2, 1, 3, 4, 5)
        qf = q.reshape(B * H, T * W, h, dh).transpose(1, 2)  # [B*H, h, TW, dh]
        kf = k.reshape(B * H, T * W, h, dh).transpose(1, 2)
        vf = v.reshape(B * H, T * W, h, dh).transpose(1, 2)
        mask = block_causal_mask(T, W, device=x.device)  # [TH, TH] True=future
        mask = mask.unsqueeze(0).unsqueeze(0)
        out = sdpa(qf, kf, vf, attn_mask=mask)
        out = (
            out.transpose(1, 2).reshape(B, H, T, W, h, dh).permute(0, 2, 1, 3, 4, 5)
        )  # [B, T, H, W, h, dh]
        return out

    def forward(self, x):
        # Pre-norm
        y = self.ln1(x)
        # Three planar attentions
        m_hw = self.planar_attn_hw(y)
        m_ht = self.planar_attn_ht(y)
        m_wt = self.planar_attn_wt(y)
        # Content-adaptive gating (per-head)
        gates = (
            self.gate(y).view(*y.shape[:-1], self.n_heads, 3).softmax(-1)
        )  # [B,T,H,W,h,3]
        m = (
            gates[..., 0].unsqueeze(-1) * m_hw
            + gates[..., 1].unsqueeze(-1) * m_ht
            + gates[..., 2].unsqueeze(-1) * m_wt
        )
        # Fuse heads -> residual
        x = x + self.drop(self.out(merge_heads(m)))
        # MLP block
        x = x + self.mlp(self.ln2(x))
        return x


class LiTrALayerMem(nn.Module):
    """
    One LiTrA layer using inducing memories for each plane.

    Factorized attention (per plane) via learned memories:
        Approximate: softmax(Q K^T) V  ≈  softmax(Q K_m^T) · softmax(Q_m K^T) · V
    where K_m, Q_m come from learned memory tokens ("axis memories").

    For HT/WT (time-causal planes), we use **time-conditioned memory queries** Q_m(t),
    and compute per-time prefix-restricted softmax(Q_m(t) K_{<=t}^T)
    to preserve exact causality.

    Shapes:
      Input/Output: [B, T, H, W, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_T: int,
        m_hw: int = 128,
        m_ht: int = 64,
        m_wt: int = 64,
        drop: float = 0.0,
    ):
        super().__init__()
        self.D = d_model
        self.Hh = n_heads
        assert d_model % n_heads == 0
        self.dh = d_model // n_heads
        self.drop = nn.Dropout(drop)

        # Projections per plane
        self.q_hw = nn.Linear(d_model, d_model)
        self.k_hw = nn.Linear(d_model, d_model)
        self.v_hw = nn.Linear(d_model, d_model)
        self.q_ht = nn.Linear(d_model, d_model)
        self.k_ht = nn.Linear(d_model, d_model)
        self.v_ht = nn.Linear(d_model, d_model)
        self.q_wt = nn.Linear(d_model, d_model)
        self.k_wt = nn.Linear(d_model, d_model)
        self.v_wt = nn.Linear(d_model, d_model)

        # Learned axis memory tokens (shared across batch/spatial/time groups)
        self.m_hw = m_hw
        self.m_ht = m_ht
        self.m_wt = m_wt
        self.mem_hw = nn.Parameter(torch.randn(m_hw, d_model) / math.sqrt(d_model))
        self.mem_ht = nn.Parameter(torch.randn(m_ht, d_model) / math.sqrt(d_model))
        self.mem_wt = nn.Parameter(torch.randn(m_wt, d_model) / math.sqrt(d_model))

        # Time conditioning for HT/WT memory queries (so masks can be prefix-dependent)
        self.max_T = max_T
        self.mem_t_ht = nn.Embedding(max_T, d_model)
        self.mem_t_wt = nn.Embedding(max_T, d_model)

        # Output + gating
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 3 * n_heads),  # per-head gates over 3 planes
        )
        self.out = nn.Linear(d_model, d_model)

        # Pre-norm + MLP
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, expansion=4, drop=drop)

    # ---------- Factorized attention via memories ----------

    def _factor_hw(self, x: torch.Tensor) -> torch.Tensor:
        """
        Intra-frame plane (HW) – non-causal.
        x: [B,T,H,W,D]
        returns: [B,T,H,W,Hh,dh]
        """
        B, T, H, W, D = x.shape
        Hh, dh = self.Hh, self.dh
        scale = dh**-0.5

        q = (
            split_heads(self.q_hw(x), Hh).view(B * T, H * W, Hh, dh).transpose(1, 2)
        )  # [B*T, Hh, N, dh]
        k = (
            split_heads(self.k_hw(x), Hh).view(B * T, H * W, Hh, dh).transpose(1, 2)
        )  # [B*T, Hh, N, dh]
        v = (
            split_heads(self.v_hw(x), Hh).view(B * T, H * W, Hh, dh).transpose(1, 2)
        )  # [B*T, Hh, N, dh]

        # Memory keys/queries (per head)
        mem_q = split_heads(self.q_hw(self.mem_hw), Hh).transpose(0, 1)  # [Hh, M, dh]
        mem_k = split_heads(self.k_hw(self.mem_hw), Hh).transpose(0, 1)  # [Hh, M, dh]

        # A = softmax(Q K_m^T)  [B*T, Hh, N, M]
        A = torch.matmul(q * scale, mem_k.transpose(-2, -1))  # [B*T, Hh, N, M]
        A = A.softmax(dim=-1)

        # B = softmax(Q_m K^T)  [B*T, Hh, M, N]
        # Broadcast mem_q across batch groups
        mem_q_bt = mem_q.unsqueeze(0).expand(B * T, -1, -1, -1)  # [B*T, Hh, M, dh]
        B_scores = torch.matmul(
            mem_q_bt * scale, k.transpose(-2, -1)
        )  # [B*T, Hh, M, N]
        B_soft = B_scores.softmax(dim=-1)

        # Out = A @ (B @ V)  -> [B*T, Hh, N, dh]
        BV = torch.matmul(B_soft, v)  # [B*T, Hh, M, dh]
        out = torch.matmul(A, BV)  # [B*T, Hh, N, dh]
        out = out.transpose(1, 2).reshape(B, T, H, W, Hh, dh)
        return out

    def _factor_ht(self, x: torch.Tensor) -> torch.Tensor:
        """
        Row-time plane (HT) – causal in T. Group by w.
        x: [B,T,H,W,D]
        returns: [B,T,H,W,Hh,dh]
        """
        B, T, H, W, D = x.shape
        Hh, dh = self.Hh, self.dh
        scale = dh**-0.5

        # Project and group by W
        q = split_heads(self.q_ht(x), Hh).permute(
            0, 3, 1, 2, 4, 5
        )  # [B, W, T, H, Hh, dh]
        k = split_heads(self.k_ht(x), Hh).permute(0, 3, 1, 2, 4, 5)
        v = split_heads(self.v_ht(x), Hh).permute(0, 3, 1, 2, 4, 5)

        # Flatten per group: N = T*H
        qf = q.reshape(B * W, T * H, Hh, dh).transpose(1, 2)  # [B*W, Hh, N, dh]
        kf = k.reshape(B * W, T * H, Hh, dh).transpose(1, 2)  # [B*W, Hh, N, dh]
        vf = v.reshape(B * W, T * H, Hh, dh).transpose(1, 2)  # [B*W, Hh, N, dh]
        N = T * H

        # Time-independent memory keys for A
        mem_k = split_heads(self.k_ht(self.mem_ht), Hh).transpose(0, 1)  # [Hh, M, dh]
        # Time-dependent memory queries for B_t
        base_mem_q = self.mem_ht  # [M, D]

        # Precompute A for all queries (cheap to do once)
        A_all = torch.matmul(qf * scale, mem_k.transpose(-2, -1)).softmax(
            dim=-1
        )  # [B*W, Hh, N, M]

        # We will fill 'out' per time t and then reshape back
        out = qf.new_zeros(B * W, Hh, N, dh)

        # Build per-time slices
        for t in range(T):
            # Queries at time t occupy positions [t*H : (t+1)*H]
            q_idx = slice(t * H, (t + 1) * H)

            # Time-conditioned memory queries (shared across groups)
            mem_q_t = base_mem_q + self.mem_t_ht.weight[t]  # [M, D]
            mem_q_t = split_heads(self.q_ht(mem_q_t), Hh).transpose(0, 1)  # [Hh, M, dh]
            mem_q_bt = mem_q_t.unsqueeze(0).expand(
                B * W, -1, -1, -1
            )  # [B*W, Hh, M, dh]

            # Prefix mask: allow keys up to time t (i.e., indices 0 : (t+1)*H)
            k_allowed = block_prefix_mask_indices(T, H, t)  # slice
            kf_pref = kf[:, :, k_allowed, :]  # [B*W, Hh, N_pref, dh]
            vf_pref = vf[:, :, k_allowed, :]  # [B*W, Hh, N_pref, dh]

            # B_t = softmax(Q_m(t) K_{<=t}^T) over key dimension
            B_scores = torch.matmul(
                mem_q_bt * scale, kf_pref.transpose(-2, -1)
            )  # [B*W, Hh, M, N_pref]
            B_soft = B_scores.softmax(dim=-1)  # [B*W, Hh, M, N_pref]

            # S_t = B_t @ V_{<=t}  -> [B*W, Hh, M, dh]
            S_t = torch.matmul(B_soft, vf_pref)

            # A_t for queries at time t  -> [B*W, Hh, H, M]
            A_t = A_all[:, :, q_idx, :]  # slice across N

            # out_t = A_t @ S_t -> [B*W, Hh, H, dh]
            out[:, :, q_idx, :] = torch.matmul(A_t, S_t)

        # Reshape back to [B,T,H,W,Hh,dh]
        out = out.transpose(1, 2).reshape(B, W, T, H, Hh, dh).permute(0, 2, 3, 1, 4, 5)
        return out

    def _factor_wt(self, x: torch.Tensor) -> torch.Tensor:
        """
        Col-time plane (WT) – causal in T. Group by h.
        x: [B,T,H,W,D]
        returns: [B,T,H,W,Hh,dh]
        """
        B, T, H, W, D = x.shape
        Hh, dh = self.Hh, self.dh
        scale = dh**-0.5

        # Project and group by H
        q = split_heads(self.q_wt(x), Hh).permute(
            0, 2, 1, 3, 4, 5
        )  # [B, H, T, W, Hh, dh]
        k = split_heads(self.k_wt(x), Hh).permute(0, 2, 1, 3, 4, 5)
        v = split_heads(self.v_wt(x), Hh).permute(0, 2, 1, 3, 4, 5)

        # Flatten per group: N = T*W
        qf = q.reshape(B * H, T * W, Hh, dh).transpose(1, 2)  # [B*H, Hh, N, dh]
        kf = k.reshape(B * H, T * W, Hh, dh).transpose(1, 2)  # [B*H, Hh, N, dh]
        vf = v.reshape(B * H, T * W, Hh, dh).transpose(1, 2)  # [B*H, Hh, N, dh]
        N = T * W

        # Memory keys (time-independent) for A; time-conditioned mem queries for B_t
        mem_k = split_heads(self.k_wt(self.mem_wt), Hh).transpose(0, 1)  # [Hh, M, dh]
        base_mem_q = self.mem_wt

        # Precompute A for all queries
        A_all = torch.matmul(qf * scale, mem_k.transpose(-2, -1)).softmax(
            dim=-1
        )  # [B*H, Hh, N, M]

        out = qf.new_zeros(B * H, Hh, N, dh)

        for t in range(T):
            q_idx = slice(t * W, (t + 1) * W)

            mem_q_t = base_mem_q + self.mem_t_wt.weight[t]  # [M, D]
            mem_q_t = split_heads(self.q_wt(mem_q_t), Hh).transpose(0, 1)  # [Hh, M, dh]
            mem_q_bt = mem_q_t.unsqueeze(0).expand(
                B * H, -1, -1, -1
            )  # [B*H, Hh, M, dh]

            k_allowed = block_prefix_mask_indices(T, W, t)
            kf_pref = kf[:, :, k_allowed, :]  # [B*H, Hh, N_pref, dh]
            vf_pref = vf[:, :, k_allowed, :]  # [B*H, Hh, N_pref, dh]

            B_scores = torch.matmul(
                mem_q_bt * scale, kf_pref.transpose(-2, -1)
            )  # [B*H, Hh, M, N_pref]
            B_soft = B_scores.softmax(dim=-1)
            S_t = torch.matmul(B_soft, vf_pref)  # [B*H, Hh, M, dh]

            A_t = A_all[:, :, q_idx, :]  # [B*H, Hh, W, M]
            out[:, :, q_idx, :] = torch.matmul(A_t, S_t)

        out = out.transpose(1, 2).reshape(B, H, T, W, Hh, dh).permute(0, 2, 1, 3, 4, 5)
        return out

    def forward(self, x):
        """
        x: [B, T, H, W, D]
        """
        y = self.ln1(x)

        # Three planar low-rank attentions
        m_hw = self._factor_hw(y)  # [B,T,H,W,Hh,dh]
        m_ht = self._factor_ht(y)  # [B,T,H,W,Hh,dh]
        m_wt = self._factor_wt(y)  # [B,T,H,W,Hh,dh]

        # Content-adaptive gating (per-head)
        gates = (
            self.gate(y).view(*y.shape[:-1], self.Hh, 3).softmax(-1)
        )  # [B,T,H,W,Hh,3]
        fused = (
            gates[..., 0].unsqueeze(-1) * m_hw
            + gates[..., 1].unsqueeze(-1) * m_ht
            + gates[..., 2].unsqueeze(-1) * m_wt
        )  # [B,T,H,W,Hh,dh]

        y = self.drop(self.out(merge_heads(fused)))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


# -----------------------------
# Full model
# -----------------------------


class LITRA(nn.Module):
    """
    Full AR model:
      - Input: int pixels [B, T, H, W] in [0..255]
      - Output: logits [B, T, H, W, 256] predicting the NEXT frame (t+1) at time t
    """

    def __init__(
        self,
        quant_levels: int = 256,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        max_T: int = 64,
        max_H: int = 64,
        max_W: int = 64,
        m_hw: int = 128,
        m_ht: int = 64,
        m_wt: int = 64,
        dropout: float = 0.0,
        layer_type: str = "LITRALayer",
    ):
        super().__init__()
        self.embed = nn.Embedding(quant_levels, d_model)
        self.axis_pos = AxisPositionalEmbedding(d_model, max_T, max_H, max_W)

        if layer_type == "LITRALayerMem":
            self.layers = nn.ModuleList(
                [
                    LiTrALayerMem(
                        d_model, n_heads, max_T, m_hw, m_ht, m_wt, drop=dropout
                    )
                    for _ in range(n_layers)
                ]
            )
        elif layer_type == "LITRALayer":
            self.layers = nn.ModuleList(
                [LiTrALayer(d_model, n_heads, drop=dropout) for _ in range(n_layers)]
            )
        else:
            raise ValueError(f"Invalid layer type: {layer_type}")

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, quant_levels)

    def forward(self, x_int: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        # x_int: [B,T,H,W] integers 0..255
        x_int = x_int.permute(0, 3, 1, 2)
        B, T, H, W = x_int.shape
        device = x_int.device

        if x is None:
            x = self.embed(x_int)  # [B,T,H,W,D]
        else:
            x = x.permute(0, 3, 1, 2, 4)

        x = x + self.axis_pos(B, T, H, W, device)  # axis-wise learned positions
        for layer in self.layers:
            x = layer(x)  # [B,T,H,W,D]
        x = self.norm(x)
        logits = self.head(x)  # [B,T,H,W,256]
        return logits.permute(0, 2, 3, 1, 4).contiguous()  # [B,H,W,T,256]

    @torch.inference_mode()
    def predict_next_logits(self, x_int: torch.Tensor) -> torch.Tensor:
        # Return logits for the NEXT frame given context up to the last frame
        return self.forward(x_int)[:, -1]  # [B,H,W,256]

    @torch.inference_mode()
    def sample(
        self, context: torch.Tensor, steps: int, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressively sample 'steps' future frames given initial context.
        context: [B, T0, H, W] ints
        returns: [B, T0+steps, H, W] ints
        """
        out = [context]
        x = context
        for _ in range(steps):
            logits = self.predict_next_logits(x) / max(1e-5, temperature)  # [B,H,W,256]
            probs = logits.softmax(-1)
            B, H, W, Q = probs.shape
            # Sample pixel-wise
            flat = torch.distributions.Categorical(
                probs.view(B, -1, Q)
            ).sample()  # [B, H*W]
            nxt = flat.view(B, H, W)
            x = torch.cat([x, nxt.unsqueeze(1)], dim=1)
            out.append(nxt.unsqueeze(1))
        return torch.cat(out, dim=1)  # [B, T0+steps, H, W]


# -----------------------------
# Synthetic dataset (moving squares)
# -----------------------------


class MovingSquares(Dataset):
    def __init__(self, num_samples=256, T=8, H=16, W=16, num_objs=2, seed=0):
        self.num_samples = num_samples
        self.T, self.H, self.W = T, H, W
        self.num_objs = num_objs
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T, H, W = self.T, self.H, self.W
        vid = torch.zeros(T, H, W, dtype=torch.long)
        # init objects
        objs = []
        for _ in range(self.num_objs):
            x = self.rng.randint(0, max(0, W - 4))
            y = self.rng.randint(0, max(0, H - 4))
            vx = self.rng.choice([-1, 1])
            vy = self.rng.choice([-1, 1])
            size = self.rng.randint(2, 4)
            intensity = self.rng.randint(100, 255)
            objs.append([x, y, vx, vy, size, intensity])
        # rollout
        for t in range(T):
            frame = torch.zeros(H, W, dtype=torch.long)
            for i in range(len(objs)):
                x, y, vx, vy, size, inten = objs[i]
                x0 = max(0, min(W - size, x))
                y0 = max(0, min(H - size, y))
                frame[y0 : y0 + size, x0 : x0 + size] = inten
                # bounce physics
                x += vx
                y += vy
                if x < 0 or x > W - size:
                    vx *= -1
                    x += 2 * vx
                if y < 0 or y > H - size:
                    vy *= -1
                    y += 2 * vy
                objs[i] = [x, y, vx, vy, size, inten]
            vid[t] = frame
        return vid  # [T,H,W] ints


# -----------------------------
# Training
# -----------------------------


@dataclass
class Config:
    T: int = 8
    H: int = 16
    W: int = 16
    d_model: int = 64
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.0
    batch_size: int = 8
    lr: float = 3e-4
    epochs: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(model, loader, opt, cfg: Config):
    model.train()
    total = 0.0
    count = 0
    for vids in loader:
        vids = vids.to(cfg.device)  # [B,T,H,W]
        logits = model(vids)  # [B,T,H,W,256] predicts next frame at each t
        # next-frame loss: compare logits[:, :-1] to vids[:, 1:]
        pred = logits[:, :-1]  # [B,T-1,H,W,256]
        tgt = vids[:, 1:]  # [B,T-1,H,W]
        loss = F.cross_entropy(pred.reshape(-1, 256), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.detach())
        count += 1
    return total / max(1, count)


@torch.inference_mode()
def quick_eval_and_sample(model, cfg: Config):
    model.eval()
    ds = MovingSquares(num_samples=1, T=cfg.T, H=cfg.H, W=cfg.W, num_objs=2, seed=123)
    seed_clip = ds[0].unsqueeze(0).to(cfg.device)  # [1,T,H,W]
    # Take the first frame as context, sample 7 future frames
    ctx = seed_clip[:, :1]  # [1,1,H,W]
    out = model.sample(ctx, steps=7, temperature=1.0)  # [1,8,H,W]
    return ctx.squeeze(0).cpu(), out.squeeze(0).cpu()  # tensors of ints


def main():
    cfg = Config()
    print("Using device:", cfg.device)
    model = LITRA(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        max_T=cfg.T,
        max_H=cfg.H,
        max_W=cfg.W,
        dropout=cfg.dropout,
    ).to(cfg.device)
    ds = MovingSquares(num_samples=256, T=cfg.T, H=cfg.H, W=cfg.W, num_objs=2, seed=42)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=1e-2
    )

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, loader, opt, cfg)
        print(f"epoch {epoch} | train CE: {loss:.4f}")

    # Quick qualitative check: sample from one seed frame
    seed, gen = quick_eval_and_sample(model, cfg)
    print("Seed frame (ints) shape:", seed.shape)  # [1,H,W]
    print("Generated clip shape:   ", gen.shape)  # [8,H,W] (1 seed + 7 sampled)

    # (Optional) visualize by printing a tiny ASCII thumbnail:
    frame = gen[0]  # first frame of sampled clip
    H, W = frame.shape
    chars = " .:-=+*#%@"
    s = ""
    for y in range(H):
        for x in range(W):
            v = frame[y, x].item() / 255.0
            s += chars[min(int(v * (len(chars) - 1)), len(chars) - 1)]
        s += "\n"
    print("Sampled frame preview:\n", s)


if __name__ == "__main__":
    main()
