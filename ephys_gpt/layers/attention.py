import torch
import torch.nn as nn
import math
from torch import Tensor
from typing import Optional

import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from .embeddings import RotaryEmbedding, rope_inv_freq, apply_rope_1d


def sdpa_gptoss(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    S: Tensor,
    sm_scale: float,
    sliding_window: int = 0,
    causal: bool = False,
) -> Tensor:
    """
    Scaled Dot-Product Attention

    Args:
        Q: (n_tokens, n_heads, q_mult, d_head)
        K: (n_tokens, n_heads, d_head)
        V: (n_tokens, n_heads, d_head)
        S: (n_heads, q_mult, 1, 1)
        sm_scale: float
        sliding_window: int

    Returns:
        (n_tokens, n_heads, q_mult, d_head)
    """
    # Supports either flattened tensors (tokens-first) or batched tensors.
    if Q.ndim == 4:
        # Q: (T, H_kv, M, D); K,V: (T, H_kv, D)
        T, H_kv, M, D = Q.shape
        assert K.shape == (T, H_kv, D)
        assert V.shape == (T, H_kv, D)

        H_total = H_kv * M
        # Flatten heads for efficient matmul: (H_total, T, D)
        Qf = Q.permute(1, 2, 0, 3).reshape(H_total, T, D).contiguous()
        Kf = (
            K.unsqueeze(2)
            .expand(T, H_kv, M, D)
            .permute(1, 2, 0, 3)
            .reshape(H_total, T, D)
            .contiguous()
        )
        Vf = (
            V.unsqueeze(2)
            .expand(T, H_kv, M, D)
            .permute(1, 2, 0, 3)
            .reshape(H_total, T, D)
            .contiguous()
        )

        # Attention logits
        logits = torch.matmul(Qf, Kf.transpose(1, 2))  # (H_total, T, T)
        logits *= sm_scale

        # Build mask once and broadcast over heads
        if causal or sliding_window > 0:
            neg_inf = -float("inf")
            base_mask = Qf.new_zeros(T, T)
            if causal:
                base_mask = base_mask + torch.triu(
                    base_mask.new_full((T, T), neg_inf), 1
                )
            if sliding_window > 0:
                base_mask = base_mask + torch.tril(
                    base_mask.new_full((T, T), neg_inf), -sliding_window
                )
            logits = logits + base_mask

        # Numerically stable softmax with extra sink mass
        max_logits, _ = logits.max(dim=-1, keepdim=True)
        exp_logits = torch.exp(logits - max_logits)
        # Sink per (head, q_mult)
        S_flat = S.view(H_total)
        S_col = S_flat.view(H_total, 1, 1).expand(-1, T, 1)
        exp_sink = torch.exp(S_col - max_logits)
        denom = exp_logits.sum(dim=-1, keepdim=True) + exp_sink
        weights = exp_logits / denom  # (H_total, T, T)

        # Output
        out = torch.matmul(weights, Vf)  # (H_total, T, D)
        out = out.view(H_kv, M, T, D).permute(2, 0, 1, 3).contiguous()
        return out.reshape(T, -1)
    elif Q.ndim == 5:
        # Q: (B, T, H_kv, M, D); K,V: (B, T, H_kv, D)
        B, T, H_kv, M, D = Q.shape
        assert K.shape == (B, T, H_kv, D)
        assert V.shape == (B, T, H_kv, D)

        H_total = H_kv * M
        Qf = Q.permute(0, 2, 3, 1, 4).reshape(B * H_total, T, D).contiguous()
        Kf = (
            K.unsqueeze(3)
            .expand(B, T, H_kv, M, D)
            .permute(0, 2, 3, 1, 4)
            .reshape(B * H_total, T, D)
            .contiguous()
        )
        Vf = (
            V.unsqueeze(3)
            .expand(B, T, H_kv, M, D)
            .permute(0, 2, 3, 1, 4)
            .reshape(B * H_total, T, D)
            .contiguous()
        )

        logits = torch.matmul(Qf, Kf.transpose(1, 2))  # (B*H_total, T, T)
        logits *= sm_scale

        if causal or sliding_window > 0:
            neg_inf = -float("inf")
            base_mask = Qf.new_zeros(T, T)
            if causal:
                base_mask = base_mask + torch.triu(
                    base_mask.new_full((T, T), neg_inf), 1
                )
            if sliding_window > 0:
                base_mask = base_mask + torch.tril(
                    base_mask.new_full((T, T), neg_inf), -sliding_window
                )
            logits = logits + base_mask

        max_logits, _ = logits.max(dim=-1, keepdim=True)
        exp_logits = torch.exp(logits - max_logits)
        S_flat = S.view(H_total)
        S_b = S_flat.unsqueeze(0).expand(B, -1).reshape(B * H_total)
        S_col = S_b.view(B * H_total, 1, 1).expand(-1, T, 1)
        exp_sink = torch.exp(S_col - max_logits)
        denom = exp_logits.sum(dim=-1, keepdim=True) + exp_sink
        weights = exp_logits / denom

        out = torch.matmul(weights, Vf)  # (B*H_total, T, D)
        out = out.view(B, H_kv, M, T, D).permute(0, 3, 1, 2, 4).contiguous()
        return out.reshape(B, T, -1)
    else:
        raise ValueError("Unexpected Q rank for sdpa: expected 4D or 5D")


def sdpa(
    q, k, v, dropout: float = 0.0, causal: bool = False, attn_mask: torch.Tensor = None
) -> torch.Tensor:
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout, is_causal=causal, attn_mask=attn_mask
        )
    return out


def sdpa_math(
    q, k, v, dropout: float = 0.0, causal: bool = False, attn_mask: torch.Tensor = None
) -> torch.Tensor:
    with sdpa_kernel([SDPBackend.MATH]):
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout, is_causal=causal, attn_mask=attn_mask
        )
    return out


class MultiHeadAttentionGPTOSS(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        d_model: int,
        dropout: float = 0.0,
        sliding_window: int = 0,
        rope_theta: float = 150000.0,
        rope_scaling_factor: float = 32.0,
        rope_ntk_alpha: float = 1.0,
        rope_ntk_beta: float = 32.0,
        initial_context_length: int = 1024,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        hidden_size = d_model
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(torch.empty(num_attention_heads, device=device))
        with torch.no_grad():
            self.sinks.normal_(std=1.0 / math.sqrt(head_dim))
        self.norm = nn.RMSNorm(hidden_size, device=device)
        qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
        self.qkv = torch.nn.Linear(
            hidden_size,
            qkv_dim,
            device=device,
        )
        self.out = torch.nn.Linear(
            head_dim * num_attention_heads,
            hidden_size,
            device=device,
        )
        self.sm_scale = 1 / math.sqrt(head_dim)
        self.rope = RotaryEmbedding(
            head_dim,
            rope_theta,
            torch.float32,
            initial_context_length=initial_context_length,
            scaling_factor=rope_scaling_factor,
            ntk_alpha=rope_ntk_alpha,
            ntk_beta=rope_ntk_beta,
            device=device,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        causal: bool = False,
    ) -> torch.Tensor:
        t = x
        qkv = self.qkv(t)
        q = qkv[..., : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            ...,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            ...,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        if x.ndim == 2:
            # (tokens, hidden)
            q = q.view(
                -1,
                self.num_key_value_heads,
                self.num_attention_heads // self.num_key_value_heads,
                self.head_dim,
            )
            k = k.view(-1, self.num_key_value_heads, self.head_dim)
            v = v.view(-1, self.num_key_value_heads, self.head_dim)
            q, k = self.rope(q, k)
            t = sdpa_gptoss(
                q,
                k,
                v,
                self.sinks,
                self.sm_scale,
                self.sliding_window,
                causal=causal,
            )
        elif x.ndim == 3:
            # (batch, tokens, hidden)
            bsz, n_tokens = x.shape[0], x.shape[1]
            q = q.view(
                bsz,
                n_tokens,
                self.num_key_value_heads,
                self.num_attention_heads // self.num_key_value_heads,
                self.head_dim,
            )
            k = k.view(bsz, n_tokens, self.num_key_value_heads, self.head_dim)
            v = v.view(bsz, n_tokens, self.num_key_value_heads, self.head_dim)
            q, k = self.rope(q, k)
            t = sdpa_gptoss(
                q,
                k,
                v,
                self.sinks,
                self.sm_scale,
                self.sliding_window,
                causal=causal,
            )
        else:
            raise ValueError(
                "Unexpected input rank for AttentionBlock: expected 2D or 3D"
            )

        t = self.out(t)
        return self.dropout(t)


class MultiHeadAttention(nn.Module):
    """
    Computes multi-head attention. Supports nested or padded tensors.

    Args:
        E_q (int): Size of embedding dim for query
        E_k (int): Size of embedding dim for key
        E_v (int): Size of embedding dim for value
        E_total (int): Total embedding dim of combined heads post input projection.
            Each head has dim E_total // nheads
        nheads (int): Number of heads
        dropout (float, optional): Dropout probability. Default: 0.0
        bias (bool, optional): Whether to add bias to input projection. Default: True
    """

    def __init__(
        self,
        d_model: int,
        nheads: int,
        E_q: Optional[int] = None,
        E_k: Optional[int] = None,
        E_v: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        rope: bool = False,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        # Set defaults if not provided
        if E_q is None:
            E_q = d_model
        if E_k is None:
            E_k = d_model
        if E_v is None:
            E_v = d_model

        self.nheads = nheads
        self.dropout = dropout
        self.drop = nn.Dropout(dropout)
        self._qkv_same_embed_dim = E_q == E_k and E_q == E_v
        if self._qkv_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, d_model * 3, bias=bias)
        else:
            self.q_proj = nn.Linear(E_q, d_model, bias=bias)
            self.k_proj = nn.Linear(E_k, d_model, bias=bias)
            self.v_proj = nn.Linear(E_v, d_model, bias=bias)
        E_out = E_q
        self.out_proj = nn.Linear(d_model, E_out, bias=bias)
        assert d_model % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = d_model // nheads
        self.bias = bias

        self.rope = rope
        self.register_buffer(
            "inv_freq_t",
            rope_inv_freq((self.E_head // 2) * 2, rope_base),
            persistent=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        causal: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass; runs the following process:
            1. Apply input projection
            2. Split heads and prepare for SDPA
            3. Run SDPA
            4. Apply output projection

        Args:
            query (torch.Tensor): query of shape (N, L_q, E_qk)
            key (torch.Tensor): key of shape (N, L_kv, E_qk)
            value (torch.Tensor): value of shape (N, L_kv, E_v)
            attn_mask (torch.Tensor, optional): attention mask of shape (N, L_q, L_kv)
                to pass to sdpa. Default: None
            is_causal (bool, optional): Whether to apply causal mask. Default: False

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        B, T, _ = query.shape
        # Step 1. Apply input projection
        if self._qkv_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim=-1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(
                    self.packed_proj.weight, 3, dim=0
                )
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(
                        self.packed_proj.bias, 3, dim=0
                    )
                else:
                    q_bias, k_bias, v_bias = None, None, None
                query, key, value = (
                    F.linear(query, q_weight, q_bias),
                    F.linear(key, k_weight, k_bias),
                    F.linear(value, v_weight, v_bias),
                )

        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nheads, self.E_head]).transpose(1, 2)

        if self.rope and self.inv_freq_t.numel() > 0:
            query, key = apply_rope_1d(query, key, self.inv_freq_t, pos_start=0, L=T)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = sdpa(query, key, value, self.dropout, causal=causal)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return self.drop(attn_output)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        attn_args: dict,
        attn_type: str = "standard",
    ):
        super().__init__()
        if attn_type == "standard":
            self.attn = MultiHeadAttention(**attn_args)
        elif attn_type == "gpt_oss":
            self.attn = MultiHeadAttentionGPTOSS(**attn_args)
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")
        self.dropout = nn.Dropout(attn_args.get("dropout", 0.0))
        self.norm = nn.RMSNorm(attn_args["d_model"])

    def forward(self, x: torch.Tensor, causal: bool = False):
        t = self.norm(x)
        t = self.attn(t, t, t, causal=causal)
        return x + t
