import torch
import math
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .attention import AttentionBlock
from .activations import swiglu


class MLPMoE(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        experts_per_token: int = 4,
        swiglu_limit: float = 7.0,
        intermediate_size: Optional[int] = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = d_model

        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit
        self.world_size = 1
        self.norm = nn.RMSNorm(d_model, device=device)
        self.gate = nn.Linear(
            d_model,
            num_experts,
            device=device,
        )

        assert intermediate_size % self.world_size == 0
        self.mlp1_weight = nn.Parameter(
            torch.empty(
                (
                    num_experts,
                    intermediate_size * 2 // self.world_size,
                    d_model,
                ),
                device=device,
            )
        )
        self.mlp1_bias = nn.Parameter(
            torch.empty(
                (num_experts, intermediate_size * 2 // self.world_size),
                device=device,
            )
        )
        self.mlp2_weight = nn.Parameter(
            torch.empty(
                (
                    num_experts,
                    d_model,
                    intermediate_size // self.world_size,
                ),
                device=device,
            )
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(
                (num_experts, d_model),
                device=device,
            )
        )

        with torch.no_grad():
            self.gate.weight.normal_(std=1.0 / math.sqrt(d_model))
            nn.init.zeros_(self.gate.bias)

            self.mlp1_weight.normal_(std=1.0 / math.sqrt(d_model))
            nn.init.zeros_(self.mlp1_bias)

            self.mlp2_weight.normal_(std=1.0 / math.sqrt(intermediate_size))
            nn.init.zeros_(self.mlp2_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.norm(x)
        original_shape = hidden.shape
        d_model = original_shape[-1]
        hidden_flat = hidden.reshape(-1, d_model)

        gate_logits = self.gate(hidden_flat)
        experts = torch.topk(gate_logits, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices

        mlp1_weight = self.mlp1_weight[expert_indices]
        mlp1_bias = self.mlp1_bias[expert_indices]
        mlp_input = hidden_flat.unsqueeze(1).unsqueeze(-1)
        t = torch.matmul(mlp1_weight, mlp_input).squeeze(-1)
        t = t + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        mlp2_weight = self.mlp2_weight[expert_indices]
        mlp2_bias = self.mlp2_bias[expert_indices]
        t = torch.matmul(mlp2_weight, t.unsqueeze(-1)).squeeze(-1)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t + mlp2_bias

        moe_output = torch.sum(t * expert_weights.unsqueeze(-1), dim=1)
        moe_output = moe_output.view(*original_shape[:-1], d_model)

        return x + moe_output


class MLP(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_model

        self.norm = nn.RMSNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = act_layer()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, residual: bool = True) -> Tensor:
        t = self.norm(x)
        t = self.fc1(t)
        t = self.act(t)
        t = self.drop(t)
        t = self.fc2(t)

        if residual:
            return x + self.drop(t)

        return self.drop(t)


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",  # must be either "standard" or "gpt_oss"
        mlp_type: str = "standard",  # must be either "standard" or "moe"
    ):
        super().__init__()
        self.attn = AttentionBlock(attn_type=attn_type, attn_args=attn_args)

        if mlp_type == "standard":
            self.mlp = MLP(**mlp_args)
        elif mlp_type == "moe":
            self.mlp = MLPMoE(**mlp_args)
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")

    def forward(self, x: torch.Tensor, causal: bool = False) -> torch.Tensor:
        x = self.attn(x, causal=causal)
        x = self.mlp(x)
        return x


class TransformerBlockCond(torch.nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "conditioned",
        mlp_type: str = "standard",
        n_cond_tok: int = 0,
        n_cond_global: int = 0,
        d_tok_emb: int = 0,
        d_glob_emb: int = 0,
    ):
        super().__init__()
        self.n_cond_tok = n_cond_tok
        self.n_cond_global = n_cond_global

        if n_cond_tok > 0:
            self.emb_tok = nn.Embedding(n_cond_tok, d_tok_emb)
            # map to FiLM params (per-token & global)
            self.mod_tok = nn.Sequential(
                nn.SiLU(), nn.Linear(d_tok_emb, 4 * attn_args["d_model"])
            )  # γ1t,β1t,γ2t,β2t  (per token)

        if n_cond_global > 0:
            self.emb_glob = nn.Embedding(n_cond_global, d_glob_emb)
            self.mod_glb = nn.Sequential(
                nn.SiLU(), nn.Linear(d_glob_emb, 4 * attn_args["d_model"])
            )  # γ1g,β1g,γ2g,β2g  (per sequence)

        attn_args["d_tok_emb"] = d_tok_emb
        attn_args["n_cond_tok"] = n_cond_tok
        self.attn = AttentionBlock(attn_type=attn_type, attn_args=attn_args)

        if mlp_type == "standard":
            self.mlp = MLP(**mlp_args)
        elif mlp_type == "moe":
            self.mlp = MLPMoE(**mlp_args)
        else:
            raise ValueError(f"Invalid MLP type: {mlp_type}")

    def forward(
        self,
        x: torch.Tensor,
        c_tok_ids: torch.Tensor = None,
        c_global_ids: torch.Tensor = None,
        causal: bool = False,
    ) -> torch.Tensor:
        # initialize FiLM params
        g1t, b1t, g2t, b2t = 0, 0, 0, 0
        g1g, b1g, g2g, b2g = 0, 0, 0, 0

        # ---- FiLM params (sum of per-token + broadcasted global) ----
        if self.n_cond_tok > 0:
            mt = self.mod_tok(self.emb_tok(c_tok_ids))  # [B,S,4D]
            g1t, b1t, g2t, b2t = mt.chunk(4, dim=-1)  # each [B,S,D]

        if self.n_cond_global > 0:
            mg = self.mod_glb(self.emb_glob(c_global_ids))  # [B,4D]
            g1g, b1g, g2g, b2g = mg.chunk(4, dim=-1)  # each [B,D]
            g1g = g1g.unsqueeze(1)
            b1g = b1g.unsqueeze(1)
            g2g = g2g.unsqueeze(1)
            b2g = b2g.unsqueeze(1)

        # broadcast global to sequence and sum
        g1 = g1t + g1g
        b1 = b1t + b1g
        g2 = g2t + g2g
        b2 = b2t + b2g

        h = self.attn(x, causal=causal, residual=False, c_tok_ids=c_tok_ids)
        x = x + h * (1 + g1) + b1

        h = self.mlp(x, residual=False)
        x = x + h * (1 + g2) + b2

        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        attn_args: dict,
        mlp_args: dict,
        vocab_size: int,
        num_layers: int,
        attn_type: str = "standard",  # must be either "standard" or "gpt_oss"
        mlp_type: str = "standard",  # must be either "standard" or "moe"
    ):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            vocab_size,
            attn_args["d_model"],
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(attn_args, mlp_args, attn_type, mlp_type)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(attn_args["d_model"])
        self.unembedding = torch.nn.Linear(
            attn_args["d_model"],
            vocab_size,
            bias=False,
        )

    def forward(
        self, x: torch.Tensor, embeds: torch.Tensor = None, causal: bool = False
    ) -> torch.Tensor:
        if embeds is None:
            x = self.embedding(x)
        else:
            x = embeds

        for block in self.block:
            x = block(x, causal=causal)
        x = self.norm(x)
        x = self.unembedding(x)
        return x
