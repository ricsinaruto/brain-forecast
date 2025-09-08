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
        orig_shape = x.shape
        t = self.norm(x)
        flattened = False
        if t.ndim > 2:
            # Flatten batch/time into token dimension to match reference math
            t = t.reshape(-1, t.shape[-1])
            x_residual = x
            flattened = True
        else:
            x_residual = x
        g = self.gate(t)

        # TODO: this likely crashes on MPS
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # MLP #1
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        # Shapes:
        # - mlp1_weight: (tokens, experts_per_token, intermediate2, hidden)
        # - t: (tokens, hidden)
        # Compute per-token, per-expert linear: (B,E,C,H) @ (B,1,H,1) -> (B,E,C)
        be, ee, c1, k = mlp1_weight.shape
        t_expanded = t.unsqueeze(1).expand(-1, ee, -1)  # (B,E,H)
        t = torch.bmm(
            mlp1_weight.reshape(be * ee, c1, k),
            t_expanded.reshape(be * ee, k, 1),
        ).reshape(be, ee, c1)
        t = t + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        # Shapes after swiglu:
        # - mlp2_weight: (B,E,hidden, intermediate)
        # - t: (B,E,intermediate)
        be2, ee2, c_out, k2 = mlp2_weight.shape
        t = torch.bmm(
            mlp2_weight.reshape(be2 * ee2, c_out, k2),
            t.reshape(be2 * ee2, k2, 1),
        ).reshape(be2, ee2, c_out)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += mlp2_bias

        # Weighted sum of experts
        t = (t * expert_weights.unsqueeze(-1)).sum(dim=1)
        if flattened:
            t = t.reshape(*orig_shape[:-1], orig_shape[-1])
        return x_residual + t


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

    def forward(self, x: Tensor) -> Tensor:
        t = self.norm(x)
        t = self.fc1(t)
        t = self.act(t)
        t = self.drop(t)
        t = self.fc2(t)
        x = x + self.drop(t)
        return x


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
