import torch
import torch.nn as nn

from ..layers.st_blocks import TASA3DBlock


class TASA3D(nn.Module):
    """
    Stacks TASA-3D blocks; outputs categorical(256) logits per pixel.
    Causality: only temporal attention mixes in T; all spatial ops use k_t=1.
    """

    def __init__(
        self,
        input_hw: tuple[int, int],
        emb_dim: int = 16,
        quant_levels: int = 256,
        depth: int = 4,
        num_down: int = 3,
        channel_grow: int = 2,
        heads: int = 8,
        drop: float = 0.0,
        rope: bool = True,
    ):
        super().__init__()
        self.token = nn.Embedding(quant_levels, emb_dim)
        self.blocks = nn.ModuleList(
            [
                TASA3DBlock(
                    C_in=emb_dim,
                    input_hw=input_hw,
                    num_down=num_down,
                    channel_grow=channel_grow,
                    heads=heads,
                    rope=rope,
                    drop=drop,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.GroupNorm(1, emb_dim)
        self.head = nn.Conv3d(emb_dim, quant_levels, 1)  # categorical logits

    def forward(
        self, x_tokens: torch.Tensor, embeds: torch.Tensor = None
    ) -> torch.Tensor:  # [B,H,W,T] ints
        if embeds is None:
            x = self.token(x_tokens)  # [B,H,W,T,D]
        else:
            x = embeds

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # [B,D,T,H,W]

        for blk in self.blocks:
            x = blk(x)  # temporal-only attention inside
        x = self.norm(x)
        logits = self.head(x)  # [B,256,T,H,W]
        return logits.permute(0, 3, 4, 2, 1).contiguous()  # [B,H,W,T,256]
