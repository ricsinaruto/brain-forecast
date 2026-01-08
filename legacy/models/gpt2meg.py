import torch
import numpy as np
from torch import nn
from typing import Tuple

from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Config

from ..layers.embeddings import Embeddings
from ..layers.st_blocks import (  # noqa: F401
    STGPTBlock,
    STGPTBlockParallel,
    STBlock,
    STConvBlock,
)
from ..layers.transformer_blocks import (
    Transformer,
    TransformerBlock,
    TransformerBlockCond,
)
from ..training.lightning import LitModel
from ..models.tokenizers.emu3 import Emu3VisionVQ  # noqa: F401


class GPT2MEG(nn.Module):
    """
    GPT2-based model for MEG (Magnetoencephalography) forecasting.

    Implements a transformer-based architecture for processing and predicting
    MEG signal sequences.
    """

    NEW_NAMES = ["head", "cond_emb", "quant_emb", "ch_emb"]

    def __init__(
        self,
        num_channels: int,
        gpt2_config: dict,
        embedding_args: dict,
    ):
        """
        Args:
            num_channels: Number of channels
            gpt2_config: GPT2 configuration
            embedding_args: Dictionary containing embedding arguments
        """
        super().__init__()
        gpt2_config = GPT2Config(**gpt2_config)

        # Initialize model parameters
        self.quant_levels = gpt2_config.vocab_size
        self.num_channels = num_channels
        embedding_args["quant_levels"] = self.quant_levels

        self.gpt2 = GPT2Model(gpt2_config)
        self.embeddings = Embeddings(num_channels, **embedding_args)
        self.head = nn.Linear(gpt2_config.n_embd, self.quant_levels, bias=False)
        # nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        chid: np.ndarray = None,
        cond: torch.Tensor = None,
        sid: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, T)
            chid: Channel IDs of shape (C,)
            cond: Conditional tensor of shape (B, 1, T)
            sid: Subject IDs of shape (B,)

        Returns:
            Model output tensor or tuple of (output, past_key_values)
        """
        # Process through model
        x = self.embeddings(x, chid, cond, sid)

        gpt_kwargs = {"inputs_embeds": x}
        if past_key_values is not None:
            gpt_kwargs.update({"past_key_values": past_key_values, "use_cache": True})

        # forward pass through GPT2
        outputs = self.gpt2(**gpt_kwargs)

        if past_key_values is None:
            outputs.past_key_values = None

        x = self.head(outputs[0])
        x = x.reshape(-1, self.num_channels, x.shape[1], x.shape[2])

        if outputs.past_key_values is None:
            return x

        return (x, outputs.past_key_values)


class GPT2MEG_Trf(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        d_model: int,
        quant_emb: int,
        attn_args: dict,
        vocab_size: int = 256,
        embedding_args: dict = None,
        mlp_args: dict = None,
        attn_type: str = "standard",  # must be either "standard" or "gpt_oss"
        mlp_type: str = "standard",  # must be either "standard" or "moe")
    ):
        super().__init__()
        embedding_args = embedding_args or {}
        mlp_args = mlp_args or {}

        embedding_args["quant_levels"] = vocab_size
        embedding_args["quant_emb"] = quant_emb
        embedding_args["num_channels"] = num_channels
        self.embedding = Embeddings(**embedding_args)

        attn_args["d_model"] = d_model
        mlp_args["d_model"] = d_model
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(attn_args, mlp_args, attn_type, mlp_type)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = torch.nn.Linear(
            quant_emb,
            vocab_size,
            bias=False,
        )

        self.num_channels = num_channels
        self.quant_emb = quant_emb
        self.d_model = d_model

    def forward(
        self, x: torch.Tensor, embeds: torch.Tensor = None, causal: bool = True
    ) -> torch.Tensor:
        if embeds is None:
            x = self.embedding(x)
        else:
            x = embeds

        for block in self.block:
            x = block(x, causal=causal)
        x = self.norm(x)
        x = self.head(x)

        return x.reshape(-1, self.num_channels, x.shape[1], x.shape[2])


class GPT2MEG_Cond(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_channels: int,
        d_model: int,
        quant_emb: int,
        attn_args: dict,
        vocab_size: int = 256,
        mlp_args: dict = None,
        attn_type: str = "standard",  # must be either "standard" or "gpt_oss"
        mlp_type: str = "standard",  # must be either "standard" or "moe")
        n_cond_tok: int = 0,
        d_tok_emb: int = 0,
        d_glob_emb: int = 0,
    ):
        super().__init__()
        mlp_args = mlp_args or {}

        self.input_emb = nn.Embedding(vocab_size, d_model)

        attn_args["d_model"] = d_model
        mlp_args["d_model"] = d_model
        self.block = torch.nn.ModuleList(
            [
                TransformerBlockCond(
                    attn_args,
                    mlp_args,
                    attn_type,
                    mlp_type,
                    n_cond_tok,
                    num_channels,
                    d_tok_emb,
                    d_glob_emb,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model)
        self.head = torch.nn.Linear(
            quant_emb,
            vocab_size,
            bias=False,
        )

        self.num_channels = num_channels
        self.quant_emb = quant_emb
        self.d_model = d_model

    def forward(
        self, x: torch.Tensor, embeds: torch.Tensor = None, causal: bool = True
    ) -> torch.Tensor:
        c_tok_ids = None
        if isinstance(x, tuple) or isinstance(x, list):
            x, c_tok_ids = x

        B, C, T = x.shape
        if c_tok_ids is not None:
            c_tok_ids = c_tok_ids.expand(-1, C, -1)  # B x C x T
            c_tok_ids = c_tok_ids.reshape(-1, T)  # (B*C) x T

        if embeds is None:
            x = self.input_emb(x)
        else:
            x = embeds
        x = x.reshape(B * C, T, -1)

        # create channel IDs for global conditioning
        ch_ids = torch.arange(self.num_channels, device=x.device)
        c_global_ids = ch_ids.expand(B, -1)  # [B, C]
        c_global_ids = c_global_ids.reshape(-1)

        for block in self.block:
            x = block(x, causal=causal, c_tok_ids=c_tok_ids, c_global_ids=c_global_ids)
        x = self.norm(x)
        x = self.head(x)

        return x.reshape(-1, self.num_channels, x.shape[1], x.shape[2])


class GPT2MEGMix(GPT2MEG_Trf):
    def __init__(self, channel_dropout_p: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.d_model == self.quant_emb * self.num_channels
        self.input_channel_dropout = nn.Dropout1d(p=channel_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        embeds: torch.Tensor = None,
        causal: bool = True,
        return_logits: bool = True,
    ) -> torch.Tensor:
        if embeds is None:
            x = self.embedding(x)
        else:
            x = embeds

        # reshape to (B, C, T, E)
        x = x.reshape(-1, self.num_channels, x.shape[1], x.shape[2])
        B, C, T, E = x.shape

        x = x.permute(0, 3, 1, 2).reshape(B * E, C, T)
        x = self.input_channel_dropout(x)
        x = x.reshape(B, E, C, T).permute(0, 3, 2, 1).reshape(B, T, -1)  # (B, T, C*E)

        for block in self.block:
            x = block(x, causal=causal)
        x = self.norm(x)

        x = x.reshape(B, T, C, E).permute(0, 2, 1, 3)  # (B, C, T, E)

        if not return_logits:
            return x

        x = self.head(x)  # (B, C, T, Q)
        return x
