import torch
import numpy as np
from torch import nn
from typing import Tuple

from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Config

from ..layers.embeddings import Embeddings
from ..layers.st_blocks import STGPTBlock, STGPTBlockParallel, STBlock  # noqa: F401
from ..layers.transformer_blocks import Transformer, TransformerBlock
from ..training.lightning import LitModel
from ..models.tokenizers.emu3visionvq import Emu3VisionVQ  # noqa: F401


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
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

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
        self.embedding = Embeddings(num_channels, **embedding_args)

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


class GPT2MEGMix(GPT2MEG_Trf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.d_model == self.quant_emb * self.num_channels

    def forward(
        self, x: torch.Tensor, embeds: torch.Tensor = None, causal: bool = True
    ) -> torch.Tensor:
        if embeds is None:
            x = self.embedding(x)
        else:
            x = embeds

        # reshape to (B, C, T, E)
        x = x.reshape(-1, self.num_channels, x.shape[1], x.shape[2])
        B, C, T, E = x.shape

        x = x.permute(0, 2, 1, 3).reshape(B, T, -1)  # (B, T, C*E)

        for block in self.block:
            x = block(x, causal=causal)
        x = self.norm(x)

        x = x.reshape(B, T, C, E).permute(0, 2, 1, 3)  # (B, C, T, E)
        x = self.head(x)  # (B, C, T, Q)
        return x


class STGPT2MEG(nn.Module):
    def __init__(
        self,
        num_channels: int,
        vocab_size: int,
        d_model: int,
        layers: int,
        trf_args: dict,
        embedding_args: dict = None,
        trf_block: str = "STGPTBlock",
    ):
        super().__init__()
        embedding_args = embedding_args or {}
        self.num_channels = num_channels
        self.quant_levels = vocab_size

        trf_class = globals()[trf_block]

        blocks = [trf_class(**trf_args) for _ in range(layers)]
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.RMSNorm(d_model)
        self.embeddings = Embeddings(
            num_channels,
            quant_levels=vocab_size,
            quant_emb=d_model,
            channel_emb=d_model,
            **embedding_args,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        chid: np.ndarray = None,
        cond: torch.Tensor = None,
        sid: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.embeddings(x, chid, cond, sid)

        _, T, E = x.shape
        x = x.reshape(-1, self.num_channels, T, E)

        for blk in self.blocks:
            x = blk(x)

        x = self.head(self.norm(x))
        return x  # (B, C, T, Q)


class VQGPT2MEG(nn.Module):
    def __init__(
        self,
        tokenizer_path: str,
        trf_args: dict,
        tok_args: dict = None,
        train_tokenizer: bool = False,
        tok_class: str = "Emu3VisionVQ",
    ):
        super().__init__()
        self.train_tokenizer = train_tokenizer

        if tokenizer_path is not None:
            lit = LitModel.load_from_checkpoint(tokenizer_path, strict=False)
            self.vqvae = lit.model

            # check if model is compiled
            if hasattr(self.vqvae, "_orig_mod"):
                self.vqvae = self.vqvae._orig_mod

        else:
            self.vqvae = globals()[tok_class](**tok_args)

        # freeze tokenizer during autoregressive training (optional)
        for p in self.vqvae.parameters():
            p.requires_grad_(False)

        self.transformer = Transformer(**trf_args)

    def forward_train(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                logits: (B, C_latent, Nq, T, K)
                codes: (B, C_latent, Nq, T') - these are the tokens to predict
        """
        embeds, _, codes = self.vqvae.encode(x)

        embeds = embeds.permute(0, 1, 3, 4, 2)  # (B, T, H, W, E)
        embeds = embeds.reshape(embeds.shape[0], -1, embeds.shape[-1])  # (B, T*H*W, E)

        outputs = self.transformer(codes, embeds=embeds, causal=True)
        return outputs[:, :-1], codes[:, 1:]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.train_tokenizer:
            return self.forward_train(x)

        self.vqvae.eval()

        with torch.no_grad():
            _, _, codes = self.vqvae.encode(x)

        outputs = self.transformer(codes, causal=True)

        return outputs[:, :-1], codes[:, 1:]
