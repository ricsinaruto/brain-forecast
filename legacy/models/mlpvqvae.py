# mlp_vqvae_hf.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from ...layers.quantizers import VectorQuantizer


# -------------------------
# Config + Output
# -------------------------


class MLPVQVAEConfig(PretrainedConfig):
    model_type = "mlp-vqvae"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        codebook_size: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        input_norm: Optional[str] = None,  # e.g. "layernorm" or None
        beta: float = 0.25,  # commitment weight
        recon_loss: str = "mse",  # "mse" | "l1" | "huber"
        huber_delta: float = 1.0,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        reset_unused_codes: bool = False,
        usage_threshold: float = 1.0,
        bypass_quantizer: bool = False,
        pre_quant_norm: bool = False,
        codebook_init_std: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.activation = activation
        self.input_norm = input_norm
        self.codebook_size = codebook_size
        self.beta = beta
        self.recon_loss = recon_loss
        self.huber_delta = huber_delta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.reset_unused_codes = reset_unused_codes
        self.usage_threshold = usage_threshold
        self.bypass_quantizer = bypass_quantizer
        self.pre_quant_norm = pre_quant_norm
        self.codebook_init_std = codebook_init_std


@dataclass
class MLPVQVAEOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    recon_loss: Optional[torch.Tensor] = None
    vq_loss: Optional[torch.Tensor] = None
    perplexity: Optional[torch.Tensor] = None

    x_hat: Optional[torch.Tensor] = None
    indices: Optional[torch.LongTensor] = None
    z_e: Optional[torch.Tensor] = None
    z_q: Optional[torch.Tensor] = None


# -------------------------
# Building blocks
# -------------------------


def _get_act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        dropout: float,
        activation: str,
        input_norm: Optional[str] = None,
    ):
        super().__init__()
        act = _get_act(activation)

        layers: list[nn.Module] = []
        if input_norm is not None:
            if input_norm.lower() == "layernorm":
                layers.append(nn.LayerNorm(in_dim))
            else:
                raise ValueError(f"Unknown input_norm: {input_norm}")

        layers += [nn.Linear(in_dim, hidden_dims[0]), act]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for i, hidden_dim in enumerate(hidden_dims[1:]):
            layers += [nn.Linear(hidden_dims[i], hidden_dim), act]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -------------------------
# HF-style Model
# -------------------------


class MLPVQVAE(PreTrainedModel):
    config_class = MLPVQVAEConfig
    base_model_prefix = "mlp_vqvae"

    def __init__(self, config: MLPVQVAEConfig | dict):
        if isinstance(config, dict):
            config = MLPVQVAEConfig(**config)

        super().__init__(config)
        self.encoder = MLP(
            in_dim=config.input_dim,
            out_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            activation=config.activation,
            input_norm=config.input_norm,
        )
        self.quantizer = VectorQuantizer(
            codebook_size=config.codebook_size,
            embed_dim=config.latent_dim,
            beta=config.beta,
            use_ema=config.use_ema,
            ema_decay=config.ema_decay,
            ema_eps=config.ema_eps,
            reset_unused_codes=config.reset_unused_codes,
            usage_threshold=config.usage_threshold,
            init_std=config.codebook_init_std,
        )

        hidden_dims = config.hidden_dims[::-1]
        self.decoder = MLP(
            in_dim=config.latent_dim,
            out_dim=config.input_dim,
            hidden_dims=hidden_dims,
            dropout=config.dropout,
            activation=config.activation,
            input_norm=config.input_norm,
        )

        self.post_init()  # HF weight init hook
        if self.quantizer.use_ema:
            # Keep EMA buffers in sync with HF re-init
            self.quantizer.ema_weight.data.copy_(self.quantizer.codebook.weight.data)
            self.quantizer.ema_cluster_size.zero_()

    def _init_weights(self, module: nn.Module):
        # Reasonable defaults
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _recon_loss(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mode = self.config.recon_loss.lower()
        if mode == "mse":
            return F.mse_loss(x_hat, x)
        if mode == "l1":
            return F.l1_loss(x_hat, x)
        if mode == "huber":
            return F.huber_loss(x_hat, x, delta=float(self.config.huber_delta))
        raise ValueError(f"Unknown recon_loss: {self.config.recon_loss}")

    @torch.no_grad()
    def encode(self, inputs: torch.Tensor) -> torch.LongTensor:
        """
        Return discrete code indices with same leading dims as inputs.
        """
        z_e = self._encode_latents(inputs)
        z_e_flat, leading_shape = self._flatten(z_e)
        if self.config.bypass_quantizer:
            return torch.zeros(*leading_shape, dtype=torch.long, device=z_e.device)
        _, idx, _, _, _ = self.quantizer(z_e_flat)
        return idx.view(*leading_shape)

    @torch.no_grad()
    def decode(self, indices: torch.LongTensor) -> torch.Tensor:
        """Decode discrete indices back to vectors."""
        z_q = self.quantizer.codebook(indices)  # (..., latent_dim)
        x_hat = self.decoder(z_q)
        return x_hat

    def _encode_latents(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (..., input_dim)
        return self.encoder(inputs)

    @staticmethod
    def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        # x: (..., D) -> (M, D), and returns leading shape
        leading_shape = x.shape[:-1]
        return x.reshape(-1, x.shape[-1]), leading_shape

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,  # optional, for Trainer compatibility
        return_dict: Optional[bool] = True,
    ) -> MLPVQVAEOutput:
        """
        inputs: (..., input_dim) float tensor
        labels: if provided, uses that as reconstruction target; else target = inputs
        """
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]

        return_dict = True if return_dict is None else return_dict
        target = inputs if labels is None else labels

        # Encode
        z_e = self._encode_latents(inputs)  # (..., latent_dim)
        z_e_flat, leading_shape = self._flatten(z_e)  # (M, latent_dim)
        if self.config.pre_quant_norm:
            z_e_flat = F.layer_norm(z_e_flat, (self.config.latent_dim,))

        # Quantize
        if self.config.bypass_quantizer:
            device = z_e_flat.device
            z_q_st_flat = z_e_flat
            z_q_flat = z_e_flat
            idx_flat = torch.zeros(z_e_flat.shape[0], device=device, dtype=torch.long)
            vq_loss = torch.zeros((), device=device, dtype=z_e_flat.dtype)
            ppl = torch.tensor(float("nan"), device=device, dtype=z_e_flat.dtype)
        else:
            z_q_st_flat, idx_flat, vq_loss, ppl, z_q_flat = self.quantizer(z_e_flat)
        z_q_st = z_q_st_flat.view(*leading_shape, self.config.latent_dim)
        z_q = z_q_flat.view(*leading_shape, self.config.latent_dim)  # (..., latent_dim)
        indices = idx_flat.view(*leading_shape)  # (...)

        # Decode
        x_hat = self.decoder(z_q_st)  # (..., input_dim)

        # Loss
        recon_loss = self._recon_loss(x_hat, target)
        loss = recon_loss + vq_loss

        if not return_dict:
            return (
                loss,
                recon_loss,
                vq_loss,
                ppl,
                x_hat,
                indices,
                z_e,
                z_q,
            )

        return MLPVQVAEOutput(
            loss=loss,
            recon_loss=recon_loss,
            vq_loss=vq_loss,
            perplexity=ppl,
            indices=indices,
            x_hat=x_hat,
            z_e=z_e,
            z_q=z_q,
        )


# -------------------------
# Modular HF-style Model
# -------------------------
