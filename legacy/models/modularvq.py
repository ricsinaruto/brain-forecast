import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput


@dataclass
class ModularVQAutoencoderOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    recon_loss: Optional[torch.Tensor] = None
    vq_loss: Optional[torch.Tensor] = None
    perplexity: Optional[torch.Tensor] = None

    x_hat: Optional[torch.Tensor] = None
    indices: Optional[torch.LongTensor] = None
    z_e: Optional[torch.Tensor] = None
    z_q: Optional[torch.Tensor] = None
    z_q_st: Optional[torch.Tensor] = None


class ModularVQAutoencoderConfig(PretrainedConfig):
    model_type = "modular-vqvae"

    def __init__(
        self,
        encoder_cls: str,
        encoder_config: Dict[str, Any],
        decoder_cls: str,
        decoder_config: Dict[str, Any],
        quantizer_cls: Optional[str] = "ephys_gpt.layers.quantizers.VectorQuantizer",
        quantizer_config: Optional[Dict[str, Any]] = None,
        recon_loss: str = "mse",  # "mse" | "l1" | "huber"
        huber_delta: float = 1.0,
        bypass_quantizer: bool = False,
        pre_quant_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_cls = encoder_cls
        self.encoder_config = encoder_config
        self.decoder_cls = decoder_cls
        self.decoder_config = decoder_config
        self.quantizer_cls = quantizer_cls
        self.quantizer_config = quantizer_config if quantizer_config is not None else {}
        self.recon_loss = recon_loss
        self.huber_delta = huber_delta
        self.bypass_quantizer = bypass_quantizer
        self.pre_quant_norm = pre_quant_norm


class ModularVQAutoencoder(PreTrainedModel):
    config_class = ModularVQAutoencoderConfig
    base_model_prefix = "modular_vqvae"

    def __init__(self, config: ModularVQAutoencoderConfig | dict):
        if isinstance(config, dict):
            config = ModularVQAutoencoderConfig(**config)
        super().__init__(config)

        self.encoder = self._build_module(config.encoder_cls, config.encoder_config)
        self.decoder = self._build_module(config.decoder_cls, config.decoder_config)
        if config.quantizer_cls is not None and not config.bypass_quantizer:
            self.quantizer = self._build_module(
                config.quantizer_cls, config.quantizer_config
            )
        else:
            self.quantizer = None

        self.post_init()

    @staticmethod
    def _build_module(class_path: str, cfg: Dict[str, Any]) -> nn.Module:
        if "." not in class_path:
            raise ValueError(
                "class_path must be a fully qualified module path like 'pkg.mod.Class'."
            )
        module_name, cls_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, cls_name)
        return cls(**cfg)

    @staticmethod
    def _coalesce(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    def _init_weights(self, module: nn.Module):
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

    @staticmethod
    def _flatten(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        leading_shape = x.shape[:-1]
        return x.reshape(-1, x.shape[-1]), leading_shape

    def _quantize(
        self, z_e_flat: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        device = z_e_flat.device
        dtype = z_e_flat.dtype
        zero = torch.zeros((), device=device, dtype=dtype)
        nan = torch.tensor(float("nan"), device=device, dtype=dtype)

        if self.config.bypass_quantizer or self.quantizer is None:
            idx_flat = torch.zeros(z_e_flat.shape[0], device=device, dtype=torch.long)
            return z_e_flat, idx_flat, zero, nan, z_e_flat

        q_out = self.quantizer(z_e_flat)

        if torch.is_tensor(q_out):
            z_q_st_flat = z_q_flat = q_out
            idx_flat = torch.zeros(z_e_flat.shape[0], device=device, dtype=torch.long)
            vq_loss = zero
            ppl = nan
        elif isinstance(q_out, (list, tuple)):
            if len(q_out) == 5:
                z_q_st_flat, idx_flat, vq_loss, ppl, z_q_flat = q_out
            elif len(q_out) == 4:
                z_q_st_flat, idx_flat, vq_loss, z_q_flat = q_out
                ppl = nan
            elif len(q_out) == 3:
                z_q_st_flat, idx_flat, vq_loss = q_out
                ppl = nan
                z_q_flat = z_q_st_flat
            else:
                raise ValueError(
                    "Quantizer outputs should be "
                    "(z_q_st, indices, vq_loss[, perplexity, z_q])."
                )
        elif isinstance(q_out, dict):
            z_q_st_flat = self._coalesce(
                q_out.get("z_q_st"), q_out.get("z_q"), q_out.get("quantized")
            )
            if z_q_st_flat is None:
                raise ValueError("Quantizer dict output missing 'z_q_st' or 'z_q'.")
            z_q_flat = q_out.get("z_q", z_q_st_flat)
            idx_flat = self._coalesce(q_out.get("indices"), q_out.get("codes"))
            if idx_flat is None:
                idx_flat = torch.zeros(
                    z_e_flat.shape[0], device=device, dtype=torch.long
                )
            vq_loss = self._coalesce(q_out.get("vq_loss"), q_out.get("loss"), zero)
            ppl = self._coalesce(q_out.get("perplexity"), q_out.get("ppl"), nan)
        else:
            raise TypeError("Unsupported quantizer output type.")

        if idx_flat.dtype != torch.long:
            idx_flat = idx_flat.long()
        vq_loss = torch.as_tensor(vq_loss, device=device, dtype=dtype)
        ppl = torch.as_tensor(ppl, device=device, dtype=dtype)
        return z_q_st_flat, idx_flat, vq_loss, ppl, z_q_flat

    @torch.no_grad()
    def encode_latents(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    @torch.no_grad()
    def quantize(
        self, latents: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        z_e_flat, leading_shape = self._flatten(latents)
        if self.config.pre_quant_norm:
            z_e_flat = F.layer_norm(z_e_flat, (z_e_flat.shape[-1],))
        z_q_st_flat, idx_flat, vq_loss, ppl, z_q_flat = self._quantize(z_e_flat)
        z_q_st = z_q_st_flat.view(*leading_shape, z_q_st_flat.shape[-1])
        z_q = z_q_flat.view(*leading_shape, z_q_flat.shape[-1])
        indices = idx_flat.view(*leading_shape)
        return z_q_st, indices, vq_loss, ppl, z_q

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    @torch.no_grad()
    def encode(self, inputs: torch.Tensor) -> torch.LongTensor:
        if self.quantizer is None or self.config.bypass_quantizer:
            raise ValueError("encode() requires an active quantizer.")
        z_e = self.encode_latents(inputs)
        _, indices, _, _, _ = self.quantize(z_e)
        return indices

    @torch.no_grad()
    def decode(self, codes_or_latents: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(codes_or_latents):
            return self.decode_latents(codes_or_latents)

        if self.quantizer is None or self.config.bypass_quantizer:
            raise ValueError("decode(indices) requires an active quantizer.")

        if hasattr(self.quantizer, "codebook"):
            z_q = self.quantizer.codebook(codes_or_latents)
        elif hasattr(self.quantizer, "decode"):
            z_q = self.quantizer.decode(codes_or_latents)
        else:
            raise ValueError(
                "Quantizer must expose 'codebook' or 'decode' to map indices."
            )
        return self.decode_latents(z_q)

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
    ) -> ModularVQAutoencoderOutput:
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]

        return_dict = True if return_dict is None else return_dict
        target = inputs if labels is None else labels

        z_e = self.encode_latents(inputs)
        z_e_flat, leading_shape = self._flatten(z_e)
        if self.config.pre_quant_norm:
            z_e_flat = F.layer_norm(z_e_flat, (z_e_flat.shape[-1],))

        z_q_st_flat, idx_flat, vq_loss, ppl, z_q_flat = self._quantize(z_e_flat)
        z_q_st = z_q_st_flat.view(*leading_shape, z_q_st_flat.shape[-1])
        z_q = z_q_flat.view(*leading_shape, z_q_flat.shape[-1])
        indices = idx_flat.view(*leading_shape)

        x_hat = self.decode_latents(z_q_st)

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
                z_q_st,
            )

        return ModularVQAutoencoderOutput(
            loss=loss,
            recon_loss=recon_loss,
            vq_loss=vq_loss,
            perplexity=ppl,
            x_hat=x_hat,
            indices=indices,
            z_e=z_e,
            z_q=z_q,
            z_q_st=z_q_st,
        )
