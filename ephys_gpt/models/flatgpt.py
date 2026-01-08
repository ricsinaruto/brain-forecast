import inspect

import torch

from torch import nn
from typing import Callable, Tuple
import numpy as np
from tqdm import tqdm

from ..training.lightning import LitModel
from .hf_adapters.llm import SmoLLM3, MiniMax  # noqa: F401
from .hf_adapters.vlm import (  # noqa: F401
    Qwen2_5_Video,
    Qwen2_5_VideoText,
    Qwen3_Video,
)
from ..layers.flatgpt_layers import (
    QuantizerEmbedding,
    JointRVQHead,
    TiedRVQHead,
    ListEmbedding,
    ChannelHead,
    MixEmbedding,
    MixHead,
)
from .tokenizers.flat_tokenizers import (  # noqa: F401
    AmplitudeTokenizer,
    AmplitudeTokenizerMix,
    DelimitedTokenizer,
    BPETokenizer,
)


class FlatGPT(nn.Module):
    def __init__(
        self,
        trf_class: str,
        trf_args: dict,
        hidden_size: int,
        vocab_size: int,
        input_shape: Tuple[int, int, int],  # T, H, W
        input_type: str = "vector",  # vector, image
        spatial_reduction: int | Tuple[int, int] = 1,
        temporal_reduction: int = 1,
        tok_class: str = "AmplitudeTokenizer",  # amplitde, cosmos, brainomni, etc.
        tok_args: dict = None,
        tokenizer_path: str = None,
        train_tokenizer: bool = False,
        token_corruption_cfg: dict | None = None,
    ):
        super().__init__()
        if isinstance(spatial_reduction, int):
            spatial_reduction = (spatial_reduction, spatial_reduction)

        self.train_tokenizer = train_tokenizer
        self.input_shape = input_shape
        self.input_type = input_type
        self.spatial_reduction = spatial_reduction
        self.temporal_reduction = temporal_reduction
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.block_size = trf_args.get("block_size", 1)
        self.reduced_shape = (
            input_shape[0] // temporal_reduction,
            input_shape[1] // spatial_reduction[0],
            input_shape[2] // spatial_reduction[1],
        )

        tok_args = tok_args or {}
        tok_args["temporal_reduction"] = temporal_reduction
        tok_args["spatial_reduction"] = spatial_reduction
        tok_args["input_shape"] = input_shape
        tok_args["vocab_size"] = vocab_size
        tok_args["hidden_size"] = hidden_size

        trf_args["hidden_size"] = hidden_size
        trf_args["vocab_size"] = vocab_size
        trf_args["reduced_shape"] = self.reduced_shape
        if trf_class == "Qwen2_5_Video_TASA3D" and "input_shape" not in trf_args:
            trf_args["input_shape"] = input_shape

        # Load tokenizer if path is given
        if tokenizer_path is not None:
            lit = LitModel.load_from_checkpoint(tokenizer_path, strict=False)
            self.tokenizer = lit.model

            # check if model is compiled
            if hasattr(self.tokenizer, "_orig_mod"):
                self.tokenizer = self.tokenizer._orig_mod

        else:
            self.tokenizer = globals()[tok_class](**tok_args)

        # freeze tokenizer during autoregressive training (optional)
        if not self.train_tokenizer:
            for p in self.tokenizer.parameters():
                p.requires_grad_(False)

        self.tokenizer.eval()
        self.transformer = globals()[trf_class](**trf_args)
        self.pre_embedding = None

        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

        # tie embeddings
        self.head.weight = self.transformer.get_embed_layer().weight
        self.token_corruption_cfg = self._init_token_corruption_cfg(
            token_corruption_cfg
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # L corresponds to T'*H'*W', the product of reduced dimensions
        inputs = self._encode_tokens(x)  # (B, L)
        codes = inputs.pop("codes")
        model_codes = self._apply_token_corruption(codes)

        if self.pre_embedding is not None:
            embeds = self.pre_embedding(model_codes)
            trf_out = self.transformer(inputs_embeds=embeds, **inputs)
        else:
            trf_out = self.transformer(model_codes, **inputs)

        hidden = trf_out
        if isinstance(trf_out, tuple):
            hidden = trf_out[0]
        elif hasattr(trf_out, "last_hidden_state"):
            hidden = trf_out.last_hidden_state

        logits = self.head(hidden)
        return logits[:, : -self.block_size], codes[:, self.block_size:]

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if self.train_tokenizer:
            return self.tokenizer.encode(x)
        with torch.no_grad():
            return self.tokenizer.encode(x)

    def _init_token_corruption_cfg(self, cfg: dict | None) -> dict:
        """Normalise corruption settings; disabled by default."""
        base_cfg = {"enabled": False, "p_start": 0.0, "p_end": 0.0}
        if not cfg:
            return base_cfg

        parsed = dict(base_cfg)
        parsed.update(cfg)
        parsed["p_start"] = max(0.0, float(parsed.get("p_start", base_cfg["p_start"])))
        parsed["p_end"] = max(0.0, float(parsed.get("p_end", base_cfg["p_end"])))
        parsed["enabled"] = bool(parsed.get("enabled", True)) and (
            parsed["p_start"] > 0.0 or parsed["p_end"] > 0.0
        )
        return parsed

    def _token_corruption_schedule(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Exponential interpolation between start/end probabilities across sequence."""
        cfg = self.token_corruption_cfg
        start = max(0.0, cfg["p_start"])
        end = max(0.0, cfg["p_end"])

        if seq_len <= 0 or (start == 0.0 and end == 0.0):
            return torch.zeros(seq_len, device=device)

        if seq_len == 1:
            return torch.tensor([max(start, end)], device=device)

        start_safe = torch.clamp(
            torch.tensor(start, device=device, dtype=torch.float32), min=1e-12
        )
        end_safe = torch.clamp(
            torch.tensor(end, device=device, dtype=torch.float32), min=1e-12
        )
        steps = torch.linspace(0, 1, steps=seq_len, device=device)
        log_start, log_end = torch.log(start_safe), torch.log(end_safe)
        probs = torch.exp(log_start + (log_end - log_start) * steps)

        if start == 0.0:
            probs[0] = 0.0
        if end == 0.0:
            probs[-1] = 0.0
        return torch.clamp(probs, 0.0, 1.0)

    def _apply_token_corruption(self, codes: torch.Tensor) -> torch.Tensor:
        """Randomly swap tokens using an exponential position-dependent schedule."""
        cfg = self.token_corruption_cfg
        if not (self.training and cfg["enabled"]):
            return codes

        seq_len = codes.shape[-1]
        if seq_len == 0 or self.vocab_size <= 1:
            return codes

        probs = self._token_corruption_schedule(seq_len, codes.device)
        if torch.all(probs == 0):
            return codes

        prob_shape = (1,) * (codes.dim() - 1) + (seq_len,)
        swap_mask = torch.rand(codes.shape, device=codes.device) < probs.view(
            prob_shape
        )
        if not swap_mask.any():
            return codes

        replacement = torch.randint(
            0,
            self.vocab_size - 1,
            codes.shape,
            device=codes.device,
            dtype=codes.dtype,
        )
        replacement = replacement + (replacement >= codes)

        return torch.where(swap_mask, replacement, codes)

    def _forecast_tokens_per_step(
        self, encoded: torch.Tensor, raw_input: torch.Tensor
    ) -> int:
        """Determine how many tokens correspond to one reduced timestep.

        Delegates to the tokenizer when available.
        """
        if hasattr(self.tokenizer, "forecast_tokens_per_step"):
            return int(
                self.tokenizer.forecast_tokens_per_step(
                    encoded, raw_input, self.reduced_shape
                )
            )
        return int(np.prod(self.reduced_shape[1:])) // self.temporal_reduction

    def _tokens_per_embedding(self, *args, **kwargs) -> int:
        return self._forecast_tokens_per_step(*args, **kwargs)

    def _forecast_strip_tokens(
        self, seq: torch.Tensor, tokens_per_step: int
    ) -> torch.Tensor:
        """Strip tokenizer-specific padding/markers after generation."""
        if hasattr(self.tokenizer, "forecast_strip_tokens"):
            return self.tokenizer.forecast_strip_tokens(seq, tokens_per_step)
        return seq

    @torch.inference_mode()
    def _call_transformer(
        self,
        token_batch: torch.Tensor,
        cache_in=None,
        cache_enabled: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, tuple | None]:
        kwargs = dict(kwargs)
        # Drop helper-only args that should not be forwarded to the transformer
        chid = kwargs.pop("chid", None)
        if cache_in is not None:
            kwargs["past_key_values"] = cache_in
            kwargs["use_cache"] = True
        elif cache_enabled:
            kwargs["use_cache"] = True

        if self.pre_embedding is not None:
            # check if pre_embedding method takes chid as an argument
            if "chid" in inspect.signature(self.pre_embedding.forward).parameters:
                embeds = self.pre_embedding(token_batch, chid=chid)
            else:
                embeds = self.pre_embedding(token_batch)

            kwargs["inputs_embeds"] = embeds
            token_batch = None

        out = self.transformer(token_batch, **kwargs)

        cache_out = None
        hidden_out = out
        if isinstance(out, tuple):
            hidden_out, cache_out = out
        elif hasattr(out, "last_hidden_state"):
            hidden_out = out.last_hidden_state
            cache_out = getattr(out, "past_key_values", None)

        return hidden_out, cache_out

    def _call_head(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        if "chid" in inspect.signature(self.head.forward).parameters:
            chid = kwargs.pop("chid", None)
            return self.head(hidden, chid=chid)
        return self.head(hidden)

    @torch.inference_mode()
    def forecast(
        self,
        initial_input: torch.Tensor,
        rollout_steps: int,
        sample_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        max_context_tokens: int = -1,
        use_cache: bool | None = None,
        sliding_window_overlap: float = 0.5,
    ) -> torch.Tensor:
        """Recursive autoregressive forecast starting from `initial_input`.

        Args:     initial_input: Raw input sample (shaped like training data) or
        integer token ids of shape (B, L).     rollout_steps: Number of *reduced*
        timesteps to roll out.     sample_fn: Callable applied to logits for the next
        token. When         generating blocks, logits are flattened across the block
        dimension before calling `sample_fn`, which is expected to         accept a
        tensor of shape (N, vocab_size) and return integer         token ids.
        max_context_tokens: Optional sliding-window cap when KV cache is
        unavailable.     use_cache: Force-enable/disable KV caching; defaults to using
        caching when the transformer supports it.     sliding_window_overlap: If float,
        interpreted as the fraction of the         current window length to SHIFT by
        once the cacheable horizon is         reached (smaller values keep more
        overlap). If int, used         directly as the shift/stride in tokens.

        Returns:     Token ids containing the original context followed by the
        generated rollout. If using DelimitedTokenizer, delimiter tokens     are
        stripped from the returned tensor.
        """

        def _make_chid_block(start_token: int, block_len: int, modulus: int):
            """Construct channel ids for embedding-aware subclasses.

            Only emit a scalar id when generating a single token so heads that expect an
            integer index (e.g., ChannelHead) continue to work.
            """
            if block_len == 1:
                return int(start_token % modulus)
            return None

        if rollout_steps < 0:
            raise ValueError("rollout_steps must be non-negative.")

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()

        seq = (
            initial_input[0]
            if isinstance(initial_input, (tuple, list))
            else initial_input
        )

        tokens = self._encode_tokens(initial_input)
        tokens = tokens["codes"]

        tokens = tokens.to(device)
        tokens_per_step = self._forecast_tokens_per_step(tokens, seq)
        toks_per_emb = self._tokens_per_embedding(tokens, seq)
        if tokens_per_step <= 0:
            raise ValueError("Invalid tokens_per_step; computed non-positive value.")

        total_new_tokens = int(rollout_steps) * tokens_per_step

        forward_params = inspect.signature(self.transformer.forward).parameters
        supports_cache = "past_key_values" in forward_params
        enable_cache = (
            supports_cache if use_cache is None else (use_cache and supports_cache)
        )

        context_seq = tokens.long()
        generated_tokens: list[torch.Tensor] = []

        window_size_limit = (
            context_seq.shape[1]
            if max_context_tokens in (-1, None)
            else max_context_tokens
        )
        window_size = max(1, int(window_size_limit))

        # Interpret overlap: int = stride tokens; float = stride as fraction of window
        if isinstance(sliding_window_overlap, int):
            stride_tokens_direct = max(1, sliding_window_overlap)
            overlap_ratio = None
        else:
            overlap_ratio = float(sliding_window_overlap)
            if overlap_ratio < 0.0:
                raise ValueError("sliding_window_overlap must be non-negative.")
            stride_tokens_direct = None

        total_generated = 0
        pbar = tqdm(total=total_new_tokens, desc="Forecast (windowed)")
        while total_generated < total_new_tokens:
            if context_seq.shape[1] > window_size:
                context_seq = context_seq[:, -window_size:]

            if stride_tokens_direct is not None:
                stride = min(window_size, stride_tokens_direct)
            else:
                stride = int(round(window_size * overlap_ratio))
                stride = max(1, min(stride, window_size))

            # Prefill on the current window to seed cache and next-block logits.
            hidden, cache = self._call_transformer(
                context_seq, cache_enabled=enable_cache
            )

            pred_hidden = hidden[:, -self.block_size:, :]
            chid_block = _make_chid_block(
                total_generated, pred_hidden.shape[1], toks_per_emb
            )
            next_logits = self._call_head(pred_hidden, chid=chid_block)
            if next_logits.dim() == 2:
                next_logits = next_logits.unsqueeze(1)

            for _ in tqdm(
                range(total_new_tokens - total_generated),
                desc="Window fill",
                leave=False,
            ):
                if total_generated >= total_new_tokens:
                    break
                if context_seq.shape[1] >= window_size:
                    break

                tokens_left = total_new_tokens - total_generated
                room_in_window = window_size - context_seq.shape[1]
                available_preds = int(next_logits.shape[1])
                tokens_this_step = min(
                    self.block_size, tokens_left, room_in_window, available_preds
                )
                if tokens_this_step <= 0:
                    break

                logits_block = next_logits[:, :tokens_this_step, :]
                next_block = sample_fn(logits_block).to(device)

                generated_tokens.append(next_block)
                context_seq = torch.cat([context_seq, next_block], dim=1)
                total_generated += int(next_block.shape[1])
                pbar.update(int(next_block.shape[1]))

                if (
                    total_generated >= total_new_tokens
                    or context_seq.shape[1] >= window_size
                ):
                    break

                chid_block = _make_chid_block(
                    total_generated - next_block.shape[1],
                    next_block.shape[1],
                    toks_per_emb,
                )
                if enable_cache:
                    hidden, cache = self._call_transformer(
                        next_block,
                        cache_in=cache,
                        cache_enabled=enable_cache,
                        chid=chid_block,
                    )
                else:
                    hidden, cache = self._call_transformer(
                        context_seq, cache_enabled=enable_cache
                    )

                pred_hidden = hidden[:, -self.block_size:, :]
                head_chid = _make_chid_block(
                    total_generated, pred_hidden.shape[1], toks_per_emb
                )
                next_logits = self._call_head(pred_hidden, chid=head_chid)

            if total_generated >= total_new_tokens:
                break

            if context_seq.shape[1] > stride:
                context_seq = context_seq[:, stride:]
            else:
                context_seq = context_seq[:, -1:]

        pbar.close()
        full_seq = torch.cat(generated_tokens, dim=1)
        full_seq = self._forecast_strip_tokens(full_seq, tokens_per_step)

        if was_training:
            self.train()

        return full_seq


class FlatGPTMix(FlatGPT):
    def __init__(self, *args, mix_method: str = "mix", **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_embedding = MixEmbedding(
            self.vocab_size,
            self.hidden_size,
            self.reduced_shape[1],
            mix_method=mix_method,
        )

        self.head = MixHead(
            self.hidden_size,
            self.vocab_size,
            self.reduced_shape[1],
            mix_method=mix_method,
            emb=self.pre_embedding.emb.quant_emb,
        )


class FlatGPTRVQ(FlatGPT):
    """FlatGPT variant that operates with per-quantizer vocabularies and concatenated
    quantizer embeddings."""

    def __init__(
        self,
        *args,
        quantizer_head: str = "joint",
        quantizer_embed_dim: int | None = None,
        quantizer_levels: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Determine quantizer meta
        if quantizer_levels is None:
            quantizer_levels = getattr(
                getattr(getattr(self.tokenizer, "quantizer", None), "rvq", None),
                "num_quantizers",
                None,
            )

        level_vocab = getattr(self.tokenizer, "codebook_size", None)
        if level_vocab is None:
            rvq = getattr(getattr(self.tokenizer, "quantizer", None), "rvq", None)
            level_vocab = getattr(rvq, "codebook_size", self.vocab_size)

        if quantizer_embed_dim is None:
            if self.hidden_size % quantizer_levels != 0:
                raise ValueError(
                    "hidden_size must be divisible by the number of quantizers or "
                    "set quantizer_embed_dim."
                )
            level_dim = self.hidden_size // quantizer_levels
        else:
            level_dim = int(quantizer_embed_dim)

        total_hidden = level_dim * quantizer_levels

        # Embeddings and heads per quantizer
        self.pre_embedding = QuantizerEmbedding(
            [int(level_vocab) for _ in range(quantizer_levels)], level_dim
        )
        head_type = quantizer_head.lower()
        if head_type == "joint":
            self.head = JointRVQHead(total_hidden, quantizer_levels, int(level_vocab))
        elif head_type == "tied":
            self.head = TiedRVQHead(self.pre_embedding.embeddings)
        else:
            raise ValueError("quantizer_head must be 'joint' or 'tied'.")

    def _encode_tokens(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        inputs = super()._encode_tokens(x)
        inputs["codes"] = inputs.pop("rvq_codes")
        return inputs

    def _call_transformer(
        self, token_batch: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, tuple | None]:
        codes = token_batch.long()
        if codes.dim() == 2:
            codes = codes.unsqueeze(1)

        return super()._call_transformer(codes, **kwargs)

    def _call_head(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        squeeze = False
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
            squeeze = True
        logits = self.head(hidden)
        if squeeze:
            logits = logits[:, -1, ...]
        return logits

    def _forecast_strip_tokens(
        self, seq: torch.Tensor, tokens_per_step: int
    ) -> torch.Tensor:
        if seq.dim() == 3:
            seq = seq.reshape(seq.shape[0], -1)
        return super()._forecast_strip_tokens(seq, tokens_per_step)


class FlatGPTEmbeds(FlatGPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_embedding = ListEmbedding(
            self.vocab_size, self.hidden_size, self.reduced_shape[1]
        )

        # Replace the shared head with a channel-aware tied head.
        self.head = ChannelHead(self.pre_embedding.emb)


class FlatGPTEmbedsRVQ(FlatGPTEmbeds):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pre_embedding = ListEmbedding(
            self.vocab_size, self.hidden_size, self.reduced_shape[2]
        )

        # Replace the shared head with a channel-aware tied head.
        self.head = ChannelHead(self.pre_embedding.emb)

    def _tokens_per_embedding(self, *args, **kwargs) -> int:
        return self.reduced_shape[2]
