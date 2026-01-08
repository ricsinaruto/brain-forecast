import torch

from transformers.configuration_utils import PretrainedConfig
from transformers.masking_utils import create_causal_mask


def make_block_causal_mask(block_len: int):
    def mask_mod(b, h, q_idx, kv_idx):
        q_blk = q_idx // block_len
        k_blk = kv_idx // block_len
        # allow attending to any token in same block (non-causal)
        # and to any token in earlier blocks (causal across blocks)
        return k_blk <= q_blk

    return mask_mod


def _block_causal_mask(
    config: PretrainedConfig,
    block_size: int,
    past_key_values: tuple,
    inputs_embeds: torch.Tensor,
    include_position_ids: bool = True,
) -> torch.Tensor:
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )

    position_ids = None
    if include_position_ids:
        position_ids = cache_position.unsqueeze(0)

    config._attn_implementation = "flex_attention"

    mask_kwargs = {
        "config": config,
        "input_embeds": inputs_embeds,
        "attention_mask": None,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "or_mask_function": make_block_causal_mask(block_size),
    }
    # Create the masks
    causal_mask_mapping = {
        "full_attention": create_causal_mask(**mask_kwargs),
    }

    return causal_mask_mapping
