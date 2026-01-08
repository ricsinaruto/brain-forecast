import pytest
import torch
import torch.nn as nn

from ephys_gpt.models import FlatGPT, FlatGPTMix
from ephys_gpt.models import flatgpt as flatgpt_module
from utils import assert_future_grad_zero


class _DummyTransformer(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, *args, **kwargs):
        super().__init__()
        self.block_size = kwargs.get("block_size", 1)
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.last_seen = None

    def forward(
        self,
        x: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs
    ):
        if (x is None) == (inputs_embeds is None):
            raise ValueError(
                "DummyTransformer expects exactly one of token ids or embeddings."
            )
        if inputs_embeds is not None:
            self.last_seen = inputs_embeds
            return inputs_embeds
        self.last_seen = x
        return self.emb(x)

    def get_embed_layer(self) -> nn.Module:
        return self.emb


flatgpt_module._DummyTransformer = _DummyTransformer


def test_qwen3_positions_match_qwen2_5_for_video_only():
    """
    Qwen3_Video uses the same 3D (T, H, W) position IDs as Qwen2_5_Video.

    Qwen3VLTextModel handles this directly: it uses position_ids[0] (T) for
    causal masking and the full (3, batch, seq) for 3D rotary embedding.
    """
    model = flatgpt_module.Qwen3_Video(
        hidden_size=16,
        vocab_size=32,
        reduced_shape=(2, 2, 1),
        max_position_embeddings=64,
        rope_scaling={"rope_type": "default"},
    )

    pos = model._build_position_ids(batch_size=1, device=torch.device("cpu"), seq_len=4)
    # Shape is (3, batch, seq) for T, H, W
    assert pos.shape == (3, 1, 4)
    assert torch.equal(pos[0, 0], torch.tensor([0, 0, 1, 1]))  # T
    assert torch.equal(pos[1, 0], torch.tensor([0, 1, 0, 1]))  # H
    assert torch.equal(pos[2, 0], torch.zeros(4, dtype=torch.long))  # W

    # With offset (for caching), positions shift within the flattened grid
    offset = model._build_position_ids(
        batch_size=1, device=torch.device("cpu"), seq_len=2, position_offset=1
    )
    assert torch.equal(offset[0, 0], torch.tensor([0, 1]))  # T
    assert torch.equal(offset[1, 0], torch.tensor([1, 0]))  # H
    assert torch.equal(offset[2, 0], torch.zeros(2, dtype=torch.long))  # W


def test_grad_causality_flatgpt():
    B, T = 2, 6
    hidden_size = 24
    vocab_size = 32

    trf_args = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 96,
        "rope_scaling": {"rope_type": "default", "mrope_section": 1},
        "max_position_embeddings": 128,
        "use_cache": False,
        "attention_dropout": 0.0,
        "use_sliding_window": False,
    }

    model = FlatGPT(
        trf_class="Qwen2_5_Video",
        trf_args=trf_args,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(T, 1, 1),
        tok_args={},
        train_tokenizer=True,
    )
    model.eval()

    # Differentiable embeddings fed directly to the decoder
    emb = torch.randn(B, T, hidden_size, requires_grad=True)
    position_ids = model.transformer._build_position_ids(
        batch_size=B, device=emb.device, seq_len=T - 1
    )
    outputs = model.transformer.model(
        inputs_embeds=emb[:, :-1],
        position_ids=position_ids,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = model.head(outputs.last_hidden_state)
    loss = logits.sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


@pytest.mark.parametrize("mix_method", ["mix", "none"])
def test_grad_causality_flatgptmix(mix_method: str):
    B, T, C = 2, 6, 4
    hidden_size = 24
    vocab_size = 32

    trf_args = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 4 * hidden_size,
        "max_position_embeddings": 128,
        "use_cache": False,
        "attention_dropout": 0.0,
        "mix_method": mix_method,
        "pad_token_id": None,
    }

    model = FlatGPTMix(
        trf_class="SmoLLM3",
        trf_args=trf_args,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(T, C, 1),
    )
    model.eval()

    # Differentiable embeddings fed directly to the decoder
    if mix_method == "mix":
        emb = torch.randn(B, T, hidden_size, requires_grad=True)
    else:
        emb = torch.randn(B * C, T, hidden_size, requires_grad=True)

    outputs = model.transformer.model(
        inputs_embeds=emb[:, :-1],
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = model.head(outputs.last_hidden_state)
    loss = logits.sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_flatgpt_block_causal():
    B, T = 2, 6
    block_size = 2
    hidden_size = 24
    vocab_size = 32

    trf_args = {
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "intermediate_size": 96,
        "rope_scaling": {"rope_type": "default", "mrope_section": 1},
        "max_position_embeddings": 128,
        "use_cache": False,
        "attention_dropout": 0.0,
        "use_sliding_window": False,
        "block_size": block_size,
    }

    model = FlatGPT(
        trf_class="Qwen2_5_Video",
        trf_args=trf_args,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        input_shape=(T, 1, 1),
        tok_args={},
        train_tokenizer=True,
    )
    model.eval()

    emb = torch.randn(B, T, hidden_size, requires_grad=True)
    hidden = model.transformer(inputs_embeds=emb, use_cache=False)
    logits = model.head(hidden)
    loss = logits[:, : T - block_size].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - block_size)


def test_token_corruption_schedule_exponential():
    seq_len = 5
    model = FlatGPT(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=16,
        input_shape=(seq_len, 1, 1),
        tok_args={},
        train_tokenizer=True,
        token_corruption_cfg={"enabled": True, "p_start": 0.01, "p_end": 0.05},
    )
    probs = model._token_corruption_schedule(seq_len, torch.device("cpu"))

    assert probs.shape == (seq_len,)
    assert float(probs[0]) == pytest.approx(0.01, rel=1e-2)
    assert float(probs[-1]) == pytest.approx(0.05, rel=1e-2)
    assert torch.all(probs[1:] >= probs[:-1])


def test_token_corruption_swaps_inputs_not_targets():
    seq_len = 6
    vocab_size = 12
    model = FlatGPT(
        trf_class="_DummyTransformer",
        trf_args={"block_size": 1},
        hidden_size=8,
        vocab_size=vocab_size,
        input_shape=(seq_len, 1, 1),
        tok_args={},
        train_tokenizer=True,
        token_corruption_cfg={"enabled": True, "p_start": 1.0, "p_end": 1.0},
    )
    model.train()

    tokens = torch.arange(seq_len).unsqueeze(0)
    model._encode_tokens = lambda x: {"codes": x}

    logits, targets = model(tokens)

    assert logits.shape[1] == seq_len - model.block_size
    assert torch.equal(tokens, torch.arange(seq_len).unsqueeze(0))
    assert torch.equal(targets, tokens[:, model.block_size :])

    corrupted = model.transformer.last_seen
    assert corrupted is not None
    assert corrupted.shape == tokens.shape
    assert torch.all(corrupted != tokens)
    assert torch.all(corrupted < vocab_size)
