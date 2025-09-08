import torch

from ephys_gpt.models.gpt2meg import GPT2MEG, STGPT2MEG
from ephys_gpt.utils.tests import assert_future_grad_zero


@torch.no_grad()
def test_gpt2meg_forward_shapes():
    B, C, T = 2, 6, 16
    x = torch.randint(0, 100, (B, C, T))

    gpt2_cfg = {
        "vocab_size": 128,
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 2,
        "n_positions": 128,
    }
    emb_args = {"quant_emb": 64}
    model = GPT2MEG(num_channels=C, gpt2_config=gpt2_cfg, embedding_args=emb_args)
    model.eval()
    out = model(x)
    # Expect logits over vocab per channel and time
    assert out.shape[:2] == (B, C)
    assert out.shape[-1] == 128


@torch.no_grad()
def test_stgpt2meg_forward_shapes():
    B, C, T = 2, 5, 12
    x = torch.randint(0, 64, (B, C, T))

    trf_args = dict(attn_args={"d_model": 32, "nheads": 4}, mlp_args={"d_model": 32})
    model = STGPT2MEG(
        num_channels=C,
        vocab_size=64,
        d_model=32,
        layers=2,
        trf_args=trf_args,
        embedding_args={},
    )
    model.eval()
    out = model(x)
    assert out.shape == (B, C, T, 64)


def test_grad_causality_gpt2meg():
    B, C, T, V = 1, 6, 64, 128
    # Test at embedding level directly for strict causal gradient
    emb_args = {"quant_emb": 32}
    gpt2_cfg = {
        "vocab_size": V,
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 2,
        "n_positions": 256,
        "resid_pdrop": 0.0,
        "embd_pdrop": 0.0,
        "attn_pdrop": 0.0,
    }
    model = GPT2MEG(num_channels=C, gpt2_config=gpt2_cfg, embedding_args=emb_args)
    model.eval()
    # Build a differentiable embedding sequence directly
    emb = torch.randn(B * C, T, 32, requires_grad=True)
    outputs = model.gpt2(inputs_embeds=emb)
    y = model.head(outputs[0]).view(B, C, T, V)
    loss = y[..., :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def _test_grad_causality_stgpt2meg(block: str):
    B, C, T, V = 1, 5, 64, 64
    # Random differentiable embeddings at block input
    emb = torch.randn(B, C, T, 32, requires_grad=True)
    trf_args = dict(attn_args={"d_model": 32, "nheads": 4}, mlp_args={"d_model": 32})
    model = STGPT2MEG(
        num_channels=C,
        vocab_size=V,
        d_model=32,
        layers=2,
        trf_args=trf_args,
        trf_block=block,
        embedding_args={},
    )
    model.eval()
    # forward through blocks
    h = emb
    for blk in model.blocks:
        h = blk(h.contiguous())
    y = model.head(model.norm(h))
    loss = y[..., :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_stgpt2meg():
    _test_grad_causality_stgpt2meg("STGPTBlockParallel")
    _test_grad_causality_stgpt2meg("STGPTBlock")
    _test_grad_causality_stgpt2meg("STBlock")
