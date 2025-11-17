import types
import torch
import torch.nn as nn

from ephys_gpt.models.gpt2meg import VQGPT2MEG
from ephys_gpt.utils.tests import assert_future_grad_zero


class _TinyVQVAE(nn.Module):
    def __init__(self, vocab_size: int = 64):
        super().__init__()
        # pretend tokenizer exposes an embedding_dim and n_codes as used by VideoGPT
        self.embedding_dim = 8
        self.n_codes = vocab_size

    def encode(self, x):
        B, C, T = x.shape
        # Deterministically map input to codes: average over channel,
        # split time into L segments
        L = max(1, T // 2)
        seg = T // L
        codes = []
        xm = x.mean(dim=1)  # (B, T)
        for i in range(L):
            s = i * seg
            e = T if i == L - 1 else (i + 1) * seg
            val = xm[:, s:e].mean(dim=1)
            norm = torch.sigmoid(val)
            code = torch.clamp(
                (norm * (self.n_codes - 1)).round().long(), 0, self.n_codes - 1
            )
            codes.append(code)
        codes = torch.stack(codes, dim=1)
        return None, None, codes


def _monkeypatch_lit_load(monkeypatch, tiny_vqvae: nn.Module):
    # Replace LitModel.load_from_checkpoint to return an object with `model`
    dummy = types.SimpleNamespace(model=tiny_vqvae)

    def fake_load_from_checkpoint(*args, **kwargs):
        return dummy

    from ephys_gpt.training import lightning as lightning_mod

    monkeypatch.setattr(
        lightning_mod.LitModel, "load_from_checkpoint", fake_load_from_checkpoint
    )


def test_vqgpt2meg_forward_shapes_and_targets(monkeypatch):
    B, C, T = 2, 1, 16
    vocab = 64
    tiny = _TinyVQVAE(vocab)
    _monkeypatch_lit_load(monkeypatch, tiny)

    config = dict(
        attn_args={"d_model": 16, "nheads": 2},
        mlp_args={"d_model": 16},
        vocab_size=vocab,
        num_layers=2,
    )

    model = VQGPT2MEG(tokenizer_path="/dev/null", trf_args=config)
    x = torch.randn(B, C, T)
    logits, targets = model(x)

    # Expect next-token setup: logits length equals targets length
    assert logits.shape[:-1] == targets.shape
    assert logits.shape[-1] == vocab


def test_vqgpt2meg_causal_prefix_invariance(monkeypatch):
    B, C, T = 1, 1, 20
    vocab = 32
    tiny = _TinyVQVAE(vocab)
    _monkeypatch_lit_load(monkeypatch, tiny)

    config = dict(
        attn_args={"d_model": 16, "nheads": 2},
        mlp_args={"d_model": 16},
        vocab_size=vocab,
        num_layers=2,
    )

    model = VQGPT2MEG(tokenizer_path="/dev/null", trf_args=config)

    # Build two inputs that share an early prefix but differ in the later half
    x_a = torch.randn(B, C, T)
    x_b = x_a.clone()
    t0 = T // 2
    x_b[:, :, t0:] = torch.randn_like(x_b[:, :, t0:])

    logits_a, tgt_a = model(x_a)
    logits_b, tgt_b = model(x_b)

    # Convert targets back to token positions. We expect the first N tokens
    # (derived from early timesteps) to match exactly.
    n_prefix = min(tgt_a.shape[1], tgt_b.shape[1]) // 2
    assert torch.equal(tgt_a[:, :n_prefix], tgt_b[:, :n_prefix])
    assert torch.allclose(logits_a[:, :n_prefix, :], logits_b[:, :n_prefix, :])


def test_grad_causality_vqgpt2meg_token_level(monkeypatch):
    # Token-level gradient causality on the internal Transformer: ensure
    # future token embeddings do not receive gradient when loss excludes last step.
    B, H, W, T = 1, 32, 32, 100
    vocab = 1024

    trf_args = dict(
        attn_args={"d_model": 16, "nheads": 2},
        mlp_args={"d_model": 16},
        vocab_size=vocab,
        num_layers=2,
    )
    tok_args = dict(
        codebook_size=vocab,
        embed_dim=16,
        z_channels=16,
        double_z=False,
        in_channels=1,
        out_channels=1,
        temporal_downsample_factor=2,
        ch=64,
        ch_mult=[1, 2, 2, 2, 2],
        num_res_blocks=2,
        attn_resolutions=[3],
    )

    model = VQGPT2MEG(
        tokenizer_path=None, trf_args=trf_args, tok_args=tok_args, train_tokenizer=True
    )
    # Build differentiable token embeddings directly at Transformer input
    x = torch.randn(B, H, W, T, requires_grad=True)

    logits, targets = model(x)

    loss = logits[:, :-3, :].sum()
    loss.backward()

    assert_future_grad_zero(x, T - 1)
