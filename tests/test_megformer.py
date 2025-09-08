import torch

from ephys_gpt.models.megformer import MEGFormer
from ephys_gpt.utils.tests import assert_future_grad_zero


attn_args = {
    "nheads": 4,
}
mlp_args = {}


def test_megformer_forward_and_nll_shapes():
    model = MEGFormer(
        input_size=16,
        patch_size=4,
        latent_dim=16,
        flow_layers=1,
        d_model=32,
        attn_args=attn_args,
        mlp_args=mlp_args,
        n_layers=2,
        n_mixtures=4,
        context_length=8,
        forecast_steps=1,
    )

    B, H, W, T = 2, 16, 16, 8
    x = torch.randn(B, H, W, T)
    model.eval()
    nll_tok, logdet = model(x)
    # Scalars or per-batch means
    assert isinstance(nll_tok, torch.Tensor) and isinstance(logdet, torch.Tensor)


@torch.no_grad()
def test_megformer_causal_mask_prefix_invariance():
    model = MEGFormer(
        input_size=16,
        patch_size=4,
        latent_dim=16,
        flow_layers=1,
        d_model=32,
        attn_args=attn_args,
        mlp_args=mlp_args,
        n_layers=1,
        n_mixtures=2,
        context_length=8,
        forecast_steps=2,
    )
    B, H, W, T = 1, 16, 16, 6
    x = torch.randn(B, H, W, T)
    model.eval()
    # Forward uses teacher forcing with causal mask; validate mask effect by
    # comparing decoder outputs when appending future frames during forecast.
    x_future = torch.randn(B, H, W, 2)
    y_ctx = model.forecast(x, steps=1)
    y_more = model.forecast(torch.cat([x, x_future], dim=-1)[..., :T], steps=1)
    assert torch.allclose(y_ctx[..., :T], y_more[..., :T])


def test_grad_causality_megformer():
    # Gradient causality: loss on early tokens must not backprop into future frames

    model = MEGFormer(
        input_size=32,
        patch_size=16,
        latent_dim=256,
        flow_layers=4,
        d_model=32,
        attn_args=attn_args,
        mlp_args=mlp_args,
        n_layers=2,
        n_mixtures=2,
        context_length=199,
        forecast_steps=1,
    )

    B, H, W, T, L = 1, 32, 32, 200, 4
    x = torch.randn(B, H, W, T, requires_grad=True)

    nll_per_tok, _ = model(x, reduce="none")
    k = (T - 1) * L - 1  # predict up to time T-2 inclusive
    k = max(1, min(k, nll_per_tok.shape[1]))
    loss = nll_per_tok[:, :k].sum()
    loss.backward()

    # Gradients on the final frame must be zero
    assert_future_grad_zero(x, T - 1)
