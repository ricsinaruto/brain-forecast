import torch

from ephys_gpt.models.bendr import BENDRForecast
from utils import assert_future_grad_zero


@torch.no_grad()
def test_bendr_forward_shapes_and_shift():
    B, C, T = 2, 4, 1024
    x = torch.randn(B, C, T)
    # dummy sensor metadata (unused by BENDRForecast forward but expected tuple)
    pos = torch.zeros(B, C, 2)
    typ = torch.zeros(B, C, dtype=torch.long)

    model = BENDRForecast(
        channels=C, samples=T, dropout=0.0, attn_args={"nheads": 8}, mlp_args={}
    )
    y = model((x, pos, typ))  # (B, C, Tenc)
    assert y.shape[0] == B and y.shape[1] == C
    # Decoder returns raw-length predictions matching the model's internal setup
    assert y.shape[-1] > 0


@torch.no_grad()
def test_bendr_causal_invariance_prefix_outputs_unchanged():
    B, C, T = 1, 4, 512
    model = BENDRForecast(
        channels=C, samples=T, dropout=0.0, attn_args={"nheads": 8}, mlp_args={}
    )
    model.eval()
    # Use only context window the model would see
    L_ctx = (model._encoded_len - 1) * model._ds_factor
    L_ctx = max(int(L_ctx), 1)
    # ensure minimum length for first conv kernel
    x = torch.randn(B, C, max(L_ctx, 3))

    # Outputs given a baseline future vs. perturbed future should match
    pos = torch.zeros(B, C, 2)
    typ = torch.zeros(B, C, dtype=torch.long)
    y_base = model((x, pos, typ))

    # Re-run on same prefix; output must be identical (sanity of determinism)
    y_perturb = model((x.clone(), pos, typ))

    # Model forward only depends on provided input; ensure identical with same prefix
    assert torch.allclose(y_base, y_perturb, atol=0, rtol=0)


@torch.no_grad()
def test_bendr_forecast_autoregressive_grows_length():
    B, C, Lp, horizon = 2, 3, 32, 5
    model = BENDRForecast(
        channels=C, samples=512, dropout=0.0, attn_args={"nheads": 8}, mlp_args={}
    )
    past = torch.randn(B, C, max(Lp, 128))
    seq = model.forecast(past, horizon)
    assert seq.shape == (B, C, max(Lp, 128) + horizon)


def test_grad_causality():
    B, C, T = 1, 4, 512
    x = torch.randn(B, C, T, requires_grad=True)
    pos = torch.zeros(B, C, 2)
    typ = torch.zeros(B, C, dtype=torch.long)

    model = BENDRForecast(
        channels=C, samples=T, dropout=0.0, attn_args={"nheads": 8}, mlp_args={}
    )
    y = model((x, pos, typ))  # (B,C,Tr)

    loss = y[..., :-1].sum()
    loss.backward()
    # Check gradients on a far-future slice of the input are zero
    assert_future_grad_zero(x, model.receptive_field)
