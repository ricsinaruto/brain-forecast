import torch

from ephys_gpt.models import ChronoFlowSSM, CK3D, LITRA, TACA, TASA3D
from ephys_gpt.models.chronoflow import ChronoFlowConfig
from ephys_gpt.utils.tests import assert_future_grad_zero


def test_grad_causality_litra():
    B, H, W, T, E = 1, 16, 16, 50, 16
    # Random differentiable embeddings at block input
    emb = torch.randn(B, H, W, T, E, requires_grad=True)
    x_int = torch.randint(0, 256, (B, H, W, T))

    model = LITRA(
        quant_levels=256,
        d_model=E,
        n_heads=4,
        n_layers=1,
        max_T=T,
        max_H=H,
        max_W=W,
        m_hw=256,
        m_ht=512,
        m_wt=512,
    )

    model.eval()
    y = model(x_int=x_int, x=emb)

    loss = y[..., :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_taca():
    B, H, W, T, E = 1, 16, 16, 50, 16
    # Random differentiable embeddings at block input
    emb = torch.randn(B, H, W, T, E, requires_grad=True)
    x_int = torch.randint(0, 256, (B, H, W, T))

    model = TACA(
        H=H,
        W=W,
        dim=E,
        heads=4,
        enc_layers=4,
        pool=(4, 4),
        dropout=0.0,
        max_T=T + 1,
    )

    model.eval()
    y = model(x_int, embeds=emb)

    loss = y[..., :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_ck3d():
    B, H, W, T, E = 1, 16, 16, 50, 16
    # Random differentiable embeddings at block input
    emb = torch.randn(B, H, W, T, E, requires_grad=True)
    x_int = torch.randint(0, 256, (B, H, W, T))

    model = CK3D(emb_dim=E, channels=128)

    model.eval()
    y = model(x_int, embeds=emb)

    loss = y[..., :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_tasa3d():
    B, H, W, T, E = 1, 16, 16, 50, 16
    # Random differentiable embeddings at block input
    emb = torch.randn(B, H, W, T, E, requires_grad=True)
    x_int = torch.randint(0, 256, (B, H, W, T))

    model = TASA3D(
        emb_dim=E,
        input_hw=(H, W),
        depth=4,
        num_down=3,
        channel_grow=2,
    )

    model.eval()
    y = model(x_int, embeds=emb)

    loss = y[..., :-1, :].sum()
    loss.backward()

    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_chronoflow():
    B, T, C, H, W = 1, 100, 1, 32, 32
    # Random differentiable embeddings at block input
    x = torch.rand(B, T, C, H, W, requires_grad=True)

    cfg = ChronoFlowConfig(
        in_channels=C,
        image_size=(H, W),
        spatial_levels=2,  # increase for higher res
        spatial_steps_per_level=2,
        spatial_hidden=160,
        emission_levels=2,
        emission_steps_per_level=2,
        emission_hidden=160,
        cond_dim=512,
        temporal_levels=4,  # strides: 1,2,4,8
        temporal_state_dim=512,
        rollout_prob=0.1,
    )

    model = ChronoFlowSSM(cfg)

    model.eval()
    y = model.sample(x, steps=1)

    loss = y[:, :-2].sum()
    loss.backward()

    assert_future_grad_zero(x, T - 1)
