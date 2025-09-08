import torch

from ephys_gpt.models.brainomni import BrainOmniSystem, BrainOmniForecast
from ephys_gpt.utils.tests import assert_future_grad_zero


def _dummy_sensor_meta(B: int, C: int):
    pos = torch.zeros(B, C, 6)
    typ = torch.zeros(B, C, dtype=torch.long)
    return pos, typ


@torch.no_grad()
def test_brainomni_system_forward_and_alignment():
    # Tiny tokenizer + forecaster
    # emb_dim of tokenizer equals make_seanet out_channels = base_channels*2**(depth-1)
    cfg = dict(
        codebook_size=64,
        num_quantizers=2,
        latent_channels=4,
        tokenizer=dict(
            in_channels=8,
            base_channels=8,  # out_channels=8*2**(1-1)=8
            encoder_depth=1,
            emb_dim=8,
        ),
        forecaster=dict(
            emb_dim=16,
            depth=2,
            attn_args={"nheads": 4, "d_model": 16},
        ),
    )

    sys = BrainOmniSystem(
        codebook_size=cfg["codebook_size"],
        num_quantizers=cfg["num_quantizers"],
        latent_channels=cfg["latent_channels"],
        tokenizer=cfg["tokenizer"],
        forecaster=cfg["forecaster"],
    )

    B, C, T = 2, 8, 48
    x = torch.randint(0, 10, (B, C, T)).float()
    pos, typ = _dummy_sensor_meta(B, C)
    sys.eval()
    logits, codes_tgt = sys((x, pos, typ))

    # Shapes: logits: (B, C_lat, Nq, T', K); codes_tgt: (B, C_lat, Nq, T')
    assert logits.ndim == 5 and codes_tgt.ndim == 4
    assert logits.shape[:-1] == codes_tgt.shape


def test_grad_causality_brainomniforecast_token_level():
    # Token-level gradient causality: treat latent token embeddings
    # as differentiable input
    B, C_latent, T = 1, 2, 64
    K = 32
    Nq = 2
    D = 16

    forecaster = BrainOmniForecast(
        codebook_size=K,
        num_quantizers=Nq,
        latent_channels=C_latent,
        emb_dim=D,
        depth=1,
        attn_args={"nheads": 2, "d_model": D},
    )

    # Differentiable token embeddings (before Transformer), shape (B, C_latent, T, D)
    emb = torch.randn(B, C_latent, T, D, requires_grad=True)
    tok = emb.view(B, C_latent * T, D)

    x = tok
    for blk in forecaster.layers:
        x = blk(x, T)
    x = forecaster.norm(x)

    # Project to per-stage logits and reshape back to (B, C_latent, T, K)
    logits_stages = []
    for head in forecaster.stage_heads:
        logit = head(x)  # (B, S, K)
        logit = logit.view(B, C_latent, T, K)
        logits_stages.append(logit)
    logits = torch.stack(logits_stages, dim=2)  # (B, C_latent, Nq, T, K)

    # Loss on early timesteps only
    loss = logits[..., :-1, :].sum()
    loss.backward()

    # Future token embeddings (time >= t0) must have zero gradient
    assert_future_grad_zero(emb, T - 1)


def test_grad_causality_brainomnisystem():

    B, C, T = 2, 272, 200

    cfg = dict(
        codebook_size=1024,
        num_quantizers=4,
        latent_channels=16,
        tokenizer=dict(in_channels=C, base_channels=32, encoder_depth=3, emb_dim=128),
        forecaster=dict(
            emb_dim=128, depth=2, dropout=0.0, attn_args={"nheads": 8, "d_model": 128}
        ),
    )

    sys = BrainOmniSystem(
        codebook_size=cfg["codebook_size"],
        num_quantizers=cfg["num_quantizers"],
        latent_channels=cfg["latent_channels"],
        tokenizer=cfg["tokenizer"],
        forecaster=cfg["forecaster"],
        train_tokenizer=True,
    )

    receptive_field = sys.tokenizer.receptive_field

    x = torch.randn((B, C, T), requires_grad=True)
    pos = torch.zeros(B, C, 6)
    typ = torch.zeros(B, C, dtype=torch.long)

    # Get logits for base input
    sys.eval()
    logits, targets = sys((x, pos, typ))

    # Loss on early timesteps only
    loss = logits.sum()
    loss.backward()

    # Future token embeddings (time >= t0) must have zero gradient
    assert_future_grad_zero(x, T - receptive_field)
