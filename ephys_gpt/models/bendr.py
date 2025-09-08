import torch
import torch.nn as nn

from ..layers.bendr import (
    ConvEncoderBENDR,
    BENDRContextualizer,
    CausalTransposeDecoder,
)


def compute_downsample_factor(strides):
    """Utility to compute the cumulative down‑sampling factor of ConvEncoderBENDR.
    Args:
        strides (Sequence[int]): strides used in the encoder (enc_downsample)
    Returns:
        int: cumulative down‑sampling factor
    """
    factor = 1
    for s in strides:
        factor *= s
    return factor


class BENDRForecast(nn.Module):
    """Forecast the next raw EEG sample for *all* channels.

    The model reuses the *core* BENDR architecture – a stack of 1‑D convolutions
    followed by a Transformer encoder – but adds a lightweight *projection* head
    that maps the contextualised embeddings back to the raw signal space.

    During training the model receives a sequence **x** with shape
    ``(batch, channels, samples)`` and is optimised with *mean‑squared error*
    to predict **x** *one timestep into the future*.
    """

    def __init__(
        self,
        channels: int,
        samples: int,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "standard",
        encoder_h: int = 512,
        enc_width=(3, 2, 2, 2, 2, 2),
        enc_downsample=(2, 2, 2, 1, 1, 1),
        transformer_layers: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Encoder (identical to original BENDR)
        # ------------------------------------------------------------------
        self.encoder = ConvEncoderBENDR(
            in_features=channels,
            encoder_h=encoder_h,
            enc_width=enc_width,
            enc_downsample=enc_downsample,
            dropout=0.0,  # keep feature extractor deterministic
        )

        # ------------------------------------------------------------------
        # Transformer contextualiser (identical hyper‑params to paper)
        # ------------------------------------------------------------------
        self.contextualiser = BENDRContextualizer(
            in_features=encoder_h,
            attn_args=attn_args,
            mlp_args=mlp_args,
            attn_type=attn_type,
            mlp_type=mlp_type,
            layers=transformer_layers,
            dropout=dropout,
            finetuning=False,  # training from scratch
            mask_p_t=0.0,  # *disable* masking for forecasting objective
            mask_p_c=0.0,
            position_encoder=25,
        )

        # ------------------------------------------------------------------
        # NEW: projection layer back to raw sample space
        # ------------------------------------------------------------------
        self.project = CausalTransposeDecoder(
            enc_width=enc_width,
            enc_stride=enc_downsample,
            in_ch=encoder_h,
            channels_out=channels,
        )

        # The cumulative down‑sampling factor so that we know which target step
        # each encoded element should predict.
        self._ds_factor = compute_downsample_factor(enc_downsample)
        self._encoded_len = samples // self._ds_factor
        self.receptive_field = (self._encoded_len - 1) * self._ds_factor
        self.output_dim = encoder_h

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor):
        x, _, _ = x
        # Inputs use only the context window up to the last encoded step.
        inputs = x[..., : (self._encoded_len - 1) * self._ds_factor]

        # Convolutional feature extractor
        z = self.encoder(inputs)  # (B, F, Tenc)
        # Contextualisation
        c = self.contextualiser(z)  # (B, F, Tenc)
        # Project back to channels
        return c

    def forward(self, x: torch.Tensor):
        """Run a forward pass.

        Args:
            x (Tensor): raw EEG with shape *(B, C, T)*.

        Returns:
            Tensor: prediction of shape *(B, C, Tenc)* corresponding to the *next*
            raw sample that starts *after* each encoded window.
        """

        c = self.encode(x)
        y_hat = self.project(c)  # (B, C, T_raw)
        return y_hat

    @torch.inference_mode()
    def forecast(self, past: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Autoregressive forecast using the trained 1‑step predictor.

        Args:
            past: (B, C, Lp) observed context
            horizon: number of future steps to generate (N)

        Returns:
            (B, C, Lp+N) concatenation of past and generated samples
        """
        device = next(self.parameters()).device
        seq = past.to(device)
        B, C, Lp = seq.shape

        # Effective receptive window the model was trained with
        win = (self._encoded_len - 1) * self._ds_factor
        win = max(int(win), 1)

        generated: list[torch.Tensor] = []
        for _ in range(horizon):
            ctx = seq[..., -win:]
            y_next = self((ctx, None, None))[..., -1]  # (B,C)
            generated.append(y_next.unsqueeze(-1))
            seq = torch.cat([seq, y_next.unsqueeze(-1)], dim=-1)

        return seq
