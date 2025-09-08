import copy

import torch
import numpy as np

from torch import nn
from math import ceil
from typing import Sequence

from .transformer_blocks import TransformerBlock


class DebugModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        print(x.shape)
        return x


class CausalTransposeDecoder(nn.Module):
    def __init__(
        self,
        enc_width: Sequence[int],
        enc_stride: Sequence[int],
        in_ch: int,
        channels_out: int,
    ):
        """Construct a causal transpose‑convolutional stack that inverts the encoder.

        Each ConvTranspose1d layer mirrors a corresponding Conv1d layer in the
        encoder but in *reverse* order.  We deliberately use `padding=0` so the
        decoder remains *causal* (i.e. the output at time *t* depends **only** on
        inputs ≤ *t*).  To compensate for the receptive‑field shift introduced by
        strides > 1 we crop the extra samples at the very end of `forward()`.
        """
        super().__init__()
        self.enc_width = enc_width
        self.enc_stride = enc_stride
        self.in_ch = in_ch
        self.channels_out = channels_out

        layers = []
        self.kernels = list(enc_width)[::-1]
        self.strides = list(enc_stride)[::-1]

        # Internal feature dimension stays constant (in_ch) throughout the decoder.
        ch = in_ch
        for k, s in zip(self.kernels, self.strides):
            # layers.append(nn.ConstantPad1d((k - 1, 0), 0))
            layers.append(
                nn.ConvTranspose1d(
                    in_channels=ch,
                    out_channels=ch,
                    kernel_size=k,
                    stride=s,  # *causal* – no future leakage
                    padding=0,
                    output_padding=s - 1,  # guarantees length doubling by stride
                ),
            )
            layers.append(nn.GELU())

        self.conv_stack = nn.ModuleList(layers)

        # Final 1 × 1 projection back to raw channel space
        self.to_raw = nn.Conv1d(in_ch, channels_out, kernel_size=1)

    def forward(self, z: torch.Tensor):
        x = z
        layer_idx = 0
        for k, s in zip(self.kernels, self.strides):
            # Apply transposed convolution
            conv = self.conv_stack[layer_idx]
            act = self.conv_stack[layer_idx + 1]
            layer_idx += 2

            x = conv(x)
            # Remove padding replicas (first k−1 samples)
            if k > 1:
                x = x[..., (k - 1) :]
            x = act(x)

        # Final projection (no cropping needed, kernel=1)
        x = self.to_raw(x)
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class ConvEncoderBENDR(nn.Module):
    def __init__(
        self,
        in_features,
        encoder_h=256,
        enc_width=(3, 3, 3, 3, 3, 3),
        enc_downsample=(2, 2, 2, 1, 1, 1),
        dropout=0.0,
    ):
        super().__init__()
        self.encoder_h = encoder_h
        self.in_features = in_features
        assert len(enc_downsample) == len(enc_width)

        self._downsampling = enc_downsample
        self._width = enc_width

        self.encoder = nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module(
                "Encoder_{}".format(i),
                nn.Sequential(
                    # add left causal padding
                    nn.ConstantPad1d((width - downsample, 0), 0),
                    nn.Conv1d(
                        in_features,
                        encoder_h,
                        width,
                        stride=downsample,
                        padding=0,
                    ),
                    nn.Dropout1d(dropout),
                    nn.GroupNorm(encoder_h // 2, encoder_h),
                    nn.GELU(),
                ),
            )
            in_features = encoder_h

    def description(self, sfreq=None, sequence_len=None):
        widths = list(reversed(self._width))[1:]
        strides = list(reversed(self._downsampling))[1:]

        rf = self._width[-1]
        for w, s in zip(widths, strides):
            rf = rf if w == 1 else (rf - 1) * s + 2 * (w // 2)

        desc = "Receptive field: {} samples".format(rf)
        if sfreq is not None:
            desc += ", {:.2f} seconds".format(rf / sfreq)

        ds_factor = np.prod(self._downsampling)
        desc += " | Downsampled by {}".format(ds_factor)
        if sfreq is not None:
            desc += ", new sfreq: {:.2f} Hz".format(sfreq / ds_factor)
        desc += " | Overlap of {} samples".format(rf - ds_factor)
        if sequence_len is not None:
            desc += " | {} encoded samples/trial".format(sequence_len // ds_factor)
        return desc

    def downsampling_factor(self, samples):
        for factor in self._downsampling:
            samples = ceil(samples / factor)
        return samples

    def forward(self, x):
        return self.encoder(x)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze


class BENDRContextualizer(nn.Module):
    def __init__(
        self,
        in_features: int,
        attn_args: dict,
        mlp_args: dict,
        attn_type: str = "standard",
        mlp_type: str = "standard",
        layers: int = 8,
        dropout: float = 0.15,
        position_encoder: int = 25,
        layer_drop: float = 0.0,
        mask_p_t: float = 0.1,
        mask_p_c: float = 0.004,
        mask_t_span: int = 6,
        mask_c_span: int = 64,
        start_token: int = -5,
        finetuning: bool = False,
    ):
        super().__init__()

        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3
        attn_args["d_model"] = in_features * 3
        mlp_args["d_model"] = in_features * 3

        encoder = TransformerBlock(
            attn_args=attn_args,
            mlp_args=mlp_args,
            attn_type=attn_type,
            mlp_type=mlp_type,
        )

        self.transformer_layers = nn.ModuleList(
            [copy.deepcopy(encoder) for _ in range(layers)]
        )
        self.layer_drop = layer_drop
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.start_token = start_token
        self.finetuning = finetuning

        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = nn.Conv1d(
                in_features,
                in_features,
                position_encoder,
                padding=position_encoder // 2,
                groups=16,
            )
            nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            nn.init.constant_(conv.bias, 0)
            conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
            self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )

        self.output_layer = nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=0.02)
            nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = (
                0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data
            )

    def forward(self, x: torch.Tensor):
        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)

        if self.start_token is not None:
            in_token = self.start_token * torch.ones((1, 1, 1), requires_grad=True).to(
                x.device
            ).expand([-1, *x.shape[1:]])
            x = torch.cat([in_token, x], dim=0)

        for layer in self.transformer_layers:
            if not self.training or torch.rand(1) > self.layer_drop:
                x = layer(x, causal=True)

        return self.output_layer(x.permute([1, 2, 0]))

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze
        if self.finetuning or finetuning:
            self.mask_replacement.requires_grad = False
