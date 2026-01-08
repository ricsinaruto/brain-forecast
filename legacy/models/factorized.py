import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput


def _as_bct(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Accepts:
      [C, T] or [C, T, 1] or [B, C, T] or [B, C, T, 1]
    Returns:
      x_bct: [B, C, T]
      was_unbatched: bool (if input had no batch dim)
    """
    if x.dim() == 2:  # [C, T]
        return x.unsqueeze(0), True
    if x.dim() == 3:
        if x.shape[-1] == 1:  # [C, T, 1]
            return x[..., 0].unsqueeze(0), True
        # [B, C, T]
        return x, False
    if x.dim() == 4:
        if x.shape[-1] != 1:
            raise ValueError(f"Expected last dim=1 for [B,C,T,1], got {x.shape}")
        # [B, C, T, 1]
        return x[..., 0], False
    raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")


def _restore_shape(
    x_bct1: torch.Tensor, was_unbatched: bool, want_last_dim_1: bool
) -> torch.Tensor:
    """
    x_bct1: [B, C, T] or [B, C, T, 1] (we'll accept either)
    returns either [C, T] / [C, T, 1] if was_unbatched else [B, C, T] / [B, C, T, 1]
    """
    if x_bct1.dim() == 3:
        x = x_bct1
        if want_last_dim_1:
            x = x.unsqueeze(-1)  # [B,C,T,1]
    elif x_bct1.dim() == 4:
        x = x_bct1 if want_last_dim_1 else x_bct1[..., 0]
    else:
        raise ValueError

    if was_unbatched:
        return x[0]
    return x


class TemporalDownsampleBlock(nn.Module):
    """
    Depthwise temporal conv (per feature channel)
     + grouped 1x1 mixing within each ROI group.
    Input:  [B, C*F, T]
    Output: [B, C*F, ceil(T/stride)] (depending on padding; we use "same-ish" padding)
    """

    def __init__(
        self, c: int, f: int, stride: int = 2, k: int = 5, dropout: float = 0.0
    ):
        super().__init__()
        assert stride in (1, 2, 4)
        self.c = c
        self.f = f
        ch = c * f

        # Depthwise conv over time for each feature channel.
        pad = (k - 1) // 2
        self.dw = nn.Conv1d(
            ch, ch, kernel_size=k, stride=stride, padding=pad, groups=ch, bias=False
        )

        # GroupNorm with groups=C =>
        # each ROI has its own group containing F feature channels.
        self.gn1 = nn.GroupNorm(num_groups=c, num_channels=ch, eps=1e-5)

        # Mix features *within* each ROI (grouped 1x1 conv).
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, groups=c, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=c, num_channels=ch, eps=1e-5)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.gn1(x)
        x = F.gelu(x)
        x = self.pw(x)
        x = self.gn2(x)
        x = F.gelu(x)
        x = self.drop(x)
        return x


class TemporalUpsampleBlock(nn.Module):
    """
    Depthwise ConvTranspose1d to upsample time
     + grouped 1x1 mixing within each ROI group.
    Input:  [B, C*F, T]
    Output: [B, C*F, ~T*2]
    """

    def __init__(
        self, c: int, f: int, stride: int = 2, k: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        assert stride in (2,)
        self.c = c
        self.f = f
        ch = c * f

        # For exact doubling with stride=2,
        # a common choice is k=4, padding=1, output_padding=0.
        self.dwt = nn.ConvTranspose1d(
            ch,
            ch,
            kernel_size=k,
            stride=stride,
            padding=1,
            output_padding=0,
            groups=ch,
            bias=False,
        )
        self.gn1 = nn.GroupNorm(num_groups=c, num_channels=ch, eps=1e-5)

        self.pw = nn.Conv1d(ch, ch, kernel_size=1, groups=c, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=c, num_channels=ch, eps=1e-5)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwt(x)
        x = self.gn1(x)
        x = F.gelu(x)
        x = self.pw(x)
        x = self.gn2(x)
        x = F.gelu(x)
        x = self.drop(x)
        return x


class MEGFactorizedEncoder(nn.Module):
    """
    Encoder for MEG ROI timeseries (C=68 typical).
    - Temporal feature lifting per ROI (depthwise/grouped)
    - Temporal downsample by factor t_down (default 4)
    - Learned channel reduction: C -> C_red = C // c_down (default 4)
    - Per-token feature projection: F -> D

    Output latent: z ∈ [B, C_red, T_red, D]
    """

    def __init__(
        self,
        c_in: int = 68,
        c_down: int = 4,
        t_down: int = 4,
        f_hidden: int = 32,
        d_latent: int = 128,
        k_stem: int = 7,
        k_down: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert (
            c_in % c_down == 0
        ), f"Channels {c_in} must be divisible by c_down={c_down}"
        assert t_down in (1, 2, 4), "This implementation supports t_down in {1,2,4}."

        self.c_in = c_in
        self.c_down = c_down
        self.c_red = c_in // c_down
        self.t_down = t_down
        self.f_hidden = f_hidden
        self.d_latent = d_latent

        # Lift 1 feature per ROI -> F features per ROI (grouped by ROI).
        # Input [B, C, T] treated as Conv1d with channels=C.
        # groups=C =>
        # each ROI independently maps 1 -> F (implemented as out_channels=C*F).
        pad = (k_stem - 1) // 2
        self.stem = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in * f_hidden,
            kernel_size=k_stem,
            stride=1,
            padding=pad,
            groups=c_in,
            bias=False,
        )
        self.stem_gn = nn.GroupNorm(
            num_groups=c_in, num_channels=c_in * f_hidden, eps=1e-5
        )

        # Temporal downsampling blocks
        downs = []
        if t_down == 1:
            pass
        elif t_down == 2:
            downs.append(
                TemporalDownsampleBlock(
                    c=c_in, f=f_hidden, stride=2, k=k_down, dropout=dropout
                )
            )
        elif t_down == 4:
            downs.append(
                TemporalDownsampleBlock(
                    c=c_in, f=f_hidden, stride=2, k=k_down, dropout=dropout
                )
            )
            downs.append(
                TemporalDownsampleBlock(
                    c=c_in, f=f_hidden, stride=2, k=k_down, dropout=dropout
                )
            )
        self.down = nn.Sequential(*downs) if downs else nn.Identity()

        # Learned channel reduction matrix W: [C_red, C]
        # Applied per time and per feature channel.
        self.W_reduce = nn.Parameter(torch.empty(self.c_red, c_in))
        nn.init.kaiming_uniform_(self.W_reduce, a=math.sqrt(5))

        # After reduction:
        # [B, C_red, F, T_red] -> project F -> D per reduced channel (grouped 1x1).
        self.proj_fd = nn.Conv1d(
            in_channels=self.c_red * f_hidden,
            out_channels=self.c_red * d_latent,
            kernel_size=1,
            groups=self.c_red,
            bias=False,
        )
        self.proj_gn = nn.GroupNorm(
            num_groups=self.c_red, num_channels=self.c_red * d_latent, eps=1e-5
        )

    @torch.no_grad()
    def expected_latent_shape(self, T: int) -> Tuple[int, int, int]:
        # approximate; exact depends on padding/strides
        return (self.c_red, math.ceil(T / self.t_down), self.d_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bct, was_unbatched = _as_bct(x)  # [B,C,T]
        B, C, T = x_bct.shape
        assert C == self.c_in, f"Expected C={self.c_in}, got {C}"

        # Stem: [B, C, T] -> [B, C*F, T]
        h = self.stem(x_bct)
        h = self.stem_gn(h)
        h = F.gelu(h)

        # Temporal downsample: [B, C*F, T] -> [B, C*F, T_red]
        h = self.down(h)
        T_red = h.shape[-1]

        # Reshape to [B, C, F, T_red]
        h = h.view(B, C, self.f_hidden, T_red)

        # Channel reduction: [B, C, F, T_red] -> [B, C_red, F, T_red] without einsum
        h_perm = h.permute(0, 2, 3, 1).contiguous()  # [B, F, T_red, C]
        h_red = F.linear(h_perm, self.W_reduce)  # [B, F, T_red, C_red]
        h_red = h_red.permute(0, 3, 1, 2).contiguous()

        # Project F -> D per reduced channel:
        # [B, C_red, F, T_red] -> [B, C_red*F, T_red] -> [B, C_red*D, T_red]
        h_red_flat = h_red.reshape(B, self.c_red * self.f_hidden, T_red)
        z = self.proj_fd(h_red_flat)
        z = self.proj_gn(z)
        z = F.gelu(z)

        # Return [B, C_red, T_red, D]
        z = z.view(B, self.c_red, self.d_latent, T_red).permute(0, 1, 3, 2).contiguous()
        return z


class MEGFactorizedDecoder(nn.Module):
    """
    Decoder inverse of MEGFactorizedEncoder:
    - Per-token projection D -> F
    - Learned channel expansion C_red -> C (optionally tied to encoder W_reduce^T)
    - Temporal upsample by factor t_up (default 4)
    - Per-ROI projection F -> 1

    Input:  z ∈ [B, C_red, T_red, D]
    Output: x̂ ∈ [B, C, T, 1] (or unbatched if input was unbatched)
    """

    def __init__(
        self,
        c_out: int = 68,
        c_down: int = 4,
        t_up: int = 4,
        f_hidden: int = 32,
        d_latent: int = 128,
        k_up: int = 4,
        k_refine: int = 7,
        dropout: float = 0.0,
        tie_channel_mats: bool = False,
    ):
        super().__init__()
        assert (
            c_out % c_down == 0
        ), f"Channels {c_out} must be divisible by c_down={c_down}"
        assert t_up in (1, 2, 4), "This implementation supports t_up in {1,2,4}."

        self.c_out = c_out
        self.c_down = c_down
        self.c_red = c_out // c_down
        self.t_up = t_up
        self.f_hidden = f_hidden
        self.d_latent = d_latent
        self.tie_channel_mats = tie_channel_mats

        # D -> F per reduced channel
        self.proj_df = nn.Conv1d(
            in_channels=self.c_red * d_latent,
            out_channels=self.c_red * f_hidden,
            kernel_size=1,
            groups=self.c_red,
            bias=False,
        )
        self.proj_gn = nn.GroupNorm(
            num_groups=self.c_red, num_channels=self.c_red * f_hidden, eps=1e-5
        )

        # Channel expansion matrix W_expand: [C, C_red]
        # If tie_channel_mats=True,
        # you can pass encoder.W_reduce at forward and use its transpose.
        if not tie_channel_mats:
            self.W_expand = nn.Parameter(torch.empty(c_out, self.c_red))
            nn.init.kaiming_uniform_(self.W_expand, a=math.sqrt(5))
        else:
            self.register_parameter("W_expand", None)

        # Temporal upsampling blocks
        ups = []
        if t_up == 1:
            pass
        elif t_up == 2:
            ups.append(
                TemporalUpsampleBlock(
                    c=c_out, f=f_hidden, stride=2, k=k_up, dropout=dropout
                )
            )
        elif t_up == 4:
            ups.append(
                TemporalUpsampleBlock(
                    c=c_out, f=f_hidden, stride=2, k=k_up, dropout=dropout
                )
            )
            ups.append(
                TemporalUpsampleBlock(
                    c=c_out, f=f_hidden, stride=2, k=k_up, dropout=dropout
                )
            )
        self.up = nn.Sequential(*ups) if ups else nn.Identity()

        # Refine + final projection F -> 1 per ROI
        # First keep [B, C*F, T] then grouped convs.
        pad = (k_refine - 1) // 2
        self.refine = nn.Sequential(
            nn.Conv1d(
                c_out * f_hidden,
                c_out * f_hidden,
                kernel_size=k_refine,
                padding=pad,
                groups=c_out * f_hidden,
                bias=False,
            ),
            nn.GroupNorm(num_groups=c_out, num_channels=c_out * f_hidden, eps=1e-5),
            nn.GELU(),
            nn.Conv1d(
                c_out * f_hidden,
                c_out * f_hidden,
                kernel_size=1,
                groups=c_out,
                bias=False,
            ),
            nn.GroupNorm(num_groups=c_out, num_channels=c_out * f_hidden, eps=1e-5),
            nn.GELU(),
        )
        self.to_x = nn.Conv1d(
            in_channels=c_out * f_hidden,
            out_channels=c_out,
            kernel_size=1,
            groups=c_out,
            bias=True,
        )

    def forward(
        self,
        z: torch.Tensor,
        T_target: Optional[int] = None,
        W_reduce_from_encoder: Optional[torch.Tensor] = None,
        return_last_dim_1: bool = True,
    ) -> torch.Tensor:
        """
        z: [B, C_red, T_red, D] or unbatched [C_red, T_red, D]
        T_target: if provided, crop/pad output time to match exactly.
        W_reduce_from_encoder: pass encoder.W_reduce if tie_channel_mats=True
        """
        was_unbatched = z.dim() == 3
        if was_unbatched:
            z = z.unsqueeze(0)
        if z.dim() != 4:
            raise ValueError(
                f"Expected z as [B,C_red,T_red,D] or [C_red,T_red,D], "
                f"got {tuple(z.shape)}"
            )

        B, C_red, T_red, D = z.shape
        assert C_red == self.c_red, f"Expected C_red={self.c_red}, got {C_red}"
        assert D == self.d_latent, f"Expected D={self.d_latent}, got {D}"

        # [B, C_red, T_red, D] -> [B, C_red*D, T_red]
        h = z.permute(0, 1, 3, 2).contiguous().view(B, C_red * D, T_red)

        # D -> F per reduced channel => [B, C_red*F, T_red]
        h = self.proj_df(h)
        h = self.proj_gn(h)
        h = F.gelu(h)

        # reshape to [B, C_red, F, T_red]
        h = h.view(B, C_red, self.f_hidden, T_red)

        # channel expansion: [B, C_red, F, T_red] -> [B, C, F, T_red]
        if self.tie_channel_mats:
            if W_reduce_from_encoder is None:
                raise ValueError(
                    "tie_channel_mats=True but W_reduce_from_encoder was not provided."
                )
            W_expand = W_reduce_from_encoder.t()  # [C, C_red]
        else:
            W_expand = self.W_expand  # [C, C_red]

        h_perm = h.permute(0, 2, 3, 1).contiguous()  # [B, F, T_red, C_red]
        h = F.linear(h_perm, W_expand)  # [B, F, T_red, C]
        h = h.permute(0, 3, 1, 2).contiguous()

        # flatten to [B, C*F, T_red]
        h = h.reshape(B, self.c_out * self.f_hidden, T_red)

        # temporal upsample to ~T
        h = self.up(h)

        # refine + project to [B, C, T]
        h = self.refine(h)
        x_hat = self.to_x(h)  # [B, C, T_hat]

        # match target length if needed
        if T_target is not None:
            T_hat = x_hat.shape[-1]
            if T_hat > T_target:
                x_hat = x_hat[..., :T_target]
            elif T_hat < T_target:
                x_hat = F.pad(x_hat, (0, T_target - T_hat))

        # return with last dim 1 if desired
        x_hat = _restore_shape(
            x_hat, was_unbatched=was_unbatched, want_last_dim_1=return_last_dim_1
        )
        return x_hat


@dataclass
class MEGFactorizedOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    recon_loss: Optional[torch.Tensor] = None

    latents: Optional[torch.Tensor] = None
    x_hat: Optional[torch.Tensor] = None


class MEGFactorizedConfig(PretrainedConfig):
    model_type = "meg-factorized"

    def __init__(
        self,
        c_in: int = 68,
        c_out: Optional[int] = None,
        c_down: int = 4,
        t_down: int = 4,
        t_up: Optional[int] = None,
        f_hidden: int = 32,
        d_latent: int = 128,
        k_stem: int = 7,
        k_down: int = 5,
        k_up: int = 4,
        k_refine: int = 7,
        dropout: float = 0.0,
        tie_channel_mats: bool = False,
        recon_loss: str = "mse",  # "mse" | "l1" | "huber"
        huber_delta: float = 1.0,
        return_last_dim_1: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.c_in = c_in
        self.c_out = c_out if c_out is not None else c_in
        self.c_down = c_down
        self.t_down = t_down
        self.t_up = t_down if t_up is None else t_up
        self.f_hidden = f_hidden
        self.d_latent = d_latent
        self.k_stem = k_stem
        self.k_down = k_down
        self.k_up = k_up
        self.k_refine = k_refine
        self.dropout = dropout
        self.tie_channel_mats = tie_channel_mats
        self.recon_loss = recon_loss
        self.huber_delta = huber_delta
        self.return_last_dim_1 = return_last_dim_1


class MEGFactorizedAutoencoder(PreTrainedModel):
    config_class = MEGFactorizedConfig
    base_model_prefix = "meg_factorized"

    def __init__(self, config: MEGFactorizedConfig | dict):
        if isinstance(config, dict):
            config = MEGFactorizedConfig(**config)
        super().__init__(config)

        self.encoder = MEGFactorizedEncoder(
            c_in=config.c_in,
            c_down=config.c_down,
            t_down=config.t_down,
            f_hidden=config.f_hidden,
            d_latent=config.d_latent,
            k_stem=config.k_stem,
            k_down=config.k_down,
            dropout=config.dropout,
        )
        self.decoder = MEGFactorizedDecoder(
            c_out=config.c_out,
            c_down=config.c_down,
            t_up=config.t_up,
            f_hidden=config.f_hidden,
            d_latent=config.d_latent,
            k_up=config.k_up,
            k_refine=config.k_refine,
            dropout=config.dropout,
            tie_channel_mats=config.tie_channel_mats,
        )

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _recon_loss(self, x_hat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mode = self.config.recon_loss.lower()
        if mode == "mse":
            return F.mse_loss(x_hat, target)
        if mode == "l1":
            return F.l1_loss(x_hat, target)
        if mode == "huber":
            return F.huber_loss(x_hat, target, delta=float(self.config.huber_delta))
        raise ValueError(f"Unknown recon_loss: {self.config.recon_loss}")

    @staticmethod
    def _infer_target_time_and_shape(
        target: torch.Tensor,
        explicit_t: Optional[int],
        channels_expected: Tuple[int, ...],
    ) -> Tuple[Optional[int], bool]:
        is_unbatched_trailing = (
            target.dim() == 3
            and target.shape[-1] == 1
            and target.shape[0] in channels_expected
            and target.shape[1] not in channels_expected
        )
        has_trailing_dim = target.dim() == 4 or is_unbatched_trailing
        if explicit_t is not None:
            return explicit_t, has_trailing_dim

        if target.dim() >= 3:
            t_dim = target.shape[-2] if has_trailing_dim else target.shape[-1]
        elif target.dim() == 2:
            t_dim = target.shape[-1]
        else:
            t_dim = None
        return t_dim, has_trailing_dim

    @torch.no_grad()
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    @torch.no_grad()
    def decode(
        self,
        latents: torch.Tensor,
        T_target: Optional[int] = None,
        return_last_dim_1: Optional[bool] = None,
    ) -> torch.Tensor:
        return_last_dim_1 = (
            self.config.return_last_dim_1
            if return_last_dim_1 is None
            else return_last_dim_1
        )
        decoder_kwargs = dict(
            T_target=T_target,
            return_last_dim_1=return_last_dim_1,
        )
        if self.config.tie_channel_mats:
            decoder_kwargs["W_reduce_from_encoder"] = self.encoder.W_reduce
        return self.decoder(latents, **decoder_kwargs)

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        T_target: Optional[int] = None,
        return_dict: Optional[bool] = True,
    ) -> MEGFactorizedOutput:
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]

        return_dict = True if return_dict is None else return_dict
        target = inputs if labels is None else labels
        input_was_unbatched = inputs.dim() == 2 or (
            inputs.dim() == 3
            and inputs.shape[-1] == 1
            and inputs.shape[0] in (self.config.c_in, self.config.c_out)
        )

        latents = self.encoder(inputs)
        tgt_time, want_last_dim_1 = self._infer_target_time_and_shape(
            target, T_target, (self.config.c_out, self.config.c_in)
        )
        decoder_kwargs = dict(
            T_target=tgt_time,
            return_last_dim_1=want_last_dim_1,
        )
        if self.config.tie_channel_mats:
            decoder_kwargs["W_reduce_from_encoder"] = self.encoder.W_reduce
        latents_for_decode = latents[0] if input_was_unbatched else latents
        x_hat = self.decoder(latents_for_decode, **decoder_kwargs)

        recon_loss = self._recon_loss(x_hat, target)
        loss = recon_loss

        if not return_dict:
            return loss, recon_loss, x_hat, latents

        return MEGFactorizedOutput(
            loss=loss,
            recon_loss=recon_loss,
            latents=latents,
            x_hat=x_hat,
        )
