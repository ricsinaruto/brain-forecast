import torch
import torch.nn as nn
from typing import Literal, Optional

from ..layers.transformer_blocks import TransformerBlock
from ..layers.conv import ConvBlock1D
from ..layers.norms import ChannelLastLayerNorm


class PerExampleChannelNorm1D(nn.Module):
    """Normalize each example and channel across time (B, C, T).

    For every sample b and channel c, computes     x[b,c,:] = (x[b,c,:] - mean_t) /
    sqrt(var_t + eps) where statistics are taken over the time dimension only.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x of shape (B, C, T)
        if x.dim() != 3:
            raise ValueError(
                f"PerExampleChannelNorm1D expects (B,C,T), got {tuple(x.shape)}"
            )
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class AttentionPool1D(nn.Module):
    """Lightweight additive attention pooling over time without einsum.

    Input:  x (B, T, D) Output: (B, D) Optional mask: (B, T) with 1 for valid, 0 for
    padded (if you ever need it).
    """

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh())
        self.psi = nn.Sequential(nn.Linear(d_model, hidden), nn.Sigmoid())
        self.out = nn.Linear(hidden, 1)

        self.tau = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        scores = self.out(self.phi(x) * self.psi(x)).squeeze(-1)  # (B, T)

        attn = torch.softmax(scores / self.tau.clamp_min(1e-3), dim=1)  # (B, T)

        # Use bmm to compute weighted sum over time:
        # (B, 1, T) @ (B, T, D) -> (B, 1, D) -> (B, D)
        x = x.contiguous()
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled


class CNNLSTM(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: tuple,
        kernels: tuple,
        strides: tuple,
        num_classes: int,
        head_inputs: int,
        lstm_num_layers: int = 3,
        conv_dropout: float = 0.0,
        channel_dropout_p: float = 0.0,
        lstm_dropout: float = 0.0,
        head_dropout: float = 0.0,
        norm: Literal["group", "batch", "layer"] = "group",
        gn_groups: int = 16,
        attn_hidden: int = 128,
        attn_type: str = "standard",
        mlp_type: str = "standard",
        attn_args: dict = None,
        mlp_args: dict = None,
        num_trf_blocks: int = 1,
        pooling: Literal["attn", "last", "flatten"] = "last",
        activation: type[nn.Module] = nn.GELU,
        causal: bool = False,
        k_conditioning: bool = False,
        ch_attn: Optional[Literal["se", "eca"]] = "se",
    ) -> None:
        super().__init__()
        assert (
            len(hidden_channels) == len(kernels) == len(strides)
        ), "conv_channels, conv_kernels, conv_strides must be same length"

        self.causal = causal
        self.lstm_num_layers = lstm_num_layers
        self.num_trf_blocks = num_trf_blocks
        self.hidden_size = hidden_channels[-1]

        # Map scalar sigma_k -> (gamma, beta) per input channel
        self.k_film = nn.Linear(1, 2 * in_channels)
        self.k_conditioning = k_conditioning

        # Channel-wise dropout after normalization/conditioning
        self.input_channel_dropout = nn.Dropout1d(p=channel_dropout_p)

        # Conv stack
        blocks = []
        ch_in = in_channels
        for ch_out, k, s in zip(hidden_channels, kernels, strides):
            blocks.append(
                ConvBlock1D(
                    ch_in,
                    ch_out,
                    kernel_size=k,
                    stride=s,
                    norm=norm,
                    gn_groups=gn_groups,
                    dropout=conv_dropout,
                    activation=activation,
                    eps=1e-5,
                    ch_attn=ch_attn,
                )
            )
            ch_in = ch_out
        self.conv_stack = nn.Sequential(*blocks)

        # Stacked LSTM layers
        if lstm_num_layers > 0:
            self.lstm = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=lstm_dropout,
                bidirectional=True,
            )
            # Layer normalization
            self.post_lstm_layer_norm = nn.LayerNorm(
                normalized_shape=self.hidden_size * 2
            )
            lstm_out_dim = self.hidden_size * 2
        else:
            lstm_out_dim = self.hidden_size

        # Self-attention layers after LSTM
        if num_trf_blocks > 0:
            self.trf_projector = nn.Linear(lstm_out_dim, attn_args["d_model"])
            self.trf_blocks = nn.ModuleList(
                [
                    TransformerBlock(
                        attn_args=attn_args,
                        mlp_args=mlp_args,
                        attn_type=attn_type,
                        mlp_type=mlp_type,
                    )
                    for _ in range(num_trf_blocks)
                ]
            )

        # Pooling
        self.pooling = pooling
        if pooling == "attn":
            self.attn_pool = AttentionPool1D(
                d_model=attn_args["d_model"], hidden=attn_hidden, dropout=head_dropout
            )
        else:
            self.attn_pool = None  # use last timestep

        # Head
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(head_inputs, num_classes * 2),
            nn.GELU(),
            nn.Linear(num_classes * 2, num_classes),
        )

        # Initialize weights
        # self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Parameters ---------- inputs : torch.Tensor     Input tensor of shape
        (batch_size, input_channels, sequence_length)

        Returns ------- output : torch.Tensor     - Output tensor of shape (batch_size,
        num_gestures, sequence_length)
        """
        sigma_k = None
        if isinstance(x, tuple) or isinstance(x, list):
            x, sigma_k = x

        if sigma_k is not None and self.k_conditioning:
            # add as an extra channel to the input
            # repeat along time dimension
            sigma_k = sigma_k.unsqueeze(-1).repeat(1, 1, x.shape[2])
            x = torch.cat([x, sigma_k], dim=1)

        # Channel-wise dropout
        x = self.input_channel_dropout(x)

        # Convolutional frontend
        for block in self.conv_stack:
            x = block(x)

        # Layer normalization
        # (batch_size, conv_output_channels, sequence_length)
        # -> (batch_size, sequence_length, conv_output_channels)
        x = x.transpose(1, 2).contiguous()

        if self.lstm_num_layers > 0:
            # Stacked LSTM layers
            x, _ = self.lstm(x)

            # Layer normalization
            x = self.post_lstm_layer_norm(x)

        if self.num_trf_blocks > 0:
            x = self.trf_projector(x)

            # Self-attention stack
            for blk in self.trf_blocks:
                x = blk(x, causal=self.causal)

        # Pooling
        if self.pooling == "attn":
            x = self.attn_pool(x)  # (B, D)
        elif self.pooling == "flatten":
            x = x.flatten(start_dim=1)
        elif self.pooling == "last":
            x = x[:, -1, :]  # last timestep (B, D)
        elif self.pooling == "ends":
            x = x[:, [0, -1], :].flatten(start_dim=1)
        elif self.pooling == "logsumexp":
            x = torch.logsumexp(x, dim=1)

        # Feedforward projection layer
        x = self.head(x)

        return x

    @staticmethod
    def _init_weights(m: nn.Module):
        # Convs: Kaiming (He) for GELU/ReLU-like
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Linear: Xavier
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # Norms
        elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, ChannelLastLayerNorm)):
            for p in m.parameters():
                # Try to set weight to 1, bias to 0 if present
                if p.dim() == 1:  # gamma or beta
                    # Heuristic: first  set all-ones, then all-zeros where appropriate
                    p.data.fill_(1.0)
            # Bias parameters (if any) set to zero
            named = dict(m.named_parameters(recurse=False))
            if "bias" in named:
                named["bias"].data.zero_()
        # LSTM: orthogonal + forget gate bias = 1
        if isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias_ih" in name or "bias_hh" in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1
                    hidden_size = param.shape[0] // 4
                    param.data[hidden_size: 2 * hidden_size].fill_(1.0)


class CNNLSTMSimple(nn.Module):
    def __init__(
        self,
        in_channels: int,
        red_channels: int,
        hidden_channels: tuple,
        kernels: tuple,
        strides: tuple,
        num_classes: int,
        lstm_num_layers: int = 0,
        dropout: float = 0.0,
        channel_dropout_p: float = 0.0,
        attention: bool = False,
        ch_attn: Optional[Literal["se", "eca"]] = None,
    ):
        super().__init__()
        self.lstm_num_layers = lstm_num_layers
        self.attention = attention

        # drop out whole channels
        self.channel_dropout = nn.Dropout1d(p=channel_dropout_p)

        # reduce channel dimension and map to common space
        self.init_layer = nn.Conv1d(
            in_channels,
            red_channels,
            kernel_size=1,
        )

        blocks = []
        ch_in = red_channels
        for ch_out, k, s in zip(hidden_channels, kernels, strides):
            blocks.append(
                ConvBlock1D(
                    ch_in,
                    ch_out,
                    kernel_size=k,
                    stride=s,
                    dropout=dropout,
                    ch_attn=ch_attn,
                    padding=True,
                )
            )
            ch_in = ch_out
        self.conv_stack = nn.Sequential(*blocks)

        if lstm_num_layers > 0:
            self.lstm = nn.LSTM(
                input_size=hidden_channels[-1],
                hidden_size=hidden_channels[-1],
                num_layers=lstm_num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
            # Layer normalization
            self.post_lstm_layer_norm = nn.LayerNorm(
                normalized_shape=hidden_channels[-1] * 2
            )
            lstm_out_dim = hidden_channels[-1] * 2
        else:
            lstm_out_dim = hidden_channels[-1]

        if attention:
            self.attn = nn.MultiheadAttention(
                embed_dim=lstm_out_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True,
            )

        self.head = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, tuple) or isinstance(x, list):
            x, sigma_k = x

        # Channel-wise dropout
        x = self.channel_dropout(x)
        x = self.init_layer(x)

        # Convolutional frontend
        for block in self.conv_stack:
            x = block(x)

        x = x.transpose(1, 2).contiguous()  # (B, C, T) -> (B, T, C)

        if self.lstm_num_layers > 0:
            x, _ = self.lstm(x)
            x = self.post_lstm_layer_norm(x)

        if self.attention:
            x, _ = self.attn(x, x, x)

        x = self.head(x[:, -1, :])
        return x


class Wavenet3DLayer(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        spatial_kernel: int,
        stride: int,
        spatial_stride: int,
        dilation_channels: int,
        residual_channels: int,
        skip_channels: int,
        dropout: float = 0.0,
        in_channels: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        in_channels = in_channels or residual_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels

        # padding to preserve spatial sizes
        pad = (spatial_kernel - 1) // 2

        self.conv_dilation = nn.Conv3d(
            in_channels,
            2 * dilation_channels,
            kernel_size=(spatial_kernel, spatial_kernel, kernel_size),
            padding=(pad, pad, 0),
            stride=(spatial_stride, spatial_stride, stride),
            bias=bias,
        )

        self.conv_res = nn.Conv3d(
            dilation_channels, residual_channels, kernel_size=1, bias=bias
        )
        self.conv_skip = nn.Conv3d(
            dilation_channels, skip_channels, kernel_size=1, bias=bias
        )

        self.conv_input = None
        if in_channels != residual_channels:
            self.conv_input = nn.Conv3d(
                in_channels,
                residual_channels,
                kernel_size=1,
                bias=bias,
                stride=(spatial_stride, spatial_stride, stride),
            )

        self.dropout = torch.nn.Dropout3d(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:

        x: (B, C, H, W, T) c: optional conditioning, (B, Cc, H, W, T) or broadcastable
        causal_pad: whether to apply causal left padding in time Returns:
        residual_out: (B, C_res, H, W, T)     skip: (B, C_skip, H, W, T)
        """
        x_dilated = self.conv_dilation(x)

        x_filter = torch.tanh(x_dilated[:, : self.dilation_channels])
        x_gate = torch.sigmoid(x_dilated[:, self.dilation_channels:])
        x_h = x_gate * x_filter

        skip = self.conv_skip(x_h)
        res = self.conv_res(x_h)

        if self.conv_input is not None:
            x = self.conv_input(x)

        out = x[..., -res.shape[-1]:] + res

        out = self.dropout(out)
        return out, skip


class Wavenet3DLogitsHead(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        head_channels: int,
        num_classes: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.transform = torch.nn.Sequential(
            nn.Dropout3d(p=dropout),
            nn.LeakyReLU(),
            nn.Conv3d(skip_channels, head_channels, kernel_size=1, bias=bias),
            nn.Dropout3d(p=dropout),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(head_channels, num_classes),
        )

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        return self.transform(sum(skips))


class Wavenet3DClassifier(nn.Module):
    """WaveNet-style model operating on 3D volumes over time.

    Expects inputs shaped as (B, C, H, W, T), where C is the channel/embedding
    dimension. The network performs causal convolutions along the temporal axis using 3D
    convolutions with kernel size (1, 1, k), so spatial dimensions are preserved while
    receptive field grows only in time.

    The model mirrors the gating, residual, and skip-connection structure of a classical
    WaveNet, but generalized to 3D.
    """

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        kernel_sizes: list[int],
        spatial_kernel_sizes: list[int],
        strides: list[int],
        spatial_strides: list[int],
        residual_channels: list[int],
        dilation_channels: list[int],
        num_classes: int,
        skip_channels: int,
        p_drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels

        # construct layers: initial 1x1x1 to get into residual space,
        # then dilated blocks
        layers: list[nn.Module] = [
            Wavenet3DLayer(
                kernel_size=1,
                in_channels=in_channels,
                residual_channels=residual_channels,
                dilation_channels=dilation_channels,
                skip_channels=skip_channels,
                bias=bias,
                dropout=p_drop,
                spatial_kernel=(1, 1),
                spatial_dilation=(1, 1),
            )
        ]

        for ks, sk, rs, ds in zip(
            kernel_sizes, spatial_kernel_sizes, residual_channels, dilation_channels
        ):
            layers.append(
                Wavenet3DLayer(
                    kernel_size=ks,
                    residual_channels=rs,
                    dilation_channels=ds,
                    skip_channels=skip_channels,
                )
            )

        self.layers = nn.ModuleList(layers)

        self.head = Wavenet3DLogitsHead(
            skip_channels=skip_channels,
            head_channels=head_channels,
            num_classes=num_classes,
            dropout=p_drop,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: (B, H, W, T) condition: optional conditioning tensor broadcastable to (B, Cc,
        H, W, T) causal_pad: whether to apply causal left padding on temporal axis
        Returns:     logits: (B, out_channels, H, W, T)
        """

        skips: list[torch.Tensor] = []
        for layer in self.layers:
            x, skip = layer(x)
            skips.append(skip)

        out = self.head(skips)  # (B, num_classes)

        return out
