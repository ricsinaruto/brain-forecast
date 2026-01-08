import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
import numpy as np

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

from .qwen3_vl import Qwen3VLTextModel
from .qwen2p5_vl import Qwen2_5_VLTextModel

from .masking import _block_causal_mask
from ...layers.st_blocks import DownStage3D, UpStage3D


def compute_inds(
    pos_2d: np.ndarray, image_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    pos = pos_2d.astype(np.float32)
    # Normalise to the unit square [0,1]²

    pos_min = pos.min(axis=0)
    pos_max = pos.max(axis=0)
    span = pos_max - pos_min
    span[span == 0] = 1.0  # avoid divide-by-zero
    pos = (pos - pos_min) / span

    col_idx = np.round(pos[:, 0] * (image_size - 1)).astype(np.int32)
    row_idx = np.round(pos[:, 1] * (image_size - 1)).astype(np.int32)

    return torch.from_numpy(row_idx), torch.from_numpy(col_idx)


class Qwen2_5_Video(nn.Module):
    CONFIG_CLS = Qwen2_5_VLTextConfig
    MODEL_CLS = Qwen2_5_VLTextModel

    def __init__(
        self,
        *args,
        CONFIG_CLS: type | None = None,
        MODEL_CLS: type | None = None,
        **kwargs,
    ):
        super().__init__()
        self.reduced_shape = kwargs.pop("reduced_shape")

        self.layout_path = kwargs.pop("layout_path", None)
        if self.layout_path is not None:
            self.layout = np.load(self.layout_path, allow_pickle=True)
            self.row_idx, self.col_idx = compute_inds(self.layout)

        spatial_size = (
            self.row_idx.shape[0]
            if self.layout_path is not None
            else self.reduced_shape[1] * self.reduced_shape[2]
        )
        self._total_positions = int(self.reduced_shape[0] * spatial_size)
        self._base_position_ids = {}
        self._position_ids_cache = {}
        self.block_size = kwargs.pop("block_size", 1)

        config_cls = CONFIG_CLS or self.CONFIG_CLS
        model_cls = MODEL_CLS or self.MODEL_CLS

        self.config = config_cls(**kwargs)
        self.model = model_cls(self.config)

    def forward(
        self,
        x: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: tuple | None = None,
        use_cache: bool | None = None,
        channel_ids: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run Qwen on token ids or precomputed embeddings."""
        if (x is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of `x` or `inputs_embeds`.")
        use_cache = bool(use_cache) if use_cache is not None else False

        main_input = x if x is not None else inputs_embeds

        past_len = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        position_ids = self._build_position_ids(
            batch_size=main_input.shape[0],
            device=main_input.device,
            seq_len=main_input.shape[1],
            position_offset=past_len,
            channel_ids=channel_ids,
        )

        if self.block_size > 1:
            attention_mask = _block_causal_mask(
                self.config,
                self.block_size,
                past_key_values,
                main_input,
                include_position_ids=False,
            )

        outputs = self.model(
            input_ids=x,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
            return_dict=True,
        )
        cache = outputs.past_key_values if use_cache else None
        if use_cache or past_key_values is not None:
            return outputs.last_hidden_state, cache
        return outputs.last_hidden_state

    def _get_base_position_ids(self, device: torch.device, *args) -> torch.Tensor:
        """Build or fetch cached [3, L] base position ids on the target device."""
        device = torch.device(device)
        cached = self._base_position_ids.get(device)
        if cached is not None:
            return cached

        t, h, w = self.reduced_shape
        t_ids = torch.arange(t, device=device).view(t, 1, 1).expand(t, h, w).reshape(-1)

        if self.layout_path is not None:
            nc = self.row_idx.shape[0]
            h_ids = self.row_idx.to(device).view(1, nc, 1).expand(t, nc, 1).reshape(-1)
            w_ids = self.col_idx.to(device).view(1, 1, nc).expand(t, 1, nc).reshape(-1)
        else:
            d = device
            h_ids = torch.arange(h, device=d).view(1, h, 1).expand(t, h, w).reshape(-1)
            w_ids = torch.arange(w, device=d).view(1, 1, w).expand(t, h, w).reshape(-1)

        pos = torch.stack([t_ids, h_ids, w_ids], dim=0)
        self._base_position_ids[device] = pos
        return pos

    def _build_position_ids(
        self,
        batch_size: int,
        device: torch.device,
        seq_len: int,
        position_offset: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Create 3D RoPE position ids for flattened (T, H, W) grids."""
        device = torch.device(device)
        start = int(position_offset)
        if start < 0:
            raise ValueError("position_offset must be non-negative.")

        cached = self._position_ids_cache.get(device)
        if cached is None:
            base_pos = self._get_base_position_ids(device)
            cached = base_pos.unsqueeze(1)  # [3,1,total_positions]
            self._position_ids_cache[device] = cached

        end = start + seq_len
        if end > self._total_positions:
            raise ValueError(
                f"Requested positions {start}:{end} exceed flattened grid "
                f"{self._total_positions} for reduced shape {self.reduced_shape}."
            )
        position_ids = cached[:, :, start:end].expand(-1, batch_size, -1)
        return position_ids.contiguous()

    def get_embed_layer(self) -> nn.Module:
        return self.model.embed_tokens


class Qwen3_Video(Qwen2_5_Video):
    """Qwen3-VL backbone for video-only (T, H, W) inputs.

    Inherits position ID generation from Qwen2_5_Video, which returns shape (3, batch,
    seq) for T, H, W dimensions. Qwen3VLTextModel accepts this directly: it uses
    position_ids[0] (T) for causal masking and the full (3, batch, seq) for the 3D
    rotary embedding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            CONFIG_CLS=Qwen3VLTextConfig, MODEL_CLS=Qwen3VLTextModel, **kwargs
        )


class Qwen2_5_VideoText(Qwen2_5_Video):
    def _get_base_position_ids(
        self, device: torch.device, channel_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build or fetch cached [3, L] base position ids on the target device."""
        device = torch.device(device)
        total_len = channel_ids.shape[-1]

        t_ids = torch.arange(total_len, device=device)
        h_ids = channel_ids[0]
        w_ids = torch.arange(1, device=device).expand(total_len)

        pos = torch.stack([t_ids, h_ids, w_ids], dim=0)
        return pos

    def _build_position_ids(
        self,
        batch_size: int,
        device: torch.device,
        seq_len: int,
        position_offset: int = 0,
        channel_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create 3D RoPE position ids for flattened (T, H, W) grids."""
        device = torch.device(device)
        start = int(position_offset)
        if start < 0:
            raise ValueError("position_offset must be non-negative.")

        pos = self._get_base_position_ids(device, channel_ids)
        pos = pos.unsqueeze(1)  # [3,1,total_positions]

        end = start + seq_len
        return pos[:, :, start:end].expand(-1, batch_size, -1)

    def get_embed_layer(self) -> nn.Module:
        return self.model.embed_tokens


class Qwen2_5_Video_TASA3D(nn.Module):
    """Wrap a Qwen2.5 backbone with a single TASA3D-style downsample/upsample block.

    Input tokens are reshaped into the original (T, H, W), passed through spatial
    downsamples, flattened to the reduced shape for Qwen2_5_Video, then unflattened and
    upsampled back to the original resolution.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        reduced_shape: Tuple[int, int, int],
        input_shape: Tuple[int, int, int] | None = None,
        num_down: int = 3,
        channel_grow: int = 2,
        drop: float = 0.0,
        use_spatial_emb: bool = False,
        **qwen_args,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.input_shape = (
            tuple(input_shape) if input_shape is not None else tuple(reduced_shape)
        )
        self.reduced_shape = tuple(reduced_shape)
        self.num_down = num_down

        t_in, h_in, w_in = self.input_shape
        t_red, h_red, w_red = self.reduced_shape
        if t_in != t_red:
            raise ValueError(
                f"Temporal length must be unchanged by the downsampler "
                f"(input T={t_in}, reduced T={t_red})."
            )

        down_factor = 2**num_down
        if (h_in % down_factor) != 0 or (w_in % down_factor) != 0:
            raise ValueError(
                f"Input H,W must be divisible by 2**num_down "
                f"(got {h_in}x{w_in} with num_down={num_down})"
            )
        expected_h = h_in // down_factor
        expected_w = w_in // down_factor
        if (expected_h, expected_w) != (h_red, w_red):
            raise ValueError(
                "reduced_shape must equal input_shape downsampled by 2**num_down "
                f"in space (got reduced {self.reduced_shape} from input "
                f"{self.input_shape} with num_down={num_down})."
            )

        self.total_tokens = t_in * h_in * w_in
        self.hb, self.wb = h_red, w_red

        # channel schedule for the conv encoder/decoder
        channels = [hidden_size]
        for _ in range(num_down):
            channels.append(channels[-1] * channel_grow)
        self.channels = channels

        self.downs = nn.ModuleList(
            [DownStage3D(channels[i], channels[i + 1]) for i in range(num_down)]
        )
        self.ups = nn.ModuleList(
            [UpStage3D(channels[i], channels[i - 1]) for i in range(num_down, 0, -1)]
        )

        self.bottleneck_in = (
            nn.Linear(channels[-1], hidden_size)
            if channels[-1] != hidden_size
            else nn.Identity()
        )
        self.bottleneck_out = (
            nn.Linear(hidden_size, channels[-1])
            if channels[-1] != hidden_size
            else nn.Identity()
        )

        self.norm_full = nn.GroupNorm(1, channels[0])
        self.mlp_full = nn.Sequential(
            nn.Conv3d(channels[0], channels[0] * 4, 1),
            nn.GELU(),
            nn.Conv3d(channels[0] * 4, channels[0], 1),
            nn.Dropout(drop),
        )
        self.out_proj = nn.Conv3d(channels[0], hidden_size, kernel_size=1, bias=False)

        qwen_kwargs = qwen_args.copy()
        qwen_kwargs["hidden_size"] = hidden_size
        qwen_kwargs["vocab_size"] = vocab_size
        qwen_kwargs["reduced_shape"] = (
            self.reduced_shape[0],
            self.hb,
            self.wb,
        )
        self.qwen = Qwen2_5_Video(**qwen_kwargs)

    def get_embed_layer(self) -> nn.Module:
        return self.qwen.get_embed_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x: Token ids of shape (B, L), where L <= T*H*W (flattened grid).

        Returns:     Hidden states of shape (B, L, hidden_size).
        """
        batch_size, seq_len = x.shape
        if seq_len > self.total_tokens:
            raise ValueError(
                f"Seq len {seq_len} exceeds flattened grid {self.total_tokens} "
                f"for input_shape={self.input_shape}."
            )

        pad_len = self.total_tokens - seq_len
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=0)

        embeds = self.get_embed_layer()(x)  # (B, total_tokens, hidden_size)
        grid = embeds.view(
            batch_size,
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.hidden_size,
        )
        grid = grid.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, H, W)

        h = grid
        for down in self.downs:
            h = down(h)

        b, c, t, h_b, w_b = h.shape
        if (h_b, w_b) != (self.hb, self.wb):
            raise RuntimeError(
                f"Downsample produced spatial {(h_b, w_b)} but expected "
                f"{(self.hb, self.wb)} from reduced_shape."
            )
        flat = h.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, -1, c)
        flat = self.bottleneck_in(flat)

        # Qwen builds its own causal mask; no extra padding mask needed here.
        qwen_out = self.qwen(inputs_embeds=flat, attention_mask=None)
        qwen_out = self.bottleneck_out(qwen_out)

        h = qwen_out.view(batch_size, t, h_b, w_b, c).permute(0, 4, 1, 2, 3)
        for up in self.ups:
            h = up(h)

        h = h + self.mlp_full(self.norm_full(h))
        y = self.out_proj(h) + grid  # (B, C, T, H, W)

        y = (
            y.permute(0, 2, 3, 4, 1)
            .contiguous()
            .view(batch_size, self.total_tokens, self.hidden_size)
        )
        return y[:, :seq_len, :]
