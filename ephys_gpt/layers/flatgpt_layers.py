import torch
from typing import Optional

from torch import nn
from typing import Sequence
from einops import rearrange

from .embeddings import Embeddings


class QuantizerEmbedding(nn.Module):
    def __init__(self, vocab_sizes: Sequence[int], embed_dim: int):
        super().__init__()
        self.vocab_sizes = list(vocab_sizes)
        self.embed_dim = int(embed_dim)
        self.num_levels = len(self.vocab_sizes)
        self.embeddings = nn.ModuleList(
            nn.Embedding(v, self.embed_dim) for v in self.vocab_sizes
        )

        for emb in self.embeddings:
            emb.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeezed_channel = False
        if codes.dim() == 3:
            codes = codes.unsqueeze(2)
            squeezed_channel = True
        if codes.dim() != 4:
            raise ValueError(
                "RVQ codes must be shaped (B, T, C, Q) or (B, L, Q) for embedding."
            )
        if codes.shape[-1] != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} quantizer levels, "
                f"got {codes.shape[-1]}."
            )

        embeds = [emb(codes[..., i]) for i, emb in enumerate(self.embeddings)]
        concat = torch.cat(embeds, dim=-1)  # (B, T, C, Q*E)

        if squeezed_channel:
            concat = concat.squeeze(2)
        return concat


class JointRVQHead(nn.Module):
    def __init__(self, hidden_dim: int, num_levels: int, vocab_size: int):
        super().__init__()
        self.num_levels = num_levels
        self.vocab_size = vocab_size
        self.proj = nn.Linear(hidden_dim, num_levels * vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.proj(hidden)
        return logits.reshape(
            hidden.shape[0], hidden.shape[1], self.num_levels, self.vocab_size
        )


class TiedRVQHead(nn.Module):
    def __init__(self, embeddings: Sequence[nn.Embedding]):
        super().__init__()
        if len(embeddings) == 0:
            raise ValueError("At least one embedding is required for a tied head.")

        self.num_levels = len(embeddings)
        self.embed_dim = embeddings[0].embedding_dim
        self.vocab_sizes = [emb.num_embeddings for emb in embeddings]
        self.heads = nn.ModuleList(
            nn.Linear(self.embed_dim, vocab, bias=False) for vocab in self.vocab_sizes
        )

        for head, emb in zip(self.heads, embeddings):
            head.weight = emb.weight

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, dim = hidden.shape
        expected_dim = self.embed_dim * self.num_levels
        if dim != expected_dim:
            raise ValueError(
                f"Hidden size {dim} incompatible with tied head (exp. {expected_dim})."
            )

        hidden = hidden.reshape(batch, length, self.num_levels, self.embed_dim)
        logits = [head(hidden[:, :, i, :]) for i, head in enumerate(self.heads)]
        return torch.stack(logits, dim=2)


class MixEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_channels: int,
        mix_method: str = "mix",
    ):
        super().__init__()
        self.mix_method = mix_method
        self.num_channels = num_channels

        hidden_size = (
            hidden_size if mix_method == "none" else hidden_size // num_channels
        )
        self.emb = Embeddings(
            num_channels,
            quant_levels=vocab_size,
            quant_emb=hidden_size,
            channel_emb=hidden_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """X: (B*C, T)"""
        x = rearrange(x, "(b c) t -> b c t", c=self.num_channels)
        x = self.emb(x)  # B*C x T x E

        if self.mix_method == "mix":
            x = rearrange(x, "(b c) t e -> b t (c e)", c=self.num_channels)
        elif self.mix_method == "none":
            pass
        else:
            raise ValueError(f"Invalid mix method: {self.mix_method}")

        return x


class MixHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_channels: int,
        mix_method: str = "mix",
        emb: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        self.mix_method = mix_method
        self.num_channels = num_channels

        hidden_size = (
            hidden_size if mix_method == "none" else hidden_size // num_channels
        )
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        if emb is not None:
            self.proj.weight = emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mix_method == "mix":
            x = rearrange(x, "b t (c e) -> (b c) t e", c=self.num_channels)
        elif self.mix_method == "none":
            pass
            # x = rearrange(x, "(b c) t e -> b (t c) e", c=self.num_channels)
        else:
            raise ValueError(f"Invalid mix method: {self.mix_method}")

        return self.proj(x)


class ListEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_channels: int):
        super().__init__()
        self.emb = nn.ModuleList(
            nn.Embedding(vocab_size, hidden_size) for _ in range(num_channels)
        )

        for emb in self.emb:
            emb.weight.data.normal_(mean=0.0, std=0.02)

        self.hidden_size = hidden_size
        self.num_channels = num_channels

    def forward(self, x: torch.Tensor, chid: torch.Tensor = None) -> torch.Tensor:
        # reshape to (B, T, num_channels)
        if chid is not None and x.shape[1] == 1:
            return self.emb[chid](x)

        x = x.reshape(x.shape[0], -1, self.num_channels)

        weight_stack = torch.stack([emb.weight for emb in self.emb], dim=0)
        channel_ids = torch.arange(self.num_channels, device=x.device).view(
            1, 1, self.num_channels
        )
        embeds = weight_stack[channel_ids, x, :]

        return embeds.reshape(x.shape[0], -1, self.hidden_size)


class ChannelHead(nn.Module):
    def __init__(self, emb: nn.ModuleList):
        super().__init__()

        vocab_size, hidden_size = emb[0].weight.shape
        self.num_channels = len(emb)

        # create module list of nn.Linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(len(emb))]
        )

        # tie to embeddings
        for i, layer in enumerate(self.layers):
            layer.weight = emb[(i + 1) % len(emb)].weight

    def forward(self, x: torch.Tensor, chid: torch.Tensor = None) -> torch.Tensor:
        """X: (B, L, H)"""
        if chid is not None and x.shape[0] == 1:
            return self.layers[(len(self.layers) - 1 + chid) % len(self.layers)](x)

        # reshape to (B, T, num_channels, H)
        x = x.reshape(x.shape[0], -1, self.num_channels, x.shape[2])

        weight_stack = torch.stack([layer.weight for layer in self.layers], dim=0)
        x = torch.einsum("btch,cvh->btcv", x, weight_stack)
        x = x.reshape(x.shape[0], -1, x.shape[-1])

        return x
