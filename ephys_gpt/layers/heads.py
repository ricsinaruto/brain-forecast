import torch
from typing import List

from torch.nn import Module


class WavenetLogitsHead(Module):
    def __init__(
        self,
        skip_channels: int,
        residual_channels: int,
        head_channels: int,
        out_channels: int,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        """Collates skip results and transforms them to logit predictions.
        Args:
            skip_channels: number of skip channels
            residual_channels: number of residual channels
            head_channels: number of hidden channels to compute result
            out_channels: number of output channels
            bias: When true, convolutions use a bias term.
        """
        del residual_channels
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Dropout1d(p=dropout),
            torch.nn.LeakyReLU(),  # note, we perform non-lin first (on sum of skips)
            torch.nn.Conv1d(
                skip_channels,
                head_channels,
                kernel_size=1,
                bias=bias,
            ),  # enlarge and squeeze (not based on paper)
            torch.nn.Dropout1d(p=dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Conv1d(
                head_channels,
                out_channels,
                kernel_size=1,
                bias=bias,
            ),  # logits
        )

    def forward(self, encoded, skips):
        """Compute logits from WaveNet layer results.
        Args:
            encoded: unused last residual output of last layer
            skips: list of skip connections of shape (B,C,T) where C is
                the number of skip channels.
        Returns:
            logits: (B,Q,T) tensor of logits, where Q is the number of output
            channels.
        """
        del encoded
        return self.transform(sum(skips))


class Wavenet3DLogitsHead(Module):
    def __init__(
        self,
        skip_channels: int,
        head_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.transform = torch.nn.Sequential(
            torch.nn.Dropout3d(p=dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(skip_channels, head_channels, kernel_size=1, bias=bias),
            torch.nn.Dropout3d(p=dropout),
            torch.nn.LeakyReLU(),
            torch.nn.Conv3d(head_channels, out_channels, kernel_size=1, bias=bias),
        )

    def forward(self, skips: List[torch.Tensor]) -> torch.Tensor:
        return self.transform(sum(skips))
