import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...layers.videogpt import Encoder, Decoder, SamePadConv3d, shift_dim
from ...layers.quantizers import VideoGPTQuantizer


class VideoGPTTokenizer(nn.Module):
    """
    VQ-VAE for video-like tensors (T, H, W) tailored for brain topomap sequences.

    Expects inputs shaped as (B, 1, T, H, W) from ChunkDatasetImage (sparse topomaps
    over time), and outputs reconstructions of the same shape. The temporal dimension
    is the third axis to align with VideoGPT 3D conv expectations.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_codes: int,
        n_hiddens: int,
        n_res_layers: int,
        sequence_length: int,
        resolution: int,
        downsample: tuple[int, int, int],
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_codes = n_codes
        self.n_hiddens = n_hiddens
        self.n_res_layers = n_res_layers
        self.downsample = downsample
        self.sequence_length = sequence_length
        self.resolution = resolution

        self.encoder = Encoder(
            n_hiddens, n_res_layers, downsample, in_channels=in_channels
        )
        self.decoder = Decoder(
            n_hiddens, n_res_layers, downsample, out_channels=out_channels
        )

        self.pre_vq_conv = SamePadConv3d(n_hiddens, embedding_dim, 1)
        self.post_vq_conv = SamePadConv3d(embedding_dim, n_hiddens, 1)

        self.codebook = VideoGPTQuantizer(n_codes, embedding_dim)

    @property
    def latent_shape(self):
        input_shape = (
            self.sequence_length,
            self.resolution,
            self.resolution,
        )
        return tuple([s // d for s, d in zip(input_shape, self.downsample)])

    def encode(self, x):
        # Ensure input is (B, C=1, T, H, W); if ChunkDatasetImage returns (H,W,T),
        # move axes

        if x.ndim == 4:  # (B, H, W, T)
            x = x.permute(0, 3, 1, 2).unsqueeze(1)
        elif x.ndim == 3:  # (H, W, T)
            x = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

        h = self.pre_vq_conv(self.encoder(x))
        vq_output = self.codebook(h)
        return vq_output

    def decode(self, vq_output: dict) -> Tensor:
        encodings = vq_output["encodings"]
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(shift_dim(h, -1, 1))
        h = self.decoder(h)

        return h.squeeze(1).permute(0, 2, 3, 1)

    def forward(self, x):
        vq_output = self.encode(x)
        x_recon = self.decoder(self.post_vq_conv(vq_output["embeddings"]))
        x_recon = x_recon.squeeze(1).permute(0, 2, 3, 1)

        return (x_recon, vq_output)
