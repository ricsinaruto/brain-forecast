from __future__ import annotations

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from .videogpt import shift_dim


class VideoGPTQuantizer(nn.Module):
    def __init__(self, n_codes: int, embedding_dim: int):
        super().__init__()
        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True

    def _tile(self, x: torch.Tensor) -> torch.Tensor:
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z: torch.Tensor) -> None:
        # z: [b, c, t, h, w]
        self._need_init = False
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        y = self._tile(flat_inputs)

        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        # z: [b, c, t, h, w]
        if self._need_init and self.training:
            self._init_embeddings(z)
        flat_inputs = shift_dim(z, 1, -1).flatten(end_dim=-2)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )

        encoding_indices = torch.argmin(distances, dim=1)
        encode_onehot = F.one_hot(encoding_indices, self.n_codes).type_as(flat_inputs)
        encoding_indices = encoding_indices.view(z.shape[0], *z.shape[2:])

        embeddings = F.embedding(encoding_indices, self.embeddings)
        embeddings = shift_dim(embeddings, -1, 1)

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training:
            n_total = encode_onehot.sum(dim=0)
            encode_sum = flat_inputs.t() @ encode_onehot
            if dist.is_initialized():
                dist.all_reduce(n_total)
                dist.all_reduce(encode_sum)

            self.N.data.mul_(0.99).add_(n_total, alpha=0.01)
            self.z_avg.data.mul_(0.99).add_(encode_sum.t(), alpha=0.01)

            n = self.N.sum()
            weights = (self.N + 1e-7) / (n + self.n_codes * 1e-7) * n
            encode_normalized = self.z_avg / weights.unsqueeze(1)
            self.embeddings.data.copy_(encode_normalized)

            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)

            usage = (self.N.view(self.n_codes, 1) >= 1).float()
            self.embeddings.data.mul_(usage).add_(_k_rand * (1 - usage))

        embeddings_st = (embeddings - z).detach() + z

        avg_probs = torch.mean(encode_onehot, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return dict(
            embeddings=embeddings_st,
            encodings=encoding_indices,
            commitment_loss=commitment_loss,
            perplexity=perplexity,
        )

    def dictionary_lookup(self, encodings):
        embeddings = F.embedding(encodings, self.embeddings)
        return embeddings


class VectorQuantizer(nn.Module):
    """
    TODO: test, this comes from EEGFormera
    Straight‑Through Vector Quantisation layer (à la VQ‑VAE).

    The codebook is updated via gradient descent; for EMA, see ``VectorQuantizerEMA``
    below.

    Parameters
    ----------
    num_embeddings : int
        Size of the codebook (K).
    embedding_dim  : int
        Dimensionality of each code (D).
    beta           : float, default 0.25
        Commitment loss multiplier.
    """

    def __init__(self, *, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings
        )

    # ------------------------------------------------------------------
    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise *z* and return (quantised, vq_loss, indices).

        ``z`` shape: (B, N, D)
        """
        B, N, D = z.shape
        flat_z = z.view(-1, D)  # (B*N, D)

        # Distance between each latent and codebook vector (B*N, K)
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        z_sq = (flat_z**2).sum(dim=1, keepdim=True)  # (B*N, 1)
        e_sq = (self.embedding.weight**2).sum(dim=1)  # (K)
        distances = z_sq + e_sq - 2 * flat_z @ self.embedding.weight.t()  # (B*N, K)

        # Closest codebook entry for each latent
        indices = distances.argmin(dim=1)  # (B*N)
        quantised = self.embedding(indices).view(B, N, D)

        # Straight‑through estimator
        quantised_st = z + (quantised - z).detach()

        # Losses (commitment + codebook)
        codebook_loss = F.mse_loss(quantised, z.detach())
        commitment_loss = F.mse_loss(z, quantised.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        return quantised_st, vq_loss, indices.view(B, N)


class ResidualVectorQuantizer(nn.Module):
    """
    This is from BrainOmni implementation.
    Multi‑stage residual VQ with straight‑through estimator.

    Outputs both the quantised vectors **and** the integer codebook indices.
    Commitment loss must be added externally during training; this forward pass
    is inference‑only (no EMA updates)."""

    def __init__(self, dim: int, num_quantizers: int = 4, codebook_size: int = 1024):
        super().__init__()
        self.dim = dim
        self.num_q = num_quantizers
        self.codebook_size = codebook_size
        self.codebooks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(codebook_size, dim))
                for _ in range(num_quantizers)
            ]
        )

    @staticmethod
    def _l2_dist(x: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
        """Computes ||x - c||^2 for every codebook vector c, in a matrix form."""
        # x: (N, D);  cb: (K, D)  ->  (N, K)
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        x_norm = (x**2).sum(dim=1, keepdim=True)  # (N, 1)
        cb_norm = (cb**2).sum(dim=1, keepdim=True).T  # (1, K)
        dist = x_norm + cb_norm - 2 * x @ cb.t()  # (N, K)
        return dist

    def forward(
        self, x: torch.Tensor, return_trace: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (*, D) float32.  Returns (quantised, codes) where
        quantised – same shape as x, codes – (*, num_q) int64 indices."""
        orig_shape = x.shape
        x = x.view(-1, self.dim)  # (N, D)

        residual = x  # start with full vector
        final_quantised = torch.zeros_like(x)
        residuals: List[torch.Tensor] = []
        nearest: List[torch.Tensor] = []
        all_codes: List[torch.Tensor] = []

        for cb in self.codebooks:
            residuals.append(residual)

            dist = self._l2_dist(residual, cb)
            codes = dist.argmin(-1)
            quant = F.embedding(codes, cb)
            all_codes.append(codes)
            nearest.append(quant)

            # straight-through view
            quant_st = residual + (quant - residual).detach()
            final_quantised += quant_st

            residual -= quant.detach()

        codes = torch.stack(all_codes, dim=-1)  # (N, num_q)

        quantised = final_quantised.view(*orig_shape)
        codes = codes.view(*orig_shape[:-1], self.num_q)

        if return_trace:
            # reshape stored traces to match input spatial dims
            residuals = [r.view(*orig_shape) for r in residuals]
            nearest = [n.view(*orig_shape) for n in nearest]
            return quantised, codes, residuals, nearest

        return quantised, codes, None, None

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (N, num_q)"""
        qs = [
            F.embedding(codes[:, i], cb)  # (N, D) for stage i
            for i, cb in enumerate(self.codebooks)
        ]
        return sum(qs)
