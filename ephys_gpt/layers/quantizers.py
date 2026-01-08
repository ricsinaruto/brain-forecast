from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from math import ceil
from einops import rearrange, pack, unpack
from vector_quantize_pytorch.vector_quantize_pytorch import rotate_to


def default(v, d):
    return v if v is not None else d


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


@torch.no_grad()
def kmeans(
    samples: torch.Tensor,
    nums_clusters: int,
    kmeans_iters: int,
    chunk_size: Optional[int] = None,
):
    device = samples.device
    samples = rearrange(samples, "... d -> (...) d")
    dim, dtype = samples.shape[1], samples.dtype
    if samples.shape[0] < nums_clusters:
        random_noise = torch.randn(
            size=(nums_clusters - samples.shape[0], dim),
            device=samples.device,
            dtype=dtype,
        )
        samples = torch.cat([samples, random_noise], dim=0)

    # pick a chunk size so that the distance matrix per step stays memory-friendly
    if chunk_size is None:
        max_dist_elements = 16_000_000  # ~64MB for float32
        chunk_size = max(1, int(max_dist_elements // max(nums_clusters, 1)))
    chunk_size = max(1, min(chunk_size, samples.shape[0]))

    centers = sample_vectors(samples, nums_clusters)
    for _ in range(kmeans_iters):
        bins = torch.zeros(nums_clusters, device=samples.device, dtype=torch.long)
        new_centers = torch.zeros(
            (nums_clusters, dim), device=samples.device, dtype=dtype
        )
        center_norm = (centers**2).sum(1)

        for batch in samples.split(chunk_size):
            batch_norm = (batch**2).sum(1, keepdim=True)
            dist = batch_norm + center_norm - 2 * (batch @ centers.t())
            buckets = dist.argmin(dim=1)
            bins.scatter_add_(0, buckets, torch.ones_like(buckets, dtype=torch.long))
            new_centers.scatter_add_(0, buckets.unsqueeze(-1).expand(-1, dim), batch)

        zero_mask = bins == 0
        bins = bins.masked_fill(zero_mask, 1)
        new_centers = new_centers / bins[..., None]
        centers = torch.where(zero_mask[..., None], centers, new_centers)

    return centers.to(device), bins.to(device)


class VectorQuantizer(nn.Module):
    """Classic VQ (nearest-neighbor) with straight-through estimator.

    Input:  z_e shape (M, D) (flattened) Output: z_q_st shape (M, D), indices shape
    (M,), vq_loss, perplexity, z_q
    """

    def __init__(
        self,
        codebook_size: int,
        embed_dim: int,
        beta: float,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        ema_eps: float = 1e-5,
        reset_unused_codes: bool = False,
        usage_threshold: float = 1.0,
        init_std: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps
        self.reset_unused_codes = reset_unused_codes
        self.usage_threshold = usage_threshold
        self.init_std = init_std

        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=self.init_std)

        if self.use_ema:
            self.codebook.weight.requires_grad = False
            self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
            self.register_buffer("ema_weight", self.codebook.weight.detach().clone())

    def forward(
        self, z_e: torch.Tensor
    ) -> Tuple[
        torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        z2 = (z_e**2).sum(dim=1, keepdim=True)  # (M,1)
        e2 = (self.codebook.weight**2).sum(dim=1).unsqueeze(0)  # (1,K)
        ze = z_e @ self.codebook.weight.t()  # (M,K)
        dist = z2 + e2 - 2 * ze  # (M,K)

        indices = dist.argmin(dim=1)  # (M,)
        z_q = self.codebook(indices)  # (M,D)

        # VQ losses + optional EMA codebook updates
        if self.use_ema:
            commit_loss = F.mse_loss(z_e, z_q.detach())
            if self.training:
                with torch.no_grad():
                    self._ema_update(z_e, indices)
            vq_loss = self.beta * commit_loss
        else:
            codebook_loss = F.mse_loss(z_q, z_e.detach())
            commit_loss = F.mse_loss(z_e, z_q.detach())
            vq_loss = codebook_loss + self.beta * commit_loss

        # straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # perplexity (code usage)
        if self.use_ema and hasattr(self, "ema_cluster_size"):
            probs = self.ema_cluster_size + 1e-10
            probs = probs / probs.sum()
            perplexity = torch.exp(-(probs * torch.log(probs)).sum())
        else:
            onehot = F.one_hot(indices, self.codebook_size).float()
            avg_probs = onehot.mean(dim=0)
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_st, indices, vq_loss, perplexity.detach(), z_q

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.LongTensor) -> None:
        onehot = F.one_hot(indices, self.codebook_size).type(z_e.dtype)
        cluster_size = onehot.sum(dim=0)  # (K,)
        embed_sum = onehot.t() @ z_e  # (K, D)

        self.ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1.0 - self.ema_decay
        )
        self.ema_weight.mul_(self.ema_decay).add_(embed_sum, alpha=1.0 - self.ema_decay)

        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.ema_eps)
            / (n + self.codebook_size * self.ema_eps)
            * n
        )
        embed_normalized = self.ema_weight / cluster_size.unsqueeze(1).clamp_min(
            self.ema_eps
        )
        self.codebook.weight.data.copy_(embed_normalized)

        if self.reset_unused_codes:
            self._reset_unused_codes(z_e)

    @torch.no_grad()
    def _reset_unused_codes(self, z_e: torch.Tensor) -> None:
        if z_e.numel() == 0:
            return
        unused = self.ema_cluster_size < self.usage_threshold
        if not unused.any():
            return

        num_new = unused.sum().item()
        rand_idx = torch.randint(0, z_e.shape[0], (num_new,), device=z_e.device)
        new_vecs = z_e[rand_idx].detach()

        self.codebook.weight.data[unused] = new_vecs
        self.ema_weight.data[unused] = new_vecs
        self.ema_cluster_size.data[unused] = torch.full_like(
            self.ema_cluster_size[unused], self.usage_threshold
        )


class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn = self._uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    def _uniform_init(self, *shape: int):
        t = torch.empty(shape)
        nn.init.kaiming_uniform_(t)
        return t

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited:
            return
        embed, cluster_size = kmeans(
            sample_vectors(data, self.codebook_size * 4),
            self.codebook_size,
            self.kmeans_iters,
        )
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)

    # flatten in
    @torch.no_grad()
    def quantize(self, x):
        x = x.float()
        embed = self.embed.t().float()
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.argmin(dim=-1)
        return embed_ind  # n

    # any in any out
    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    # any in any out
    def encode(self, x):
        shape = x.shape
        # pre-process
        x = rearrange(x, "... d -> (...) d")
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = embed_ind.view(*shape[:-1])
        return embed_ind

    # equals dequantize
    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def _ema_inplace(self, moving_avg: torch.Tensor, new: torch.Tensor, decay: float):
        moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

    def _laplace_smoothing(self, x, epsilon: float = 1e-6):
        return (x + epsilon) / (x.sum() + epsilon * len(x))

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = rearrange(x, "... d -> (...) d")
        self.init_embed_(x)
        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])
        quantize = self.dequantize(embed_ind).type(dtype)

        if self.training:
            self.expire_codes_(x)
            one_hot_sum = embed_onehot.sum(0)

            self._ema_inplace(self.cluster_size, one_hot_sum, self.decay)
            embed_sum = embed_onehot.t() @ x
            embed_sum = embed_sum.to(torch.float32)
            self._ema_inplace(self.embed_avg, embed_sum, self.decay)

            cluster_size = (
                self._laplace_smoothing(self.cluster_size, self.epsilon)
                * self.cluster_size.sum()
            )

            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind


class VQ(nn.Module):
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: int = None,
        decay: float = 0.99,
        epsilon: float = 1e-6,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        rotation_trick: bool = True,
    ):
        super().__init__()
        self.rotation_trick = rotation_trick
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    # any in any out
    def encode(self, x):
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    # any in any out
    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        return quantize

    def forward(self, x):
        input_dtype = x.dtype
        x = self.project_in(x)
        quantize, embed_ind = self._codebook(x)
        if self.training:
            if self.rotation_trick:
                quantize = rotate_to(x, quantize).to(input_dtype)
            else:
                quantize = x + (quantize - x).detach()

        loss = F.mse_loss(x.float(), quantize.detach().float()) * 0.25
        if not self.training:
            loss = loss.detach()

        quantize = self.project_out(quantize)
        return quantize, embed_ind, loss


class RVQ(nn.Module):
    def __init__(
        self,
        dim,
        codebook_dim: int,
        codebook_size: int,
        num_quantizers: int,
        quantize_dropout=False,
        quantize_dropout_cutoff_index=0,
        quantize_dropout_multiple_of=1,
        rotation_trick=True,  # rotation trick from @cfifty
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList([])
        for _ in range(num_quantizers):
            self.layers.append(
                VQ(
                    dim=dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    rotation_trick=rotation_trick,
                )
            )
        # quantize dropout
        self.quantize_dropout = quantize_dropout and num_quantizers > 1
        assert quantize_dropout_cutoff_index >= 0
        self.quantize_dropout_cutoff_index = quantize_dropout_cutoff_index
        self.quantize_dropout_multiple_of = quantize_dropout_multiple_of
        # encodec paper proposes structured dropout, believe this was set to 4

    @property
    def codebook_size(self):
        return self.layers[0].codebook_size

    @property
    def codebook_dim(self):
        return self.layers[0].codebook_dim

    @property
    def codebooks(self):
        codebooks = [layer.codebook for layer in self.layers]
        codebooks = torch.stack(codebooks)
        return codebooks

    def _round_up_multiple(self, num, mult):
        return ceil(num / mult) * mult

    def forward(self, x: torch.Tensor):
        num_quant, quant_dropout_multiple_of, device = (
            self.num_quantizers,
            self.quantize_dropout_multiple_of,
            x.device,
        )

        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        should_quantize_dropout = self.training and self.quantize_dropout

        # sample a layer index at which to dropout further residual quantization
        # also prepare null indices and loss

        if should_quantize_dropout:

            # check if seed is manually passed in

            rand_quantize_dropout_fixed_seed = torch.randint(
                0, 10_000, (), device=device
            ).item()

            rand = random.Random(rand_quantize_dropout_fixed_seed)

            rand_quantize_dropout_index = rand.randrange(
                self.quantize_dropout_cutoff_index, num_quant
            )

            if quant_dropout_multiple_of != 1:
                rand_quantize_dropout_index = (
                    self._round_up_multiple(
                        rand_quantize_dropout_index + 1, quant_dropout_multiple_of
                    )
                    - 1
                )

        # go through the layers
        for quantizer_index, vq in enumerate(self.layers):

            if (
                should_quantize_dropout
                and quantizer_index > rand_quantize_dropout_index
            ):
                continue

            # sim vq forward
            quantized, indices, loss = vq(residual)

            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_losses.append(loss)
            all_indices.append(indices)

        # stack all losses and indices
        all_losses = torch.stack(all_losses, dim=-1)
        all_indices = torch.stack(all_indices, dim=-1)

        return quantized_out, all_indices, all_losses.mean()

    def encode(self, x: torch.Tensor):
        """X: B W D."""
        quantized_out = 0.0
        residual = x
        all_indices = []
        for quantizer_index, vq in enumerate(self.layers):
            quantized, indices, _ = vq(residual)
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
        all_indices = torch.stack(all_indices, dim=-1)
        return all_indices


class FSQuantizer(nn.Module):
    """Finite Scalar Quantization: VQ-VAE Made Simple -
    https://arxiv.org/abs/2309.15505.

    Code adapted from Jax version in Appendix A.1.

    Adapted from:
    https://github.com/lucidrains/
     vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/
    vector_quantize_pytorch/finite_scalar_quantization.py [Copyright (c) 2020 Phil Wang]
    https://github.com/lucidrains/vector-quantize-pytorch/
    blob/9502a1f447876d53fd37685b226bf28f250dc4a3/LICENSE
    """

    def __init__(
        self,
        levels: list[int],
        dim: Optional[int] = None,
        num_codebooks=1,
        keep_num_codebooks_dim: Optional[bool] = None,
        scale: Optional[float] = None,
        **ignore_kwargs,
    ):
        super().__init__()
        self.dtype = ignore_kwargs.get("dtype", torch.bfloat16)
        self.persistent = ignore_kwargs.get("persistent_quantizer", True)
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=self.persistent)

        _basis = torch.cumprod(
            torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32
        )
        self.register_buffer("_basis", _basis, persistent=self.persistent)

        self.scale = scale

        codebook_dim = len(levels)
        self.codebook_dim = codebook_dim

        effective_codebook_dim = codebook_dim * num_codebooks
        self.num_codebooks = num_codebooks
        self.effective_codebook_dim = effective_codebook_dim

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        self.dim = default(dim, len(_levels) * num_codebooks)

        has_projections = self.dim != effective_codebook_dim
        self.project_in = (
            nn.Linear(self.dim, effective_codebook_dim)
            if has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(effective_codebook_dim, self.dim)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.codebook_size = self._levels.prod().item()

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer(
            "implicit_codebook", implicit_codebook, persistent=self.persistent
        )

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound `z`, an array of shape (..., d)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: torch.Tensor) -> torch.Tensor:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(self, indices: torch.Tensor, project_out=True) -> torch.Tensor:
        """Inverse of `codes_to_indices`."""
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)

        if self.keep_num_codebooks_dim:
            codes = rearrange(codes, "... c d -> ... (c d)")

        if project_out:
            codes = self.project_out(codes)

        if is_img_or_video:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes.to(self.dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Einstein notation.

        b - batch n - sequence (or flattened spatial dimensions) d - feature dimension,
        which is also log2(codebook size) c - number of codebook dim
        """
        is_img_or_video = z.ndim >= 4

        # standardize image or video into (batch, seq, dimension)

        if is_img_or_video:
            z = rearrange(z, "b d ... -> b ... d")
            z, ps = pack_one(z, "b * d")

        assert (
            z.shape[-1] == self.dim
        ), f"expected dimension of {self.dim} but found dimension of {z.shape[-1]}"

        z = self.project_in(z)

        z = rearrange(z, "b n (c d) -> b n c d", c=self.num_codebooks)

        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)

        codes = rearrange(codes, "b n c d -> b n (c d)")

        out = self.project_out(codes)

        # reconstitute image or video dimensions

        if is_img_or_video:
            out = unpack_one(out, ps, "b * d")
            out = rearrange(out, "b ... d -> b d ...")
            indices = unpack_one(indices, ps, "b * c")
            dummy_loss = torch.zeros_like(out.mean(dim=[1, 2, 3], keepdim=True))
        else:
            dummy_loss = torch.zeros_like(out.mean(dim=[1, 2], keepdim=True)).unsqueeze(
                1
            )

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        return (indices, out.to(self.dtype), dummy_loss)


class ResidualFSQuantizer(nn.Module):
    """Residual Finite Scalar Quantization.

    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, levels: list[int], num_quantizers: int, **ignore_kwargs):
        super().__init__()
        self.dtype = ignore_kwargs.get("dtype", torch.float32)
        self.layers = nn.ModuleList(
            [FSQuantizer(levels=levels) for _ in range(num_quantizers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indices_stack = []
        residual = x
        quantized_out = 0
        loss_out = 0
        for i, layer in enumerate(self.layers):
            quant_indices, z, loss = layer(residual)
            indices_stack.append(quant_indices)
            residual = residual - z.detach()
            quantized_out = quantized_out + z
            loss_out = loss_out + loss
        self.residual = residual
        indices = torch.stack(indices_stack, dim=1)
        return indices, quantized_out.to(self.dtype), loss_out.to(self.dtype)

    def indices_to_codes(self, indices_stack: torch.Tensor) -> torch.Tensor:
        quantized_out = 0
        for layer, indices in zip(self.layers, indices_stack.transpose(0, 1)):
            quantized_out += layer.indices_to_codes(indices)
        return quantized_out
