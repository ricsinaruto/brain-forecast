import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange, reduce
import math


class VectorQuantizer(nn.Module):
    """Discretization bottleneck part of the VQ-VAE."""

    # NOTE: this class contains a bug regarding beta; see VectorQuantizer2 for
    # a fix and use legacy=False to apply that fix. VectorQuantizer2 can be
    # used wherever VectorQuantizer has been used before and is additionally
    # more efficient.
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """Inputs the output of the encoder network z and maps it to a discrete one-hot
        vector that is the index of the closest embedding vector e_j z (continuous) ->
        z_q (discrete) z.shape = (batch, channel, height, width)"""
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantize(nn.Module):
    """Gumbel Softmax trick quantizer."""

    def __init__(
        self,
        num_hiddens,
        embedding_dim,
        n_embed,
        straight_through=True,
        kl_weight=5e-4,
        temp_init=1.0,
        use_vqinterface=True,
        remap=None,
        unknown_index="random",
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            full_zeros = torch.zeros_like(logits)
            logits = logits[:, self.used, ...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            full_zeros[:, self.used, ...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum("b n h w, n d -> b d h w", soft_one_hot, self.embed.weight)

        qy = F.softmax(logits, dim=1)
        diff = (
            self.kl_weight
            * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        )

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b * h * w == indices.shape[0]
        indices = rearrange(indices, "(b h w) -> b h w", b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = (
            F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        )
        z_q = einsum("b n h w, n d -> b d h w", one_hot, self.embed.weight)
        return z_q


class VectorQuantizer2(nn.Module):
    """Improved version over VectorQuantizer, can be used as a drop-in replacement."""

    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
        l2_normalize=False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e
        self.sane_index_shape = sane_index_shape
        self.l2_normalize = l2_normalize

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        if self.l2_normalize:
            z_flattened = F.normalize(z_flattened, p=2, dim=1)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(
            F.one_hot(min_encoding_indices, self.n_e).type(z.dtype), dim=0
        )
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        min_encoding_indices = min_encoding_indices.view(
            z.shape[0], z.shape[1], z.shape[2]
        )
        min_encoding_indices = (
            min_encoding_indices
            if self.sane_index_shape
            else min_encoding_indices.view(z.shape[0], -1)
        )
        min_encoding_indices = min_encoding_indices.int()
        if self.remap is not None:
            min_encoding_indices = self.remap_to_used(min_encoding_indices)

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None):
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        if len(indices.shape) == 2:
            indices = indices[:, :, None]
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class VectorQuantizerWithWeakTrick(nn.Module):
    def __init__(self, n_e, e_dim, beta, weak_trick=0.0):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.weak_trick = weak_trick

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def weak_scatter(self, z, min_encoding_indices, min_encodings):
        z_shape = z.shape
        z_flattened = z.view(-1, self.e_dim)
        z_q_scattered = torch.zeros(z_flattened.shape[0], self.e_dim, device=z.device)
        min_encoding_indices = min_encoding_indices.view(-1, 1).repeat(1, 2)
        arange = torch.arange(z_flattened.shape[0])[:, None].to(device=z.device)
        inds_to_fill = torch.bernoulli(self.weak_trick * torch.ones_like(z_flattened))
        random_inds = torch.randint(0, self.n_e, size=(z_flattened.shape[0], 1)).to(
            device=z.device
        )
        random_one_hots = torch.zeros(z_flattened.shape[0], self.n_e, device=z.device)
        random_one_hots.scatter_(1, random_inds, 1)
        min_encodings = (
            1 - inds_to_fill
        ) * min_encodings + inds_to_fill * random_one_hots
        min_encoding_indices = (
            1 - inds_to_fill
        ) * min_encoding_indices + inds_to_fill * random_inds
        min_encoding_indices = min_encoding_indices[:, 0].long()
        min_encodings[min_encodings.sum(-1) == 0, 0] = 1

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z_shape)
        return z_q, min_encodings, min_encoding_indices

    def forward(self, z, use_weak_trick=False):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if use_weak_trick and self.training and self.weak_trick > 0:
            z_q, min_encodings, min_encoding_indices = self.weak_scatter(
                z, min_encoding_indices, min_encodings
            )
        else:
            z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        min_encoding_indices = min_encoding_indices.view(
            z.shape[0], z.shape[1], z.shape[2]
        )
        min_encoding_indices = min_encoding_indices.view(z.shape[0], -1)
        min_encoding_indices = min_encoding_indices.int()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None):
        if len(indices.shape) == 2:
            indices = indices[:, :, None]
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)

            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class ScaledL2CrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", temperature=1.0, scale=1.0):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.scale = scale

    def forward(self, x, labels):
        norm_temp = 2 * (self.temperature**2)
        x = -torch.sum((x / self.scale) ** 2, dim=1) / norm_temp

        if self.reduction == "none":
            ce = F.cross_entropy(x, labels, reduction="none")
            return ce, x
        return F.cross_entropy(x, labels, reduction=self.reduction), x


class VectorQuantizerWithKld(nn.Module):
    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        reduction="mean",
        temperature=1.0,
        use_l2_norm=True,
        scale=1.0,
        normalize=False,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.normal_()

        self.cross_entropy = ScaledL2CrossEntropyLoss(
            reduction=reduction, temperature=temperature, scale=scale
        )
        self.use_l2_norm = use_l2_norm
        self.normalize = normalize

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        if self.normalize:
            emb_norm = torch.norm(self.embedding.weight, dim=1)
            self.embedding.weight.data = self.embedding.weight / emb_norm.unsqueeze(1)

        if self.use_l2_norm:
            z_norm = torch.norm(z_flattened, dim=1, keepdim=True)
            z_l2 = z_flattened / z_norm
            loss, logits = self.cross_entropy(z_l2, z_norm.squeeze())
        else:
            logits = torch.matmul(z_flattened, self.embedding.weight.t())
            loss = F.cross_entropy(logits, torch.arange(self.n_e).to(z))
        min_encoding_indices = torch.argmax(logits, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
            (z_q - z.detach()) ** 2
        )

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(
            F.one_hot(min_encoding_indices, self.n_e).type(z.dtype), dim=0
        )
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        min_encoding_indices = min_encoding_indices.view(
            z.shape[0], z.shape[1], z.shape[2]
        )
        min_encoding_indices = min_encoding_indices.view(z.shape[0], -1)
        min_encoding_indices = min_encoding_indices.int()

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None):
        if len(indices.shape) == 2:
            indices = indices[:, :, None]
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)

            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class EMAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, decay):
        ctx.decay = decay
        return x

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return grad_output, None
        return grad_output * ctx.decay, None


def EMA(x, decay):
    return EMAFunction.apply(x, decay)


class VectorQuantizerCosine(nn.Module):
    def __init__(self, n_e, e_dim, decay=0.99, eps=1e-5, beta=0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.decay = decay
        self.eps = eps
        self.beta = beta

        self.embed = nn.Embedding(n_e, e_dim)
        self.embed.weight.data.normal_()
        self.register_buffer("cluster_size", torch.zeros(n_e))
        self.register_buffer("embed_avg", self.embed.weight.data.clone())

    def forward(self, z):
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, self.e_dim)
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embed.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embed.weight.t())
        )
        _, min_encoding_indices = (-d).max(1)
        z_q = self.embed_code(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
            (z_q - z.detach()) ** 2
        )
        z_q = z + (z_q - z).detach()
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device
        )
        min_encodings.scatter_(1, min_encoding_indices.view(-1, 1), 1)
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        min_encoding_indices = min_encoding_indices.view(
            z.shape[0], z.shape[2], z.shape[3]
        )

        return z_q, loss, (perplexity, min_encoding_indices)

    def embed_code(self, embed_id):
        with torch.no_grad():
            if self.training:
                encodings = torch.zeros(
                    embed_id.shape[0], self.n_e, device=embed_id.device
                )
                encodings.scatter_(1, embed_id.unsqueeze(1), 1)
                cluster_size = encodings.sum(0)
                self.cluster_size.data.mul_(self.decay).add_(
                    cluster_size, alpha=1 - self.decay
                )
                embed_sum = z_flattened.t() @ encodings
                self.embed_avg.data.mul_(self.decay).add_(
                    embed_sum, alpha=1 - self.decay
                )
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_e * self.eps) * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
                self.embed.weight.data.copy_(embed_normalized)
        return self.embed(embed_id)


class IndexPropagationQuantize(nn.Module):
    """Vector quantizer with index propagation and optional entropy regularization."""

    def __init__(
        self,
        n_e,
        e_dim,
        beta,
        use_entropy_loss=False,
        remap=None,
        unknown_index="random",
        sane_index_shape=False,
        legacy=True,
        cosine_similarity=False,
        entropy_temperature=0.01,
        sample_minimization_weight=1.0,
        batch_maximization_weight=1.0,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index  # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed + 1
            print(
                f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                f"Using {self.unknown_index} for unknown indices."
            )
        else:
            self.re_embed = n_e
        self.sane_index_shape = sane_index_shape
        self.cosine_similarity = cosine_similarity

        # entropy loss
        self.use_entropy_loss = use_entropy_loss
        self.entropy_temperature = entropy_temperature
        self.sample_minimization_weight = sample_minimization_weight
        self.batch_maximization_weight = batch_maximization_weight

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        match = (inds[:, :, None] == used[None, None, ...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2) < 1
        if self.unknown_index == "random":
            new[unknown] = torch.randint(0, self.re_embed, size=new[unknown].shape).to(
                device=new.device
            )
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape) > 1
        inds = inds.reshape(ishape[0], -1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:  # extra token
            inds[inds >= self.used.shape[0]] = 0  # simply set to zero
        back = torch.gather(used[None, :][inds.shape[0] * [0], :], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # [B, D, H, W] -> [B*H*W, D]
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        embedding = self.embedding.weight

        if self.cosine_similarity:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(embedding, p=2, dim=-1)

        # compute distances
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2 * torch.matmul(z_flattened, embedding.t())
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if not self.legacy:
            loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(
            F.one_hot(min_encoding_indices, self.n_e).type(z.dtype), dim=0
        )
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        min_encoding_indices = min_encoding_indices.view(
            z.shape[0], z.shape[1], z.shape[2]
        )
        min_encoding_indices = (
            min_encoding_indices
            if self.sane_index_shape
            else min_encoding_indices.view(z.shape[0], -1)
        )
        min_encoding_indices = min_encoding_indices.int()
        if self.remap is not None:
            min_encoding_indices = self.remap_to_used(min_encoding_indices)

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        if self.use_entropy_loss:
            sample_logits = (
                -d.view(*z.shape[:-1], d.shape[-1]) / self.entropy_temperature
            )
            sample_probs = F.softmax(sample_logits, dim=-1)

            # per-sample entropy (maximize)
            sample_entropy = -torch.sum(
                sample_probs * torch.log(sample_probs + 1e-10), dim=-1
            )
            avg_sample_entropy = torch.mean(sample_entropy)

            # batch-wise code usage (maximize entropy)
            batch_probs = torch.mean(sample_probs.view(-1, d.shape[-1]), dim=0)
            batch_entropy = -torch.sum(batch_probs * torch.log(batch_probs + 1e-10))

            entropy_loss = (
                self.sample_minimization_weight * avg_sample_entropy
                - self.batch_maximization_weight * batch_entropy
            )
            loss = (loss, sample_entropy.mean(), batch_entropy, entropy_loss)

        return z_q, loss, (perplexity, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None):
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        if len(indices.shape) == 2:
            indices = indices[:, :, None]
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
