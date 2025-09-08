import itertools
from tqdm import tqdm

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.videogpt import (
    AttentionStack,
    LayerNorm,
    shift_dim,
)
from ..training.lightning import LitModel
from ..models.tokenizers.videogpttokenizer import VideoGPTTokenizer


class VideoGPT(nn.Module):
    def __init__(
        self,
        tokenizer_path: str,
        hidden_dim: int,
        heads: int,
        layers: int,
        dropout: float = 0.0,
        attn_type: str = "full",
        attn_dropout: float = 0.0,
        tokenizer_args: dict = None,
    ):
        super().__init__()

        if tokenizer_path is not None:
            lit = LitModel.load_from_checkpoint(tokenizer_path, strict=False)
            self.vqvae = lit.model

            # check if model is compiled
            if hasattr(self.vqvae, "_orig_mod"):
                self.vqvae = self.vqvae._orig_mod

        else:
            self.vqvae = VideoGPTTokenizer(**tokenizer_args)

        # freeze tokenizer during autoregressive training (optional)
        for p in self.vqvae.parameters():
            p.requires_grad_(False)

        self.vqvae.codebook._need_init = False

        # VideoGPT transformer
        self.shape = self.vqvae.latent_shape

        self.fc_in = nn.Linear(self.vqvae.embedding_dim, hidden_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_stack = AttentionStack(
            self.shape,
            hidden_dim,
            heads,
            layers,
            dropout,
            attn_type,
            attn_dropout,
        )

        self.norm = LayerNorm(hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, self.vqvae.n_codes, bias=False)
        # self.fc_out.weight.data.copy_(torch.zeros(self.vqvae.n_codes, hidden_dim))

    def get_reconstruction(self, videos):
        return self.vqvae.decode(self.vqvae.encode(videos))

    def sample(self, n, batch=None):
        device = self.fc_in.weight.device

        cond = dict()
        if self.use_frame_cond or self.args.class_cond:
            assert batch is not None
            video = batch["video"]

            if self.args.class_cond:
                label = batch["label"]
                cond["class_cond"] = F.one_hot(label, self.args.class_cond_dim).type_as(
                    video
                )
            if self.use_frame_cond:
                cond["frame_cond"] = video[:, :, : self.args.n_cond_frames]

        samples = torch.zeros((n,) + self.shape).long().to(device)
        idxs = list(itertools.product(*[range(s) for s in self.shape]))

        with torch.no_grad():
            prev_idx = None
            for i, idx in enumerate(tqdm(idxs)):
                batch_idx_slice = (slice(None, None), *[slice(i, i + 1) for i in idx])
                batch_idx = (slice(None, None), *idx)
                embeddings = self.vqvae.codebook.dictionary_lookup(samples)

                if prev_idx is None:
                    # set arbitrary input values for the first token
                    # does not matter what value since it will be shifted anyways
                    embeddings_slice = embeddings[batch_idx_slice]
                    samples_slice = samples[batch_idx_slice]
                else:
                    embeddings_slice = embeddings[prev_idx]
                    samples_slice = samples[prev_idx]

                logits = self(
                    embeddings_slice, samples_slice, cond, decode_step=i, decode_idx=idx
                )[1]
                # squeeze all possible dim except batch dimension
                logits = (
                    logits.squeeze().unsqueeze(0)
                    if logits.shape[0] == 1
                    else logits.squeeze()
                )
                probs = F.softmax(logits, dim=-1)
                samples[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)

                prev_idx = batch_idx_slice
            samples = self.vqvae.decode(samples)
            samples = torch.clamp(samples, -0.5, 0.5) + 0.5

        return samples  # BCTHW in [0, 1]

    def forward(self, x: Tensor, test_mode: bool = False) -> tuple[Tensor, Tensor]:
        self.vqvae.eval()

        if not test_mode:
            with torch.no_grad():
                vq_output = self.vqvae.encode(x)
        else:
            vq_output = self.vqvae.encode(x)

        x = shift_dim(vq_output["embeddings"], 1, -1)

        h = self.fc_in(x)
        h = self.attn_stack(h)
        h = self.norm(h)
        logits = self.fc_out(h)

        # logits = shift_dim(logits, -1, 1)
        return (logits, vq_output["encodings"])
