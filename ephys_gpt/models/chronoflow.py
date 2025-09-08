# chronoflow_ssm.py
import math
import random
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from ..layers.chronoflow import (
    SpatialFlow,
    TemporalBackbone,
    ConditionalFlow,
)

# -------------------------
# Utils
# -------------------------


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def numel_except_batch(x):
    return x[0].numel() * x.shape[0] if isinstance(x, tuple) else x[0].numel()


# Logit transform for continuous images in [0,1]
def dequantize_and_logit(x, alpha=1e-6):
    # uniform dequantization + logit
    if x.dtype != torch.float32:
        x = x.float()
    x = x.clamp(0.0, 1.0)
    x = (x * (1 - 2 * alpha)) + alpha
    y = torch.log(x) - torch.log(1 - x)
    logdet = -F.softplus(-y) - F.softplus(y)  # log(sigmoid(y)*(1-sigmoid(y))) with sign
    logdet = logdet.sum(dim=[1, 2, 3])  # per-sample
    return y, logdet


def logit_inverse(y):
    x = torch.sigmoid(y)
    # logdet of inverse is negative of forward's logdet
    logdet = (F.softplus(-y) + F.softplus(y)).sum(dim=[1, 2, 3])
    return x, logdet


# -------------------------
# Full model
# -------------------------


@dataclass
class ChronoFlowConfig:
    in_channels: int = 3
    image_size: Tuple[int, int] = (64, 64)
    spatial_levels: int = 3
    spatial_steps_per_level: int = 4
    spatial_hidden: int = 192

    emission_levels: int = 2
    emission_steps_per_level: int = 4
    emission_hidden: int = 192
    cond_dim: int = 512

    temporal_levels: int = 4
    temporal_state_dim: int = 512

    rollout_prob: float = 0.1  # scheduled sampling prob
    alpha_logit: float = 1e-6  # dequantization epsilon


class ChronoFlowSSM(nn.Module):
    def __init__(self, cfg: ChronoFlowConfig):
        super().__init__()
        self.cfg = cfg
        H, W = cfg.image_size

        # Spatial flow g_theta
        self.g = SpatialFlow(
            in_channels=cfg.in_channels,
            levels=cfg.spatial_levels,
            steps_per_level=cfg.spatial_steps_per_level,
            hidden=cfg.spatial_hidden,
        )

        # Determine latent spatial shape after g
        H_lat = H // (
            2 ** (cfg.spatial_levels + 1 - 1)
        )  # levels + final (no squeeze): we squeezed len(levels) times
        W_lat = W // (2 ** (cfg.spatial_levels + 1 - 1))
        u_channels = cfg.in_channels * (
            4 ** (cfg.spatial_levels + 0)
        )  # last level's channels
        self.u_shape = (u_channels, H_lat, W_lat)

        # Temporal backbone
        self.temporal = TemporalBackbone(
            u_channels=u_channels,
            u_spatial_hw=(H_lat, W_lat),
            levels=cfg.temporal_levels,
            state_dim=cfg.temporal_state_dim,
            cond_dim=cfg.cond_dim,
        )

        # Emission flow h_phi (conditional): maps z -> u
        self.h = ConditionalFlow(
            channels=u_channels,
            levels=cfg.emission_levels,
            steps_per_level=cfg.emission_steps_per_level,
            hidden=cfg.emission_hidden,
            cond_dim=cfg.cond_dim,
        )

    # ----- Framewise encode/decode -----
    def encode_frame(self, x):
        # x in logit space already
        u, ldj_g = self.g(x, reverse=False)  # x -> u
        return u, ldj_g

    def decode_frame(self, u):
        x, ldj_g_inv = self.g(u, reverse=True)  # u -> x
        return x, ldj_g_inv

    # ----- Likelihood of a sequence -----
    def nll_sequence(self, x_seq):
        """
        x_seq: [B, T, C, H, W] in [0,1]
        Returns: nll (scalar), stats dict
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # Dequantize + logit transform
        x_logit = []
        logdet_pre = []
        for t in range(T):
            y, ldj = dequantize_and_logit(x_seq[:, t], alpha=self.cfg.alpha_logit)
            x_logit.append(y)
            logdet_pre.append(ldj)
        x_logit = torch.stack(x_logit, dim=1)  # [B,T,C,H,W]
        logdet_pre = torch.stack(
            logdet_pre, dim=1
        )  # [B,T,Batch] actually [B] per step, stack -> [B,T]

        # Temporal states
        states = self.temporal.init_state(B, device)
        total_logprob = torch.zeros(B, device=device)

        # Rollout augmentation toggles
        rollout_prob = self.cfg.rollout_prob

        u_prev_for_state = None
        for t in range(T):
            # 1) conditioning from states (s_{t} depends on u_{t-1})
            cond, states, boundary = self.temporal(states, u_prev_for_state, step_idx=t)

            # 2) encode current frame into u_t
            u_t, ldj_g = self.encode_frame(x_logit[:, t])

            # 3) emission log-prob p(u_t | cond_t)
            logp_u = self.h.log_prob(u_t, cond)  # exact

            # 4) accumulate: log p(x_t | x_<t) = log p(u_t | s_t) + log |det dg/dx|
            total_logprob = total_logprob + logp_u + ldj_g + logdet_pre[:, t]

            # 5) teacher-forced state input for next step (with rollout aug)
            if (t < T - 1) and (random.random() < rollout_prob):
                # sample u_hat ~ p(u | cond) to self-condition
                u_hat = self.h.sample(self.u_shape_with_batch(B, device), cond)
                u_prev_for_state = u_hat.detach()
            else:
                u_prev_for_state = u_t.detach()

        nll = -(total_logprob.mean())
        stats = dict(
            bits_per_dim=nll / (math.log(2) * C * H * W),
            avg_boundary=boundary.mean().item(),
        )
        return nll, stats

    def u_shape_with_batch(self, B, device):
        C, H, W = self.u_shape
        return (B, C, H, W)

    # ----- Sampling / forecasting -----
    def sample(self, x_context, steps: int):
        """
        x_context: [B, T0, C, H, W] in [0,1]
        Returns: samples [B, T0+steps, C, H, W] in [0,1]
        """
        self.eval()
        B, T0, C, H, W = x_context.shape
        device = x_context.device

        # preprocess context
        x_logit = []
        for t in range(T0):
            y, _ = dequantize_and_logit(x_context[:, t], alpha=self.cfg.alpha_logit)
            x_logit.append(y)
        x_logit = torch.stack(x_logit, dim=1)

        # init states via teacher forcing over context
        states = self.temporal.init_state(B, device)
        u_prev = None
        for t in range(T0):
            cond, states, _ = self.temporal(states, u_prev, step_idx=t)
            u_t, _ = self.encode_frame(x_logit[:, t])
            u_prev = u_t

        # now autoregressive generation
        frames_out = [x_context[:, t] for t in range(T0)]
        for k in range(steps):
            step_idx = T0 + k
            cond, states, _ = self.temporal(states, u_prev, step_idx=step_idx)
            u_sample = self.h.sample(self.u_shape_with_batch(B, device), cond)
            x_logit_sample, _ = self.decode_frame(u_sample)
            # inverse logit -> [0,1]
            x_sample, _ = logit_inverse(x_logit_sample)
            frames_out.append(x_sample.clamp(0, 1))
            u_prev = u_sample

        return torch.stack(frames_out, dim=1)


# -------------------------
# Minimal dataset & training loop
# -------------------------


class TensorVideoDataset(Dataset):
    """
    A simple dataset wrapper around a preloaded tensor: [N, T, C, H, W] in [0,1].
    For real use, replace with a streaming video dataset that returns contiguous chunks.
    """

    def __init__(self, videos: torch.Tensor, chunk_len: int):
        super().__init__()
        assert videos.ndim == 5
        self.videos = videos
        self.chunk_len = chunk_len

    def __len__(self):
        N, T, _, _, _ = self.videos.shape
        return N * (T // self.chunk_len)

    def __getitem__(self, idx):
        N, T, C, H, W = self.videos.shape
        n = idx % N
        k = (idx // N) * self.chunk_len
        k = k % (T - self.chunk_len + 1)
        x = self.videos[n, k : k + self.chunk_len]  # [chunk_len, C, H, W]
        return x  # [T,C,H,W]


@dataclass
class TrainConfig:
    batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 10
    grad_clip: float = 1.0
    log_every: int = 50
    amp: bool = True


def train_one_epoch(
    model: ChronoFlowSSM,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    scaler,
    device,
    epoch,
    tcfg: TrainConfig,
):
    model.train()
    running = 0.0
    for it, x in enumerate(loader):
        # x: [B,T,C,H,W]
        x = x.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(tcfg.amp and device.type == "cuda")):
            nll, stats = model.nll_sequence(x)
        scaler.scale(nll).backward()
        if tcfg.grad_clip is not None:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        running += nll.item()
    return running / max(1, len(loader))


def fit(model: ChronoFlowSSM, train_loader: DataLoader, tcfg: TrainConfig, epochs=None):
    device = next(model.parameters()).device
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(tcfg.amp and device.type == "cuda"))
    max_epochs = epochs or tcfg.max_epochs
    for ep in range(1, max_epochs + 1):
        avg = train_one_epoch(model, train_loader, opt, scaler, device, ep, tcfg)
        print(f"epoch {ep} avg_nll={avg:.3f}")


# -------------------------
# Example usage (toy)
# -------------------------

if __name__ == "__main__":
    device = default_device()
    cfg = ChronoFlowConfig(
        in_channels=3,
        image_size=(64, 64),
        spatial_levels=2,  # increase for higher res
        spatial_steps_per_level=4,
        spatial_hidden=160,
        emission_levels=2,
        emission_steps_per_level=4,
        emission_hidden=160,
        cond_dim=512,
        temporal_levels=4,  # strides: 1,2,4,8
        temporal_state_dim=512,
        rollout_prob=0.1,
    )
    model = ChronoFlowSSM(cfg).to(device)

    # Dummy data: 8 videos, each with 256 frames (64x64 RGB)
    N, T, C, H, W = 8, 256, 3, 64, 64
    videos = torch.rand(N, T, C, H, W)
    ds = TensorVideoDataset(videos, chunk_len=64)
    dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True)

    tcfg = TrainConfig(batch_size=2, max_epochs=2, lr=2e-4, log_every=5)
    fit(model, dl, tcfg)

    # Sampling demo from 16-frame context:
    with torch.no_grad():
        ctx = videos[:2, :16].to(device)
        samples = model.sample(ctx, steps=32)  # [2, 48, 3, 64, 64]
        print("Generated samples:", samples.shape)
