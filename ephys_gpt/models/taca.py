# taca_video_ar.py
# Tri-Axial Causal Attention (TACA) — autoregressive video likelihood model
# Minimal, self-contained PyTorch demo with training & sampling.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ..layers.attention import MultiHeadAttention

# -----------------------------
# Utilities
# -----------------------------


def sinusoid_position_embedding(length, dim, device):
    """Standard 1D sinusoidal position embeddings."""
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [length, dim]


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mult * dim),
            nn.GELU(),
            nn.Linear(mult * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AxialAttentionBlock(nn.Module):
    """
    Non-causal spatial attention within a frame: row-attn then column-attn, plus MLP.
    I/O per call: [B, H, W, D]
    """

    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.row_attn = MultiHeadAttention(dim, heads, dropout=dropout)
        self.col_attn = MultiHeadAttention(dim, heads, dropout=dropout)
        self.ff = FeedForward(dim, mult=4, dropout=dropout)

    def forward(self, x):  # [B,H,W,D]
        B, H, W, D = x.shape
        # Row attention across width
        y = x.reshape(B * H, W, D)
        y = y + self.row_attn(self.norm(y), self.norm(y), self.norm(y))
        y = y.reshape(B, H, W, D)
        # Column attention across height
        z = y.permute(0, 2, 1, 3).reshape(B * W, H, D)  # [B*W,H,D]
        z = z + self.col_attn(self.norm(z), self.norm(z), self.norm(z))
        z = z.reshape(B, W, H, D).permute(0, 2, 1, 3)  # [B,H,W,D]
        # FF
        z = z + self.ff(z.reshape(B * H * W, D)).reshape(B, H, W, D)
        return z


class FrameEncoder(nn.Module):
    """
    Embeds 8-bit pixels -> D, adds separable 2D pos, then L axial-attn blocks.
    Vectorized over time by flattening [B*T,...]
    """

    def __init__(self, H, W, dim=192, quant_levels=256, heads=8, layers=4, dropout=0.0):
        super().__init__()
        self.H, self.W, self.D = H, W, dim
        self.h_pos = nn.Parameter(torch.zeros(H, dim))
        self.w_pos = nn.Parameter(torch.zeros(W, dim))
        nn.init.normal_(self.h_pos, std=0.02)
        nn.init.normal_(self.w_pos, std=0.02)
        self.blocks = nn.ModuleList(
            [AxialAttentionBlock(dim, heads, dropout) for _ in range(layers)]
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:  # [B,T,H,W,D] (float)
        B, T, H, W, D = frames.shape
        x = frames.reshape(B * T, H, W, D)  # [B*T,H,W,D]
        x = x + self.h_pos.view(1, H, 1, self.D) + self.w_pos.view(1, 1, W, self.D)
        for blk in self.blocks:
            x = blk(x)  # [B*T,H,W,D]
        return x.view(B, T, H, W, self.D)  # [B,T,H,W,D]


class MemoryCompressor(nn.Module):
    """
    Average-pools features into patch tokens of size (ph,pw).
    Vectorized over time.
    """

    def __init__(self, H, W, dim, pool=(4, 4)):
        super().__init__()
        ph, pw = pool
        assert H % ph == 0 and W % pw == 0
        self.ph, self.pw = ph, pw
        self.H2, self.W2 = H // ph, W // pw

    def forward(self, feats):  # [B,T,H,W,D] -> [B,T,M,D], M=H2*W2
        B, T, H, W, D = feats.shape
        x = feats.permute(0, 1, 4, 2, 3).reshape(B * T, D, H, W)  # [B*T,D,H,W]
        x = F.avg_pool2d(
            x, kernel_size=(self.ph, self.pw), stride=(self.ph, self.pw)
        )  # [B*T,D,H2,W2]
        x = (
            x.view(B, T, D, self.H2, self.W2)
            .permute(0, 1, 3, 4, 2)
            .reshape(B, T, self.H2 * self.W2, D)
        )
        return x  # [B,T,M,D]


class DecoderVectorized(nn.Module):
    """
    Vectorized next-frame decoder:
      - Builds queries for all t=1..T-1 at once (predict frame t from keys <= t-1).
      - One big cross-attention with a block-lower-triangular time mask.
      - Optional spatial refine, logits per pixel.
    """

    def __init__(
        self,
        H,
        W,
        dim=192,
        heads=8,
        dropout=0.0,
        max_T=2048,
        refine=True,
        quant_levels=256,
    ):
        super().__init__()
        self.H, self.W, self.D = H, W, dim
        self.Nq = H * W
        self.query_grid = nn.Parameter(torch.randn(self.Nq, dim) * 0.02)
        self.t_pos = nn.Embedding(max_T, dim)
        nn.init.normal_(self.t_pos.weight, std=0.02)
        self.cross = MultiHeadAttention(dim, heads, dropout=dropout)
        self.refine = (
            AxialAttentionBlock(dim, heads, dropout) if refine else nn.Identity()
        )
        self.head = nn.Linear(dim, quant_levels)

    def forward(self, mem_tokens):
        """
        mem_tokens: [B, T, M, D]  (M = memory tokens per frame)
        Returns logits for t=1..T-1: [B, T-1, H, W, 256]
        """
        B, T, M, D = mem_tokens.shape
        device = mem_tokens.device

        # --- Build queries for "predict frames 1..T-1" (i.e., next-frame targets) ---
        # q_time indices (the *predicted* frame index) = 1..T-1
        t_idx = torch.arange(1, T, device=device)  # [T-1]
        # Base query grid repeated over time & batch
        q_base = self.query_grid.view(1, 1, self.Nq, D).expand(
            B, T - 1, self.Nq, D
        )  # [B,T-1,Nq,D]
        q = q_base + self.t_pos(t_idx).view(1, T - 1, 1, D)  # [B,T-1,Nq,D]
        q = q.reshape(B, (T - 1) * self.Nq, D)  # [B,(T-1)*Nq,D]

        # --- Keys/values: memory tokens for all frames (0..T-1) ---
        kv = mem_tokens.reshape(B, T * M, D)  # [B,T*M,D]

        # --- Block-lower-triangular time mask (causality over time) ---
        # Query times repeated per pixel, Key times repeated per memory token
        q_times = torch.arange(1, T, device=device).repeat_interleave(
            self.Nq
        )  # [(T-1)*Nq]
        k_times = torch.arange(0, T, device=device).repeat_interleave(M)  # [T*M]
        # Allow attendance only if key_time <= query_time-1
        allow = k_times.view(1, 1, 1, T * M) <= (
            q_times.view(1, 1, (T - 1) * self.Nq, 1) - 1
        )
        # Broadcast to batch; mask shape: [B,1,(T-1)*Nq,T*M]
        attn_mask = allow.expand(B, 1, (T - 1) * self.Nq, T * M)

        # --- Single cross-attention call for all times at once ---
        y = self.cross(q, kv, kv, attn_mask=attn_mask)  # [B,(T-1)*Nq,D]
        y = y.view(B, T - 1, self.Nq, D).view(
            B * (T - 1), self.H, self.W, D
        )  # [B*(T-1),H,W,D]

        # --- Optional spatial refine per predicted frame (vectorized over B*(T-1)) ---
        y = self.refine(y)  # [B*(T-1),H,W,D]

        # --- Per-pixel 256-way logits ---
        logits = self.head(y.reshape(B * (T - 1) * self.H * self.W, D)).reshape(
            B, T - 1, self.H, self.W, 256
        )
        return logits


class TACA(nn.Module):
    """
    Vectorized TACA:
      - FrameEncoder processes all frames in parallel
      - MemoryCompressor pools all frames in parallel
      - DecoderVectorized predicts all next frames in a single cross-attention
    """

    def __init__(
        self,
        H: int,
        W: int,
        quant_levels: int = 256,
        dim: int = 192,
        heads: int = 8,
        enc_layers: int = 4,
        pool: tuple[int, int] = (4, 4),
        dropout: float = 0.0,
        max_T: int = 400,
    ):
        super().__init__()
        self.H, self.W = H, W
        self.embed = nn.Embedding(quant_levels, dim)
        self.encoder = FrameEncoder(H, W, dim, quant_levels, heads, enc_layers, dropout)
        self.compressor = MemoryCompressor(H, W, dim, pool)
        self.decoder = DecoderVectorized(
            H,
            W,
            dim,
            heads,
            dropout,
            max_T,
            quant_levels=quant_levels,
        )

    def forward_old(self, frames, embeds=None):
        """
        frames: [B,T,H,W] uint8/long
        Returns logits for t=1..T-1 stacked: [B,T-1,H,W,256]
        """
        frames = frames.permute(0, 3, 1, 2)
        B, T, H, W = frames.shape

        assert H == self.H and W == self.W

        if embeds is None:
            embeds = self.embed(frames)  # [B,T,H,W,D]
        else:
            embeds = embeds.permute(0, 3, 1, 2, 4)

        logits_list = []
        mem_list = []
        # loop over time, build memory from encoded frames <= t
        for t in range(T):  # last index T-2 used to predict T-1
            xt = embeds[:, t]  # [B,H,W]
            feats = self.encoder(xt)  # [B,H,W,D]
            mem_t = self.compressor(feats)  # [B,M,D]
            mem_list.append(mem_t)
            # concatenate memory from frames <= t
            mem = torch.cat(mem_list, dim=1)  # [B, (t+1)*M, D]
            # predict next frame logits at time t+1
            logits_t1 = self.decoder(mem, t_index=t + 1, batch_size=B)  # [B,H,W,256]
            logits_list.append(logits_t1.unsqueeze(1))
        logits = torch.cat(logits_list, dim=1)  # [B,T-1,H,W,256]
        return logits.permute(0, 2, 3, 1, 4).contiguous()  # [B,H,W,T-1,256]

    def forward(
        self, frames: torch.Tensor, embeds: torch.Tensor = None
    ) -> torch.Tensor:  # [B,T,H,W] (long)
        frames = frames.permute(0, 3, 1, 2)
        B, T, H, W = frames.shape

        assert H == self.H and W == self.W

        if embeds is None:
            embeds = self.embed(frames)  # [B,T,H,W,D]
        else:
            embeds = embeds.permute(0, 3, 1, 2, 4)

        # Encode all frames at once
        feats = self.encoder(embeds)  # [B,T,H,W,D]
        # Pool to memory for all frames at once
        mem = self.compressor(feats)  # [B,T,M,D]
        # Predict logits for frames 1..T-1 in a single pass
        logits = self.decoder(mem)  # [B,T-1,H,W,256]
        return logits.permute(0, 2, 3, 1, 4).contiguous()  # [B,H,W,T-1,256]

    @torch.no_grad()
    def generate(self, init_frames: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Autoregressive sampling is inherently sequential (kept as a loop).
        """
        self.eval()
        B, T0, H, W = init_frames.shape
        device = next(self.parameters()).device
        frames = init_frames.to(device)
        # Build memory from context
        feats0 = self.encoder(frames)  # [B,T0,H,W,D]
        mem = self.compressor(feats0)  # [B,T0,M,D]
        outs = []
        for s in range(steps):
            # Decoder expects mem over all frames so far;
            # we call it with T = current length
            logits = self.decoder(
                mem
            )  # predicts for times 1..Tcur-1; take the *last* prediction
            last = logits[:, -1]  # [B,H,W,256]
            probs = torch.softmax(last, dim=-1)
            flat = probs.view(B, -1, 256)
            sample = (
                torch.multinomial(flat.reshape(B * -1, 256), 1)
                .squeeze(-1)
                .view(B, H, W)
                .long()
            )
            outs.append(sample.unsqueeze(1))
            # Append to memory
            feats_new = self.encoder(sample.unsqueeze(1))  # [B,1,H,W,D]
            mem_new = self.compressor(feats_new)  # [B,1,M,D]
            mem = torch.cat([mem, mem_new], dim=1)
        return torch.cat(outs, dim=1)


# -----------------------------
# Synthetic dataset
# -----------------------------


class MovingShapesDataset(Dataset):
    """
    Generates sequences of moving squares on a black background.
    Each sequence: [T,H,W] with values in {0..255}.
    """

    def __init__(
        self,
        num_sequences=1000,
        T=16,
        H=32,
        W=32,
        num_objects=2,
        max_speed=2,
        square_size=4,
        seed=0,
    ):
        super().__init__()
        self.T, self.H, self.W = T, H, W
        self.data = []
        rng = np.random.RandomState(seed)
        for _ in range(num_sequences):
            frames = np.zeros((T, H, W), dtype=np.uint8)
            # initialize objects
            objs = []
            for i in range(num_objects):
                x = rng.randint(0, H - square_size)
                y = rng.randint(0, W - square_size)
                vx = rng.choice([-max_speed, -max_speed + 1, max_speed - 1, max_speed])
                vy = rng.choice([-max_speed, -max_speed + 1, max_speed - 1, max_speed])
                vx = 1 if vx == 0 else vx
                vy = 1 if vy == 0 else vy
                intensity = rng.randint(128, 256)
                objs.append([x, y, vx, vy, intensity])
            for t in range(T):
                for x, y, vx, vy, intensity in objs:
                    frames[t, x : x + square_size, y : y + square_size] = intensity
                # update positions with bounce
                for idx in range(len(objs)):
                    x, y, vx, vy, intensity = objs[idx]
                    x_new = x + vx
                    y_new = y + vy
                    if x_new < 0 or x_new + square_size > H:
                        vx = -vx
                        x_new = x + vx
                    if y_new < 0 or y_new + square_size > W:
                        vy = -vy
                        y_new = y + vy
                    objs[idx] = [x_new, y_new, vx, vy, intensity]
            self.data.append(frames)
        self.data = np.stack(self.data, axis=0)  # [N,T,H,W]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.data[idx].astype(np.int64))  # [T,H,W] long
        return seq


# -----------------------------
# Minimal training loop
# -----------------------------


def train_minimal():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # hyperparameters
    T, H, W = 12, 32, 32
    dim, heads, enc_layers = 192, 8, 4
    pool = (4, 4)
    batch_size = 8
    train_steps = 200  # keep small for demo
    lr = 3e-4

    # data
    ds = MovingShapesDataset(
        num_sequences=512, T=T, H=H, W=W, num_objects=2, square_size=4, seed=123
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # model
    model = TACA(
        H,
        W,
        dim=dim,
        heads=heads,
        enc_layers=enc_layers,
        pool=pool,
        dropout=0.1,
        max_T=1024,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    step = 0
    for epoch in range(9999):
        for batch in dl:
            batch = batch.to(device)  # [B,T,H,W]
            logits = model(batch)  # [B,T-1,H,W,256]
            targets = batch[:, 1:, :, :]  # next frames [B,T-1,H,W]
            B, Tp, Hh, Ww, Q = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * Tp * Hh * Ww, Q), targets.reshape(B * Tp * Hh * Ww)
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            if step % 20 == 0:
                print(f"step {step}: loss={loss.item():.4f}")
            step += 1
            if step >= train_steps:
                break
        if step >= train_steps:
            break

    # sampling demo
    model.eval()
    with torch.no_grad():
        batch = next(iter(dl)).to(device)
        context = batch[:, :4]  # first 4 frames as context
        sampled = model.generate(context, steps=8)  # [B,8,H,W]
        print("Generated sample shape:", sampled.shape)
        # You can visualize or save 'sampled' as needed.
    return model


if __name__ == "__main__":
    train_minimal()
