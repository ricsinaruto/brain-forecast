# nsr_brain_nextstep.py
# ------------------------------------------------------------
# NeuroSSM-Reasoner (NSR) core, ready for next-step prediction
# on brain signals (EEG/MEG/ECoG) with spectral + time losses.
# ------------------------------------------------------------
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


@dataclass
class TrainConfig:
    seq_len: int = 1024
    batch_size: int = 8
    hidden_dim: int = 256
    num_layers: int = 4
    micro_steps: int = 2
    n_fft: int = 256
    band_weight: float = 0.5


class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm (no bias)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., d)
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.scale * x


class OscSSMCell(nn.Module):
    """
    Oscillatory diagonal state-space cell using 2x2 blocks per mode.
    Continuous-time A_c block per mode i:
        A_i = [[-alpha_i, -omega_i],
               [ omega_i, -alpha_i]]
    Discrete-time with dt:
        A_d = exp(-alpha_i*dt) * R(theta), theta = omega_i * dt
    Input coupling is learned and modulated by a tiny gate network.

    Args
    ----
    input_dim:   channels in the input u_t
    hidden_dim:  even number (2 * n_modes)
    dt:          time step (relative units; keep =1.0 for indexed samples)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dt: float = 1.0):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even (2x2 oscillatory blocks)."
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_modes = hidden_dim // 2
        self.dt = dt

        # Positive damping (alpha) via softplus; omega unconstrained (radians / sample)
        self.log_alpha = nn.Parameter(torch.zeros(self.n_modes))
        self.omega = nn.Parameter(0.10 * torch.randn(self.n_modes))

        # Input coupling and bias (learned)
        # B maps u_t->[hidden_dim]; initialized like a 1x1 conv
        self.B = nn.Parameter(torch.randn(input_dim, hidden_dim) / math.sqrt(input_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        # Input-dependent modulation (Mamba-like selectivity, lightweight)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),  # bounded modulation in [-1, 1]
        )

        # A little stabilization on the hidden
        self.hnorm = RMSNorm(hidden_dim)

    def forward(self, u_t: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """
        One micro-step update.
        u_t: (B, input_dim)
        x_t: (B, hidden_dim)
        returns x_{t+1}: (B, hidden_dim)
        """
        B = u_t.shape[0]
        n = self.n_modes

        # Precompute decay and rotation per mode
        alpha = F.softplus(self.log_alpha)  # (n,)
        theta = self.omega * self.dt  # (n,)
        decay = torch.exp(-alpha * self.dt)  # (n,)
        cos_t = torch.cos(theta)  # (n,)
        sin_t = torch.sin(theta)  # (n,)

        # Apply block-wise A_d to x_t; view into pairs
        x = x_t.view(B, n, 2)  # (B, n, 2)
        x1 = x[..., 0]  # (B, n)
        x2 = x[..., 1]  # (B, n)

        x1p = decay * (cos_t * x1 - sin_t * x2)
        x2p = decay * (sin_t * x1 + cos_t * x2)
        x_next = torch.stack([x1p, x2p], dim=-1).reshape(B, self.hidden_dim)

        # Input injection with selective modulation
        mod = self.gate(u_t)  # (B, hidden_dim) in [-1, 1]
        Bu = F.linear(u_t, self.B.t())  # (B, hidden_dim)

        x_next = x_next + (1.0 + mod) * Bu + self.bias
        x_next = self.hnorm(F.silu(x_next))
        return x_next


class StateSpaceBlock(nn.Module):
    """
    Residual block that runs an OscSSMCell for K micro-steps per input sample.

    u_t -> (K micro-steps) -> x_{t+1} -> project back to input space -> residual
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.10,
        micro_steps: int = 1,
    ):
        super().__init__()
        self.micro_steps = micro_steps
        self.cell = OscSSMCell(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.res_norm = RMSNorm(input_dim)

    def forward(
        self, u_t: torch.Tensor, h_t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        u_t: (B, input_dim)       -- current sample
        h_t: (B, hidden_dim)      -- block hidden state
        returns:
          y_t: (B, input_dim)     -- block output (residual to next layer)
          h_next: (B, hidden_dim)
        """
        x = h_t
        # Same u_t injected at each micro-step (cheap but effective)
        for _ in range(self.micro_steps):
            x = self.cell(u_t, x)
        y = self.out_proj(x)
        y = self.dropout(y)
        y = self.res_norm(u_t + y)  # residual to input space
        return y, x


class NSR(nn.Module):
    """
    Multi-layer oscillatory SSM network for sequence modeling.
    Next-step prediction: given x[:, :-1, :], predict x[:, 1:, :]

    Args
    ----
    input_dim:   channels in the brain signal
    hidden_dim:  size of SSM latent (must be even)
    num_layers:  number of SSM blocks (stacked)
    micro_steps: micro-steps per sample in each block
    dropout:     dropout inside blocks
    out_dim:     defaults to input_dim (predict next sample for each channel)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        micro_steps: int = 1,
        dropout: float = 0.0,
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        assert hidden_dim % 2 == 0, "hidden_dim must be even."

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim or input_dim

        self.blocks = nn.ModuleList(
            [
                StateSpaceBlock(
                    input_dim, hidden_dim, dropout=dropout, micro_steps=micro_steps
                )
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Linear(input_dim, self.out_dim)

    def forward(
        self,
        x: torch.Tensor,
        h: Optional[List[torch.Tensor]] = None,
        return_states: bool = False,
    ):
        """
        x: (B, T, C_in)
        h: optional list of hidden states per layer, each (B, H)
        return:
          yseq: (B, T, C_out)
          h:    updated hidden states
          (optional) states_over_time: (B, T, L, H)
        """
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.permute(0, 2, 1).contiguous()  # (B, C_in, T)

        B, T, C = x.shape
        if h is None:
            h = [x.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)]

        outs = []
        states = []
        for t in range(T):
            u = x[:, t, :]  # (B, C_in)
            for i, block in enumerate(self.blocks):
                u, h[i] = block(u, h[i])
            y = self.readout(u)  # (B, C_out)
            outs.append(y.unsqueeze(1))
            if return_states:
                states.append(torch.stack(h, dim=1).unsqueeze(1))  # (B, 1, L, H)

        yseq = torch.cat(outs, dim=1)  # (B, T, C_out)
        if return_states:
            st = torch.cat(states, dim=1)  # (B, T, L, H)
            return yseq, h, st
        return yseq.permute(0, 2, 1).contiguous()  # (B, C_out, T)

    def step(
        self,
        u: torch.Tensor,
        h: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run a single causal update step of the NSR stack.

        Args
        ----
        u: (B, C_in)
            Current input sample.
        h: Optional list of hidden states per layer, each of shape (B, hidden_dim).
           If None, initializes zeros.

        Returns
        -------
        y: (B, C_out)
            Next-step prediction.
        h: list[Tensor]
            Updated hidden states.
        """
        B, _ = u.shape
        if h is None:
            h = [u.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)]

        for i, block in enumerate(self.blocks):
            u, h[i] = block(u, h[i])
        y = self.readout(u)
        return y, h

    @torch.no_grad()
    def forecast(
        self,
        x_context: torch.Tensor,
        steps: int,
        h: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Recursive free-running generation beyond a context.

        The model is assumed to have been trained for next-step prediction:
            x_t  ->  x_{t+1}

        Semantics:
          - We first run over the context with teacher forcing to warm up the hidden
            states.
          - Let T_ctx = context length.
          - The first generated point x_{T_ctx} is the prediction from the LAST
            context input x_{T_ctx-1}.
          - Subsequent points are generated autoregressively from the model's own
            outputs.

        Args
        ----
        x_context: (B, T_ctx, C_in) or (T_ctx, C_in)
            Observed context sequence.
        steps: int
            Number of future steps to generate *beyond* the context.
            steps = 0 returns an empty tensor of shape (B, 0, C_out).
        h: optional list[Tensor]
            Initial hidden states per layer, each (B, hidden_dim).
            If None, initializes zeros and updates while scanning the context.

        Returns
        -------
        y_future: (B, steps, C_out)  if input was batched
                  (steps, C_out)     if input was (T_ctx, C_in)
        """
        if steps < 0:
            raise ValueError("steps must be non-negative")

        if isinstance(x_context, (list, tuple)):
            x_context = x_context[0]

        squeeze_batch = False
        if x_context.ndim == 2:
            # (T_ctx, C) -> (1, T_ctx, C)
            x_context = x_context.unsqueeze(0)
            squeeze_batch = True
        elif x_context.ndim != 3:
            raise ValueError(
                f"x_context must be (B, T, C) or (T, C), got shape {x_context.shape}"
            )

        B, T_ctx, _ = x_context.shape

        # Preserve training/eval mode and disable dropout during generation
        was_training = self.training
        self.eval()

        # Initialize hidden states if needed
        if h is None:
            h = [
                x_context.new_zeros(B, self.hidden_dim) for _ in range(self.num_layers)
            ]

        # 1) Warm-up: scan the context with teacher forcing
        last_y = None
        for t in range(T_ctx):
            u_t = x_context[:, t, :]  # ground-truth input
            last_y, h = self.step(u_t, h)

        # If no future steps requested, bail out early
        if steps == 0:
            if was_training:
                self.train()
            out = x_context.new_zeros(B, 0, self.out_dim)
            return out.squeeze(0) if squeeze_batch else out

        # 2) First future step: x_{T_ctx} predicted from last context input
        preds = [last_y.unsqueeze(1)]  # (B, 1, C_out)
        u = last_y

        # 3) Subsequent steps: purely autoregressive
        for _ in tqdm(range(steps - 1), desc="Generating future steps"):
            u, h = self.step(u, h)  # feed previous prediction
            preds.append(u.unsqueeze(1))

        y_future = torch.cat(preds, dim=1)  # (B, steps, C_out)

        if was_training:
            self.train()

        if squeeze_batch:
            y_future = y_future.squeeze(0)  # (steps, C_out)

        return y_future.transpose(-1, -2).contiguous()  # (B, C_out, steps)
