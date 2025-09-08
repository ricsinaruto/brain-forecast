# chronoflow_ssm.py
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Utils
# -------------------------


# -------------------------
# Invertible building blocks (Glow-ish)
# -------------------------


class ActNorm2d(nn.Module):
    """
    Per-channel affine transform with log-det.
    Initialized to zero-mean/unit-var on first batch.
    """

    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.initialized = False
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    @torch.no_grad()
    def _init(self, x):
        # x: [B,C,H,W]
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
        self.bias.data = -mean
        self.log_scale.data = -0.5 * torch.log(var + self.eps)
        self.initialized = True

    def forward(self, x, reverse=False):
        if not self.initialized:
            self._init(x)
        if reverse:
            y = (x - self.bias) * torch.exp(-self.log_scale)
            ldj = -self._logdet(x)
        else:
            y = x * torch.exp(self.log_scale) + self.bias
            ldj = self._logdet(x)
        return y, ldj

    def _logdet(self, x):
        B, C, H, W = x.shape
        return (self.log_scale.sum() * H * W).repeat(B)


class Invertible1x1Conv2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        # Orthonormal init with torch.linalg.qr to avoid deprecation warnings
        q, _ = torch.linalg.qr(torch.randn(num_channels, num_channels))
        self.weight = nn.Parameter(q)

    def forward(self, x, reverse=False):
        B, C, H, W = x.shape
        if reverse:
            W_inv = torch.inverse(self.weight.double()).float()
            y = F.conv2d(x, W_inv.view(C, C, 1, 1))
            ldj = -(H * W) * torch.slogdet(self.weight)[1].repeat(B)
        else:
            y = F.conv2d(x, self.weight.view(C, C, 1, 1))
            ldj = (H * W) * torch.slogdet(self.weight)[1].repeat(B)
        return y, ldj


class FiLM(nn.Module):
    def __init__(self, in_dim, cond_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, in_dim * 2),
            nn.SiLU(),
            nn.Linear(in_dim * 2, in_dim * 2),
        )

    def forward(self, h, cond):
        gamma_beta = self.mlp(cond)  # [B, 2*D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return h * (1 + gamma) + beta


class CouplingNet(nn.Module):
    """
    ConvNet used inside affine coupling. Optional FiLM conditioning.
    """

    def __init__(
        self, in_channels, hidden_channels=192, out_channels=None, cond_dim: int = 0
    ):
        super().__init__()
        out_channels = out_channels or (in_channels * 2)
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, 3, padding=1),
        )
        self.film = FiLM(hidden_channels, cond_dim) if cond_dim > 0 else None
        # tiny last layer init for stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, cond: Optional[torch.Tensor] = None):
        # Inject FiLM on the middle activation if cond is provided
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        if self.film is not None and cond is not None:
            # cond: [B, cond_dim]
            h = self.film(h, cond)
        h = self.net[3](h)
        h = self.net[4](h)
        return h


class AffineCoupling2d(nn.Module):
    """
    Affine coupling with optional conditioning (FiLM).
    Splits channels into [x_a, x_b]; transforms x_b.
    """

    def __init__(
        self, channels, hidden=192, cond_dim: int = 0, scale_clamp: float = 5.0
    ):
        super().__init__()
        assert channels >= 2, (
            "Coupling requires at least 2 channels. "
            "Make sure your FlowLevel squeezes BEFORE the first coupling."
        )
        self.scale_clamp = scale_clamp
        self.channels = channels
        self.nn = CouplingNet(
            channels // 2,
            hidden_channels=hidden,
            out_channels=channels,
            cond_dim=cond_dim,
        )

    def forward(self, x, cond: Optional[torch.Tensor] = None, reverse=False):
        xa, xb = x.chunk(2, dim=1)  # safe now (>=2 channels guaranteed)
        h = self.nn(xa, cond)
        s, t = h.chunk(2, dim=1)
        s = torch.tanh(s) * self.scale_clamp
        if reverse:
            yb = (xb - t) * torch.exp(-s)
            ldj = -s.flatten(1).sum(dim=1)
        else:
            yb = xb * torch.exp(s) + t
            ldj = s.flatten(1).sum(dim=1)
        y = torch.cat([xa, yb], dim=1)
        return y, ldj


class Squeeze2d(nn.Module):
    """
    Space-to-depth, increases channels by 4, halves H and W.
    """

    def forward(self, x, reverse=False):
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H and W must be even for squeeze"
        if not reverse:
            x = x.view(B, C, H // 2, 2, W // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(B, C * 4, H // 2, W // 2)
            ldj = torch.zeros(B, device=x.device)
            return x, ldj
        else:
            x = x.view(B, C // 4, 2, 2, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(B, C // 4, H * 2, W * 2)
            ldj = torch.zeros(B, device=x.device)
            return x, ldj


class FlowStep(nn.Module):
    def __init__(self, channels, hidden=192, cond_dim: int = 0):
        super().__init__()
        self.actnorm = ActNorm2d(channels)
        self.inv1x1 = Invertible1x1Conv2d(channels)
        self.coupling = AffineCoupling2d(channels, hidden=hidden, cond_dim=cond_dim)

    def forward(self, x, cond=None, reverse=False):
        ldj_total = torch.zeros(x.shape[0], device=x.device)
        if reverse:
            x, ldj = self.coupling(x, cond, reverse=True)
            ldj_total += ldj
            x, ldj = self.inv1x1(x, reverse=True)
            ldj_total += ldj
            x, ldj = self.actnorm(x, reverse=True)
            ldj_total += ldj
        else:
            x, ldj = self.actnorm(x, reverse=False)
            ldj_total += ldj
            x, ldj = self.inv1x1(x, reverse=False)
            ldj_total += ldj
            x, ldj = self.coupling(x, cond, reverse=False)
            ldj_total += ldj
        return x, ldj_total


class FlowLevel(nn.Module):
    """
    A level: SQUEEZE first (so channel-split coupling always sees >=2 channels),
    then K flow steps. Reverse: undo steps, then UNSQUEEZE.
    """

    def __init__(self, channels, K=4, hidden=192, cond_dim: int = 0, do_squeeze=True):
        super().__init__()
        self.do_squeeze = do_squeeze
        self.squeeze = Squeeze2d() if do_squeeze else None

        # After squeeze, channels are multiplied by 4
        channels_after = channels * (4 if do_squeeze else 1)
        self.steps = nn.ModuleList(
            [
                FlowStep(channels_after, hidden=hidden, cond_dim=cond_dim)
                for _ in range(K)
            ]
        )

    def forward(self, x, cond=None, reverse=False):
        ldj_total = torch.zeros(x.shape[0], device=x.device)

        if reverse:
            # Reverse of (SQUEEZE -> STEPS) is (reverse STEPS -> UNSQUEEZE)
            for step in reversed(self.steps):
                x, ldj = step(x, cond, reverse=True)
                ldj_total += ldj
            if self.do_squeeze:
                x, ldj = self.squeeze(x, reverse=True)
                ldj_total += ldj
        else:
            # Forward: SQUEEZE first, then steps
            if self.do_squeeze:
                x, ldj = self.squeeze(x, reverse=False)
                ldj_total += ldj
            for step in self.steps:
                x, ldj = step(x, cond, reverse=False)
                ldj_total += ldj

        return x, ldj_total


class SpatialFlow(nn.Module):
    """
    g_theta: Invertible per-frame flow (levels with squeeze-first).
    """

    def __init__(self, in_channels=3, levels=3, steps_per_level=4, hidden=192):
        super().__init__()
        ch = in_channels
        levs = []
        for _ in range(levels):
            levs.append(
                FlowLevel(
                    ch, K=steps_per_level, hidden=hidden, cond_dim=0, do_squeeze=True
                )
            )
            ch = ch * 4  # after squeeze in that level
        # final level without squeeze
        levs.append(
            FlowLevel(
                ch, K=steps_per_level, hidden=hidden, cond_dim=0, do_squeeze=False
            )
        )
        self.levels = nn.ModuleList(levs)

    def forward(self, x, reverse=False):
        ldj_total = torch.zeros(x.shape[0], device=x.device)
        if reverse:
            for level in reversed(self.levels):
                x, ldj = level(x, cond=None, reverse=True)
                ldj_total += ldj
        else:
            for level in self.levels:
                x, ldj = level(x, cond=None, reverse=False)
                ldj_total += ldj
        return x, ldj_total


# -------------------------
# Conditional emission flow (h_phi)
# -------------------------


class ConditionalFlow(nn.Module):
    """
    Shape-preserving conditional flow: no internal squeezes.
    Forward(reverse=False): z -> u
    Forward(reverse=True):  u -> z, also returns log|det J|
    """

    def __init__(self, channels, levels=2, steps_per_level=4, hidden=192, cond_dim=512):
        super().__init__()
        assert channels >= 2, "ConditionalFlow requires at least 2 channels."
        # Just stack 'levels' groups; each group has 'steps_per_level'
        # FlowSteps, no squeeze.
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        FlowStep(channels, hidden=hidden, cond_dim=cond_dim)
                        for _ in range(steps_per_level)
                    ]
                )
                for _ in range(levels)
            ]
        )

    def forward(self, x, cond, reverse=False):
        ldj_total = torch.zeros(x.shape[0], device=x.device)
        if reverse:
            # Inverse: reverse order of all steps in all blocks
            for block in reversed(self.blocks):
                for step in reversed(block):
                    x, ldj = step(x, cond, reverse=True)
                    ldj_total += ldj
            return x, ldj_total
        else:
            # Forward: z -> u
            for block in self.blocks:
                for step in block:
                    x, ldj = step(x, cond, reverse=False)
                    ldj_total += ldj
            return x, ldj_total

    def log_prob(self, u, cond):
        z, ldj = self.forward(u, cond, reverse=True)
        logpz = -0.5 * (z**2 + math.log(2 * math.pi)).flatten(1).sum(dim=1)
        return logpz + ldj

    def sample(self, shape, cond):
        z = torch.randn(shape, device=cond.device)
        u, _ = self.forward(z, cond, reverse=False)
        return u


# -------------------------
# Hierarchical Selective SSM (minutes-scale memory)
# -------------------------


class SelectiveSSMLevel(nn.Module):
    """
    Lightweight, stable, gated state-space update:
      s_t = (1-g)*s_{t-1} + g*( d*s_{t-1} + W_in * inp + W_td * td )
    where d in (0,1) is learned (per-dim decay), g is a gate in (0,1).
    Updates happen every 'stride' steps; otherwise state is carried forward.
    """

    def __init__(self, inp_dim, td_dim, state_dim, stride: int):
        super().__init__()
        self.stride = stride
        self.td_dim = td_dim
        self.inp_proj = nn.Linear(inp_dim, state_dim, bias=False)
        self.td_proj = nn.Linear(td_dim, state_dim, bias=False) if td_dim > 0 else None
        self.decay = nn.Parameter(torch.zeros(state_dim))  # sigmoid -> (0,1)
        self.gate = nn.Sequential(
            nn.Linear(inp_dim + td_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(self, s_prev, inp, td, step_idx, boundary_scalar=None):
        # s_prev: [B, D], inp: [B, inp_dim], td: [B, td_dim] or None
        if (step_idx % self.stride) != 0:
            return s_prev  # no update on this step
        d = torch.sigmoid(self.decay)  # [D]
        base = d * s_prev + self.inp_proj(inp)
        if self.td_proj is not None and td is not None:
            base = base + self.td_proj(td)
        # gate
        cat = torch.cat([inp, td] if td is not None else [inp], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        if boundary_scalar is not None:
            # sharpen gates on events
            g = torch.clamp(g + boundary_scalar.unsqueeze(-1), 0.0, 1.0)
        s = (1 - g) * s_prev + g * base
        return s


class TemporalBackbone(nn.Module):
    """
    Multi-level selective SSM with strides [1,2,4,...].
    Aggregates states -> conditioning vector for emission flow.
    """

    def __init__(
        self,
        u_channels,
        u_spatial_hw: Tuple[int, int],
        levels=4,
        state_dim=512,
        cond_dim=512,
    ):
        super().__init__()
        H, W = u_spatial_hw
        self.levels = levels
        self.state_dim = state_dim
        self.cond_dim = cond_dim

        inp_dim = u_channels  # we will use global avgpool over HxW, so channels suffice
        td_dim = state_dim  # top-down comes from previous (slower) level

        self.down = nn.Sequential(  # embed u -> inp vector
            nn.Conv2d(u_channels, u_channels, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.event_head = nn.Sequential(
            nn.Linear(inp_dim, 128), nn.SiLU(), nn.Linear(128, 1)
        )

        self.level_modules = nn.ModuleList()
        strides = [2**i for i in range(levels)]  # e.g., 1,2,4,8,...
        for i in range(levels):
            self.level_modules.append(
                SelectiveSSMLevel(
                    inp_dim,
                    td_dim if i < levels - 1 else 0,
                    state_dim,
                    stride=strides[i],
                )
            )

        self.agg = nn.Sequential(
            nn.Linear(levels * state_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.inp_dim = inp_dim

    def init_state(self, batch_size, device):
        return [
            torch.zeros(batch_size, self.state_dim, device=device)
            for _ in range(self.levels)
        ]

    def forward(
        self, states: List[torch.Tensor], u_prev: Optional[torch.Tensor], step_idx: int
    ):
        """
        states: list of [B, D]
        u_prev: latent of previous frame [B, C, H, W] (or None for t=0)
        step_idx: int
        Returns:
          cond: [B, cond_dim], new_states, boundary_prob: [B,1]
        """
        B = states[0].shape[0]
        if u_prev is None:
            inp_vec = torch.zeros(B, self.inp_dim, device=states[0].device)
            boundary = torch.zeros(B, 1, device=states[0].device)
        else:
            h = self.down(u_prev)  # [B, C, 1, 1]
            inp_vec = h.view(B, -1)
            boundary = torch.sigmoid(self.event_head(inp_vec))  # [B,1]

        # top-down: slower level = higher index
        new_states: List[torch.Tensor] = []
        td = None
        for i in reversed(range(self.levels)):
            s = self.level_modules[i](
                states[i], inp_vec, td, step_idx, boundary_scalar=boundary.squeeze(1)
            )
            new_states.insert(0, s)
            td = s  # next faster level receives this as top-down

        # aggregate
        cat = torch.cat(new_states, dim=-1)
        cond = self.agg(cat)  # [B, cond_dim]
        return cond, new_states, boundary
