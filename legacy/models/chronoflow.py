import math
import os
from dataclasses import dataclass
from typing import Tuple, Mapping, Optional, List
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Utilities: logit transform for continuous frames in [0,1]
# -------------------------


def _recover_discrete_values(
    x: torch.Tensor, quant_levels: Optional[int]
) -> torch.Tensor:
    if quant_levels is None:
        return x
    # Dataset stores (v + 1)/(Q + 1); invert back to v ∈ {0,...,Q-1}
    q = float(quant_levels)
    x_disc = torch.round(x * (q + 1.0) - 1.0)
    return x_disc.clamp_(0.0, q - 1.0)


def dequantize_and_logit(
    x: torch.Tensor, alpha: float = 1e-6, quant_levels: Optional[int] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optional uniform dequantization (if quant_levels is provided) followed by logit.
    Returns (y, log|det d y / d x|) per-sample.
    """
    if not (0.0 <= alpha < 0.5):
        raise ValueError("alpha must be in [0, 0.5) for a valid logit transform")

    x = x.clamp(0.0, 1.0)
    if quant_levels is not None and quant_levels > 0:
        x_disc = _recover_discrete_values(x, quant_levels)
        x = (x_disc + torch.rand_like(x_disc)) / float(quant_levels)

    x = (x * (1 - 2 * alpha)) + alpha
    x = x.clamp(alpha, 1.0 - alpha)
    y = torch.log(x) - torch.log1p(-x)

    per_elem_ldj = F.softplus(-y) + F.softplus(y)
    logdet = per_elem_ldj.flatten(1).sum(dim=1)
    if alpha > 0:
        numel = x.shape[1] * x.shape[2] * x.shape[3]
        logdet = logdet + x.new_tensor(numel * math.log1p(-2.0 * alpha))
    return y, logdet


def logit_inverse(
    y: torch.Tensor, alpha: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of the above: returns (x, log|det dx/dy|)."""
    if not (0.0 <= alpha < 0.5):
        raise ValueError("alpha must be in [0, 0.5) for a valid logit transform")

    x_scaled = torch.sigmoid(y)
    logdet = -(F.softplus(-y) + F.softplus(y)).flatten(1).sum(dim=1)
    if alpha > 0:
        numel = y.shape[1] * y.shape[2] * y.shape[3]
        logdet = logdet - y.new_tensor(numel * math.log1p(-2.0 * alpha))
    x = (x_scaled - alpha) / (1 - 2 * alpha)
    return x.clamp(0.0, 1.0), logdet


# -------------------------
# Invertible building blocks (Glow-ish)
# -------------------------


class ActNorm2d(nn.Module):
    """
    Per-channel affine transform with log-det.
    Initialized to zero-mean/unit-var on the first batch.
    """

    def __init__(
        self, num_channels: int, eps: float = 1e-5, max_log_scale: float = 3.0
    ):
        super().__init__()
        self.initialized = False
        self.eps = eps
        self.max_log_scale = max_log_scale
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def _centered_log_scale(self) -> torch.Tensor:
        log_scale = self.log_scale
        return log_scale - log_scale.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def _init(self, x: torch.Tensor) -> None:
        # x: [B,C,H,W]
        mean = x.mean(dim=[0, 2, 3], keepdim=True)
        var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
        self.bias.data = -mean
        # scale so that std becomes 1 -> log_scale = -log(std),
        # but clamp to avoid huge gains
        log_scale = -0.5 * torch.log(var.clamp_min(self.eps))
        if self.max_log_scale is not None:
            log_scale = log_scale.clamp(-self.max_log_scale, self.max_log_scale)
        # Remove global scale bias so ActNorm doesn't introduce huge log-dets.
        log_scale = log_scale - log_scale.mean(dim=1, keepdim=True)
        self.log_scale.data = log_scale

    def _logdet(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # log|det J| = sum_c log_scale_c * H * W
        centered = self._centered_log_scale()
        per_sample = (centered.sum() * H * W).expand(B)
        return per_sample

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.initialized:
            self._init(x)
            self.initialized = True
        log_scale = self._centered_log_scale()
        if reverse:
            y = (x - self.bias) * torch.exp(-log_scale)
            ldj = -self._logdet(x)
        else:
            y = x * torch.exp(log_scale) + self.bias
            ldj = self._logdet(x)
        return y, ldj


class Invertible1x1Conv2d(nn.Module):
    """
    1x1 invertible convolution with LU parameterization (Glow-style).

    W is parameterized as P @ L @ U, where:
      - P is a fixed permutation matrix (buffer)
      - L has ones on its diagonal (stored as strictly lower + I)
      - U has its diagonal stored separately via log_s and sign_s
    This makes log|det W| = sum(log_s) cheap and stable.
    """

    def __init__(self, num_channels: int):
        super().__init__()
        w_shape = (num_channels, num_channels)

        # random orthonormal init with positive determinant
        w_init = torch.randn(w_shape)
        q, _ = torch.linalg.qr(w_init)
        if torch.det(q) < 0:
            q[:, 0] *= -1

        # LU factorization: q = P @ L @ U
        try:
            P, L, U = torch.linalg.lu(q)
        except AttributeError:
            # fallback for older PyTorch
            P, L, U = torch.lu(q)

        s = torch.diag(U)
        sign_s = torch.sign(s)
        log_s = torch.log(torch.abs(s))

        # store strictly lower/upper parts; diag handled via sign_s/log_s
        L = torch.tril(L, -1)
        U = torch.triu(U, 1)

        self.register_buffer("P", P)
        self.register_buffer("sign_s", sign_s)
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        self.log_s = nn.Parameter(log_s)

    def _get_weight(self) -> torch.Tensor:
        C = self.L.shape[0]
        device = self.L.device
        dtype = self.L.dtype

        L = self.L + torch.eye(C, device=device, dtype=dtype)
        U = self.U + torch.diag(self.sign_s * torch.exp(self.log_s))
        W = self.P @ L @ U
        return W

    def _apply_linear(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        B, C, H, W_spatial = x.shape
        compute_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        x_mat = x.permute(1, 0, 2, 3).reshape(C, -1)
        if x_mat.dtype != compute_dtype:
            x_mat = x_mat.to(compute_dtype)
        W = W.to(compute_dtype)
        y_mat = W @ x_mat
        y = y_mat.view(C, B, H, W_spatial).permute(1, 0, 2, 3).contiguous()
        return y.to(x.dtype)

    def _apply_inverse(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        B, C, H, W_spatial = x.shape
        compute_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        x_mat = x.permute(1, 0, 2, 3).reshape(C, -1)
        if x_mat.dtype != compute_dtype:
            x_mat = x_mat.to(compute_dtype)
        W = W.to(compute_dtype)
        y_mat = torch.linalg.solve(W, x_mat)
        y = y_mat.view(C, B, H, W_spatial).permute(1, 0, 2, 3).contiguous()
        return y.to(x.dtype)

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W_spatial = x.shape
        W = self._get_weight()

        hw = H * W_spatial
        if reverse:
            y = self._apply_inverse(x, W)
            logdet = -hw * self.log_s.sum()
        else:
            y = self._apply_linear(x, W)
            logdet = hw * self.log_s.sum()

        ldj = logdet.to(x.dtype).expand(B)
        return y, ldj


class FiLM(nn.Module):
    """Simple FiLM conditioning: h * (1+gamma) + beta"""

    def __init__(self, in_dim: int, cond_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 2 * in_dim),
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(cond).chunk(2, dim=-1)  # [B,D],[B,D]
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return h * (1 + gamma) + beta


class CouplingNet(nn.Module):
    """
    ConvNet used inside affine coupling. Optional FiLM conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 192,
        out_channels: Optional[int] = None,
        cond_dim: int = 0,
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
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # inject FiLM on the middle activation if conditioning is provided
        h = self.net[0](x)
        h = self.net[1](h)
        h = self.net[2](h)
        if self.film is not None and cond is not None:
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
        self,
        channels: int,
        hidden: int = 192,
        cond_dim: int = 0,
        scale_clamp: float = 5.0,
    ):
        super().__init__()
        assert channels >= 2, "Coupling requires at least 2 channels."
        self.scale_clamp = scale_clamp
        self.nn = CouplingNet(
            channels // 2,
            hidden_channels=hidden,
            out_channels=channels,
            cond_dim=cond_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xa, xb = x.chunk(2, dim=1)
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
    """Space-to-depth: increases channels by 4, halves H and W."""

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    def __init__(self, channels: int, hidden: int = 192, cond_dim: int = 0):
        super().__init__()
        self.actnorm = ActNorm2d(channels)
        self.inv1x1 = Invertible1x1Conv2d(channels)
        self.coupling = AffineCoupling2d(channels, hidden=hidden, cond_dim=cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(
        self,
        channels: int,
        K: int = 4,
        hidden: int = 192,
        cond_dim: int = 0,
        do_squeeze: bool = True,
    ):
        super().__init__()
        self.do_squeeze = do_squeeze
        self.squeeze = Squeeze2d() if do_squeeze else None
        ch = channels * (4 if do_squeeze else 1)
        self.steps = nn.ModuleList(
            [FlowStep(ch, hidden=hidden, cond_dim=cond_dim) for _ in range(K)]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ldj_total = torch.zeros(x.shape[0], device=x.device)
        if reverse:
            for step in reversed(self.steps):
                x, ldj = step(x, cond, reverse=True)
                ldj_total += ldj
            if self.do_squeeze:
                x, ldj = self.squeeze(x, reverse=True)
                ldj_total += ldj
        else:
            if self.do_squeeze:
                x, ldj = self.squeeze(x, reverse=False)
                ldj_total += ldj
            for step in self.steps:
                x, ldj = step(x, cond, reverse=False)
                ldj_total += ldj
        return x, ldj_total


# -------------------------
# Spatial per-frame flow (g_theta)
# -------------------------


class SpatialFlow(nn.Module):
    """
    g_theta: Invertible per-frame flow (levels with squeeze-first).
    """

    def __init__(
        self,
        in_channels: int = 3,
        levels: int = 3,
        steps_per_level: int = 4,
        hidden: int = 192,
    ):
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
        # terminal level with no additional squeeze (keeps latent spatial HW)
        levs.append(
            FlowLevel(
                ch, K=steps_per_level, hidden=hidden, cond_dim=0, do_squeeze=False
            )
        )
        self.levels = nn.ModuleList(levs)

    def forward(
        self, x: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __init__(
        self,
        channels: int,
        levels: int = 2,
        steps_per_level: int = 4,
        hidden: int = 192,
        cond_dim: int = 512,
        base_std: float = 1.0,
    ):
        super().__init__()
        assert channels >= 2, "ConditionalFlow requires at least 2 channels."
        if base_std <= 0:
            raise ValueError("base_std must be positive")
        self.base_std = float(base_std)
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

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ldj_total = torch.zeros(x.shape[0], device=x.device)
        if reverse:
            for block in reversed(self.blocks):
                for step in reversed(block):
                    x, ldj = step(x, cond, reverse=True)
                    ldj_total += ldj
        else:
            for block in self.blocks:
                for step in block:
                    x, ldj = step(x, cond, reverse=False)
                    ldj_total += ldj
        return x, ldj_total

    def log_prob(self, u: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        z, ldj = self.forward(u, cond, reverse=True)
        z_scaled = z / self.base_std
        log_base = math.log(2 * math.pi * (self.base_std**2))
        logpz = -0.5 * (z_scaled**2 + log_base).flatten(1).sum(dim=1)
        return logpz + ldj

    def sample(
        self,
        shape: tuple[int, int, int, int],
        cond: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        z = torch.randn(shape, device=cond.device) * float(temperature) * self.base_std
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

    def __init__(self, inp_dim: int, td_dim: int, state_dim: int, stride: int):
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

    def forward(
        self,
        s_prev: torch.Tensor,
        inp: torch.Tensor,
        td: Optional[torch.Tensor],
        step_idx: int,
        boundary_scalar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # s_prev: [B, D], inp: [B, inp_dim], td: [B, td_dim] or None
        if (step_idx % self.stride) != 0:
            return s_prev  # carry forward
        d = torch.sigmoid(self.decay)  # [D] in (0,1)
        base = s_prev * d + self.inp_proj(inp)
        if self.td_proj is not None and td is not None:
            base = base + self.td_proj(td)
        # gate depends on both input and top-down; add optional boundary bias
        if td is None:
            gate_in = torch.cat(
                [inp, torch.zeros(inp.shape[0], 0, device=inp.device)], dim=-1
            )
        else:
            gate_in = torch.cat([inp, td], dim=-1)
        g = torch.sigmoid(self.gate(gate_in))
        if boundary_scalar is not None:
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
        u_channels: int,
        u_spatial_hw: Tuple[int, int],
        levels: int = 4,
        state_dim: int = 512,
        cond_dim: int = 512,
    ):
        super().__init__()
        H, W = u_spatial_hw
        self.levels = levels
        self.state_dim = state_dim
        self.cond_dim = cond_dim

        inp_dim = u_channels  # we will use global avgpool over HxW, so channels suffice
        td_dim = state_dim  # top-down comes from previous (slower) level

        self.down = nn.Sequential(
            nn.Conv2d(u_channels, u_channels, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),  # [B,C,1,1]
        )

        self.event_head = nn.Sequential(  # boundary detector (optional)
            nn.Linear(inp_dim, 128), nn.SiLU(), nn.Linear(128, 1)
        )

        self.level_modules = nn.ModuleList()
        strides = [2**i for i in range(levels)]  # e.g. 1,2,4,8,...
        for i in range(levels):
            self.level_modules.append(
                SelectiveSSMLevel(
                    inp_dim=inp_dim,
                    td_dim=(td_dim if i < levels - 1 else 0),
                    state_dim=state_dim,
                    stride=strides[i],
                )
            )

        self.agg = nn.Sequential(
            nn.Linear(levels * state_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.inp_dim = inp_dim

    def init_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        return [
            torch.zeros(batch_size, self.state_dim, device=device)
            for _ in range(self.levels)
        ]

    def forward(
        self, states: List[torch.Tensor], u_prev: Optional[torch.Tensor], step_idx: int
    ) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
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

        # aggregate to conditioning vector
        cat = torch.cat(new_states, dim=-1)
        cond = self.agg(cat)  # [B, cond_dim]
        return cond, new_states, boundary


# -------------------------
# Full model
# -------------------------


@dataclass
class ChronoFlowConfig:
    in_channels: int = 1
    image_size: Tuple[int, int] = (32, 32)
    quant_levels: Optional[int] = None
    base_std: float = 0.5

    # Spatial per-frame flow g_theta
    spatial_levels: int = 3
    spatial_steps_per_level: int = 4
    spatial_hidden: int = 160

    # Conditional emission flow h_phi
    emission_levels: int = 2
    emission_steps_per_level: int = 4
    emission_hidden: int = 160
    cond_dim: int = 512

    # Temporal backbone
    temporal_levels: int = 4
    temporal_state_dim: int = 512

    # Training / transform
    rollout_prob: float = 0.1
    alpha_logit: float = 1e-6
    logdet_penalty_weight: float = 0.0


class ChronoFlowSSM(nn.Module):
    """
    Flow-based causal model for sequences of frames x_t in [0,1]^{C,H,W}.
    p(x_{1:T}) = Π_t p(u_t | s_t) |det dg/dx_t|,
    where u_t = g_theta(x_t), s_t = f_phi(s_{t-1}, u_{t-1}),
    and p(u | s) is given by a conditional flow h_psi.
    """

    def __init__(self, cfg: ChronoFlowConfig | Mapping | None = None, **kwargs):
        super().__init__()
        # Flexible constructor
        if cfg is not None and not isinstance(cfg, ChronoFlowConfig):
            if isinstance(cfg, Mapping):
                if kwargs:  # merge mapping with kwargs
                    cfg = {**cfg, **kwargs}
                else:
                    cfg = dict(cfg)
            else:
                raise TypeError("cfg must be ChronoFlowConfig, mapping or None")
            cfg = ChronoFlowConfig(**cfg)
        elif cfg is None:
            cfg = ChronoFlowConfig(**kwargs)
        self.cfg = cfg

        C, (H, W) = cfg.in_channels, cfg.image_size
        stride = 2**cfg.spatial_levels
        if (H % stride) != 0 or (W % stride) != 0:
            raise ValueError(
                "image_size must be divisible by 2**spatial_levels in both dims"
            )
        if cfg.quant_levels is not None and cfg.quant_levels <= 0:
            raise ValueError("quant_levels must be positive when provided")
        self.quant_levels: Optional[int] = cfg.quant_levels

        self.debug_enabled = os.environ.get("CHRONOFLOW_DEBUG", "0") == "1"
        self.debug_limit = int(os.environ.get("CHRONOFLOW_DEBUG_LIMIT", "20"))
        self._debug_counter = 0

        # Spatial flow g_theta
        self.g = SpatialFlow(
            in_channels=C,
            levels=cfg.spatial_levels,
            steps_per_level=cfg.spatial_steps_per_level,
            hidden=cfg.spatial_hidden,
        )

        # Determine latent spatial shape after g (after L squeezes)
        H_lat = H // (2**cfg.spatial_levels)
        W_lat = W // (2**cfg.spatial_levels)
        u_channels = C * (4**cfg.spatial_levels)
        self.u_shape: tuple[int, int, int] = (u_channels, H_lat, W_lat)

        # Temporal backbone f_phi
        self.temporal = TemporalBackbone(
            u_channels=u_channels,
            u_spatial_hw=(H_lat, W_lat),
            levels=cfg.temporal_levels,
            state_dim=cfg.temporal_state_dim,
            cond_dim=cfg.cond_dim,
        )

        # Emission flow h_psi: p(u | cond) exact
        self.h = ConditionalFlow(
            channels=u_channels,
            levels=cfg.emission_levels,
            steps_per_level=cfg.emission_steps_per_level,
            hidden=cfg.emission_hidden,
            cond_dim=cfg.cond_dim,
            base_std=cfg.base_std,
        )

        self.rollout_prob = cfg.rollout_prob
        self.register_buffer("_dummy", torch.zeros(1))  # to locate device easily

    # ----- Helpers -----
    def u_shape_with_batch(self, B: int) -> tuple[int, int, int, int]:
        C, H, W = self.u_shape
        return (B, C, H, W)

    def encode_frame(self, x_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x in LOGIT space
        u, ldj_g = self.g(x_logit, reverse=False)  # x -> u
        return u, ldj_g

    def decode_frame(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_logit, ldj = self.g(u, reverse=True)  # u -> x (logit)
        return x_logit, ldj

    # ----- Likelihood of a sequence -----
    def nll_sequence(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        x_seq: [B,T,C,H,W] in [0,1]
        Returns mean NLL (in nats) over the batch and diagnostics.
        """
        B, T, C, H, W = x_seq.shape

        # 0) dequantize+logit for all frames at once (vectorized)
        x_flat = x_seq.view(B * T, C, H, W)
        y_flat, logdet_flat = dequantize_and_logit(
            x_flat, alpha=self.cfg.alpha_logit, quant_levels=self.quant_levels
        )
        x_logit = y_flat.view(B, T, C, H, W)
        logdet_pre = logdet_flat.view(B, T)
        logdet_const = logdet_pre.sum(dim=1)

        # 1) iterate time
        states = self.temporal.init_state(B, x_seq.device)
        total_logprob = x_seq.new_zeros(B)
        boundary_sum = x_seq.new_tensor(0.0)
        count = 0
        ldj_reg = x_seq.new_tensor(0.0)

        u_prev_for_state = None
        for t in range(T):
            cond, states, boundary = self.temporal(
                states, u_prev_for_state, step_idx=t
            )  # s_t
            u_t, ldj_g = self.encode_frame(x_logit[:, t])  # u_t
            logp_u = self.h.log_prob(u_t, cond)  # exact
            total_logprob = total_logprob + logp_u + ldj_g
            if self.cfg.logdet_penalty_weight > 0:
                ldj_reg = ldj_reg + ldj_g.pow(2).mean()

            if self.debug_enabled and self._debug_counter < self.debug_limit:
                self._debug_counter += 1
                print(
                    f"[ChronoFlow debug] step={t} "
                    f"logp_u_mean={logp_u.mean().item():.3e} "
                    f"ldj_g_mean={ldj_g.mean().item():.3e} "
                    f"u_abs_max={u_t.abs().max().item():.3e}"
                )

            boundary_sum = boundary_sum + boundary.mean()
            count += 1

            # scheduled sampling for state input
            rollout_draw = torch.rand(1, device=x_seq.device).item()
            if (t < T - 1) and (rollout_draw < self.rollout_prob):
                u_hat = self.h.sample(
                    self.u_shape_with_batch(B), cond
                )  # sample u ~ p(u|cond)
                u_prev_for_state = u_hat.detach()
            else:
                u_prev_for_state = u_t

        nll_model = -(total_logprob.mean())
        true_nll = -(total_logprob + logdet_const).mean()
        penalty = (
            (ldj_reg / max(1, count)) * self.cfg.logdet_penalty_weight
            if self.cfg.logdet_penalty_weight > 0
            else x_seq.new_tensor(0.0)
        )
        # Optimise the constant-free NLL to avoid the fixed logit offset dominating.
        loss = nll_model + penalty
        stats = {
            "bits_per_dim": (nll_model / (math.log(2.0) * T * C * H * W)),
            "bits_per_dim_true": (true_nll / (math.log(2.0) * T * C * H * W)),
            "avg_boundary": boundary_sum / max(1, count),
            "logdet_const": logdet_const.mean().detach(),
            "ldj_penalty": penalty.detach(),
        }
        return loss, stats

    def forward(
        self, x: torch.Tensor | tuple[torch.Tensor, ...], *, return_stats: bool = True
    ) -> dict:
        """
        Compute negative log-likelihood for a batch of sequences.
        Accepts x as [B,T,C,H,W].
        """
        if isinstance(x, (tuple, list)):
            x = x[0]

        nll, stats = self.nll_sequence(x)
        out: dict = {"nll": nll}
        if return_stats:
            out["stats"] = {
                "bits_per_dim": stats["bits_per_dim"],
                "avg_boundary": torch.as_tensor(
                    stats["avg_boundary"], device=nll.device, dtype=nll.dtype
                ),
            }
        return out

    def forecast(
        self, x_context: torch.Tensor, steps: int, *, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Autoregressively sample 'steps' future frames.
        Args:
            x_context: [B,T0,C,H,W] in [0,1]
            steps: number of frames to generate
            temperature: latent sampling temperature for h (default 1.0)
        Returns:
            [B, T0+steps, C, H, W] in [0,1]
        """
        self.eval()
        B, T0, C, H, W = x_context.shape
        device = x_context.device

        # preprocess context -> logit
        xy, _ = (
            dequantize_and_logit(
                x_context.view(B * T0, C, H, W),
                alpha=self.cfg.alpha_logit,
                quant_levels=self.quant_levels,
            )
            if T0 > 0
            else (x_context.new_empty((0,)), x_context.new_empty((0,)))
        )
        x_logit = (
            xy.view(B, T0, C, H, W) if T0 > 0 else x_context.new_empty((B, 0, C, H, W))
        )

        # 1) init temporal state by teacher-forcing the context
        states = self.temporal.init_state(B, device)
        u_prev = None
        for t in range(T0):
            cond, states, _ = self.temporal(states, u_prev, step_idx=t)
            u_t, _ = self.encode_frame(x_logit[:, t])
            u_prev = u_t

        # 2) start from original context frames
        frames_out: list[torch.Tensor] = [f for f in x_context.unbind(dim=1)]

        # 3) autoregressive rollout
        for k in tqdm(range(steps), desc="Generating future steps"):
            step_idx = T0 + k
            cond, states, _ = self.temporal(states, u_prev, step_idx=step_idx)
            u_sample = self.h.sample(
                self.u_shape_with_batch(B), cond, temperature=temperature
            )
            x_logit_sample, _ = self.decode_frame(u_sample)
            x_sample, _ = logit_inverse(x_logit_sample, alpha=self.cfg.alpha_logit)
            frames_out.append(x_sample.clamp(0, 1))
            u_prev = u_sample

        # Return stacked frames as (B, T_total, C, H, W) in [0, 1]
        return torch.stack(frames_out, dim=1).contiguous()

    def sample(
        self, x_context: torch.Tensor, steps: int, *, temperature: float = 1.0
    ) -> torch.Tensor:
        """Alias for forecast() to match previous public API."""
        return self.forecast(x_context, steps, temperature=temperature)
