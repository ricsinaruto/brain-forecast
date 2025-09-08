import torch
import torch.nn as nn
from typing import Optional

from ..layers.ntd import AdaConv, OUProcess, WhiteNoiseProcess


class NTD(nn.Module):
    """
    Neurophysiological Time-series Diffusion (NTD)

    Parameters
    ----------
    signal_length : int
        Length (L) of the training signals.
    signal_channel : int
        Number of channels (C).
    cond_dim : int, default 0
        Dimensionality of optional conditioning channels (C_cond). If you pass
        extra channels for conditioning, they must match (B, cond_dim, L).
    diffusion_time_steps : int
        Number of diffusion steps T.
    schedule : {"linear", "quad"}
        Noise schedule type for betas.
    start_beta, end_beta : float
        Range for beta schedule.
    ou_sigma2 : float
        OU process variance.
    ou_ell : float
        OU process length scale.
    net_hidden_channel : int
        Hidden channels per signal channel for AdaConv.
    net_in_kernel_size, net_out_kernel_size : int
    net_slconv_kernel_size : int
    net_num_scales : int
    net_num_blocks : int
    net_num_off_diag : int
    net_use_pos_emb : bool
    net_padding_mode : str
    net_use_fft_conv : bool
    """

    def __init__(
        self,
        *,
        signal_length: int,
        signal_channel: int,
        cond_dim: int = 0,
        diffusion_time_steps: int = 1000,
        schedule: str = "linear",
        start_beta: float = 1e-4,
        end_beta: float = 2e-2,
        ou_sigma2: float = 1.0,
        ou_ell: float = 10.0,
        net_hidden_channel: int = 8,
        net_in_kernel_size: int = 1,
        net_out_kernel_size: int = 1,
        net_slconv_kernel_size: int = 17,
        net_num_scales: int = 5,
        net_num_blocks: int = 3,
        net_num_off_diag: int = 8,
        net_use_pos_emb: bool = False,
        net_padding_mode: str = "circular",
        net_use_fft_conv: bool = False,
        noise_process: str = "ou",
        mask_channel: int = 0,
        p_forecast: float = 0.0,
    ):
        super().__init__()

        # --- Noise process (sampler + Mahalanobis) ---
        if noise_process == "ou":
            ou_process = OUProcess(ou_sigma2, ou_ell, signal_length)
        elif noise_process == "white":
            ou_process = WhiteNoiseProcess(ou_sigma2, signal_length)
        else:
            raise ValueError(f"Unknown noise process: {noise_process}")

        # --- Denoiser network ---
        self.network = AdaConv(
            signal_length=signal_length,
            mask_channel=mask_channel,
            signal_channel=signal_channel,
            cond_dim=cond_dim,
            hidden_channel=net_hidden_channel,
            in_kernel_size=net_in_kernel_size,
            out_kernel_size=net_out_kernel_size,
            slconv_kernel_size=net_slconv_kernel_size,
            num_scales=net_num_scales,
            num_blocks=net_num_blocks,
            num_off_diag=net_num_off_diag,
            use_pos_emb=net_use_pos_emb,
            padding_mode=net_padding_mode,
            use_fft_conv=net_use_fft_conv,
        )
        assert self.network.signal_length == ou_process.signal_length

        self.noise_sampler = ou_process
        self.mal_dist_computer = ou_process
        self.schedule = schedule
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.mask_channel = mask_channel
        self.p_forecast = p_forecast

        self._create_schedule(diffusion_time_steps)

    def _create_schedule(self, diffusion_time_steps: int):
        self.diffusion_time_steps = diffusion_time_steps

        if self.schedule == "linear":
            betas = torch.linspace(self.start_beta, self.end_beta, diffusion_time_steps)
        elif self.schedule == "quad":
            betas = (
                torch.linspace(
                    self.start_beta**0.5, self.end_beta**0.5, diffusion_time_steps
                )
                ** 2.0
            )
        else:
            raise ValueError("Unknown schedule type.")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("unormalized_probs", torch.ones(self.diffusion_time_steps))

    def _get_beta(self, timestep: int) -> torch.Tensor:
        return self.betas[timestep] ** 0.5

    def _get_alpha_beta(self, timestep: int) -> torch.Tensor:
        if timestep == 0:
            return self._get_beta(timestep)
        return (
            ((1.0 - self.alpha_bars[timestep - 1]) / (1.0 - self.alpha_bars[timestep]))
            * self.betas[timestep]
        ) ** 0.5

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device_ = args[0] if args else kwargs.get("device", "cpu")
        self.noise_sampler.to(*args, **kwargs)
        self.mal_dist_computer.to(*args, **kwargs)
        return self

    def sample_mask(
        self,
        x: torch.Tensor,
        min_L: int = 1,
        max_L: int = -1,
    ) -> torch.Tensor:
        """
        shape  : (B, C, L_total)
        Returns a binary mask tensor with the same shape.

        Args:
            shape: (B, C, L_total)
            p_forecast: probability of forecast scenario
            min_L: minimum length of forecast scenario
            max_L: maximum length of forecast scenario

        Returns:
            mask: (B, C, L_total)

        """
        B, C, L = x.shape
        mask = torch.zeros(x.shape, device=x.device)

        max_L = L - 1 if max_L == -1 else max_L

        if torch.rand(1).item() < self.p_forecast:
            # suffix mask (forecast scenario)
            Lf = torch.randint(min_L, max_L + 1, ()).item()
            mask[..., : L - Lf] = 1.0
        return mask

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        One forward diffusion step loss per sample (returns per-item losses).

        Args:
            batch: (B, C, L)
            cond : (B, C_cond, L) or None
            mask : (B, C, L) -> 1 for observed (loss computed), 0 ignored

        Returns:
            torch.Tensor: Per-sample losses.
        """
        if isinstance(x, (tuple, list)):
            x = x[0]
        B = x.shape[0]

        if mask is None and self.mask_channel > 0:
            mask = self.sample_mask(x)

        t_idx = self.unormalized_probs.multinomial(num_samples=B, replacement=True)
        a_bar = self.alpha_bars[t_idx].unsqueeze(-1).unsqueeze(-1)
        noise = self.noise_sampler.sample(
            sample_shape=(B, self.network.signal_channel, self.network.signal_length),
            device=x.device,
        )
        assert noise.shape == x.shape

        # forward diffusion
        noisy = torch.sqrt(a_bar) * x + torch.sqrt(1.0 - a_bar) * noise

        if mask is not None:
            noisy = torch.cat([noisy, mask[:, :1, :]], dim=1)  # suggested by chatgpt

        pred_noise = self.network(noisy, t_idx, cond=cond)

        return noise, pred_noise, mask

    def loss(
        self,
        noise: torch.Tensor,
        pred_noise: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduce: str = "mean",
    ) -> torch.Tensor:
        diff = noise - pred_noise
        maha = self.mal_dist_computer.sqrt_mal(diff)
        if mask is not None:
            maha = maha * (1 - mask)

        # per-sample loss
        # return torch.einsum("bcl,bcl->b", maha, maha)

        if reduce == "mean":
            return (maha**2).mean()
        elif reduce == "sum":
            return (maha**2).sum()

        return maha**2

    def sample(
        self,
        B: int,
        device: torch.device = "cuda",
        cond: Optional[torch.Tensor] = None,
        sample_length: Optional[int] = None,
        sampler: Optional[OUProcess] = None,
        noise_type: str = "alpha_beta",
    ) -> torch.Tensor:
        channels = self.network.signal_channel
        mask = None
        if sampler is None:
            sampler = self.noise_sampler
        if sample_length is None:
            sample_length = sampler.signal_length

        if cond is not None:
            b_cond, c_cond, l_cond = cond.shape
            assert l_cond == sample_length
            if b_cond == 1:
                cond = cond.repeat(B, 1, 1)
            else:
                assert b_cond == B

        self.eval()
        with torch.no_grad():
            state = sampler.sample((B, channels, sample_length), device=device)
            if self.mask_channel > 0:
                mask = self.sample_mask(state, p_forecast=0.0)

            for i in range(self.diffusion_time_steps):
                t = self.diffusion_time_steps - i - 1
                t_vec = torch.full((B,), t, device=device, dtype=torch.long)
                eps = sampler.sample((B, channels, sample_length), device=device)

                x_in = state
                if mask is not None:
                    x_in = torch.cat([x_in, mask[:, :1, :]], dim=1)

                res = self.network(x_in, t_vec, cond=cond)

                state = (1 / torch.sqrt(self.alphas[t])) * (
                    state
                    - ((1.0 - self.alphas[t]) / torch.sqrt(1.0 - self.alpha_bars[t]))
                    * res
                )

                if t > 0:
                    sigma = (
                        self._get_alpha_beta(t)
                        if noise_type == "alpha_beta"
                        else self._get_beta(t)
                    )
                    state = state + sigma * eps
            return state

    def impute(
        self,
        signal: torch.Tensor,
        mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        noise_type: str = "alpha_beta",
    ) -> torch.Tensor:
        """
        signal : (B,C,L) observed target values (fill anywhere mask==1)
        mask   : (B,C,L) 1 = observed (keep), 0 = missing (sample)
        cond   : optional conditioning channels (B,Cc,L)
        Returns completed signal.
        """
        B, C, L = signal.shape
        assert mask.shape == signal.shape
        if cond is not None:
            assert cond.shape[0] == B and cond.shape[2] == L

        self.eval()
        with torch.no_grad():
            # ---- before the loop ----
            sqrt_alphas = torch.sqrt(self.alphas).to(signal.device)
            sqrt_alpha_bars = torch.sqrt(self.alpha_bars).to(signal.device)
            sqrt_one_minus_ab = torch.sqrt(1.0 - self.alpha_bars).to(signal.device)

            # Optional: initialise observed coords at x_T
            eps_init = self.noise_sampler.sample((B, C, L), device=signal.device)
            state = sqrt_alpha_bars[-1] * signal + sqrt_one_minus_ab[-1] * eps_init

            for i in range(self.diffusion_time_steps):
                t = self.diffusion_time_steps - i - 1
                t_vec = torch.full((B,), t, dtype=torch.long, device=signal.device)

                # independent noises
                eps_pred = self.noise_sampler.sample((B, C, L), device=signal.device)
                eps_obs = self.noise_sampler.sample((B, C, L), device=signal.device)

                x_in = state
                if mask is not None:
                    x_in = torch.cat([x_in, mask[:, :1, :]], dim=1)

                # network prediction ε_θ(x_t,t)
                eps_theta = self.network(x_in, t_vec, cond=cond)

                # ------ unknown / missing coordinates ------
                mu_tilde = (1.0 / sqrt_alphas[t]) * (
                    state - ((1.0 - self.alphas[t]) / sqrt_one_minus_ab[t]) * eps_theta
                )

                if t > 0:
                    sigma = torch.sqrt(
                        self.betas[t]
                        * (1.0 - self.alpha_bars[t - 1])
                        / (1.0 - self.alpha_bars[t])
                    )
                    mu_tilde += sigma * eps_pred

                # ------ known / observed coordinates ------
                noisy_obs = sqrt_alpha_bars[t] * signal + sqrt_one_minus_ab[t] * eps_obs
                mu_known = (1.0 / sqrt_alphas[t]) * (
                    noisy_obs - ((1.0 - self.alphas[t]) / sqrt_one_minus_ab[t]) * signal
                )

                # combine
                state = mask * mu_known + (1.0 - mask) * mu_tilde

        return state

    def forecast(
        self,
        past: torch.Tensor,
        horizon: int,
        cond: Optional[torch.Tensor] = None,
        noise_type: str = "alpha_beta",
    ) -> torch.Tensor:
        """Autoregressive-style multi-step forecast via diffusion imputation.
        past: (B,C,L_p)
        horizon: number of future steps L_f to sample
        Returns: (B,C,L_p+L_f)
        """
        B, C, Lp = past.shape
        L = Lp + horizon
        # build signal & mask
        signal = torch.zeros(B, C, L, device=past.device, dtype=past.dtype)
        signal[..., :Lp] = past
        mask = torch.zeros_like(signal)
        mask[..., :Lp] = 1.0
        cond_full = None
        if cond is not None:
            assert cond.shape[-1] == Lp
            # pad cond with zeros for future unless user passes full
            pad = torch.zeros(
                B, cond.shape[1], horizon, device=cond.device, dtype=cond.dtype
            )
            cond_full = torch.cat([cond, pad], dim=-1)
        return self.impute(signal, mask, cond=cond_full, noise_type=noise_type)
