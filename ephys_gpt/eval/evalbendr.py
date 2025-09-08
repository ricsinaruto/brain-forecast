from __future__ import annotations

import numpy as np
import torch

from .evalquant import EvalQuant

DEBUG = False


class EvalBENDR(EvalQuant):
    """Evaluation for BENDRForecast (continuous next‑sample model).
    TODO: not tested

    Implements sliding‑window 1‑step and N‑step evaluation using the model's
    forward pass and recursive re‑feeding of predictions.
    """

    def _get_max_hist(self) -> int:  # type: ignore[override]
        # Use the model's configured sample window length if available
        if hasattr(self.model, "_encoded_len") and hasattr(self.model, "_ds_factor"):
            return int(self.model._encoded_len * self.model._ds_factor)
        # fall back to config 'samples'
        return int(self.model_cfg.get("samples", 0))

    @torch.inference_mode()
    def _predict_next(self, past: torch.Tensor) -> torch.Tensor:
        """Return 1‑step prediction y_{t} given past[:, :, :t]."""
        B, C, L = past.shape
        # BENDR forward expects a triple; second and third items are unused
        y_seq = self.model((past, None, None))  # (B,C,Tenc)
        # treat the last decoded sample as the next step
        return y_seq[..., -1]  # (B,C)

    @torch.inference_mode()
    def step_history_sweep(self) -> None:  # type: ignore[override]
        H = self._get_max_hist()
        hist_lengths = list(range(1, H + 1))
        mse_hist = torch.zeros((self.num_channels, H), device=self.device)
        counts = torch.zeros(H, device=self.device)

        for i, (inputs, _) in enumerate(self.test_loader):
            x = (
                inputs[0].to(self.device)
                if isinstance(inputs, (list, tuple))
                else inputs.to(self.device)
            )
            B, C, L = x.shape
            for t in range(1, L):
                max_h = min(H, t)
                for h in range(1, max_h + 1):
                    past = x[..., t - h : t]
                    pred = self._predict_next(past)  # (B,C)
                    gt = x[..., t]
                    mse_hist[:, h - 1] += ((pred - gt) ** 2).sum(dim=0)
                    counts[h - 1] += B
                if DEBUG and t > H + 10:
                    break
            if DEBUG and i == 0:
                break

        counts = torch.clamp_min(counts, 1)
        mse_hist = (mse_hist / counts).cpu().numpy()
        np.save(self.out_dir / "hist_mse.npy", mse_hist.mean(0))
        np.save(self.out_dir / "hist_lengths.npy", np.array(hist_lengths))
        self._plot_horizon_lines(
            mse_hist, "mse_hist.pdf", "MSE", "History length (samples)", hist_lengths
        )

    @torch.inference_mode()
    def step_recursive_future(self) -> None:  # type: ignore[override]
        N = int(self.eval_args.get("future_steps", 1))
        H = self._get_max_hist() - N
        H = max(H, 1)

        mse_horizon = torch.zeros((self.num_channels, N), device=self.device)
        counts = torch.zeros(N, device=self.device)

        for i, (inputs, _) in enumerate(self.test_loader):
            x = (
                inputs[0].to(self.device)
                if isinstance(inputs, (list, tuple))
                else inputs.to(self.device)
            )
            B, C, L = x.shape
            if L <= N:
                continue
            for t in range(H, L - N + 1):
                ctx = x[..., t - H : t].clone()
                preds: list[torch.Tensor] = []
                for k in range(N):
                    y_next = self._predict_next(ctx).unsqueeze(-1)  # (B,C,1)
                    preds.append(y_next)
                    ctx = torch.cat([ctx, y_next], dim=-1)
                    ctx = ctx[..., -H:]
                pred = torch.cat(preds, dim=-1)  # (B,C,N)
                gt = x[..., t : t + N]
                mse_horizon += ((pred - gt) ** 2).sum(dim=0)
                counts += B
                if DEBUG and t > H + 10:
                    break
            if DEBUG and i == 0:
                break

        counts = torch.clamp_min(counts, 1)
        mse_horizon = (mse_horizon / counts).cpu().numpy()
        np.save(self.out_dir / "future_mse.npy", mse_horizon.mean(0))
        self._plot_horizon_lines(
            mse_horizon, "mse_future.pdf", "MSE", "Forecast horizon (samples)"
        )

    def step_free_running(self) -> None:  # type: ignore[override]
        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        # start from a random 1‑sample context
        x, _ = next(iter(self.test_loader))
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.to(self.device)
        ctx = x[..., :1]
        gen = self.model.forecast(ctx, horizon=total_steps - 1)
        arr = gen.squeeze(0).cpu().numpy()
        np.save(self.out_dir / "generated_bendr.npy", arr)
        self._eval_psd_cov(arr, prefix="gen")
        self._eval_psd_cov(self._get_test_deq(), prefix="test")
