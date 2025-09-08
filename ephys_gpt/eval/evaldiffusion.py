from __future__ import annotations

import numpy as np
import torch

from .evalquant import EvalQuant
from typing import List

DEBUG = False


class EvalDiffusion(EvalQuant):
    """Evaluation utilities specialised for the diffusion-based NTD model.
    TODO: not tested
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # self.model.create_schedule(200)

    def run_all(self) -> None:
        print("[EVAL] 1/3  history length sweep …")
        self.step_history_sweep()

        print("[EVAL] 2/3  recursive N-step forecasting …")
        self.step_recursive_future()

        print("[EVAL] 3/3  free-running generation …")
        self.step_free_running()

        print("[EVAL] finished ✓")

    # ------------------------------------------------------------------
    # Special-case helpers (override base implementations)
    # ------------------------------------------------------------------
    def _get_max_hist(self) -> int:  # type: ignore[override]
        """Return the maximum context length supported by the diffusion model.

        For NTD, the context length is fixed and equal to the configured
        ``signal_length`` hyper-parameter.  We first try to read this value from
        the instantiated model (``self.model.signal_length``) and fall back to
        the raw configuration if the attribute is missing.
        """
        if hasattr(self.model, "signal_length"):
            return int(self.model.signal_length)
        return int(self.model_cfg.get("signal_length", 0))

    def _get_test_deq(self) -> np.ndarray:
        """
        Get the de-quantised test set.

        Returns:
            Array of shape (C, T_total) containing the de-quantised test set
        """
        if self._cached_test_deq is None:
            print("[EVAL] Assembling de-quantised full test set …")
            chans: List[np.ndarray] = [np.empty((self.num_channels, 0))]
            for inp, _ in self.test_dataset:
                if isinstance(inp, (list, tuple)):
                    inp = inp[0]
                chans.append(inp.cpu().numpy())
            self._cached_test_deq = np.concatenate(chans, axis=1)

        # save to file
        np.save(self.out_dir / "test_deq.npy", self._cached_test_deq)
        return self._cached_test_deq

    @torch.inference_mode()
    def generate(self) -> np.ndarray:  # type: ignore[override]
        """Generate a long sequence by **recursive diffusion forecasting**.

        Algorithm
        ---------
        1. Draw an *initial* sample of length ``signal_length`` with
           ``model.sample``.
        2. Repeatedly:
           a. Take the **second half** of the current signal as context.
           b. Call :py:meth:`model.forecast` to predict the *next* half-length
              segment.
           c. Append the new segment to the running signal.
        3. Stop when the desired duration (``gen_seconds``) is reached.
        """
        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        seg_len = self._get_max_hist()  # == signal_length
        past_len = seg_len // 2
        future_len = seg_len // 2

        # 1) Initial ancestral sample (shape: 1,C,seg_len)
        generated = self.model.sample(
            B=1,
            device=self.device,
            sample_length=seg_len,
        )

        steps_left = total_steps - seg_len
        while steps_left > 0:
            horizon = min(future_len, steps_left)
            ctx = generated[:, :, -past_len:]
            new = self.model.forecast(
                past=ctx,
                horizon=horizon,
            )  # (B, C, L_ctx + N)

            generated = torch.cat([generated, new[:, :, -horizon:]], dim=-1)
            steps_left -= horizon

        gen_np = generated.squeeze(0).cpu().numpy()  # (C, T)
        np.save(self.out_dir / "generated_diffusion.npy", gen_np)
        return gen_np

    # ------------------------------------------------------------------
    # History-length sweep (diffusion)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def step_history_sweep(self) -> None:  # type: ignore[override]
        """
        Evaluate 1‑step reconstruction MSE across multiple history lengths,
        mirroring GPT2MEG's per‑timepoint sweep but for continuous signals.
        TODO: test
        """
        seg_len = self._get_max_hist()
        # Use 1..H (like GPT) but cap at seg_len-1 for 1‑step forecast
        H = seg_len - 1
        hist_list = list(range(1, H + 1))

        mse_hist = torch.zeros((self.num_channels, H), device=self.device)
        counts = torch.zeros(H, device=self.device)  # number of contexts per h

        for i, (inputs, targets) in enumerate(self.test_loader):
            signal = (
                inputs[0].to(self.device)
                if isinstance(inputs, (list, tuple))
                else inputs.to(self.device)
            )
            B, C, L = signal.shape
            if L <= 1:
                continue

            # sweep t from 1..L-1, and for each history length h<=min(H, t)
            for t in range(1, L):
                max_h = min(H, t)
                for h in range(1, max_h + 1):
                    past = signal[..., t - h : t]  # (B,C,h)
                    pred_full = self.model.forecast(past=past, horizon=1)  # (B,C,h+1)
                    pred = pred_full[..., -1]  # (B,C)
                    gt = signal[..., t]  # (B,C)
                    mse_hist[:, h - 1] += ((pred - gt) ** 2).sum(dim=0)
                    counts[h - 1] += B

                if DEBUG and t > H + 10:
                    break

            if DEBUG and i == 0:
                break

        # safe divide
        counts = torch.clamp_min(counts, 1)
        mse_hist = (mse_hist / counts).cpu().numpy()
        np.save(self.out_dir / "hist_mse.npy", mse_hist.mean(0))
        np.save(self.out_dir / "hist_lengths.npy", np.array(hist_list))

        self._plot_horizon_lines(
            mse_hist,
            "mse_hist.pdf",
            "MSE",
            "History length (timesteps)",
            hist_lengths=hist_list,
        )

    # ------------------------------------------------------------------
    # Recursive N-step forecasting (diffusion)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def step_recursive_future(self) -> None:  # type: ignore[override]
        """
        Per‑horizon MSE for continuous forecasts, matching GPT2MEG logic:
        iterate over all valid timepoints and measure N‑step recursive forecasts.
        TODO: test
        """
        N = int(self.eval_args.get("future_steps", self._get_max_hist() // 4))
        mse_horizon = torch.zeros((self.num_channels, N), device=self.device)
        counts = torch.zeros(N, device=self.device)

        for i, (inputs, _) in enumerate(self.test_loader):
            signal = (
                inputs[0].to(self.device)
                if isinstance(inputs, (list, tuple))
                else inputs.to(self.device)
            )
            B, C, L = signal.shape
            if L <= N:
                continue

            # choose context length so that ctx+N <= seg_len
            ctx_len = min(self._get_max_hist() - N, L - N)
            if ctx_len <= 0:
                continue

            for t in range(ctx_len, L - N + 1):
                past = signal[..., t - ctx_len : t]
                pred_full = self.model.forecast(past=past, horizon=N)  # (B,C,ctx_len+N)
                pred = pred_full[..., -N:]  # (B,C,N)
                gt = signal[..., t : t + N]
                mse_horizon += ((pred - gt) ** 2).sum(dim=0)
                counts += B

                if DEBUG and t > ctx_len + 10:
                    break

            if DEBUG and i == 0:
                break

        counts = torch.clamp_min(counts, 1)
        mse_horizon = (mse_horizon / counts).cpu().numpy()  # (C, N)
        np.save(self.out_dir / "future_mse.npy", mse_horizon.mean(0))

        self._plot_horizon_lines(
            mse_horizon,
            "mse_future.pdf",
            ylabel="MSE",
            xlabel="Forecast horizon (steps)",
        )
