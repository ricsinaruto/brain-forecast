from __future__ import annotations

import random

import numpy as np
import torch
from tqdm import tqdm

from torch import Tensor
from typing import List
from .evalquant import EvalQuant

DEBUG = False


class EvalCont(EvalQuant):
    """Evaluation for BENDRForecast (continuous next‑sample model).

    Implements sliding‑window 1‑step and N‑step evaluation using the model's forward
    pass and recursive re‑feeding of predictions.
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
                    past = x[..., t - h: t]
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
                ctx = x[..., t - H: t].clone()
                preds: list[torch.Tensor] = []
                for k in range(N):
                    y_next = self._predict_next(ctx).unsqueeze(-1)  # (B,C,1)
                    preds.append(y_next)
                    ctx = torch.cat([ctx, y_next], dim=-1)
                    ctx = ctx[..., -H:]
                pred = torch.cat(preds, dim=-1)  # (B,C,N)
                gt = x[..., t: t + N]
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

    def step_free_running(self) -> None:
        """Generate a free-running sequence of tokens."""
        recursive_gen = self.generate()
        self._eval_psd_cov(recursive_gen, prefix="gen")

    @torch.inference_mode()
    def generate(self) -> np.ndarray:
        """Generate a sequence using the model's native forecast routine when
        available."""
        rollout_gen = self.generate_one_step_rollout()
        self._eval_psd_cov(rollout_gen, prefix="gen_1step")

        if hasattr(self.model, "forecast"):
            return self._generate_with_forecast()
        return super().generate()

    @torch.inference_mode()
    def generate_one_step_rollout(self) -> np.ndarray:
        """Generate a sequence via 1-step teacher-forced rollouts.

        This stitches together contiguous windows from the **test** dataset, feeds each
        window through the model, and records the final predicted timestep. Doing so
        yields a sequence of predictions with the same length as the requested
        generation horizon while avoiding recursive error accumulation.
        """

        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        if total_steps <= 0:
            raise ValueError("`gen_seconds` must be positive for rollouts.")

        dataset_window = getattr(self.test_dataset, "length", None)
        if dataset_window is None:
            window_len = int(self._fallback_max_hist)
        else:
            window_len = max(int(dataset_window) - 1, 1)

        required_tokens = window_len + total_steps - 1
        quant_seq = self._assemble_quant_sequence(
            self.test_dataset,
            min_total_length=required_tokens,
            drop_first_overlap=False,
        )

        seq_len = int(quant_seq.shape[-1])
        if seq_len < required_tokens:
            raise RuntimeError(
                "Insufficient test data to run 1-step rollouts: "
                f"need {required_tokens} tokens, got {seq_len}."
            )

        num_windows = total_steps
        predictions: List[Tensor] = []
        pbar = tqdm(total=num_windows, desc="1-step-rollout")
        start = 0
        while start < num_windows:
            end = min(start + self.batch_size, num_windows)
            window_batch = torch.stack(
                [quant_seq[..., idx: idx + window_len] for idx in range(start, end)],
                dim=0,
            )
            window_batch = window_batch.to(self.device, non_blocking=True)
            preds = self._fwd(window_batch)
            predictions.append(preds[..., -1].cpu())
            pbar.update(end - start)
            start = end
        pbar.close()

        pred_tokens = torch.cat(predictions, dim=0)  # (num_windows, C)
        gen_deq = pred_tokens.transpose(0, 1).contiguous()  # (C, num_windows)

        if hasattr(self.test_dataset, "postprocessor"):
            gen_deq = self.test_dataset.postprocessor.reshape(gen_deq)

        gen_np = gen_deq.cpu().numpy()
        np.save(self.out_dir / "generated_1step.npy", gen_np)
        return gen_np

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _generate_with_forecast(self) -> np.ndarray:
        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        if total_steps <= 0:
            raise ValueError("`gen_seconds` must be positive for generation.")

        ctx = self._sample_continuous_context()
        ctx_seq = ctx.transpose(0, 1).contiguous().unsqueeze(0)  # (1, T, C)

        print(
            f"Running NSR.forecast with context of {ctx_seq.shape[1]} steps "
            f"to generate {total_steps} future samples."
        )
        future = self.model.forecast(ctx_seq, total_steps)
        future = future.squeeze(0).detach()  # (C, total_steps)

        if hasattr(self.test_dataset, "postprocessor"):
            future = self.test_dataset.postprocessor.reshape(future)

        gen_np = future.cpu().numpy()
        strategy = self.eval_args.get("gen_sampling", "forecast")
        np.save(self.out_dir / f"generated_{strategy}.npy", gen_np)
        return gen_np

    def _sample_continuous_context(self) -> torch.Tensor:
        dataset = self.dataset
        dataset_len = len(dataset)
        if dataset_len == 0:
            raise RuntimeError("Dataset is empty; cannot sample generation context.")

        sample_idx = random.randrange(dataset_len)
        inputs, _ = dataset[sample_idx]
        if isinstance(inputs, (tuple, list)):
            ctx = inputs[0]
        else:
            ctx = inputs

        if ctx.ndim != 2:
            raise ValueError(
                "Continuous datasets must yield tensors shaped (C, T); "
                f"got shape {ctx.shape}"
            )

        return ctx.to(self.device).float().clone()
