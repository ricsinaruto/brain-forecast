from __future__ import annotations

import numpy as np
import torch

from .evalquant import EvalQuant

DEBUG = False


class EvalFlow(EvalQuant):
    """Evaluation for MEGFormer image‑based autoregressive model. TODO: not tested.

    Maps image forecasts back to channel space via dataset pixel indices and computes
    the same metrics (MSE curves) as the GPT2MEG evaluation.
    """

    def _get_max_hist(self) -> int:  # type: ignore[override]
        # context_length is the AR context in frames
        if hasattr(self.model, "context_length"):
            return int(self.model.context_length)
        return int(self.model_cfg.get("context_length", 0))

    def _images_to_channels(self, img: torch.Tensor) -> torch.Tensor:
        """Convert (B,H,W,T) to (B,C,T) by indexing sensor pixels."""
        row_idx = torch.as_tensor(
            getattr(self.test_dataset, "row_idx"), device=img.device
        )
        col_idx = torch.as_tensor(
            getattr(self.test_dataset, "col_idx"), device=img.device
        )

        # gather per time slice
        out = img[:, row_idx, col_idx, :]  # (B,C,T)
        return out

    @torch.inference_mode()
    def step_history_sweep(self) -> None:  # type: ignore[override]
        H = self._get_max_hist()
        hist_lengths = list(range(1, H + 1))
        mse_hist = torch.zeros((self.num_channels, H), device=self.device)
        counts = torch.zeros(H, device=self.device)

        for i, (inputs, _) in enumerate(self.test_loader):
            # dataset returns (img, img)
            if isinstance(inputs, (list, tuple)):
                img = inputs[0].to(self.device)
            else:
                img = inputs.to(self.device)
            B, H_img, W_img, T = img.shape
            # sweep t across time
            for t in range(1, T):
                max_h = min(H, t)
                for h in range(1, max_h + 1):
                    ctx = img[..., t - h: t]
                    # forecast 1 step
                    pred_img = self.model.forecast(ctx, steps=1)[..., -1:]
                    pred = self._images_to_channels(pred_img)[..., -1]  # (B,C)
                    gt = self._images_to_channels(img[..., t: t + 1])[..., -1]
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
            mse_hist, "mse_hist.pdf", "MSE", "History length (frames)", hist_lengths
        )

    @torch.inference_mode()
    def step_recursive_future(self) -> None:  # type: ignore[override]
        N = int(self.eval_args.get("future_steps", 1))
        ctx_len = self._get_max_hist() - N
        ctx_len = max(ctx_len, 1)

        mse_horizon = torch.zeros((self.num_channels, N), device=self.device)
        counts = torch.zeros(N, device=self.device)

        for i, (inputs, _) in enumerate(self.test_loader):
            img = (
                inputs[0].to(self.device)
                if isinstance(inputs, (list, tuple))
                else inputs.to(self.device)
            )
            B, H_img, W_img, T = img.shape
            if T <= N:
                continue
            for t in range(ctx_len, T - N + 1):
                ctx = img[..., t - ctx_len: t]
                pred_imgs = self.model.forecast(ctx, steps=N)[..., -N:]  # (B,H,W,N)
                pred = self._images_to_channels(pred_imgs)  # (B,C,N)
                gt = self._images_to_channels(img[..., t: t + N])  # (B,C,N)
                mse_horizon += ((pred - gt) ** 2).sum(dim=0)
                counts += B
                if DEBUG and t > ctx_len + 10:
                    break
            if DEBUG and i == 0:
                break

        counts = torch.clamp_min(counts, 1)
        mse_horizon = (mse_horizon / counts).cpu().numpy()
        np.save(self.out_dir / "future_mse.npy", mse_horizon.mean(0))
        self._plot_horizon_lines(
            mse_horizon, "mse_future.pdf", "MSE", "Forecast horizon (frames)"
        )

    @torch.inference_mode()
    def step_free_running(self) -> None:  # type: ignore[override]
        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        # pick a random trial image and start from first frame
        img, _ = next(iter(self.test_loader))
        if isinstance(img, (list, tuple)):
            img = img[0]
        ctx = img[:1].to(self.device)

        gen = self.model.forecast(ctx, steps=total_steps)

        print(gen.shape)

        # remove input context
        gen = gen[..., ctx.shape[-1]:]

        chans = self._images_to_channels(gen).squeeze(0).cpu().numpy()
        np.save(self.out_dir / "generated.npy", chans)
        self._eval_psd_cov(chans, prefix="gen")
        self._eval_psd_cov(self._get_test_deq(), prefix="test")
