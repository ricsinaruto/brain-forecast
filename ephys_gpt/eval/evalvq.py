from __future__ import annotations

import numpy as np
import torch

from .evalquant import EvalQuant

DEBUG = False


class EvalVQ(EvalQuant):
    """Evaluation for BrainOmniSystem (tokeniser + AR forecaster).
    TODO: not tested

    Computes reconstruction MSE curves by reconstructing predicted latent
    codes back to channel space using the tokenizer decoder. Metrics mirror
    GPT2MEG outputs (hist_mse.npy, future_mse.npy, PSD/Cov for free‑running).
    """

    # ------------------------ helpers ------------------------
    def _get_latent_codes(self, inputs) -> torch.Tensor:
        """Return latent codes with shape (B, C_latent, Nq, T_lat)."""
        # Tokeniser returns (B, C_latent, T_lat, Nq)
        codes = self.model.tokenizer(inputs, return_reconstruction=False)
        return codes.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _stage_argmax(logits: torch.Tensor) -> torch.Tensor:
        """Argmax over codebook dim K for stage‑wise logits.

        logits: (B, C_latent, Nq, T, K) -> indices (B, C_latent, Nq, T)
        """
        return logits.argmax(dim=-1)

    def _reconstruct_codes(self, codes_btqn: torch.Tensor) -> torch.Tensor:
        """codes_btqn: (B, C_latent, T, Nq) -> x: (B, C, L)"""
        return self.model.tokenizer.reconstruct(codes_btqn)

    # ------------------------ tasks -------------------------
    @torch.inference_mode()
    def step_history_sweep(self) -> None:  # type: ignore[override]
        mse_hist = None
        counts = None

        for i, (inputs, _) in enumerate(self.test_loader):
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(self.device) for x in inputs)
            else:
                inputs = inputs.to(self.device)

            codes_bcq_t = self._get_latent_codes(inputs)  # (B,Cq,Nq,T)
            B, Cq, Nq, T = codes_bcq_t.shape

            H = T - 1
            if mse_hist is None:
                mse_hist = torch.zeros((self.num_channels, H), device=self.device)
                counts = torch.zeros(H, device=self.device)

            for t in range(1, T):
                max_h = min(H, t)
                for h in range(1, max_h + 1):
                    # context codes: (B,Cq,Nq,h)
                    ctx = codes_bcq_t[..., t - h : t]
                    logits = self.model.forecaster(ctx)  # (B,Cq,Nq,h,K)
                    next_idx = self._stage_argmax(logits)[..., -1]  # (B,Cq,Nq)

                    # recon predicted one latent step
                    pred_codes = next_idx.unsqueeze(2)  # (B,Cq,1,Nq)
                    gt_codes = (
                        codes_bcq_t[..., t].permute(0, 1, 2).unsqueeze(2).contiguous()
                    )  # (B,Cq,1,Nq)
                    x_pred = self._reconstruct_codes(pred_codes)
                    x_gt = self._reconstruct_codes(gt_codes)

                    se = (x_pred - x_gt) ** 2  # (B,C,L)
                    mse_hist[:, h - 1] += se.sum(dim=(0, 2))
                    counts[h - 1] += B * se.shape[-1]

                if DEBUG and t > H + 10:
                    break

            if DEBUG and i == 0:
                break

        counts = torch.clamp_min(counts, 1)
        mse_hist = (mse_hist / counts).detach().cpu().numpy()
        np.save(self.out_dir / "hist_mse.npy", mse_hist.mean(0))
        np.save(self.out_dir / "hist_lengths.npy", np.arange(1, mse_hist.shape[1] + 1))
        self._plot_horizon_lines(
            mse_hist, "mse_hist.pdf", "MSE", "History length (latent steps)"
        )

    @torch.inference_mode()
    def step_recursive_future(self) -> None:  # type: ignore[override]
        N = int(self.eval_args.get("future_steps", 4))
        mse_horizon = torch.zeros((self.num_channels, N), device=self.device)
        counts = torch.zeros(N, device=self.device)

        for i, (inputs, _) in enumerate(self.test_loader):
            if isinstance(inputs, (list, tuple)):
                inputs = tuple(x.to(self.device) for x in inputs)
            else:
                inputs = inputs.to(self.device)

            codes = self._get_latent_codes(inputs)  # (B,Cq,Nq,T)
            B, Cq, Nq, T = codes.shape
            if T <= N:
                continue

            ctx_len = T - N
            for t in range(ctx_len, T - N + 1):
                seq = codes[..., t - ctx_len : t]  # (B,Cq,Nq,ctx)
                for k in range(N):
                    logits = self.model.forecaster(seq)  # (B,Cq,Nq,S,K)
                    next_idx = self._stage_argmax(logits)[..., -1]  # (B,Cq,Nq)

                    pred_codes = next_idx.unsqueeze(2)  # (B,Cq,1,Nq)
                    gt_codes = codes[..., t + k].unsqueeze(2).permute(0, 1, 3, 2)
                    x_pred = self._reconstruct_codes(pred_codes)
                    x_gt = self._reconstruct_codes(gt_codes)

                    se = (x_pred - x_gt) ** 2
                    mse_horizon[:, k] += se.sum(dim=(0, 2))
                    counts[k] += B * se.shape[-1]

                    # append predicted step to sequence
                    seq = torch.cat([seq, pred_codes.permute(0, 1, 3, 2)], dim=-1)

                if DEBUG and t > ctx_len + 10:
                    break

            if DEBUG and i == 0:
                break

        counts = torch.clamp_min(counts, 1)
        mse_horizon = (mse_horizon / counts).detach().cpu().numpy()
        np.save(self.out_dir / "future_mse.npy", mse_horizon.mean(0))
        self._plot_horizon_lines(
            mse_horizon, "mse_future.pdf", "MSE", "Forecast horizon (latent)"
        )

    def step_free_running(self) -> None:  # type: ignore[override]
        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        # seed from a random trial
        inputs, _ = next(iter(self.test_loader))
        if isinstance(inputs, (list, tuple)):
            inputs = tuple(x.to(self.device) for x in inputs)
        else:
            inputs = inputs.to(self.device)

        codes = self._get_latent_codes(inputs)  # (B,Cq,Nq,T)
        B, Cq, Nq, T = codes.shape
        seq = codes[..., :1]  # start with one latent step

        # grow until reconstructed length covers desired seconds
        while True:
            x_rec = self._reconstruct_codes(seq.permute(0, 1, 3, 2))  # (B,C,L)
            if x_rec.shape[-1] >= total_steps:
                break
            logits = self.model.forecaster(seq)
            next_idx = self._stage_argmax(logits)[..., -1].unsqueeze(2)
            seq = torch.cat([seq, next_idx], dim=-1)

        arr = x_rec.squeeze(0).detach().cpu().numpy()
        np.save(self.out_dir / "generated_brainomni.npy", arr)
        self._eval_psd_cov(arr, prefix="gen")
        self._eval_psd_cov(self._get_test_deq(), prefix="test")
