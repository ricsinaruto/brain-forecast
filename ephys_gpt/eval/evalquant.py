from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import split_datasets
from ..training.utils import get_model_class
from ..utils.quantizers import mulaw_inv_torch
from ..utils.plotting import plot_psd
from ..utils.eval import sample

DEBUG = False


class EvalQuant:
    """Run and store the three evaluation tasks required by the specification."""

    def __init__(self, cfg: Dict) -> None:
        """
        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg
        self.eval_args = {**cfg.get("eval", {})}

        with open(cfg["model_config"]) as f:
            self.model_cfg = yaml.safe_load(f)

        # --------------------------- data ---------------------------
        split = split_datasets(**cfg["datasplitter"])
        dl_kwargs = {"shuffle": False, **cfg["dataloader"]}
        self.test_dataset = split.test
        self.test_loader = DataLoader(split.test, **dl_kwargs)
        first_inputs, _ = split.test[0]

        if isinstance(first_inputs, (list, tuple)):
            first_inputs = first_inputs[0]

        self.channel_shape = self.eval_args["channel_shape"]
        self.num_channels = self.eval_args["num_channels"]
        self._fallback_max_hist = int(first_inputs.shape[-1])
        self.pos_2d = np.array(split.test.pos_2d)
        self.sfreq = split.test.sfreq
        self.device = self.eval_args["accelerator"]

        # --------------------------- model --------------------------
        self.model = self._load_model()
        self.model.eval().to(self.device)

        if DEBUG and hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod  # for compiled models

        self.max_hist = self._get_max_hist() - 1
        self.mu = getattr(self.model, "quant_levels", 256) - 1

        # --------------------------- output dir ---------------------
        self.out_dir = Path(cfg["save_dir"]) / "evals"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._cached_test_deq: np.ndarray | None = None  # (C,T_total)

    def run_all(self) -> None:
        print("[EVAL] 1/3  history length sweep …")
        self.step_history_sweep()

        print("[EVAL] 2/3  recursive N-step forecasting …")
        self.step_recursive_future()

        print("[EVAL] 3/3  free-running generation …")
        self.step_free_running()

        print("[EVAL] finished ✓")

    # ---------------------------------------------------------------------
    #  Task 1 – per‑time‑point accuracy + MSE (topomaps)
    # ---------------------------------------------------------------------
    def step_basic_metrics(
        self, acc_sum: Tensor, mse_sum: Tensor, total_tokens: int
    ) -> None:

        channel_acc = (acc_sum / total_tokens).cpu().numpy()
        channel_mse = (mse_sum / total_tokens).cpu().numpy()

        # ---------- persist ----------
        np.save(self.out_dir / "channel_accuracy.npy", channel_acc)
        np.save(self.out_dir / "channel_mse.npy", channel_mse)
        with open(self.out_dir / "basic_metrics.json", "w") as f:
            json.dump(
                {
                    "mean_accuracy": float(channel_acc.mean()),
                    "mean_mse": float(channel_mse.mean()),
                },
                f,
            )

        # ---------- visualise ----------
        self._plot_topomap(channel_acc, "accuracy_topomap.pdf", ch_type="mag")
        self._plot_topomap(channel_mse, "mse_topomap.pdf", ch_type="mag")

    # ---------------------------------------------------------------------
    #  Task 2 – recursive N‑step forecasting at every timepoint
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def step_recursive_future(self) -> None:
        N = int(self.eval_args["future_steps"])

        strategy = self.eval_args["gen_sampling"]
        self.eval_args["gen_sampling"] = "argmax"

        # We use a shorter context (max_hist - N) so that, after generating N
        # future tokens, the overall sequence length does not exceed the model
        # cache window size. This allows us to leverage the model's KV-cache
        # across the recursive generation steps without overflow.
        ctx_len = self.max_hist - N

        acc_horizon = torch.zeros((*self.channel_shape, N), device=self.device)
        mse_horizon = torch.zeros_like(acc_horizon)
        total_tokens = 0  # number of (batch, context) pairs processed

        for inputs, targets in tqdm(self.test_loader, desc="recursive-future"):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            B, T = inputs.shape[0], inputs.shape[-1]

            # Number of valid contexts in this trial
            L = T - ctx_len - N + 1
            if L <= 0:
                continue

            # ------------------------------------------------------------------
            # Iterate over each start index t where we have a full context of
            # length `ctx_len` *before* t and at least N future tokens *after* t.
            # ------------------------------------------------------------------
            for t in tqdm(range(ctx_len, T - N + 1), desc="recursive-future-trial"):
                # Context slice: tokens [t-ctx_len, t)
                ctx = inputs[..., t - ctx_len : t]  # (B, ..., ctx_len)

                # Recursive forecast N future steps using the model's KV-cache
                fut_pred = self._recursive_forecast(ctx, N)  # (B, ..., N)

                # Ground-truth tokens at positions t-1..t+N-1
                fut_gt = torch.stack(
                    [targets[..., t - 1 + i] for i in range(N)], dim=-1
                )  # (B, ..., N)

                # ---- per-horizon accuracy ----
                eq = (fut_pred == fut_gt).float()  # (B, ..., N)
                acc_horizon += eq.sum(dim=0)

                # ---- per-horizon MSE (after µ-law inversion) ----
                pred_deq = mulaw_inv_torch(fut_pred, self.mu)
                gt_deq = mulaw_inv_torch(fut_gt, self.mu)
                mse_horizon += ((pred_deq - gt_deq) ** 2).sum(dim=0)

                total_tokens += B  # processed B contexts for each horizon

                # break early for testing
                if DEBUG and t > ctx_len + 10:
                    break

        self.eval_args["gen_sampling"] = strategy

        if hasattr(self.test_dataset, "reshape"):
            acc_horizon = self.test_dataset.reshape(acc_horizon)
            mse_horizon = self.test_dataset.reshape(mse_horizon)

        acc_horizon = (acc_horizon / total_tokens).cpu().numpy()  # (C, N)
        mse_horizon = (mse_horizon / total_tokens).cpu().numpy()  # (C, N)

        np.save(self.out_dir / "future_accuracy.npy", acc_horizon.mean(0))
        np.save(self.out_dir / "future_mse.npy", mse_horizon.mean(0))

        self._plot_horizon_lines(
            acc_horizon,
            "accuracy_future.pdf",
            ylabel="Accuracy",
            xlabel="Forecast horizon (steps)",
        )
        self._plot_horizon_lines(
            mse_horizon,
            "mse_future.pdf",
            ylabel="MSE",
            xlabel="Forecast horizon (steps)",
        )

    # ---------------------------------------------------------------------
    #  Task 3 – history length sweep at every timepoint
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def step_history_sweep(self) -> None:
        """
        Compute accuracy and MSE for all history lengths at every timepoint.
        """
        H = self.max_hist
        hist_lengths = torch.arange(1, H + 1, device=self.device)  # (H,)

        acc_sum = torch.zeros(self.channel_shape, device=self.device)
        mse_sum = torch.zeros(self.channel_shape, device=self.device)

        acc_hist = torch.zeros((*self.channel_shape, H), device=self.device)
        mse_hist = torch.zeros_like(acc_hist)
        total_tokens = 0  # tokens counted per history len

        for inputs, targets in tqdm(self.test_loader, desc="history-sweep"):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            B, T = inputs.shape[0], inputs.shape[-1]
            if T <= 1:
                continue

            # We will evaluate *all* histories for positions 1..T-1
            # (since context must be >=1)
            # For memory reasons process in chunks along time dimension
            for t in tqdm(range(H, T), desc="history-sweep-trial"):
                # prepare slice (B,...,H)
                ctx_full = inputs[..., t - H : t]
                logits = self._fwd(ctx_full)  # (B,...,H,Q)
                preds = logits.argmax(dim=-1)  # (B,...,H)
                gt = targets[..., t - H : t]

                # accuracy + mse for all histories at this t
                acc_hist += (preds == gt).float().sum(dim=0)
                pred_deq = mulaw_inv_torch(preds, self.mu)
                gt_deq = mulaw_inv_torch(gt, self.mu)
                mse_hist += ((pred_deq - gt_deq) ** 2).sum(dim=0)

                # last-timepoint accuracy + mse
                acc_sum += (preds[..., -1] == gt[..., -1]).float().sum(dim=0)
                mse_sum += ((pred_deq[..., -1] - gt_deq[..., -1]) ** 2).sum(dim=0)

                total_tokens += B

                # break early for testing
                if DEBUG and t > H + 10:
                    break

        if hasattr(self.test_dataset, "reshape"):
            acc_hist = self.test_dataset.reshape(acc_hist)
            mse_hist = self.test_dataset.reshape(mse_hist)
            acc_sum = self.test_dataset.reshape(acc_sum)
            mse_sum = self.test_dataset.reshape(mse_sum)

        # compute metrics for last-timepoint
        self.step_basic_metrics(acc_sum, mse_sum, total_tokens)

        # divide safely
        acc_hist = (acc_hist / total_tokens).cpu().numpy()
        mse_hist = (mse_hist / total_tokens).cpu().numpy()
        np.save(self.out_dir / "hist_accuracy.npy", acc_hist.mean(0))
        np.save(self.out_dir / "hist_mse.npy", mse_hist.mean(0))
        np.save(self.out_dir / "hist_lengths.npy", hist_lengths.cpu().numpy())

        self._plot_horizon_lines(
            acc_hist,
            "accuracy_hist.pdf",
            "Accuracy",
            "History length (tokens)",
        )
        self._plot_horizon_lines(
            mse_hist,
            "mse_hist.pdf",
            "MSE",
            "History length (tokens)",
        )

    # ---------------------------------------------------------------------
    #  Task 4 – free-running generation
    # ---------------------------------------------------------------------
    def step_free_running(self) -> None:
        """
        Generate a free-running sequence of tokens.
        """
        self._eval_psd_cov(self.generate(), prefix="gen")
        self._eval_psd_cov(self._get_test_deq(), prefix="test")

    # ---------------------------------------------------------------------
    #  Generation utilities
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self) -> np.ndarray:
        """Generate a sequence of length `1 + total_steps` (channels x time).

        • Picks a **single random starting token** from the test set.
        • Grows the sample in a `while` loop, each iteration requesting at most
          `max_hist` new tokens from `_recursive_forecast`.
        • Saves generated .npy and PSD/cov plots, then returns the de-quantised
          array `(C, 1+total_steps)`.

        Returns:
            Array of shape (C, 1+total_steps) containing the generated sequence
        """
        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        unroll_steps = self.eval_args.get("unroll_steps", 10)

        # ------- random single-token context -------
        trial_idx = random.randrange(len(self.test_dataset))
        quant_seq, _ = self.test_dataset[trial_idx]  # (...,T)
        t = random.randrange(quant_seq.shape[-1])
        ctx_quant = quant_seq[..., t : t + 1]  # (...,1)
        ctx = ctx_quant.unsqueeze(0).to(self.device)  # (1,...,1)

        # ------- iterative growth -------
        steps_left = total_steps

        pbar = tqdm(total=total_steps, desc="Generating sequence")
        while steps_left > 0:
            if steps_left < unroll_steps:
                break

            N = unroll_steps
            if ctx.shape[-1] < self.max_hist:
                N = self.max_hist - 1  # full generation first

            new_tokens = self._recursive_forecast(ctx[..., -self.max_hist + N :], N)
            ctx = torch.cat([ctx, new_tokens], dim=-1)
            steps_left -= N
            pbar.update(N)
        pbar.close()

        gen_deq = mulaw_inv_torch(ctx.squeeze(0), self.mu)

        if hasattr(self.test_dataset, "reshape"):
            gen_deq = self.test_dataset.reshape(gen_deq)

        gen_deq = gen_deq.cpu().numpy()

        strategy = self.eval_args["gen_sampling"]
        np.save(self.out_dir / f"generated_{strategy}.npy", gen_deq)
        return gen_deq

    @torch.inference_mode()
    def _recursive_forecast(self, ctx: Tensor, N: int) -> Tensor:
        """Forecast *N* tokens given `ctx`; supports various sampling strategies.

        Args:
            ctx: Context tensor of shape (B, C, max_hist)
            N: Forecast horizon

        Returns:
            Tensor of shape (B, C, N) containing the integer-bin predictions for
            the *N* future steps.
        """
        sample_args = {
            "strategy": self.eval_args["gen_sampling"],
            "temperature": self.eval_args.get("temperature", 1.0),
            "top_k": self.eval_args.get("top_k", 0),
            "top_p": self.eval_args.get("top_p", 0.0),
        }

        if hasattr(self.model, "generate"):
            return self.model.generate(ctx.clone(), N, sample_args)

        seq = ctx.clone()
        generated: List[Tensor] = []
        cache = {}
        for _ in range(N):
            try:
                logits, cache = self._fwd(seq, past_key_values=cache)  # (B,...,T,Q)
            except (TypeError, ValueError):
                logits = self._fwd(seq)  # type: ignore[assignment]

            next_logits = logits[..., -1, :]  # (B,...,Q)
            next_tok = sample(next_logits, **sample_args)  # (B,...)
            generated.append(next_tok)

            if cache:
                seq = next_tok.unsqueeze(-1)
            else:
                seq = torch.cat([seq, next_tok.unsqueeze(-1)], dim=-1)
        return torch.stack(generated, dim=-1)  # (B,...,N)

    # -----------------------------------
    # Helper functions
    # -----------------------------------
    def _eval_psd_cov(self, data: np.ndarray, prefix: str) -> None:
        """
        Evaluate the PSD and covariance of the data.

        Args:
            data: Array of shape (C, T) containing the data
            prefix: Prefix for the filenames
        """

        fig = plot_psd(data, self.sfreq, prefix)

        fig.savefig(self.out_dir / f"{prefix}_psd.pdf", bbox_inches="tight")
        fig.savefig(self.out_dir / f"{prefix}_psd.png", bbox_inches="tight")
        plt.close(fig)

        # Covariance
        cov = np.cov(data)
        np.save(self.out_dir / f"{prefix}_cov.npy", cov)
        fig, ax = plt.subplots(figsize=(15, 15))
        im = ax.imshow(cov, cmap="viridis")
        ax.set_title(f"Covariance - {prefix}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(self.out_dir / f"{prefix}_cov.pdf", bbox_inches="tight")
        fig.savefig(self.out_dir / f"{prefix}_cov.png", bbox_inches="tight")
        plt.close(fig)

    def _load_model(self) -> nn.Module:
        ckpt = self.eval_args.get("ckpt_path")
        if ckpt is None:
            raise ValueError("'ckpt_path' must be provided in the eval config")

        # Lightning first
        try:
            from ..training.train import LitModel

            lit = LitModel.load_from_checkpoint(ckpt, strict=False)
            return lit.model
        except Exception:
            pass

        # Plain state_dict
        model_cls = get_model_class(self.cfg["model_name"])
        with open(self.cfg["model_config"]) as f:
            model_cfg = yaml.safe_load(f)
        model = model_cls(**model_cfg)
        sd = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(sd)
        return model

    def _get_max_hist(self) -> int:
        # 1) user override in config
        if (mh := self.model_cfg.get("gpt2_config", {}).get("n_positions")) is not None:
            return int(mh)
        # 2) attribute on model
        if hasattr(self.model, "max_history"):
            return int(self.model.max_history)
        # 3) gpt‑like model
        if hasattr(self.model, "config") and hasattr(self.model.config, "n_positions"):
            return int(self.model.config.n_positions)
        # 4) fallback – derive from dataset example length
        return self._fallback_max_hist

    def _plot_topomap(self, data: np.ndarray, fname: str, **kwargs) -> None:
        """
        Plot a topomap of the data.

        Args:
            data: Array of shape (C, N) containing the metric value for *C* channels
            across *N* horizon steps.
            fname: Filename (within ``self.out_dir``) for the figure.
            **kwargs: Additional arguments for ``mne.viz.plot_topomap``.
        """
        fig = plt.figure(figsize=(15, 15), dpi=300)

        # axes for the topomap, filling most of the canvas
        ax_map = fig.add_axes([0.05, 0.05, 0.80, 0.90])  # [left, bottom, width, height]

        # recenter pos2d, not sure if needed
        pos = self.pos_2d.copy()
        pos -= pos.mean(axis=0)  # centre at (0, 0)
        pos /= np.abs(pos).max()  # scale to roughly unit circle

        # draw the map – im is the mappable object we’ll attach the colour-bar to
        im, _ = mne.viz.plot_topomap(
            data,
            pos,
            axes=ax_map,
            show=False,
            cmap="RdBu_r",  # or your favourite
            vlim=(np.min(data), np.max(data)),
            **kwargs,  # any other styling args you need
        )

        # colour-bar axes filling the remaining slice of canvas
        ax_cbar = fig.add_axes([0.87, 0.05, 0.03, 0.90])
        cbar = fig.colorbar(im, cax=ax_cbar)
        cbar.set_label("Amplitude (a.u.)", rotation=270, labelpad=15)

        ax_map.set_title(fname.split(".")[0])  # keep your title
        ax_map.set_axis_off()  # no frame/ticks if you like

        fig.savefig(self.out_dir / fname, bbox_inches="tight", pad_inches=0)
        fig.savefig(
            self.out_dir / f"{fname.split('.')[0]}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

    def _plot_horizon_lines(
        self,
        data: np.ndarray,
        fname: str,
        ylabel: str,
        xlabel: str,
        hist_lengths: Optional[List[int]] = None,
    ) -> None:
        """Plot each channel's forecast-horizon curve together with the mean.

        Parameters
        ----------
        arr : np.ndarray
            Array of shape (C, N) containing the metric value for *C* channels
            across *N* horizon steps.  The legacy shape (N, C) is still
            accepted and automatically transposed.
        fname : str
            Filename (within ``self.out_dir``) for the figure.
        ylabel : str
            Y-axis label, e.g. "Accuracy" or "MSE".
        xlabel : str
            X-axis label, e.g. "Forecast horizon (steps)" or "History length (tokens)".
        hist_lengths : Optional[List[int]]
            List of history lengths to plot.
        """

        if data.ndim != 2:
            raise ValueError(f"Expected a 2-D array, got shape {data.shape}.")

        C, N = data.shape
        horizon = np.arange(1, N + 1)

        if hist_lengths is not None:
            horizon = np.array(hist_lengths)

        sns.set_style(style="whitegrid")
        fig, ax = plt.subplots(figsize=(15, 10))

        # Individual channel curves
        palette = sns.color_palette("husl", C)
        for c in range(C):
            sns.lineplot(
                x=horizon,
                y=data[c],
                ax=ax,
                color=palette[c],
                alpha=0.3,
                linewidth=1,
                legend=False,
            )

        # Mean curve across channels
        mean_curve = data.mean(axis=0)
        sns.lineplot(
            x=horizon,
            y=mean_curve,
            ax=ax,
            color="black",
            linewidth=2,
            label="Mean",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(fname.split(".")[0])
        ax.legend(frameon=False)
        fig.savefig(self.out_dir / fname, bbox_inches="tight")
        fig.savefig(self.out_dir / f"{fname.split('.')[0]}.png", bbox_inches="tight")
        plt.close(fig)

    def _get_test_deq(self) -> np.ndarray:
        """
        Get the de-quantised test set.

        Returns:
            Array of shape (C, T_total) containing the de-quantised test set
        """
        if self._cached_test_deq is None:
            print("[EVAL] Assembling de-quantised full test set …")
            chans: List[np.ndarray] = [np.empty((*self.channel_shape, 0))]
            for inp, _ in self.test_dataset:
                dec = mulaw_inv_torch(inp, self.mu).numpy()
                chans.append(dec)
            self._cached_test_deq = np.concatenate(chans, axis=-1)

            if hasattr(self.test_dataset, "reshape"):
                self._cached_test_deq = torch.from_numpy(self._cached_test_deq)
                self._cached_test_deq = self.test_dataset.reshape(self._cached_test_deq)
                self._cached_test_deq = self._cached_test_deq.numpy()

        return self._cached_test_deq

    @torch.inference_mode()
    def _fwd(self, *args, **kwargs):
        device_type = "cuda" if self.device == "cuda" else "cpu"
        ctx_mgr = torch.autocast(
            device_type, dtype=torch.float16, enabled=self.device == "cuda"
        )
        with ctx_mgr:
            return self.model(*args, **kwargs)
