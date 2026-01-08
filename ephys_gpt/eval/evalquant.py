from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import random
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset  # noqa: F401
from ..dataset.dataloaders import TextDataLoader  # noqa: F401
from scipy import signal
from tqdm import tqdm

from ..dataset import split_datasets
from ..training.utils import get_model_class
from ..utils.quantizers import mulaw_inv_torch, mulaw_inv
from ..utils.plotting import plot_psd
from ..utils.eval import sample

DEBUG = False


class EvalQuant:
    """Run and store the three evaluation tasks required by the specification."""

    def __init__(self, cfg: Dict) -> None:
        """Args:

        cfg: Configuration dictionary
        """
        self.cfg = cfg
        self.eval_args = {**cfg.get("eval", {})}

        with open(cfg["model_config"]) as f:
            self.model_cfg = yaml.safe_load(f)

        # --------------------------- data ---------------------------
        split = split_datasets(**cfg["datasplitter"])
        dl_kwargs = {"shuffle": False, **cfg["dataloader"]}
        self.dataloader_args = dl_kwargs
        self.test_dataset = split.test
        self.train_dataset = split.train

        dataloader_class = cfg.get("dataloader_class", "DataLoader")
        self.test_loader = globals()[dataloader_class](split.test, **dl_kwargs)
        self.train_loader = globals()[dataloader_class](split.train, **dl_kwargs)

        first_inputs, _ = split.test[0]
        if isinstance(first_inputs, (list, tuple)):
            first_inputs = first_inputs[0]

        input_overlap = cfg["datasplitter"]["overlap_seconds"]
        self.input_overlap = int(input_overlap * split.test.sfreq)

        self.dtype = self.eval_args.get("dtype", "float16")
        self.channel_shape = self.eval_args["channel_shape"]
        self.num_channels = self.eval_args["num_channels"]
        self.save_test_data = self.eval_args.get("save_test_data", False)
        self.use_test_dataset = self.eval_args.get("use_test_dataset", False)
        if self.use_test_dataset:
            self.dataset = self.test_dataset
        else:
            self.dataset = self.train_dataset

        try:
            self._fallback_max_hist = int(first_inputs.shape[-1])
        except AttributeError:
            self._fallback_max_hist = 0
        self.pos_2d = np.array(split.test.pos_2d)
        self.sfreq = split.test.sfreq
        self.device = self.eval_args["accelerator"]
        self.batch_size = cfg["dataloader"]["batch_size"]

        # --------------------------- model --------------------------
        self.model = self._load_model()
        self.model.eval().to(self.device)

        if DEBUG and hasattr(self.model, "_orig_mod"):
            self.model = self.model._orig_mod  # for compiled models

        self.max_hist = self._get_max_hist()
        self.mu = getattr(self.model, "quant_levels", None)

        if self.mu is None:
            self.mu = self.eval_args.get("quant_levels", 256) - 1

        # --------------------------- output dir ---------------------
        self.out_dir = Path(cfg["save_dir"]) / "evals"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._cached_test_deq: np.ndarray | None = None
        self._cached_quant_test_deq: np.ndarray | None = None

        # build mask for image-models
        self.postprocessor = getattr(self.train_dataset, "postprocessor", None)

    def run_all(self) -> None:
        # print("[EVAL] 1/3  history length sweep …")
        # self.step_history_sweep()

        # print("[EVAL] 2/3  recursive N-step forecasting …")
        # self.step_recursive_future()

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
        N = int(self.eval_args.get("future_steps", 1))

        strategy = self.eval_args.get("gen_sampling", "top_p")
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
                ctx = inputs[..., t - ctx_len: t]  # (B, ..., ctx_len)

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

        if hasattr(self.test_dataset, "postprocessor"):
            acc_horizon = self.test_dataset.postprocessor.reshape(acc_horizon)
            mse_horizon = self.test_dataset.postprocessor.reshape(mse_horizon)

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
        """Compute accuracy and MSE for all history lengths at every timepoint."""
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
                ctx_full = inputs[..., t - H: t]
                logits = self._fwd(ctx_full)  # (B,...,H,Q)
                preds = logits.argmax(dim=-1)  # (B,...,H)
                gt = targets[..., t - H: t]

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

        if hasattr(self.test_dataset, "postprocessor"):
            acc_hist = self.test_dataset.postprocessor.reshape(acc_hist)
            mse_hist = self.test_dataset.postprocessor.reshape(mse_hist)
            acc_sum = self.test_dataset.postprocessor.reshape(acc_sum)
            mse_sum = self.test_dataset.postprocessor.reshape(mse_sum)

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
        """Generate a free-running sequence of tokens."""
        # rollout_gen = self.generate_one_step_rollout()
        # self._eval_psd_cov(rollout_gen, prefix="gen_1step")

        # self.evaluate_random_recursive_rollouts(self.eval_args["gen_seconds"])

        recursive_gen = self.generate()
        self._eval_psd_cov(recursive_gen, prefix="gen")

        self._eval_psd_cov(self._get_test_deq(), prefix="test")

    def _get_initial_example(
        self, total_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        # ------- random single-token context -------
        trial_idx = random.randrange(len(self.train_dataset))
        quant_seq, _ = self.train_dataset[trial_idx]  # (...,T) or ((...,T), cond)

        cond_sequence_full: torch.Tensor | None = None
        cond_history: torch.Tensor | None = None
        cond_pointer: int | None = None

        if isinstance(quant_seq, (tuple, list)):
            ctx_quant = quant_seq[0]
            initial_cond = quant_seq[1]
            cond_needed = ctx_quant.shape[-1] + total_steps
            cond_sequence_full = self._assemble_condition_sequence(
                trial_idx, cond_needed, initial_cond
            ).unsqueeze(
                0
            )  # (1, C_cond, L_total)
            cond_history = cond_sequence_full[..., : ctx_quant.shape[-1]].to(
                self.device
            )
            cond_pointer = ctx_quant.shape[-1]
        else:
            ctx_quant = quant_seq

        ctx = ctx_quant.unsqueeze(0).to(self.device)  # (1,...,1)

        return ctx, cond_history, cond_pointer, cond_sequence_full

    # ---------------------------------------------------------------------
    #  Generation utilities
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def generate(self) -> np.ndarray:
        """Generate a sequence of length `1 + total_steps` (channels x time).

        • Picks a **single random starting token** from the test set. • Grows the sample
        in a `while` loop, each iteration requesting at most   `max_hist` new tokens
        from `_recursive_forecast`. • Saves generated .npy and PSD/cov plots, then
        returns the de-quantised   array `(C, 1+total_steps)`.

        Returns:     Array of shape (C, 1+total_steps) containing the generated sequence
        """
        total_steps = int(self.eval_args["gen_seconds"] * self.sfreq)
        unroll_steps = 1
        print(f"Unroll steps during recursive forecasting: {unroll_steps}")

        ctx, cond_history, cond_pointer, cond_sequence_full = self._get_initial_example(
            total_steps
        )
        init_length = ctx.shape[-1]

        # ------- iterative growth -------
        steps_left = total_steps

        pbar = tqdm(total=total_steps, desc="Generating sequence")
        while steps_left > 0:
            if steps_left < unroll_steps:
                break

            N = unroll_steps

            ctx_slice = ctx[..., -self.max_hist + N - 1:]
            cond_future_chunk = None
            condition_inputs: Tuple[torch.Tensor, torch.Tensor] | None = None
            if cond_history is not None and cond_sequence_full is not None:
                cond_slice = cond_history[..., -self.max_hist + N - 1:]
                assert cond_pointer is not None
                cond_future_chunk = cond_sequence_full[
                    ..., cond_pointer: cond_pointer + N
                ].to(self.device)
                if cond_future_chunk.shape[-1] < N:
                    raise RuntimeError(
                        "Insufficient condition labels assembled for generation."
                    )
                condition_inputs = (cond_slice, cond_future_chunk)

            new_tokens = self._recursive_forecast(
                ctx_slice,
                N,
                condition=condition_inputs,
            )
            ctx = torch.cat([ctx, new_tokens], dim=-1)
            if cond_history is not None and cond_future_chunk is not None:
                cond_history = torch.cat([cond_history, cond_future_chunk], dim=-1)
                cond_pointer += N
            steps_left -= N
            pbar.update(N)
        pbar.close()

        gen_deq = mulaw_inv_torch(ctx.squeeze(0), self.mu)
        gen_deq = gen_deq[..., init_length:]

        if hasattr(self.test_dataset, "postprocessor"):
            gen_deq = self.test_dataset.postprocessor.reshape(gen_deq)

        gen_deq = gen_deq.cpu().numpy()

        strategy = self.eval_args.get("gen_sampling", "top_p")
        np.save(self.out_dir / f"generated_{strategy}.npy", gen_deq)
        return gen_deq

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

        strategy = self.eval_args.get("gen_sampling", "top_p")
        sample_args = {
            "strategy": strategy,
            "temperature": self.eval_args.get("temperature", 1.0),
            "top_k": self.eval_args.get("top_k", 0),
            "top_p": self.eval_args.get("top_p", 0.8),
        }

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
            logits = self._fwd(window_batch)
            next_logits = logits[..., -1, :]
            next_tok = sample(next_logits, **sample_args)
            predictions.append(next_tok.cpu())
            pbar.update(end - start)
            start = end
        pbar.close()

        pred_tokens = torch.cat(predictions, dim=0)  # (num_windows, C)
        pred_tokens = pred_tokens.transpose(0, 1).contiguous()  # (C, num_windows)

        gen_deq = mulaw_inv_torch(pred_tokens, self.mu)
        if hasattr(self.test_dataset, "postprocessor"):
            gen_deq = self.test_dataset.postprocessor.reshape(gen_deq)

        gen_np = gen_deq.cpu().numpy()
        np.save(self.out_dir / f"generated_1step_{strategy}.npy", gen_np)
        return gen_np

    @torch.inference_mode()
    def evaluate_random_recursive_rollouts(self, num_samples: int) -> None:
        """Evaluate PSD/Cov for 1-second recursive rollouts from random contexts."""

        if num_samples <= 0:
            return

        horizon = max(int(round(self.sfreq)), self.eval_args.get("unroll_steps", 1))
        dataset_len = len(self.test_dataset)
        if dataset_len == 0:
            raise RuntimeError("Test dataset is empty; cannot sample contexts.")

        replace = num_samples > dataset_len
        rng = np.random.default_rng()
        indices = rng.choice(dataset_len, size=num_samples, replace=replace)

        contexts: List[Tensor] = []
        lengths: List[int] = []
        for idx in tqdm(indices.tolist(), desc="random-rollout-contexts"):
            inputs, _ = self.test_dataset[int(idx)]
            if isinstance(inputs, (tuple, list)):
                ctx = inputs[0]
            else:
                ctx = inputs
            ctx = ctx.to(torch.long)
            contexts.append(ctx)
            lengths.append(int(ctx.shape[-1]))

        context_len = int(min(lengths))
        if context_len <= 0:
            raise RuntimeError("Context length must be positive for rollouts.")

        ctx_tensor = torch.stack([ctx[..., -context_len:] for ctx in contexts], dim=0)

        preds: List[Tensor] = []
        batch_iter = range(0, num_samples, self.batch_size)
        for start in tqdm(batch_iter, desc="random-rollout-forecast"):
            end = min(start + self.batch_size, num_samples)
            ctx_batch = ctx_tensor[start:end].to(self.device, non_blocking=True)
            new_tokens = self._recursive_forecast(ctx_batch, horizon)
            preds.append(new_tokens.cpu())

        pred_tokens = torch.cat(preds, dim=0)
        pred_float = mulaw_inv_torch(pred_tokens, self.mu)

        if hasattr(self.test_dataset, "postprocessor"):
            pred_float = self.test_dataset.postprocessor.reshape(pred_float)

        rollouts = pred_float.cpu().numpy()

        if rollouts.ndim == 2:
            rollouts = rollouts[None, ...]

        nperseg = max(1, min(horizon, int(round(self.sfreq))))
        freq_axis, psd = signal.welch(
            rollouts,
            fs=self.sfreq,
            axis=-1,
            nperseg=nperseg,
            scaling="density",
        )
        psd_mean = psd.mean(axis=0)

        cov_chunks = np.stack([np.cov(window) for window in rollouts], axis=0)
        cov_mean = cov_chunks.mean(axis=0)

        np.save(self.out_dir / "random_rollout_psd_mean.npy", psd_mean)
        np.save(self.out_dir / "random_rollout_psd_freqs.npy", freq_axis)
        np.save(self.out_dir / "random_rollout_cov_mean.npy", cov_mean)

        prefix = "random_rollout_mean"
        self._plot_psd_from_values(freq_axis, psd_mean, prefix)
        self._plot_covariance_matrix(cov_mean, prefix)

    @torch.inference_mode()
    def _recursive_forecast(
        self,
        ctx: Tensor,
        N: int,
        condition: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """Forecast *N* tokens given `ctx`; supports various sampling strategies.

        Args:     ctx: Context tensor of shape (B, C, max_hist)     N: Forecast horizon

        Returns:     Tensor of shape (B, C, N) containing the integer-bin predictions
        for     the *N* future steps.
        """
        sample_args = {
            "strategy": self.eval_args.get(["gen_sampling"], "top_p"),
            "temperature": self.eval_args.get("temperature", 1.0),
            "top_k": self.eval_args.get("top_k", 0),
            "top_p": self.eval_args.get("top_p", 0.8),
        }

        cond_hist: Tensor | None = None
        cond_future: Tensor | None = None
        if condition is not None:
            if not isinstance(condition, (tuple, list)) or len(condition) != 2:
                raise ValueError("Condition must be a (history, future) tuple.")
            cond_hist = condition[0].clone()
            cond_future = condition[1]
            if cond_future.shape[-1] < N:
                raise ValueError(
                    "Cond future chunk must be at least as long as forecast horizon."
                )

        if hasattr(self.model, "generate") and cond_hist is None:
            return self.model.generate(ctx.clone(), N, sample_args)

        seq = ctx.clone()
        cond_seq = cond_hist.clone() if cond_hist is not None else None
        generated: List[Tensor] = []
        cache = {}
        for i in range(N):
            model_input: Tensor | Tuple[Tensor, Tensor]

            # truncate seq to length of ctx
            seq = seq[..., -ctx.shape[-1]:]

            if cond_seq is not None:
                if cond_future is None:
                    raise ValueError(
                        "Future condition sequence missing while conditioning enabled."
                    )
                cond_seq = cond_seq[..., -ctx.shape[-1]:]
                model_input = (seq, cond_seq)
            else:
                model_input = seq

            # try:
            #    logits, cache = self._fwd(
            #        model_input, past_key_values=cache
            #    )  # (B,...,T,Q)
            # except (TypeError, ValueError):
            logits = self._fwd(model_input)  # type: ignore[assignment]

            next_logits = logits[..., -1, :]  # (B,...,Q)
            next_tok = sample(next_logits, **sample_args)  # (B,...)
            if self.postprocessor is not None:
                H = self.train_dataset.image_size
                row_idx = self.postprocessor.row_idx
                col_idx = self.postprocessor.col_idx

                img = (
                    torch.ones((1, H, H), dtype=next_tok.dtype, device=next_tok.device)
                    * self.train_dataset.fill_value
                )
                img[:, row_idx, col_idx] = next_tok[:, row_idx, col_idx]
                next_tok = img

            generated.append(next_tok)

            if cache:
                seq = next_tok.unsqueeze(-1)
                if cond_seq is not None:
                    next_cond = cond_future[..., i: i + 1]
                    cond_seq = next_cond
            else:
                seq = torch.cat([seq, next_tok.unsqueeze(-1)], dim=-1)
                if cond_seq is not None:
                    next_cond = cond_future[..., i: i + 1]
                    cond_seq = torch.cat([cond_seq, next_cond], dim=-1)
        return torch.stack(generated, dim=-1)  # (B,...,N)

    # -----------------------------------
    # Helper functions
    # -----------------------------------
    def _assemble_quant_sequence(
        self,
        dataset: Dataset,
        *,
        min_total_length: Optional[int] = None,
        drop_first_overlap: bool = True,
    ) -> torch.Tensor:
        """Concatenate dataset windows into a continuous quantised sequence."""

        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty; cannot assemble sequence.")

        chunks: List[torch.Tensor] = []
        total = 0
        first_chunk = True
        for idx in range(len(dataset)):
            inputs, _ = dataset[idx]
            if isinstance(inputs, (tuple, list)):
                chunk = inputs[0]
            else:
                chunk = inputs

            chunk = chunk.detach().cpu()
            overlap = (
                self.input_overlap if (drop_first_overlap or not first_chunk) else 0
            )
            if overlap > 0:
                chunk = chunk[..., overlap:]

            if chunk.shape[-1] == 0:
                first_chunk = False
                continue

            chunks.append(chunk)
            total += int(chunk.shape[-1])
            first_chunk = False

            if min_total_length is not None and total >= min_total_length:
                break

        if not chunks:
            raise RuntimeError("Unable to assemble seq; encountered empty windows.")

        sequence = torch.cat(chunks, dim=-1)

        if min_total_length is not None:
            if sequence.shape[-1] < min_total_length:
                raise RuntimeError(
                    "Insufficient tokens for requested sequence: "
                    f"need {min_total_length}, got {sequence.shape[-1]}."
                )
            sequence = sequence[..., :min_total_length]

        return sequence

    def _eval_psd_cov(self, data: np.ndarray, prefix: str) -> None:
        """Evaluate the PSD and covariance of the data.

        Args:     data: Array of shape (C, T) containing the data     prefix: Prefix for
        the filenames
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

    def _plot_psd_from_values(
        self, freqs: np.ndarray, psd: np.ndarray, prefix: str
    ) -> None:
        """Save PSD plot given explicit frequency and PSD arrays."""

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(freqs, psd.T, alpha=0.3)
        ax.set_xlabel("Hz")
        ax.set_ylabel("Power")
        ax.set_title(f"PSD - {prefix}")
        ax.set_yscale("log")

        psd_flat = psd.flatten()
        psd_flat = psd_flat[psd_flat > 0]
        if psd_flat.size > 0:
            lower = np.percentile(psd_flat, 0.1)
            upper = np.percentile(psd_flat, 99.9)
            if lower > 0 and upper > lower:
                ax.set_ylim([lower, upper])

        fig.savefig(self.out_dir / f"{prefix}_psd.pdf", bbox_inches="tight")
        fig.savefig(self.out_dir / f"{prefix}_psd.png", bbox_inches="tight")
        plt.close(fig)

    def _plot_covariance_matrix(self, cov: np.ndarray, prefix: str) -> None:
        """Save a covariance heatmap with consistent styling."""

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
        # 1) attribute on model
        if hasattr(self.model, "max_history"):
            return int(self.model.max_history)
        # 2) fallback – derive from dataset example length
        return self._fallback_max_hist

    def _plot_topomap(self, data: np.ndarray, fname: str, **kwargs) -> None:
        """Plot a topomap of the data.

        Args:     data: Array of shape (C, N) containing the metric value for *C*
        channels     across *N* horizon steps.     fname: Filename (within
        ``self.out_dir``) for the figure.     **kwargs: Additional arguments for
        ``mne.viz.plot_topomap``.
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

        Parameters ---------- arr : np.ndarray     Array of shape (C, N) containing the
        metric value for *C* channels     across *N* horizon steps.  The legacy shape
        (N, C) is still     accepted and automatically transposed. fname : str
        Filename (within ``self.out_dir``) for the figure. ylabel : str     Y-axis
        label, e.g. "Accuracy" or "MSE". xlabel : str     X-axis label, e.g. "Forecast
        horizon (steps)" or "History length (tokens)". hist_lengths :
        Optional[List[int]]     List of history lengths to plot.
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

    def _assemble_condition_sequence(
        self, start_idx: int, required_length: int, initial_cond: Tensor
    ) -> Tensor:
        """Concatenate condition labels starting from `start_idx` until
        `required_length`."""

        dataset_len = len(self.train_dataset)
        if dataset_len == 0:
            raise RuntimeError(
                "Test dataset is empty; cannot assemble condition labels."
            )

        cond_chunks = [initial_cond.clone()]
        total_len = int(initial_cond.shape[-1])
        idx = (start_idx + 1) % dataset_len

        while total_len < required_length:
            inputs, _ = self.train_dataset[idx]
            if not isinstance(inputs, (tuple, list)) or len(inputs) < 2:
                raise RuntimeError(
                    "Encountered sample without condition labels during assembly."
                )
            cond_chunk = inputs[1]
            if cond_chunk.shape[-1] == 0:
                raise RuntimeError("Encountered empty condition chunk in dataset.")
            cond_chunks.append(cond_chunk.clone())
            total_len += int(cond_chunk.shape[-1])
            idx = (idx + 1) % dataset_len

        cond_seq = torch.cat(cond_chunks, dim=-1)
        return cond_seq[..., :required_length]

    def _get_test_deq(self) -> np.ndarray:
        """Get the de-quantised test set.

        Returns:     Array of shape (C, T_total) containing the de-quantised test set
        """
        if self._cached_test_deq is None:
            print("[EVAL] Assembling de-quantised full test set …")
            quant_tensor = self._assemble_quant_sequence(
                self.dataset,
                drop_first_overlap=True,
            )
            self._cached_quant_test_deq = quant_tensor.numpy()

            # save to file
            if self.save_test_data:
                np.save(self.out_dir / "test_quant.npy", self._cached_quant_test_deq)

            self._cached_test_deq = mulaw_inv(self._cached_quant_test_deq, self.mu)

            if hasattr(self.test_dataset, "postprocessor"):
                self._cached_test_deq = torch.from_numpy(self._cached_test_deq)
                self._cached_test_deq = self.test_dataset.postprocessor.reshape(
                    self._cached_test_deq
                )
                self._cached_test_deq = self._cached_test_deq.numpy()

        return self._cached_test_deq

    @torch.inference_mode()
    def _fwd(self, *args, **kwargs):
        device_type = "cuda" if self.device == "cuda" else "cpu"
        dtype = torch.float16 if self.dtype == "float16" else torch.float32
        ctx_mgr = torch.autocast(
            device_type, dtype=dtype, enabled=self.device == "cuda"
        )
        with ctx_mgr:
            return self.model(*args, **kwargs)
