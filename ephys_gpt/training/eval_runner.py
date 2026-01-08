from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
import logging
import scipy.signal as signal
from scipy.stats import wasserstein_distance
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data._utils.collate import default_collate

from ..dataset import TextDataLoader  # noqa: F401
from ..dataset import split_datasets
from ..dataset.datasets import _load_chunk_cached
from ..utils.eval import sample
from .lightning import LitModel
from .vidtok import VidtokLightning  # noqa: F401
from ..utils.quantizers import mulaw_inv_torch

logger = logging.getLogger(__name__)


def _chunk_sort_key(filename: str) -> tuple[int, object]:
    stem = Path(filename).stem
    if stem.isdigit():
        return 0, int(stem)
    return 1, stem


def _unpack_index_entry(entry: Any) -> tuple[str, str, str, int] | None:
    if entry is None:
        return None

    if isinstance(entry, tuple) and len(entry) >= 4:
        dataset_key, session, chunk, start = entry[:4]
        return str(dataset_key), str(session), str(chunk), int(start)

    if all(hasattr(entry, attr) for attr in ("dataset", "session", "chunk", "start")):
        return (
            str(getattr(entry, "dataset")),
            str(getattr(entry, "session")),
            str(getattr(entry, "chunk")),
            int(getattr(entry, "start")),
        )

    return None


def _select_first_chunk_indices(
    indices: list[Any], num_runs: int, rng: np.random.Generator
) -> list[int]:
    if not indices or num_runs <= 0:
        return []

    unpacked: list[tuple[int, tuple[str, str, str, int]]] = []
    for idx, entry in enumerate(indices):
        spec = _unpack_index_entry(entry)
        if spec is not None:
            unpacked.append((idx, spec))

    if not unpacked:
        return []

    first_chunk: dict[tuple[str, str], tuple[tuple[int, object], str]] = {}
    for _, (dataset_key, session, chunk, _) in unpacked:
        key = (dataset_key, session)
        order = _chunk_sort_key(chunk)
        prev = first_chunk.get(key)
        if prev is None or order < prev[0]:
            first_chunk[key] = (order, chunk)

    candidates: dict[tuple[str, str], list[int]] = {}
    for idx, (dataset_key, session, chunk, _) in unpacked:
        key = (dataset_key, session)
        target_chunk = first_chunk.get(key, (None, None))[1]
        if chunk != target_chunk:
            continue
        candidates.setdefault(key, []).append(idx)

    session_keys = list(candidates.keys())
    rng.shuffle(session_keys)

    selected: list[int] = []
    for key in session_keys:
        if len(selected) >= num_runs:
            break
        options = sorted(
            candidates.get(key, []),
            key=lambda i: _unpack_index_entry(indices[i])[3],  # type: ignore[index]
        )
        if options:
            selected.append(int(options[0]))

    return selected


class EvaluationRunner:
    """Lightweight evaluation loop used by automated checkpoint triggers."""

    def __init__(
        self,
        cfg: dict,
        device: str | None = "cuda",
    ) -> None:
        eval_cfg = cfg.get("eval_runner", {})
        self.lit_module_name = eval_cfg.get("lit_module", None)
        self.ckpt_path = self._resolve_checkpoint(eval_cfg.get("ckpt_path"))
        self.eval_step = eval_cfg.get("step")
        self.eval_epoch = eval_cfg.get("epoch")
        self.run_version = eval_cfg.get("version")
        self.max_batches = eval_cfg.get("max_batches", 2)
        self.num_examples = eval_cfg.get("num_examples", 2)
        self.mu = eval_cfg.get("mu", None)
        self.generate_cfg = self._parse_generate(eval_cfg.get("generate"))

        self.cfg = cfg
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                and cfg.get("trainer", {}).get("accelerator", "cpu") != "cpu"
                else "cpu"
            )
        )
        self.save_dir = Path(cfg.get("save_dir", "."))
        self.sfreq: float | None = None
        self._prepare_data()

    def _parse_generate(self, cfg: Any) -> dict[str, Any]:
        """Normalise optional generation settings."""
        base = {"enabled": False, "params": {}}
        if cfg is None:
            return base

        if isinstance(cfg, bool):
            base["enabled"] = cfg
            return base

        if not isinstance(cfg, dict):
            raise TypeError("eval_runner.generate must be a bool or a dict of args.")

        enabled = bool(cfg.get("enabled", True))
        params = {k: v for k, v in cfg.items() if k != "enabled"}
        base.update({"enabled": enabled, "params": params})
        return base

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _prepare_data(self) -> None:
        datasets = split_datasets(**self.cfg["datasplitter"])

        dataloader_cls_name = self.cfg.get("dataloader_class", "DataLoader")
        dataloader_cls = globals().get(dataloader_cls_name, DataLoader)

        args = self.cfg.get("dataloader", {})
        shuffle = None if isinstance(datasets.val, IterableDataset) else False
        bs = args.pop("batch_size", 1)
        bs = 1

        self.val_loader = dataloader_cls(
            datasets.val, shuffle=shuffle, batch_size=bs, **args
        )
        self.val_dataset = datasets.val
        self.postprocessor = getattr(datasets.train, "postprocessor", None)
        self.sfreq = getattr(datasets.val, "sfreq", None)

    def _resolve_checkpoint(self, ckpt_path: str | None) -> str:
        """Fallback to the newest matching checkpoint if the requested one is gone."""
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint path provided for evaluation.")

        path = Path(ckpt_path)
        if path.exists():
            return str(path)

        alt = self._wait_for_checkpoint(path, timeout=30)
        if alt:
            logger.warning(
                "Requested checkpoint %s missing; using latest available %s",
                path,
                alt,
            )
            return str(alt)

        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    def _find_latest_checkpoint(self, path: Path) -> Path | None:
        prefix = path.stem.split("-epoch")[0]
        pattern = f"{prefix}-epoch*{path.suffix}"
        candidates = sorted(path.parent.glob(pattern))
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _load_model(self, ckpt_path: str) -> LitModel:
        path = Path(ckpt_path)
        if not path.exists():
            alt = self._wait_for_checkpoint(path, timeout=60)
            if alt:
                logger.warning(
                    "Checkpoint %s disappeared; falling back to %s", path, alt
                )
                path = alt
            else:
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if self.lit_module_name is not None:
            lit_module = globals().get(self.lit_module_name)
            # Pass postprocessor for modules that need it (e.g., VidtokLightning)
            lit_model = lit_module.load_from_checkpoint(
                str(path), strict=False, postprocessor=self.postprocessor
            )
        else:
            lit_model = LitModel.load_from_checkpoint(str(path), strict=False)
        lit_model.model.to(self.device)
        lit_model.model.eval()
        return lit_model

    def _wait_for_checkpoint(self, path: Path, timeout: int = 30) -> Path | None:
        """Poll for a matching checkpoint for up to `timeout` seconds."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            alt = self._find_latest_checkpoint(path)
            if alt:
                return alt
            time.sleep(1)
        return None

    def _move_to_device(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = batch

        if isinstance(inputs, (tuple, list)):
            inputs = tuple(x.to(self.device) for x in inputs)
        else:
            inputs = inputs.to(self.device)

        return inputs, targets.to(self.device)

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #
    def _build_sample_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        params = self.generate_cfg.get("params", {})
        strategy = str(params.get("strategy", params.get("sampling", "top_p"))).lower()
        sample_args = {
            "strategy": strategy,
            "temperature": float(params.get("temperature", 1.0)),
            "top_k": int(params.get("top_k", 0)),
            "top_p": float(params.get("top_p", 0.8)),
        }

        def _sample_fn(logits: torch.Tensor) -> torch.Tensor:
            return sample(logits, **sample_args)

        return _sample_fn

    def _extract_tensor(self, output: Any) -> torch.Tensor | None:
        if torch.is_tensor(output):
            return output

        if isinstance(output, dict):
            for key in ("logits", "output", "outputs", "pred", "preds"):
                val = output.get(key)
                if torch.is_tensor(val):
                    return val
            for val in output.values():
                if torch.is_tensor(val):
                    return val

        if isinstance(output, (tuple, list)):
            for item in output:
                if torch.is_tensor(item):
                    return item

        return None

    def _normalise_timeseries(self, tensor: torch.Tensor | Any) -> torch.Tensor | None:
        """Convert arbitrary tensors to a (C, T) float tensor without a batch dim."""
        if tensor is None:
            return None

        try:
            arr = (
                tensor.detach() if torch.is_tensor(tensor) else torch.as_tensor(tensor)
            )
        except Exception:
            return None

        if arr.ndim >= 3:
            arr = arr[0]

        if arr.ndim == 0:
            return None

        if arr.ndim == 1:
            arr = arr.unsqueeze(0)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        if arr.shape[0] > arr.shape[1]:
            arr = arr.transpose(0, 1)

        return arr.to(torch.float32)

    def _to_timeseries(self, tensor: torch.Tensor | Any) -> np.ndarray | None:
        arr = self._normalise_timeseries(tensor)
        if arr is None:
            return None
        return arr.cpu().numpy()

    def _collect_target_rollout(
        self,
        initial_targets: torch.Tensor,
        rollout_steps: int,
        loader_iter: Iterator[Any],
    ) -> torch.Tensor | None:
        """Stitch together consecutive validation samples until the target length
        matches the rollout horizon."""
        seq = self._normalise_timeseries(initial_targets)
        if seq is None:
            return None

        while seq.shape[1] < rollout_steps:
            try:
                _, next_targets = self._move_to_device(next(loader_iter))
            except StopIteration:
                break

            next_seq = self._normalise_timeseries(next_targets)
            if next_seq is None:
                continue
            seq = torch.cat([seq, next_seq], dim=1)

        if seq.shape[1] < rollout_steps:
            print("[eval_runner] Val data shorter than rollout; truncating targets.")

        return seq[:, :rollout_steps]

    def _resolve_divergence_window(self, params: dict[str, Any]) -> int:
        """Determine the window (in steps) used for rollout divergence."""
        steps = params.get("divergence_window_steps")
        if steps is not None:
            steps_int = int(steps)
            if steps_int > 0:
                return steps_int

        seconds = params.get("divergence_window_seconds")
        if seconds is not None and self.sfreq is not None:
            sec_val = float(seconds)
            if sec_val > 0:
                approx_steps = int(sec_val * float(self.sfreq))
                if approx_steps > 0:
                    return approx_steps

        if self.sfreq is not None:
            return max(5, int(float(self.sfreq) * 0.5))

        return 20

    def _window_corr_divergence(
        self, generated: np.ndarray, target: np.ndarray
    ) -> float:
        """Compute a correlation-based divergence for a window of samples.

        The window is z-scored before computing the correlation to make the metric
        insensitive to absolute amplitude and drift. The result is scaled to [0, 1],
        where 0 = perfect alignment and 1 = perfect anti-correlation.
        """
        gen_flat = generated.reshape(-1).astype(np.float64)
        tgt_flat = target.reshape(-1).astype(np.float64)

        gen_flat -= gen_flat.mean()
        tgt_flat -= tgt_flat.mean()
        gen_std = gen_flat.std()
        tgt_std = tgt_flat.std()
        if gen_std < 1e-8 or tgt_std < 1e-8:
            return 1.0

        corr = float(
            np.clip(np.mean(gen_flat * tgt_flat) / (gen_std * tgt_std), -1.0, 1.0)
        )
        # Convert correlation in [-1, 1] to divergence in [0, 1]
        return float((1.0 - corr) * 0.5)

    def _relative_l2(self, generated: np.ndarray, target: np.ndarray) -> float:
        """Compute a relative L2 error, guarding against zero norms."""
        denom = float(np.linalg.norm(target))
        if denom < 1e-12:
            denom = 1e-12
        return float(np.linalg.norm(generated - target) / denom)

    def _phase_distance(self, gen_angle: np.ndarray, tgt_angle: np.ndarray) -> float:
        """Mean absolute wrapped phase difference."""
        wrapped = np.angle(np.exp(1j * (gen_angle - tgt_angle)))
        return float(np.mean(np.abs(wrapped)))

    def _window_cov_distance(self, generated: np.ndarray, target: np.ndarray) -> float:
        gen_cov = np.cov(generated)
        tgt_cov = np.cov(target)
        return self._relative_l2(gen_cov, tgt_cov)

    def _spectral_distance(
        self, gen_spec: np.ndarray, tgt_spec: np.ndarray
    ) -> tuple[float, float]:
        """Compare two complex spectra via magnitude and phase."""
        if gen_spec.size == 0 or tgt_spec.size == 0:
            return np.nan, np.nan

        mag_diff = self._relative_l2(np.abs(gen_spec), np.abs(tgt_spec))
        ang_diff = self._phase_distance(
            np.angle(gen_spec),
            np.angle(tgt_spec),
        )
        return mag_diff, ang_diff

    def _window_stft_distance(
        self, generated: np.ndarray, target: np.ndarray
    ) -> tuple[float, float]:
        timesteps = generated.shape[1]
        if timesteps < 2:
            return np.nan, np.nan

        nperseg = min(128, timesteps)
        noverlap = int(nperseg * 0.75) if nperseg > 1 else 0
        fs_attr = getattr(self, "sfreq", None)
        fs = float(fs_attr) if fs_attr is not None else 1.0

        _, _, gen_spec = signal.stft(
            generated, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
        )
        _, _, tgt_spec = signal.stft(
            target, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=1
        )
        return self._spectral_distance(gen_spec, tgt_spec)

    def _window_fft_distance(
        self, generated: np.ndarray, target: np.ndarray
    ) -> tuple[float, float]:
        """Compare amplitude spectra and phase via FFT.

        Note: This computes FFT-based spectral distance (complex coefficients), not PSD
        (which is |FFT|² and has no phase).
        """
        timesteps = generated.shape[1]
        if timesteps < 2:
            return np.nan, np.nan

        norm = float(max(1, timesteps))
        gen_fft = np.fft.rfft(generated, axis=1) / norm
        tgt_fft = np.fft.rfft(target, axis=1) / norm
        return self._spectral_distance(gen_fft, tgt_fft)

    def _rollout_divergence_curve(
        self, generated: np.ndarray, target: np.ndarray, window_steps: int
    ) -> dict[str, np.ndarray]:
        """Compute a divergence curve over the rollout horizon.

        For each timestep, we compare the generated samples against the ground-truth
        continuation within a short sliding window. A small divergence means the local
        dynamics still match; the curve should climb as the free-run drifts away from
        the reference.

        Returns a mapping of metric name -> divergence values for each stride.
        """
        if generated.shape != target.shape:
            raise ValueError(
                "Generated and target shapes must match,"
                f"got {generated.shape} vs {target.shape}"
            )

        total_steps = generated.shape[1]
        curves: dict[str, list[float]] = {
            "correlation": [],
            "covariance": [],
            "stft_magnitude": [],
            "stft_angle": [],
            "fft_magnitude": [],
            "fft_angle": [],
        }
        for step in range(window_steps, total_steps, window_steps):
            window_gen = generated[:, :step]
            window_tgt = target[:, :step]
            curves["correlation"].append(
                self._window_corr_divergence(window_gen, window_tgt)
            )
            curves["covariance"].append(
                self._window_cov_distance(window_gen, window_tgt)
            )

            stft_mag, stft_angle = self._window_stft_distance(window_gen, window_tgt)
            curves["stft_magnitude"].append(stft_mag)
            curves["stft_angle"].append(stft_angle)

            fft_mag, fft_angle = self._window_fft_distance(window_gen, window_tgt)
            curves["fft_magnitude"].append(fft_mag)
            curves["fft_angle"].append(fft_angle)

        return {key: np.asarray(vals, dtype=np.float32) for key, vals in curves.items()}

    def _serialise_numpy(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: self._serialise_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serialise_numpy(v) for v in obj]
        return obj

    def _plot_timeseries_divergence_metrics(
        self, metrics: dict[str, Any], out_dir: Path
    ) -> None:
        times = np.asarray(metrics.get("times_s", []), dtype=np.float32)
        if times.size == 0:
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        metric_order = ["psd_jsd", "band_jsd", "stft_wass"]
        for ax, name in zip(axes, metric_order):
            pred_vals = np.asarray(metrics["pred"].get(name, []), dtype=np.float32)
            null_vals = np.asarray(metrics["null"].get(name, []), dtype=np.float32)
            thresh = float(metrics["thresholds"].get(name, np.nan))
            t_div = metrics["summary"]["time_to_divergence_s"].get(name)

            ax.plot(times, pred_vals, label="pred vs target", color="C0")
            ax.plot(times, null_vals, label="null (target vs other)", color="C1")
            if np.isfinite(thresh):
                ax.axhline(thresh, color="red", linestyle="--", label="threshold")
            if t_div is not None:
                ax.axvline(t_div, color="black", linestyle=":", label="divergence")
            ax.set_ylabel(name)
            ax.grid(False)

        axes[-1].set_xlabel("Time (s)")
        axes[0].legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / "rollout_divergence_timeseries.png", bbox_inches="tight")
        plt.close(fig)

    def _compute_timeseries_divergence_metrics(
        self,
        x_true: np.ndarray,
        x_pred: np.ndarray,
        x_true_other: np.ndarray,
        window_steps: int,
        out_dir: Path,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        if self.sfreq is None:
            return None

        cfg = params.get("timeseries_divergence", {})
        fs = float(cfg.get("fs", self.sfreq))
        win_s = float(
            cfg.get("win_seconds", max(window_steps / fs, 8.0 / max(fs, 1e-12)))
        )
        stride_s = float(cfg.get("stride_seconds", max(window_steps / fs, 1.0 / fs)))
        fmin = float(cfg.get("fmin", 0.5))
        fmax = cfg.get("fmax")
        fmax_val = float(fmax) if fmax is not None else None
        welch_nperseg_s = float(cfg.get("welch_nperseg_s", 2.0))
        welch_noverlap_frac = float(cfg.get("welch_noverlap_frac", 0.5))
        stft_nperseg_s = float(cfg.get("stft_nperseg_s", 1.0))
        stft_noverlap_frac = float(cfg.get("stft_noverlap_frac", 0.5))
        threshold_k = float(cfg.get("threshold_k", 2.0))
        consecutive_windows = int(cfg.get("consecutive_windows", 3))
        bands_cfg = cfg.get("bands")
        eps = 1e-12
        ch_names = getattr(getattr(self, "val_dataset", None), "ch_names", None)
        num_channels_hint = len(ch_names) if ch_names is not None else None

        def _to_TxC(x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x)
            if arr.ndim == 1:
                return arr[:, None]
            if arr.ndim != 2:
                raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
            # Inputs are channel-first (C, T); transpose to (T, C)
            if num_channels_hint:
                if (
                    arr.shape[0] == num_channels_hint
                    and arr.shape[1] != num_channels_hint
                ):
                    return arr.T
                if (
                    arr.shape[1] == num_channels_hint
                    and arr.shape[0] != num_channels_hint
                ):
                    return arr
            return arr.T

        def _jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
            p = np.clip(p, eps, None)
            q = np.clip(q, eps, None)
            p = p / np.sum(p, axis=-1, keepdims=True)
            q = q / np.sum(q, axis=-1, keepdims=True)
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=-1)
            kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=-1)
            return 0.5 * (kl_pm + kl_qm)

        def _first_crossing_time(
            metric_series: np.ndarray, thresh: float, times_s: np.ndarray
        ) -> float | None:
            above = metric_series > thresh
            if consecutive_windows <= 1:
                idx = np.argmax(above) if np.any(above) else None
                if idx is None or not np.any(above):
                    return None
                return float(times_s[idx])
            run = 0
            for i, a in enumerate(above):
                run = run + 1 if a else 0
                if run >= consecutive_windows:
                    return float(times_s[i - consecutive_windows + 1])
            return None

        if bands_cfg is None:
            bands = {
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 45.0),
            }
        else:
            bands = {str(k): tuple(v) for k, v in bands_cfg.items()}

        x_true = _to_TxC(x_true)
        x_pred = _to_TxC(x_pred)
        x_true_other = _to_TxC(x_true_other)

        T = min(x_true.shape[0], x_pred.shape[0], x_true_other.shape[0])
        C = min(x_true.shape[1], x_pred.shape[1], x_true_other.shape[1])
        x_true = x_true[:T, :C]
        x_pred = x_pred[:T, :C]
        x_true_other = x_true_other[:T, :C]

        if fmax_val is None:
            fmax_val = 0.95 * (fs / 2.0)

        W = int(round(win_s * fs))
        S = int(round(stride_s * fs))
        if W < 8 or T < W:
            return None
        S = max(1, S)
        starts = np.arange(0, T - W + 1, S, dtype=int)
        if starts.size == 0:
            return None
        times_s = starts / fs

        welch_nperseg = int(round(welch_nperseg_s * fs))
        welch_nperseg = max(8, min(welch_nperseg, W))
        welch_noverlap = int(round(welch_noverlap_frac * welch_nperseg))
        welch_noverlap = min(welch_noverlap, welch_nperseg - 1)

        stft_nperseg = int(round(stft_nperseg_s * fs))
        stft_nperseg = max(8, min(stft_nperseg, W))
        stft_noverlap = int(round(stft_noverlap_frac * stft_nperseg))
        stft_noverlap = min(stft_noverlap, stft_nperseg - 1)

        band_names = list(bands.keys())
        band_edges = np.array([bands[k] for k in band_names], dtype=float)
        B = band_edges.shape[0]

        pred_psd_jsd = np.zeros(len(starts), dtype=float)
        null_psd_jsd = np.zeros(len(starts), dtype=float)
        pred_band_jsd = np.zeros(len(starts), dtype=float)
        null_band_jsd = np.zeros(len(starts), dtype=float)
        pred_stft_wass = np.zeros(len(starts), dtype=float)
        null_stft_wass = np.zeros(len(starts), dtype=float)

        for i, s0 in enumerate(starts):
            sl = slice(s0, s0 + W)
            xt = x_true[sl, :]
            xp = x_pred[sl, :]
            xo = x_true_other[sl, :]

            f, P_t = signal.welch(
                xt, fs=fs, nperseg=welch_nperseg, noverlap=welch_noverlap, axis=0
            )
            _, P_p = signal.welch(
                xp, fs=fs, nperseg=welch_nperseg, noverlap=welch_noverlap, axis=0
            )
            _, P_o = signal.welch(
                xo, fs=fs, nperseg=welch_nperseg, noverlap=welch_noverlap, axis=0
            )

            fmask = (f >= fmin) & (f <= fmax_val)
            ff = f[fmask]
            P_t = P_t[fmask, :] + eps
            P_p = P_p[fmask, :] + eps
            P_o = P_o[fmask, :] + eps

            C_psd = min(C, P_t.shape[1], P_p.shape[1], P_o.shape[1])
            if C_psd <= 0:
                return None
            P_t = P_t[:, :C_psd]
            P_p = P_p[:, :C_psd]
            P_o = P_o[:, :C_psd]

            Pt_dist = (P_t / np.sum(P_t, axis=0, keepdims=True)).T
            Pp_dist = (P_p / np.sum(P_p, axis=0, keepdims=True)).T
            Po_dist = (P_o / np.sum(P_o, axis=0, keepdims=True)).T

            pred_psd_jsd[i] = float(np.mean(_jsd(Pt_dist, Pp_dist)))
            null_psd_jsd[i] = float(np.mean(_jsd(Pt_dist, Po_dist)))

            band_pow_t = np.zeros((C_psd, B), dtype=float)
            band_pow_p = np.zeros((C_psd, B), dtype=float)
            band_pow_o = np.zeros((C_psd, B), dtype=float)

            for bi, (lo, hi) in enumerate(band_edges):
                bm = (ff >= lo) & (ff < hi)
                if not np.any(bm):
                    band_pow_t[:, bi] = eps
                    band_pow_p[:, bi] = eps
                    band_pow_o[:, bi] = eps
                    continue
                band_pow_t[:, bi] = np.trapezoid(P_t[bm, :], ff[bm], axis=0)
                band_pow_p[:, bi] = np.trapezoid(P_p[bm, :], ff[bm], axis=0)
                band_pow_o[:, bi] = np.trapezoid(P_o[bm, :], ff[bm], axis=0)

            bt = band_pow_t + eps
            bp = band_pow_p + eps
            bo = band_pow_o + eps
            bt = bt / np.sum(bt, axis=1, keepdims=True)
            bp = bp / np.sum(bp, axis=1, keepdims=True)
            bo = bo / np.sum(bo, axis=1, keepdims=True)

            pred_band_jsd[i] = float(np.mean(_jsd(bt, bp)))
            null_band_jsd[i] = float(np.mean(_jsd(bt, bo)))

            f_st, _, Zt = signal.stft(
                xt, fs=fs, nperseg=stft_nperseg, noverlap=stft_noverlap, axis=0
            )
            _, _, Zp = signal.stft(
                xp, fs=fs, nperseg=stft_nperseg, noverlap=stft_noverlap, axis=0
            )
            _, _, Zo = signal.stft(
                xo, fs=fs, nperseg=stft_nperseg, noverlap=stft_noverlap, axis=0
            )

            sm = (f_st >= fmin) & (f_st <= fmax_val)
            Mt = np.log(np.abs(Zt[sm, :, :]) + eps)
            Mp = np.log(np.abs(Zp[sm, :, :]) + eps)
            Mo = np.log(np.abs(Zo[sm, :, :]) + eps)

            C_stft = min(C_psd, Mt.shape[2], Mp.shape[2], Mo.shape[2])
            if C_stft <= 0:
                return None

            wp: list[float] = []
            wo: list[float] = []
            for c in range(C_stft):
                a = Mt[:, :, c].ravel()
                b = Mp[:, :, c].ravel()
                d = Mo[:, :, c].ravel()
                wp.append(wasserstein_distance(a, b))
                wo.append(wasserstein_distance(a, d))
            pred_stft_wass[i] = float(np.mean(wp))
            null_stft_wass[i] = float(np.mean(wo))

        def _zscore(
            pred: np.ndarray, null: np.ndarray
        ) -> tuple[np.ndarray, float, float]:
            mu = float(np.mean(null))
            sd = float(np.std(null) + eps)
            return (pred - mu) / sd, mu, sd

        z_psd, mu_psd, sd_psd = _zscore(pred_psd_jsd, null_psd_jsd)
        z_band, mu_band, sd_band = _zscore(pred_band_jsd, null_band_jsd)
        z_stft, mu_stft, sd_stft = _zscore(pred_stft_wass, null_stft_wass)

        thr_psd = mu_psd + threshold_k * sd_psd
        thr_band = mu_band + threshold_k * sd_band
        thr_stft = mu_stft + threshold_k * sd_stft

        tdiv_psd = _first_crossing_time(pred_psd_jsd, thr_psd, times_s)
        tdiv_band = _first_crossing_time(pred_band_jsd, thr_band, times_s)
        tdiv_stft = _first_crossing_time(pred_stft_wass, thr_stft, times_s)

        result = {
            "times_s": times_s,
            "pred": {
                "psd_jsd": pred_psd_jsd,
                "band_jsd": pred_band_jsd,
                "stft_wass": pred_stft_wass,
            },
            "null": {
                "psd_jsd": null_psd_jsd,
                "band_jsd": null_band_jsd,
                "stft_wass": null_stft_wass,
            },
            "z": {
                "psd_jsd": z_psd,
                "band_jsd": z_band,
                "stft_wass": z_stft,
            },
            "thresholds": {
                "k": float(threshold_k),
                "consecutive_windows": int(consecutive_windows),
                "psd_jsd": float(thr_psd),
                "band_jsd": float(thr_band),
                "stft_wass": float(thr_stft),
            },
            "summary": {
                "time_to_divergence_s": {
                    "psd_jsd": tdiv_psd,
                    "band_jsd": tdiv_band,
                    "stft_wass": tdiv_stft,
                },
                "null_mean_std": {
                    "psd_jsd": (mu_psd, sd_psd),
                    "band_jsd": (mu_band, sd_band),
                    "stft_wass": (mu_stft, sd_stft),
                },
                "auc_pred": {
                    "psd_jsd": float(np.trapezoid(pred_psd_jsd, times_s)),
                    "band_jsd": float(np.trapezoid(pred_band_jsd, times_s)),
                    "stft_wass": float(np.trapezoid(pred_stft_wass, times_s)),
                },
                "auc_null": {
                    "psd_jsd": float(np.trapezoid(null_psd_jsd, times_s)),
                    "band_jsd": float(np.trapezoid(null_band_jsd, times_s)),
                    "stft_wass": float(np.trapezoid(null_stft_wass, times_s)),
                },
                "bands_hz": {
                    name: tuple(map(float, bands[name])) for name in band_names
                },
            },
            "params": {
                "fs": float(fs),
                "win_s": float(win_s),
                "stride_s": float(stride_s),
                "fmin": float(fmin),
                "fmax": float(fmax_val),
                "welch_nperseg_s": float(welch_nperseg_s),
                "stft_nperseg_s": float(stft_nperseg_s),
            },
        }

        serialised = self._serialise_numpy(result)
        with open(out_dir / "rollout_divergence_timeseries.json", "w") as f:
            json.dump(serialised, f, indent=2)
        self._plot_timeseries_divergence_metrics(serialised, out_dir)
        return serialised

    def _sample_independent_baseline_target(self, steps: int) -> np.ndarray | None:
        """Pull a fresh target window from the validation dataset without consuming the
        loader iterator."""
        dataset = getattr(self, "val_loader", None)
        dataset = getattr(dataset, "dataset", None)
        if dataset is None or steps <= 0:
            return None
        rng = np.random.default_rng(0)
        idx = 0
        if hasattr(dataset, "__len__"):
            try:
                idx = int(rng.integers(low=0, high=len(dataset)))
            except Exception:
                idx = 0

        sample = self._load_dataset_sample(dataset, idx)
        if sample is None or len(sample) < 2:
            return None

        _, targets, spec = sample
        if isinstance(targets, (tuple, list)):
            targets = targets[0]

        context_steps = self._infer_context_steps(targets)
        continuation = self._build_target_continuation(
            dataset, spec, context_steps, steps
        )
        if continuation is None:
            return None

        continuation = continuation.to(self.device)
        continuation = self._normalise_timeseries(continuation.squeeze())
        if continuation is None or continuation.numel() == 0:
            return None

        if self.mu is not None:
            continuation = mulaw_inv_torch(continuation, self.mu)

        arr = continuation.cpu().numpy()
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr[:, :steps]

    def _plot_single_divergence_curve(
        self,
        runs: list[np.ndarray],
        mean_curve: np.ndarray,
        std_curve: np.ndarray,
        out_dir: Path,
        window_steps: int,
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        if mean_curve.size == 0:
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(mean_curve.shape[0], dtype=np.float32)
        if self.sfreq:
            x = x / float(self.sfreq) * window_steps  # to account for the stride
            ax.set_xlabel("Rollout time (s)")
        else:
            ax.set_xlabel("Rollout steps")

        for curve in runs:
            xs = x[: curve.shape[0]]
            ax.plot(xs, curve, color="gray", alpha=0.2, linewidth=0.7)

        lower = mean_curve - std_curve
        upper = mean_curve + std_curve
        ax.plot(x, mean_curve, color="C0", label="mean distance", linewidth=2.0)
        ax.fill_between(
            x, lower, upper, color="C0", alpha=0.2, label="±1 std across runs"
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(out_dir / filename, bbox_inches="tight")
        plt.close(fig)

    def _plot_divergence_curves(
        self,
        metrics: dict[str, dict[str, Any]],
        out_dir: Path,
        window_steps: int,
    ) -> None:
        if not metrics:
            return

        ordered = [
            "correlation",
            "covariance",
            "stft_magnitude",
            "stft_angle",
            "fft_magnitude",
            "fft_angle",
        ]
        metric_names = [m for m in ordered if m in metrics] + [
            m for m in metrics.keys() if m not in ordered
        ]
        cols = min(3, len(metric_names))
        rows = int(np.ceil(len(metric_names) / cols)) if cols > 0 else 0
        if rows == 0:
            return

        fig, axes = plt.subplots(
            rows, cols, figsize=(8 * cols, 6 * rows), squeeze=False
        )
        axes_list = axes.flatten().tolist()
        xlabel = "Rollout time (s)" if self.sfreq else "Rollout steps"

        for ax, name in zip(axes_list, metric_names):
            data = metrics[name]
            mean_curve = data["mean"]
            std_curve = data["std"]
            runs = data["runs"]
            if mean_curve.size == 0:
                ax.set_title(f"No data for {name}")
                ax.axis("off")
                continue

            x = np.arange(mean_curve.shape[0], dtype=np.float32)
            if self.sfreq:
                x = x / float(self.sfreq) * window_steps

            for curve in runs:
                xs = x[: curve.shape[0]]
                ax.plot(xs, curve, color="gray", alpha=0.2, linewidth=0.7)

            lower = mean_curve - std_curve
            upper = mean_curve + std_curve
            ax.plot(x, mean_curve, color="C0", label="mean distance", linewidth=2.0)
            ax.fill_between(
                x, lower, upper, color="C0", alpha=0.2, label="±1 std across runs"
            )
            ax.set_title(f"{name.replace('_', ' ')} vs rollout", loc="left")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Distance")
            ax.legend(loc="upper left")
            ax.grid(False)

        for ax in axes_list[len(metric_names) :]:
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(out_dir / "rollout_divergence.png", bbox_inches="tight")
        plt.close(fig)

    def _evaluate_rollout_divergence(
        self,
        generated_runs: list[np.ndarray],
        target_runs: list[np.ndarray],
        out_dir: Path,
        params: dict[str, Any],
    ) -> None:
        if not generated_runs or not target_runs:
            return

        window_steps = self._resolve_divergence_window(params)
        metric_runs: dict[str, list[np.ndarray]] = defaultdict(list)
        lengths: dict[str, list[int]] = defaultdict(list)
        timeseries_metrics: dict[str, Any] | None = None
        for gen, tgt in zip(generated_runs, target_runs):
            if gen.shape != tgt.shape:
                print(
                    "[eval_runner] Skipping divergence: shape mismatch "
                    f"{gen.shape} vs {tgt.shape}"
                )
                continue
            curves = self._rollout_divergence_curve(gen, tgt, window_steps)
            for name, curve in curves.items():
                metric_runs[name].append(curve)
                lengths[name].append(int(curve.shape[0]))

        if not metric_runs:
            print("[eval_runner] No valid runs for divergence metric.")
            return

        aggregated: dict[str, dict[str, Any]] = {}
        for name, runs in metric_runs.items():
            if not runs:
                continue
            max_len = max(len(c) for c in runs)
            stacked = np.full((len(runs), max_len), np.nan, dtype=np.float32)
            for idx, curve in enumerate(runs):
                stacked[idx, : curve.shape[0]] = curve

            aggregated[name] = {
                "runs": runs,
                "mean": np.nanmean(stacked, axis=0),
                "std": np.nanstd(stacked, axis=0),
                "lengths": lengths[name],
            }

        payload = {
            "window_steps": int(window_steps),
            "window_seconds": (
                float(window_steps) / float(self.sfreq)
                if self.sfreq is not None
                else None
            ),
            "sfreq": self.sfreq,
            "metrics": {
                name: {
                    "per_run_lengths": data["lengths"],
                    "mean": data["mean"].tolist(),
                    "std": data["std"].tolist(),
                }
                for name, data in aggregated.items()
            },
        }
        if "correlation" in aggregated:
            payload["per_run_lengths"] = aggregated["correlation"]["lengths"]
            payload["mean"] = aggregated["correlation"]["mean"].tolist()
            payload["std"] = aggregated["correlation"]["std"].tolist()
        if timeseries_metrics is not None:
            payload["timeseries_metrics_file"] = "rollout_divergence_timeseries.json"
        with open(out_dir / "rollout_divergence.json", "w") as f:
            json.dump(payload, f, indent=2)

        self._plot_divergence_curves(aggregated, out_dir, window_steps)

        baseline_target = self._sample_independent_baseline_target(
            target_runs[0].shape[1]
        )
        if baseline_target is not None:
            timeseries_metrics = self._compute_timeseries_divergence_metrics(
                target_runs[0],
                generated_runs[0],
                baseline_target,
                window_steps,
                out_dir,
                params,
            )

    def _plot_timeseries_grid(
        self,
        data: np.ndarray,
        prefix: str,
        out_dir: Path,
        channel_indices: np.ndarray | list[int] | None = None,
        context_len: int | None = None,
    ) -> None:
        if data is None or data.size == 0:
            return

        indices = (
            np.asarray(channel_indices)
            if channel_indices is not None
            else np.arange(data.shape[0])
        )
        data = data[indices]
        n_channels, timesteps = data.shape
        width = 30
        height = max(5, int(n_channels * 5))
        fig, axes = plt.subplots(n_channels, 1, figsize=(width, height), sharex=True)
        if isinstance(axes, np.ndarray):
            axes_list = axes.flatten().tolist()
        else:
            axes_list = [axes]

        palette = sns.color_palette("husl", n_channels)
        time_axis = (
            np.arange(timesteps) / float(self.sfreq)
            if self.sfreq
            else np.arange(timesteps)
        )
        xlabel = "Time (s)" if self.sfreq else "Samples"

        for idx, ax in enumerate(axes_list):
            ax.plot(
                time_axis,
                data[idx],
                color=palette[idx % len(palette)],
                linewidth=0.8,
                alpha=0.9,
            )
            label = int(indices[idx])
            ax.set_ylabel(f"ch {label}")
            ax.set_title(f"{prefix} channel {label}", loc="left", fontsize=10)
            ax.grid(False)
            if context_len and context_len < data.shape[1]:
                cutoff = (
                    context_len / float(self.sfreq)
                    if self.sfreq
                    else float(context_len)
                )
                ax.axvline(
                    cutoff, color="red", linestyle="--", linewidth=4.0, alpha=0.9
                )

        axes_list[-1].set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_timeseries.png", bbox_inches="tight")
        plt.close(fig)

    def _plot_stft_grid(
        self,
        data: np.ndarray,
        prefix: str,
        out_dir: Path,
        channel_indices: np.ndarray | list[int] | None = None,
        context_len: int | None = None,
    ) -> None:
        if data is None or data.size == 0:
            return

        indices = (
            np.asarray(channel_indices)
            if channel_indices is not None
            else np.arange(data.shape[0])
        )
        data = data[indices]
        n_channels, timesteps = data.shape
        if timesteps < 2:
            return

        nfft = min(128, timesteps)
        noverlap = int(nfft * 0.75) if nfft > 1 else 0
        fig, axes = plt.subplots(
            n_channels, 1, figsize=(30, max(5, int(n_channels * 5))), sharex=True
        )
        if isinstance(axes, np.ndarray):
            axes_list = axes.flatten().tolist()
        else:
            axes_list = [axes]
        last_im = None

        for idx, ax in enumerate(axes_list):
            _, _, t, im = ax.specgram(
                data[idx],
                NFFT=nfft,
                Fs=float(self.sfreq) if self.sfreq else 1.0,
                noverlap=noverlap,
                cmap="magma",
                mode="psd",
                scale="dB",
            )
            last_im = im
            label = int(indices[idx])
            ax.set_ylabel(f"ch {label}")
            ax.set_title(f"{prefix} STFT ch {label}", loc="left", fontsize=10)
            ax.grid(False)
            if context_len and context_len < data.shape[1]:
                cutoff = (
                    context_len / float(self.sfreq)
                    if self.sfreq
                    else float(context_len)
                )
                ax.axvline(
                    cutoff, color="black", linestyle="--", linewidth=4.0, alpha=0.9
                )

        axes_list[-1].set_xlabel("Time (s)" if self.sfreq else "Samples")
        if last_im is not None:
            fig.colorbar(
                last_im,
                ax=axes_list,
                orientation="vertical",
                fraction=0.02,
                pad=0.01,
                label="Power (dB)",
            )

        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_stft.png", bbox_inches="tight")
        plt.close(fig)

    def _dataset_supports_generation(self, dataset: Any) -> bool:
        required = (
            "indices",
            "root_dirs",
            "ch_names",
            "fill_value",
            "_get_session_indices",
            "_resolve_index",
        )
        return all(hasattr(dataset, attr) for attr in required)

    def _load_dataset_sample(
        self, dataset: Any, idx: int
    ) -> tuple[Any, Any, tuple[str, str, str, int] | None] | None:
        sample = dataset[idx]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            return None

        batched = default_collate([sample])
        if not isinstance(batched, (tuple, list)) or len(batched) < 2:
            return None

        inputs, targets = batched[:2]
        spec = None
        if hasattr(dataset, "_resolve_index"):
            spec = _unpack_index_entry(dataset._resolve_index(idx))

        return inputs, targets, spec

    def _max_continuation_steps(
        self, dataset: Any, spec: tuple[str, str, str, int] | None, context_steps: int
    ) -> int:
        if spec is None:
            return 0
        dataset_key, session, chunk, start = spec
        session_dir = Path(dataset.root_dirs[dataset_key]) / session
        chunk_files = sorted(
            [f for f in session_dir.iterdir() if f.suffix == ".npy"],
            key=_chunk_sort_key,
        )

        total = 0
        seen_current = False
        for chunk_file in chunk_files:
            chunk_dict = _load_chunk_cached(str(chunk_file))
            total_samples = int(chunk_dict["data"].shape[1])
            if chunk_file.name == chunk:
                offset = int(start) + int(context_steps)
                if offset < total_samples:
                    total += total_samples - offset
                seen_current = True
            elif seen_current:
                total += total_samples

        return max(0, total)

    def _build_target_continuation(
        self,
        dataset: Any,
        spec: tuple[str, str, str, int] | None,
        context_steps: int,
        steps: int,
    ) -> torch.Tensor | None:
        if spec is None or steps <= 0:
            return None

        dataset_key, session, chunk, start = spec
        session_dir = Path(dataset.root_dirs[dataset_key]) / session
        chunk_files = sorted(
            [f for f in session_dir.iterdir() if f.suffix == ".npy"],
            key=_chunk_sort_key,
        )
        try:
            start_idx = chunk_files.index(session_dir / chunk)
        except ValueError:
            return None

        remaining = int(steps)
        segments: list[np.ndarray] = []
        for chunk_path in chunk_files[start_idx:]:
            chunk_dict = _load_chunk_cached(str(chunk_path))
            data = chunk_dict["data"]

            offset = 0
            if chunk_path.name == chunk:
                offset = int(start) + int(context_steps)
            if offset >= data.shape[1]:
                continue

            take = min(remaining, data.shape[1] - offset)
            window = data[:, offset : offset + take]
            mapped = np.ones(
                (len(dataset.ch_names), window.shape[1]), dtype=window.dtype
            )
            mapped *= dataset.fill_value
            indices = dataset._get_session_indices(
                dataset_key, session, window.shape[0]
            )
            if len(indices) != window.shape[0]:
                raise ValueError(
                    f"Channel mismatch for session {session} ({dataset_key}): "
                    f"expected {len(indices)}, got {window.shape[0]}"
                )

            mapped[indices, :] = window
            segments.append(mapped)
            remaining -= take
            if remaining <= 0:
                break

        if not segments:
            return None

        return torch.from_numpy(np.concatenate(segments, axis=1))

    def _infer_context_steps(self, targets: torch.Tensor) -> int:
        seq = self._normalise_timeseries(targets)
        if seq is None:
            return 0
        return int(seq.shape[1])

    def _maybe_generate(self, lit_model: LitModel, out_dir: Path) -> None:
        if not self.generate_cfg.get("enabled", False):
            return

        model = lit_model.model
        forecast_fn = getattr(model, "forecast", None)
        if not callable(forecast_fn):
            print("[eval_runner] Model has no forecast() method; skipping generation.")
            return

        params = dict(self.generate_cfg.get("params", {}))
        seconds = float(params.get("seconds", 1))
        num_runs = int(params.get("num_runs", 3))
        num_runs = max(1, num_runs)
        rng = np.random.default_rng(params.get("seed"))
        sfreq = float(self.sfreq) if self.sfreq is not None else None
        rollout_steps = int(seconds * (sfreq if sfreq else 1.0))
        rollout_steps = max(1, rollout_steps)
        sample_fn = self._build_sample_fn()

        forecast_kwargs = {
            "use_cache": True,
            "sliding_window_overlap": params.get("kv_overlap", 0.5),
            "max_context_tokens": params.get("max_context_tokens", -1),
        }

        dataset = getattr(self.val_loader, "dataset", None)
        candidate_indices: list[int] = []
        if dataset is not None and self._dataset_supports_generation(dataset):
            candidate_indices = _select_first_chunk_indices(
                list(getattr(dataset, "indices", [])), num_runs, rng
            )
            if not candidate_indices and getattr(dataset, "indices", None):
                candidate_indices = [0]

            if len(candidate_indices) < num_runs:
                print(
                    f"[eval_runner] Only {len(candidate_indices)} unique sessions "
                    f"available for generation (requested {num_runs})."
                )

        if not candidate_indices:
            print("[eval_runner] No valid validation samples for generation.")
            return

        gen_runs: list[np.ndarray] = []
        tgt_runs: list[np.ndarray] = []
        context_runs: list[np.ndarray] = []

        for run_idx, ds_idx in enumerate(candidate_indices):
            sample = (
                self._load_dataset_sample(dataset, ds_idx)
                if dataset is not None
                else None
            )
            if sample is None:
                print(f"[eval_runner] Skipping run {run_idx}: could not load sample.")
                continue

            inputs, targets, spec = sample
            inputs, targets = self._move_to_device((inputs, targets))
            context_steps = self._infer_context_steps(targets)
            if context_steps <= 0:
                print(f"[eval_runner] Skipping run {run_idx}: empty context.")
                continue

            context_tensor = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            context_tensor = self._normalise_timeseries(context_tensor)
            if context_tensor is None:
                print(f"[eval_runner] Skipping run {run_idx}: invalid context tensor.")
                continue
            context_tensor = context_tensor[:, :context_steps]

            max_available = self._max_continuation_steps(dataset, spec, context_steps)
            rollout_horizon = min(rollout_steps, max_available) if max_available else 0
            if rollout_horizon <= 0:
                print(
                    f"[eval_runner] Skipping run {run_idx}: not enough data for "
                    "ground truth continuation."
                )
                continue

            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    generated = forecast_fn(
                        inputs, rollout_horizon, sample_fn, **forecast_kwargs
                    )

                if torch.is_tensor(generated):
                    generated = generated.to(torch.float32)
            except Exception as exc:
                print(f"[eval_runner] Generation failed for run {run_idx}: {exc}")
                continue

            gen_tensor = self._extract_tensor(generated)
            if gen_tensor is None:
                print("[eval_runner] Unable to extract tensor from forecast output.")
                continue

            gen_tensor = self._normalise_timeseries(gen_tensor.squeeze())
            if gen_tensor is None:
                print("[eval_runner] Could not normalise generated tensor.")
                continue

            effective_steps = min(rollout_horizon, gen_tensor.shape[1], max_available)
            if effective_steps <= 0:
                print(f"[eval_runner] Skipping run {run_idx}: zero-length generation.")
                continue
            gen_tensor = gen_tensor[:, :effective_steps]

            if self.mu is not None:
                gen_tensor = mulaw_inv_torch(gen_tensor, self.mu)
                context_tensor = mulaw_inv_torch(context_tensor, self.mu)
            gen_arr = gen_tensor.cpu().numpy()
            context_arr = context_tensor.cpu().numpy()
            gen_runs.append(gen_arr)
            context_runs.append(context_arr)
            np.save(out_dir / f"generated_run{run_idx}.npy", gen_arr)

            tgt_tensor = self._build_target_continuation(
                dataset, spec, context_steps, effective_steps
            )
            if tgt_tensor is not None:
                if self.mu is not None:
                    tgt_tensor = mulaw_inv_torch(tgt_tensor, self.mu)

                tgt_arr = tgt_tensor.cpu().numpy()
                tgt_runs.append(tgt_arr)
                np.save(out_dir / f"target_run{run_idx}.npy", tgt_arr)

            # Per-run plots (context + continuation)
            max_channels = min(context_arr.shape[0], gen_arr.shape[0])
            if tgt_tensor is not None:
                max_channels = min(max_channels, tgt_arr.shape[0])
            if max_channels <= 0:
                continue

            num_plot_channels = min(10, max_channels)
            channel_indices = rng.choice(
                max_channels, size=num_plot_channels, replace=False
            )

            combined_gen = np.concatenate([context_arr, gen_arr], axis=1)
            self._plot_timeseries_grid(
                combined_gen,
                f"generated_run{run_idx}",
                out_dir,
                channel_indices,
                context_len=context_arr.shape[1],
            )
            self._plot_stft_grid(
                combined_gen,
                f"generated_run{run_idx}",
                out_dir,
                channel_indices,
                context_len=context_arr.shape[1],
            )

            if tgt_tensor is not None:
                combined_tgt = np.concatenate([context_arr, tgt_arr], axis=1)
                self._plot_timeseries_grid(
                    combined_tgt,
                    f"target_run{run_idx}",
                    out_dir,
                    channel_indices,
                    context_len=context_arr.shape[1],
                )
                self._plot_stft_grid(
                    combined_tgt,
                    f"target_run{run_idx}",
                    out_dir,
                    channel_indices,
                    context_len=context_arr.shape[1],
                )

        if not gen_runs:
            print("[eval_runner] No successful generation runs.")
            return

        self._plot_psd_cov_pair(gen_runs, tgt_runs, out_dir, prefix="gen_vs_target")
        self._evaluate_rollout_divergence(gen_runs, tgt_runs, out_dir, params)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def run(self) -> Dict[str, Any]:
        lit_model = self._load_model(self.ckpt_path)
        out_dir = self._prepare_output_dir()
        self._maybe_generate(lit_model, out_dir)

        if hasattr(lit_model.model, "set_eval_mode"):
            lit_model.model.set_eval_mode()

        losses: List[float] = []
        metrics: dict[str, list[float]] = defaultdict(list)
        example_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for batch_idx, batch in enumerate(self.val_loader):
            inputs, targets = self._move_to_device(batch)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = lit_model.model(inputs)
            if torch.is_tensor(outputs):
                logits = outputs
            elif isinstance(outputs, dict):
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                raise ValueError(f"Invalid outputs type: {type(outputs)}")

            if self.postprocessor is not None:
                inputs, logits, targets = self.postprocessor(
                    inputs, logits, targets, gaussian=False
                )
            logits = logits.to(torch.float32)

            if isinstance(outputs, (tuple, list)):
                outputs = [logits] + [out for out in outputs[1:]]

            loss = lit_model.loss(outputs, targets, model=lit_model.model)
            losses.append(float(loss.detach().cpu()))

            for name, metric in lit_model.loss.metrics.items():
                metric_val = metric(outputs, targets)
                metrics[name].append(float(metric_val.detach().cpu()))

            if isinstance(inputs, (tuple, list)):
                inputs = inputs[0]

            if len(example_batches) < self.num_examples:
                example_batches.append(
                    (
                        inputs.detach().cpu(),
                        logits.detach().cpu(),
                        targets.detach().cpu(),
                    )
                )

            if self.max_batches and (batch_idx + 1) >= self.max_batches:
                break

        summary = self._summarise_results(losses, metrics)
        self._persist_results(summary, out_dir)
        self._log(summary, example_batches, out_dir)
        return summary

    # ------------------------------------------------------------------ #
    # Logging helpers
    # ------------------------------------------------------------------ #
    def _summarise_results(
        self, losses: list[float], metrics: dict[str, list[float]]
    ) -> Dict[str, Any]:
        summary = {
            "loss_mean": float(np.mean(losses)) if losses else float("nan"),
            "loss_std": float(np.std(losses)) if losses else float("nan"),
            "losses": losses,
            "metrics": {},
        }
        for name, values in metrics.items():
            summary["metrics"][name] = {
                "mean": float(np.mean(values)) if values else float("nan"),
                "std": float(np.std(values)) if values else float("nan"),
                "values": values,
            }
        return summary

    def _prepare_output_dir(self) -> Path:
        base_dir = self.save_dir / "logs" / f"version_{self.run_version}"
        ckpt_name = Path(self.ckpt_path).stem
        if self.eval_epoch is not None:
            subdir = f"epoch_{int(self.eval_epoch):03d}"
        else:
            subdir = ckpt_name
        out_dir = base_dir / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _persist_results(
        self,
        summary: Dict[str, Any],
        out_dir: Path,
    ) -> None:
        payload = dict(summary)
        payload["step"] = self.eval_step
        payload["epoch"] = self.eval_epoch
        with open(out_dir / "summary.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _plot_metric_grid(
        self, losses: list[float], metrics: dict[str, list[float]]
    ) -> plt.Figure:
        """Plot each metric on its own subplot within a single figure."""
        metric_items = [("loss", losses)] + list(metrics.items())
        n = len(metric_items)
        fig, axes = plt.subplots(1, n, figsize=(4 * max(2, n), 4))
        # Normalize axes handling for the n==1 case
        if isinstance(axes, np.ndarray):
            axes_list = axes.flatten().tolist()
        else:
            axes_list = [axes]

        for ax, (name, values) in zip(axes_list, metric_items):
            if values:
                sns.violinplot(y=values, cut=0, inner="box", ax=ax)
                ax.set_ylabel(name)
                ax.set_title(f"{name} distribution")
            else:
                ax.set_title(f"No data for {name}")
                ax.axis("off")
                continue
            ax.grid(False)

        fig.tight_layout()
        return fig

    def _compute_psd(
        self, tensor: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray] | None:
        seq = self._normalise_timeseries(tensor)
        if seq is None or seq.numel() == 0:
            return None

        arr = seq.cpu().numpy()
        fs = float(self.sfreq) if self.sfreq is not None else 1.0
        nperseg = min(arr.shape[-1], max(1, int(fs)))
        if nperseg <= 0:
            return None

        freqs, psd = signal.welch(
            arr,
            fs=fs,
            axis=-1,
            nperseg=nperseg,
            scaling="density",
        )
        return freqs, psd

    def _compute_psd_cov(
        self, data_runs: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not data_runs or self.sfreq is None:
            return None

        psd_list: list[np.ndarray] = []
        cov_list: list[np.ndarray] = []
        freqs_ref: np.ndarray | None = None

        for data in data_runs:
            freqs, psd = signal.welch(
                data,
                fs=self.sfreq,
                axis=-1,
                nperseg=self.sfreq,
                scaling="density",
            )
            freqs_ref = freqs if freqs_ref is None else freqs_ref
            psd_list.append(psd)
            cov_list.append(np.cov(data))

        psd_mean = np.mean(np.stack(psd_list, axis=0), axis=0)
        cov_mean = np.mean(np.stack(cov_list, axis=0), axis=0)
        return freqs_ref if freqs_ref is not None else np.array([]), psd_mean, cov_mean

    def _save_psd_pair(
        self, preds: torch.Tensor, target: torch.Tensor, prefix: str, out_dir: Path
    ) -> None:
        pred_res = self._compute_psd(preds)
        tgt_res = self._compute_psd(target)
        if pred_res is None or tgt_res is None:
            return

        pred_freqs, pred_psd = pred_res
        tgt_freqs, tgt_psd = tgt_res

        fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
        plots = [
            ("Preds", pred_freqs, pred_psd, axes[0]),
            ("Target", tgt_freqs, tgt_psd, axes[1]),
        ]
        psd_flat = np.concatenate([pred_psd.flatten(), tgt_psd.flatten()])
        psd_flat = psd_flat[psd_flat > 0]
        ylims = None
        if psd_flat.size > 0:
            lower = np.percentile(psd_flat, 0.1)
            upper = np.percentile(psd_flat, 99.9)
            if lower > 0 and upper > lower:
                ylims = (lower, upper)

        for title, freqs, psd, ax in plots:
            ax.plot(freqs, psd.T, alpha=0.3)
            ax.set_xlabel("Hz")
            ax.set_ylabel("Power")
            ax.set_title(f"{title} PSD - {prefix}")
            ax.set_yscale("log")
            if ylims:
                ax.set_ylim(ylims)

        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_psd.png", bbox_inches="tight")
        plt.close(fig)

    def _plot_psd_cov_pair(
        self,
        gen_runs: list[np.ndarray],
        tgt_runs: list[np.ndarray],
        out_dir: Path,
        prefix: str = "gen_target",
    ) -> None:
        gen_res = self._compute_psd_cov(gen_runs)
        tgt_res = self._compute_psd_cov(tgt_runs)
        if gen_res is None or tgt_res is None:
            return

        gen_freqs, gen_psd, gen_cov = gen_res
        tgt_freqs, tgt_psd, tgt_cov = tgt_res

        # PSD side-by-side
        fig_psd, ax_psd = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
        psd_sets = [
            ("Generated PSD", gen_freqs, gen_psd, ax_psd[0]),
            ("Target PSD", tgt_freqs, tgt_psd, ax_psd[1]),
        ]
        psd_flat = np.concatenate([gen_psd.flatten(), tgt_psd.flatten()])
        psd_flat = psd_flat[psd_flat > 0]
        ylims = None
        if psd_flat.size > 0:
            lower = np.percentile(psd_flat, 0.1)
            upper = np.percentile(psd_flat, 99.9)
            if lower > 0 and upper > lower:
                ylims = (lower, upper)

        for title, freqs, psd, ax in psd_sets:
            ax.plot(freqs, psd.T, alpha=0.3)
            ax.set_xlabel("Hz")
            ax.set_ylabel("Power")
            ax.set_title(title)
            ax.set_yscale("log")
            if ylims:
                ax.set_ylim(ylims)

        fig_psd.tight_layout()
        fig_psd.savefig(out_dir / f"{prefix}_psd.png", bbox_inches="tight")
        plt.close(fig_psd)

        # Covariance side-by-side
        fig_cov, ax_cov = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True)
        cov_sets = [
            ("Generated Covariance", gen_cov, ax_cov[0]),
            ("Target Covariance", tgt_cov, ax_cov[1]),
        ]
        for title, cov_mat, ax in cov_sets:
            im = ax.imshow(cov_mat, cmap="viridis")
            ax.set_title(title)
            ax.grid(False)
        fig_cov.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)
        fig_cov.savefig(out_dir / f"{prefix}_cov.png", bbox_inches="tight")
        plt.close(fig_cov)

    def _images_to_channels(self, img: torch.Tensor) -> torch.Tensor:
        """Convert (B,T,H,W) to (B,T,C) by indexing sensor pixels."""
        row_idx = torch.as_tensor(self.val_dataset.row_idx, device=img.device)
        col_idx = torch.as_tensor(self.val_dataset.col_idx, device=img.device)

        img = img.squeeze()

        # gather per time slice
        return img[..., row_idx, col_idx]

    def _plot_examples(
        self,
        examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        out_dir: Path,
    ) -> list[plt.Figure]:
        figures: list[plt.Figure] = []
        for idx, (inp, out, tgt) in enumerate(examples):
            fig, axes = plt.subplots(3, 1, figsize=(100, 12), sharex=True)

            # convert to channels if shape[-2] = shape[-1], i.e. images
            if inp.shape[-2] == inp.shape[-1]:
                inp = self._images_to_channels(inp)
                out = self._images_to_channels(out)
                tgt = self._images_to_channels(tgt)

            self._plot_sequence(inp, axes[0], "Input")

            preds = out
            if torch.is_tensor(out) and out.dim() == tgt.dim() + 1:
                preds = out.argmax(dim=-1)
            self._plot_sequence(preds, axes[1], "Model output")
            self._plot_sequence(tgt, axes[2], "Target")
            fig.suptitle(f"Example {idx}")
            axes[-1].set_xlabel("timestep")

            try:
                self._save_psd_pair(preds, tgt, f"example{idx}", out_dir)
            except Exception as exc:  # pragma: no cover - logging only
                print(f"[eval_runner] Failed to save PSD for example {idx}: {exc}")

            figures.append(fig)
        return figures

    def _plot_sequence(self, tensor: torch.Tensor, ax: plt.Axes, title: str) -> None:
        arr = tensor.detach().cpu()
        if arr.dim() == 1:
            arr = arr.unsqueeze(0)
        if arr.dim() > 2:
            arr = arr.reshape(arr.shape[0], -1)

        max_series = min(arr.shape[0], 6)
        for i in range(max_series):
            ax.plot(arr[i].numpy(), alpha=0.5, linewidth=0.5)
        ax.set_title(title)
        ax.grid(False)

    def _log(
        self,
        summary: Dict[str, Any],
        examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        out_dir: Path,
    ) -> None:
        metric_fig = self._plot_metric_grid(
            summary["losses"], {k: v["values"] for k, v in summary["metrics"].items()}
        )
        example_figs = self._plot_examples(examples, out_dir)

        metrics_path = out_dir / "metrics_distributions.png"
        metric_fig.savefig(metrics_path, bbox_inches="tight")
        plt.close(metric_fig)

        for idx, fig in enumerate(example_figs):
            fig_path = out_dir / f"example_{idx}.png"
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", required=True, type=Path, help="Path to YAML config"
    )
    parser.add_argument("--ckpt", type=Path, help="Checkpoint to evaluate")
    parser.add_argument(
        "--step", type=int, default=None, help="Global step for logging"
    )
    args = parser.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("eval_runner", {})
    max_batches = eval_cfg.get("max_batches", 8)
    num_examples = eval_cfg.get("num_examples", 3)

    ckpt_path = args.ckpt or eval_cfg.get("ckpt_path")
    if ckpt_path is None:
        raise ValueError("Provide --ckpt or set eval_runner.ckpt_path in the config.")

    runner = EvaluationRunner(
        cfg,
        max_batches=max_batches,
        num_examples=num_examples,
    )
    runner.run(
        str(ckpt_path),
        step=args.step or eval_cfg.get("step"),
        epoch=eval_cfg.get("epoch"),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
