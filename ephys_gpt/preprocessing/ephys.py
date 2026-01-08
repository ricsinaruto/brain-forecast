"""Base classes for ephys recordings."""

import warnings
from pathlib import Path
from typing import Any

from scipy import signal

import mne

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from mne.io.constants import FIFF
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw
from osl_ephys import preprocessing

from ..utils.quantizers import mulaw, mulaw_inv
from ..utils.utils import compute_roi_layout_2d

from .base import Preprocessing


def predictive_residual_encode(x: np.ndarray) -> np.ndarray:
    """Second-order predictor residuals with seeds stored in the first two steps.

    ẑ[t] = 2*z[t-1] - z[t-2] e[t] = z[t] - ẑ[t]
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (C, T); got shape {arr.shape}")

    C, T = arr.shape
    residual = np.empty_like(arr, dtype=np.float32)
    if T == 0:
        return residual

    residual[:, 0] = arr[:, 0]
    if T == 1:
        return residual

    residual[:, 1] = arr[:, 1]
    if T > 2:
        residual[:, 2:] = arr[:, 2:] - 0.95 * arr[:, 1:-1]  # + arr[:, :-2]
    return residual


def quantize_deadzone_linear(
    e: np.ndarray, deadzone: float, Emax: float = 1.0, bins: int = 256
) -> np.ndarray:
    q = np.zeros_like(e, dtype=np.uint8)
    center = bins // 2  # pick center code

    # deadzone -> exactly center
    mask = np.abs(e) <= deadzone
    q[mask] = center

    # outside deadzone: quantize linearly to remaining codes
    # map (-Emax..-deadzone) -> (0..127), (deadzone..Emax) -> (129..255)
    pos = e > deadzone
    neg = e < -deadzone

    # positive side
    u = (e[pos] - deadzone) / (Emax - deadzone)  # 0..1
    q[pos] = (center + 1 + np.round(u * (bins - 1 - center - 1))).astype(np.uint8)

    # negative side
    u = (-e[neg] - deadzone) / (Emax - deadzone)  # 0..1
    q[neg] = (center - 1 - np.round(u * (center - 1))).astype(np.uint8)

    return q


def predictive_residual_decode(residual: np.ndarray) -> np.ndarray:
    """Invert the predictor residuals via causal integration."""
    res = np.asarray(residual, dtype=np.float32)
    if res.ndim != 2:
        raise ValueError(f"Expected 2D array (C, T); got shape {res.shape}")

    C, T = res.shape
    out = np.empty_like(res, dtype=np.float32)
    if T == 0:
        return out

    out[:, 0] = res[:, 0]
    if T == 1:
        return out

    out[:, 1] = res[:, 1]
    for t in range(2, T):
        out[:, t] = res[:, t] + 2 * out[:, t - 1] - out[:, t - 2]
    return out


def predictive_mulaw_decode(
    tokens: np.ndarray, *, scale: float | np.ndarray = 1.0, mu: int = 255
) -> np.ndarray:
    """Inverse µ-law then integrate residuals back to continuous values."""
    residual = mulaw_inv(tokens, mu)
    residual = residual * np.asarray(scale, dtype=np.float32)
    return predictive_residual_decode(residual)


class Ephys(Preprocessing):
    # OSL pipeline config for basic preprocessing
    default_config_maxwell = """
        meta:
            event_codes:
        preproc:
            - apply_gradient_compensation: {grade: 0}
            - maxwell_filter: {
                origin: "auto",
                int_order: 8,
                ext_order: 3,
                st_duration: 10.0,
                st_correlation: 0.98,
                regularize: "in",
                verbose: "warning",
            }
            - notch_filter:       {
                freqs: 60 120 180 240 300,
                phase: 'minimum'
            }
            - filter:             {l_freq: 0.1, h_freq: 250, phase: 'minimum'}
            - resample:           {sfreq: 500}
            - bad_channels:       {picks: 'mag'}
            - bad_channels:       {picks: 'grad'}
            - interpolate_bads:   {}
    """

    default_config = """
        meta:
            event_codes:
        preproc:
            - apply_gradient_compensation: {grade: 3}
            - notch_filter:       {
                freqs: 60 120 180 240 300,
                phase: 'minimum'
            }
            - filter:             {l_freq: 0.1, h_freq: 250, phase: 'minimum'}
            - resample:           {sfreq: 500}
            - bad_channels:       {picks: 'mag'}
            - bad_channels:       {picks: 'grad'}
            - interpolate_bads:   {}
    """

    def __init__(
        self,
        *args,
        maxwell: bool = False,
        source_space: bool = False,
        sfreq: int = None,
        interpolate_bads: bool = False,
        get_fsaverage_data: bool = False,
        residual_scale: float = 1.0,
        fsaverage_dir: str = "/vol/data/datasets/mne_data",
        bad_handling: str = "zero",
        **kwargs,
    ) -> None:
        """Args:

        maxwell: Whether to use the maxwell filter
        """
        super().__init__(*args, **kwargs)

        self.residual_scale = residual_scale
        self.source_space = source_space
        self.sfreq = sfreq
        self.interpolate_bads = interpolate_bads
        # keep backwards compat: interpolate_bads flag overrides default handling
        if interpolate_bads and bad_handling == "zero":
            self.bad_handling = "interpolate"
        else:
            self.bad_handling = bad_handling
        if self.osl_config is None:
            if maxwell:
                self.osl_config = self.default_config_maxwell
            else:
                self.osl_config = self.default_config

        if get_fsaverage_data:
            fetch_fsaverage(Path(fsaverage_dir))

        self.subjects_dir = Path(fsaverage_dir)

    def find_events_safe(
        self, data: dict[str, Any], min_duration: float = 0.005
    ) -> dict[str, Any]:
        """Find events if stim channels are present, otherwise continue without events.

        Args:     data: Dictionary containing raw MNE object     min_duration: Minimum
        duration between events in seconds
        """
        raw = data["raw"]

        # Try to find stim channels
        stim_picks = mne.pick_types(raw.info, stim=True)

        if len(stim_picks) > 0:
            try:
                # Attempt to find events
                events = mne.find_events(raw, min_duration=min_duration)
                data["events"] = events
                data["has_events"] = True
                print(f"INFO: Found {len(events)} events")
            except Exception as e:
                print(f"WARNING: Failed to find events: {str(e)}")
                data["has_events"] = False
        else:
            print("INFO: No stim channels found, continuing without events")
            data["has_events"] = False

        return data

    def source_space_proj(
        self,
        raw,
        subject: str,
        parc: str = "aparc",  # Desikan-Killiany (68)
        spacing: str = "ico5",  # ~10k verts/hemis
        snr: float = 3.0,  # inverse regularization
    ):
        # Optional: fast artifact annotations (muscle bursts) — safe defaults
        if False:
            try:
                ann_muscle, scores = mne.preprocessing.annotate_muscle_zscore(
                    raw.copy().pick("meg"), threshold=4.0, min_length_good=0.1
                )
                raw.set_annotations(raw.annotations + ann_muscle)
                raw = raw.copy().load_data().annotate_bad_segments(ann_muscle)
            except Exception:
                pass

        sfreq = raw.info["sfreq"]

        # --- extract fiducials + extra scalp points from info['dig'] ---
        digs = raw.info.get("dig", None)
        if not digs:
            raise RuntimeError("No digitization points found in raw.info['dig'].")

        nas = lpa = rpa = None
        hsp = []
        for d in digs:
            if d["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
                continue
            if d["kind"] == FIFF.FIFFV_POINT_CARDINAL:
                if d["ident"] == FIFF.FIFFV_POINT_NASION:
                    nas = d["r"]
                elif d["ident"] == FIFF.FIFFV_POINT_LPA:
                    lpa = d["r"]
                elif d["ident"] == FIFF.FIFFV_POINT_RPA:
                    rpa = d["r"]
            elif d["kind"] == FIFF.FIFFV_POINT_EEG:
                hsp.append(d["r"])
        if nas is None or lpa is None or rpa is None:
            raise RuntimeError("Missing fiducials (NAS/LPA/RPA) in digitization.")
        hsp = np.array(hsp) if len(hsp) else None

        montage = mne.channels.make_dig_montage(
            nasion=nas, lpa=lpa, rpa=rpa, hsp=hsp, coord_frame="head"
        )
        raw.set_montage(montage, on_missing="ignore")

        # --- coregister to fsaverage (MRI-less), conservative ICP ---
        coreg = mne.coreg.Coregistration(
            info=raw.info, subject="fsaverage", subjects_dir=self.subjects_dir
        )
        coreg.fit_fiducials()
        if hsp is not None and len(hsp) >= 4:
            coreg.fit_icp(n_iterations=10)  # light refinement only
        trans = coreg.trans
        mne.write_trans("ctf_to_fsaverage-trans.fif", trans, overwrite=True)

        # --- forward & inverse on fsaverage ---
        src = mne.setup_source_space(
            "fsaverage", spacing=spacing, subjects_dir=self.subjects_dir, add_dist=False
        )
        bem_model = mne.make_bem_model(
            "fsaverage", ico=4, conductivity=[0.3], subjects_dir=self.subjects_dir
        )
        bem = mne.make_bem_solution(bem_model)

        # forward solution
        fwd = mne.make_forward_solution(
            raw.info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0
        )

        # Use empty-room if you have it; otherwise ad-hoc is fine for rhythms
        noise_cov = mne.make_ad_hoc_cov(raw.info)

        inv = make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
        lambda2 = 1.0 / (snr**2)
        stc = apply_inverse_raw(
            raw, inv, lambda2=lambda2, method="dSPM", pick_ori="normal"
        )

        # --- parcel time series (stable, low-dimensional) ---
        labels = mne.read_labels_from_annot(
            "fsaverage", parc=parc, subjects_dir=self.subjects_dir
        )
        # Drop labels that are not cortical parcels you actually want
        labels = [
            lab
            for lab in labels
            if "unknown" not in lab.name.lower()
            and "corpuscallosum" not in lab.name.lower()
        ]
        ts = mne.extract_label_time_course(stc, labels, src, mode="mean_flip")

        pos2d = compute_roi_layout_2d(labels, src)

        # Post-processing
        # 1) detrend
        ts = signal.detrend(ts, axis=1, type="linear")

        # 2) resample
        if self.sfreq is not None:
            gcd = int(np.gcd(int(round(sfreq)), int(round(self.sfreq))))
            up = int(round(self.sfreq / gcd))
            down = int(round(sfreq / gcd))
            ts = signal.resample_poly(ts, up, down, axis=1, padtype="line")
        else:
            self.sfreq = sfreq

        # assemble dict
        data = {
            "raw_array": ts,
            "sfreq": self.sfreq,
            "ch_names": [f"src{i}" for i in range(ts.shape[0])],
            "ch_types": ["parcel"] * ts.shape[0],
            "pos_2d": pos2d,
            "session": subject,
            "decimate": int(sfreq / self.sfreq),
        }
        return data

    def _interpolate_bads(self, raw):
        """Interpolate bad channels using MNE's interpolate_bads function."""
        for picks in ("mag", "grad"):
            try:
                raw = preprocessing.osl_wrappers.bad_channels(
                    raw, picks=picks, ref_meg=False
                )
            except Exception:
                continue

        # print how many bad channels were interpolated
        print(f"INFO: Detected {len(raw.info['bads'])} bad channels")

        # if number of bad channels is more than 25, return -> bad session
        if len(raw.info["bads"]) > 25:
            print(
                f"INFO: Too many bad channels ({len(raw.info['bads'])}), skipping sess"
            )
            return None

        # don't interpolate bads if we are doing source space projection
        if self.source_space:
            return raw

        print(f"INFO: Interpolating {len(raw.info['bads'])} bad channels")
        return raw.copy().interpolate_bads(reset_bads=True)

    def _detect_bad_channels(self, raw):
        """Run bad channel detection while excluding reference channels."""
        bads: set[str] = set(raw.info.get("bads", []))
        for picks in ("mag", "grad"):
            try:
                tmp = preprocessing.osl_wrappers.bad_channels(
                    raw.copy(), picks=picks, ref_meg=False
                )
                bads.update(tmp.info.get("bads", []))
            except Exception as exc:  # pragma: no cover - detection failures are logged
                print(f"WARNING: bad channel detection failed for picks={picks}: {exc}")
                continue
        raw.info["bads"] = list(bads)
        return raw

    def _safe_interpolate_bads(self, raw):
        """Interpolate bad channels; if interpolation fails, return None."""
        try:
            return self._interpolate_bads(raw)
        except Exception as exc:  # pragma: no cover - mne interpolation errors
            print(f"WARNING: interpolate_bads failed: {exc}")
            return None

    def extract_raw(self, fif_file: str, subject: str) -> dict[str, Any]:
        """Extract raw data and metadata from MNE Raw object with memory efficiency.

        Args:
        fif_file: Path to the fif file
        subject: Subject name

        Returns:
        Dictionary containing raw data and metadata
        """
        data = {}
        raw = mne.io.read_raw_fif(fif_file, preload=True)

        # keep only the MEG channels (drop reference channels early)
        keep_chn = [
            raw.ch_names[idx]
            for idx in mne.pick_types(raw.info, meg=True, ref_meg=False)
        ]
        raw.pick(picks=keep_chn)

        detected_bads: list[str] = []
        if self.bad_handling in {"zero", "interpolate"}:
            raw = self._detect_bad_channels(raw)
            detected_bads = [
                name for name in raw.info.get("bads", []) if name in raw.ch_names
            ]
            if self.bad_handling == "interpolate":
                raw_interp = self._safe_interpolate_bads(raw)
                if raw_interp is None:
                    return None
                raw = raw_interp

        if self.source_space:
            return self.source_space_proj(raw, subject)

        # Use memory-efficient data loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data["raw_array"] = raw.get_data()

        # Extract metadata
        data["sfreq"] = raw.info["sfreq"]
        data["ch_names"] = raw.ch_names
        data["ch_types"] = [
            raw.info["chs"][idx]["kind"] for idx in range(len(raw.ch_names))
        ]
        data["bad_channels"] = detected_bads

        # Get 2D sensor positions
        layout = mne.channels.find_layout(raw.info)

        # Filter layout to only keep position of channels in ch_names
        positions = []
        positions_3d = []
        orientations = []
        for ch_name in data["ch_names"]:
            pos = layout.pos[layout.names.index(ch_name.split("-")[0])]
            positions.append(pos[:2])
            ch_info = raw.info["chs"][raw.ch_names.index(ch_name)]
            loc = ch_info.get("loc", np.zeros(12, dtype=float))
            positions_3d.append(loc[:3])

            # MNE stores a 3x3 rotation matrix in loc[3:], columns (ex, ey, ez).
            # The coil "normal" aligns with the ez column for MEG sensors.
            ori_vec = np.array(loc[9:12], dtype=float)
            norm = np.linalg.norm(ori_vec)
            orientations.append(ori_vec / norm if norm > 0 else ori_vec)
        data["pos_2d"] = np.array(positions)
        data["pos_3d"] = np.array(positions_3d)
        data["ori_3d"] = np.array(orientations)

        # Get bad channels
        # data["bad_chs"] = raw.info["bads"]

        data["session"] = subject

        return data

    def normalize(self, data, method: str = "robust") -> dict[str, Any]:
        """Memory-efficient normalization using batches.

        Args:     data: Dictionary containing raw data and metadata     method: Method
        to use for normalization

        Returns:     Dictionary containing normalized data
        """
        cont_data = data["raw_array"]

        if method == "robust":
            scaler = RobustScaler(with_centering=True, with_scaling=True)
        else:
            scaler = StandardScaler()

        # Fit on transposed data for channel-wise normalization
        cont_data = scaler.fit_transform(cont_data.T).T

        # Store normalization parameters
        if method == "robust":
            data["scaler_centers"] = scaler.center_
            data["scaler_scales"] = scaler.scale_
        data["raw_array"] = cont_data

        return data

    def clip(self, data: dict[str, Any], std_factor: float = 3) -> dict[str, Any]:
        """Outlier clipping over the whole data array at once.

        Args:     data: Dictionary containing raw data and metadata     std_factor:
        Factor to multiply the standard deviation by

        Returns:     Dictionary containing clipped data
        """
        arr = data["raw_array"]
        mean = arr.mean(axis=1, keepdims=True)
        std = arr.std(axis=1, keepdims=True)
        lower_bound = mean - std * std_factor
        upper_bound = mean + std * std_factor

        n_clipped = np.sum((arr < lower_bound) | (arr > upper_bound))
        np.clip(arr, lower_bound, upper_bound, out=arr)
        percent_clipped = (n_clipped / arr.size) * 100
        print(f"INFO: {percent_clipped:.2f}% of data clipped")
        data["clipped_percent"] = percent_clipped
        data["raw_array"] = arr
        return data

    def mulaw_quantize(self, data: dict[str, Any], n_bits: int = 8) -> dict[str, Any]:
        """Args: data: Dictionary containing raw data and metadata n_bits: Number of
        bits to use for quantization.

        Returns:     Dictionary containing quantized data
        """
        # first do max scaling for the data to be in -1, 1 range
        max_val = np.max(np.abs(data["raw_array"]))
        data["raw_array"] = data["raw_array"] / max_val

        mu = 2**n_bits - 1
        quant, recon = mulaw(data["raw_array"], mu)

        # compute the mean squared error
        mse = np.mean((data["raw_array"] - recon) ** 2)
        print(f"INFO: Mean squared error of quantization: {mse}")
        data["mse"] = mse

        data["raw_array"] = quant
        return data

    @staticmethod
    def decode_predictive_tokens(
        tokens: np.ndarray,
        *,
        scale: float | np.ndarray = 1.0,
        mu: int = 255,
    ) -> np.ndarray:
        """Inverse µ-law tokens and integrate predictor residuals."""
        return predictive_mulaw_decode(tokens, scale=scale, mu=int(mu))

    # Override stage-3 chunk quantization to use predictor residuals + µ-law
    def _quantize_chunks(self, src_dir: str, dst_dir: str, chunk_files: list[str]):
        """Quantize chunks using predictive residuals + µ-law compression."""
        residuals: list[np.ndarray] = []
        chunk_records: list[tuple[str, dict[str, Any], np.ndarray]] = []

        for chunk_name in chunk_files:
            chunk_path = Path(src_dir) / chunk_name
            try:
                chunk = np.load(chunk_path, allow_pickle=True).item()
            except Exception as exc:  # pragma: no cover - IO errors logged
                print(f"WARNING: Failed to load {chunk_path}: {exc}")
                continue

            if not isinstance(chunk, dict) or "data" not in chunk:
                print(f"WARNING: {chunk_path} missing 'data'; skipping")
                continue

            data = np.asarray(chunk["data"], dtype=np.float32)
            residual = predictive_residual_encode(data)
            residuals.append(residual)
            chunk_records.append((chunk_name, chunk, residual))

        if not residuals:
            print("INFO: No valid chunks to quantize.")
            return

        residual_cat = np.concatenate(residuals, axis=-1)

        # print percentage of residuals clipped
        n_clipped = np.sum(np.abs(residual_cat) > self.residual_scale)
        print(f"INFO: {n_clipped / residual_cat.size * 100:.2f}% of residuals clipped")

        for chunk_name, chunk, residual in chunk_records:
            res_norm = residual / self.residual_scale
            res_norm = np.clip(res_norm, -0.99, 0.99)
            # quant, recon = mulaw(res_norm, mu)
            quant = quantize_deadzone_linear(res_norm, 0.01, bins=self.text_num_bins)

            chunk["data"] = quant.astype(np.uint8, copy=False)
            chunk["residual_scale"] = self.residual_scale
            # chunk["mulaw_mu"] = int(mu)
            # chunk["mulaw_mse"] = float(np.mean((res_norm - recon) ** 2))

            out_path = Path(dst_dir) / chunk_name
            np.save(out_path, chunk)
