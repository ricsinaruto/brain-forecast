"""Base classes for ephys recordings."""

import warnings
from typing import Any
from scipy import signal
from pathlib import Path

import mne

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from mne.io.constants import FIFF
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_raw

from ..utils.quantizers import mulaw

from .base import Preprocessing


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
        get_fsaverage_data: bool = False,
        fsaverage_dir: str = "/vol/data/datasets/mne_data",
        **kwargs,
    ) -> None:
        """
        Args:
            maxwell: Whether to use the maxwell filter
        """
        super().__init__(*args, **kwargs)

        self.source_space = source_space
        self.sfreq = sfreq
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

        Args:
            data: Dictionary containing raw MNE object
            min_duration: Minimum duration between events in seconds
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

        # Post-processing
        # 1) detrend
        ts = signal.detrend(ts, axis=1, type="linear")

        # 2) resample
        gcd = int(np.gcd(int(round(sfreq)), int(round(self.sfreq))))
        up = int(round(self.sfreq / gcd))
        down = int(round(sfreq / gcd))
        ts = signal.resample_poly(ts, up, down, axis=1, padtype="line")

        # fake pos_2d
        pos_2d = np.array(
            [[i / ts.shape[0], i / ts.shape[0]] for i in range(ts.shape[0])]
        )

        # assemble dict
        data = {
            "raw_array": ts,
            "sfreq": self.sfreq,
            "ch_names": [f"src{i}" for i in range(ts.shape[0])],
            "ch_types": ["parcel"] * ts.shape[0],
            "pos_2d": pos_2d,
            "session": subject,
            "decimate": int(sfreq / self.sfreq),
        }
        return data

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

        if self.source_space:
            return self.source_space_proj(raw, subject)

        # keep only the MEG channels
        keep_chn = [
            raw.ch_names[idx]
            for idx in mne.pick_types(raw.info, meg=True, ref_meg=False)
        ]
        raw.pick(picks=keep_chn)

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

        # Get 2D sensor positions
        layout = mne.channels.find_layout(raw.info)

        # Filter layout to only keep position of channels in ch_names
        positions = []
        for ch_name in data["ch_names"]:
            pos = layout.pos[layout.names.index(ch_name.split("-")[0])]
            positions.append(pos[:2])
        data["pos_2d"] = np.array(positions)

        # Get bad channels
        # data["bad_chs"] = raw.info["bads"]

        data["session"] = subject

        return data

    def normalize(self, data, method: str = "robust") -> dict[str, Any]:
        """Memory-efficient normalization using batches.

        Args:
            data: Dictionary containing raw data and metadata
            method: Method to use for normalization

        Returns:
            Dictionary containing normalized data
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

        Args:
            data: Dictionary containing raw data and metadata
            std_factor: Factor to multiply the standard deviation by

        Returns:
            Dictionary containing clipped data
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
        """
        Args:
            data: Dictionary containing raw data and metadata
            n_bits: Number of bits to use for quantization

        Returns:
            Dictionary containing quantized data
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
