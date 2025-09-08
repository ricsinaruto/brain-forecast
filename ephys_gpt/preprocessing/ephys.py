"""Base classes for ephys recordings."""

import warnings
from typing import Any

import mne
import numpy as np
from sklearn.preprocessing import RobustScaler

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
        **kwargs,
    ) -> None:
        """
        Args:
            maxwell: Whether to use the maxwell filter
        """
        super().__init__(*args, **kwargs)

        if self.osl_config is None:
            if maxwell:
                self.osl_config = self.default_config_maxwell
            else:
                self.osl_config = self.default_config

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

        scaler = RobustScaler(with_centering=True, with_scaling=True)

        # Fit on transposed data for channel-wise normalization
        cont_data = scaler.fit_transform(cont_data.T).T

        # Store normalization parameters
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
