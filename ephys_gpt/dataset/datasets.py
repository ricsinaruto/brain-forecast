import os
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentations import Augmentations
from ..utils.quantizers import mulaw_torch

SENSOR_TYPES = {
    "GRAD_CTF": 0,
    "MAG": 1,
    "GRAD_ELEKTA_X": 2,
    "GRAD_ELEKTA_Y": 3,
}

DATASET_NAMES = {
    "omega": "GRAD_CTF",
}


class Postprocessor:
    def __init__(
        self, pos_2d: List[Tuple[float, float]], image_size: int, tmp_dir: str = "tmp"
    ):
        # --- Pre-compute the (row, col) pixel for each channel -----------------
        pos = pos_2d.astype(np.float32)
        # Normalise to the unit square [0,1]²
        pos_min = pos.min(axis=0)
        pos_max = pos.max(axis=0)
        span = pos_max - pos_min
        span[span == 0] = 1.0  # avoid divide-by-zero
        pos_norm = (pos - pos_min) / span

        col_idx = np.round(pos_norm[:, 0] * (image_size - 1)).astype(np.int64)
        row_idx = np.round(pos_norm[:, 1] * (image_size - 1)).astype(np.int64)

        self.row_idx = torch.from_numpy(row_idx)
        self.col_idx = torch.from_numpy(col_idx)

        # save these to a tmp file
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        tmp_file = os.path.join(tmp_dir, "img_inds.npy")
        np.save(tmp_file, {"row_idx": row_idx, "col_idx": col_idx})

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape an image into a channel-wise tensor. This is the opposite of
        the vectorised scatter operation in __getitem__. It should create a tensor
        with shape (C, T), where C is the number of channels, always lower
        than H * W.

        x: [H, W, T] -> [C, T]
        """
        if isinstance(x, tuple) or isinstance(x, list):
            x = x[0]  # kind of hacky assuming tensor is first element of tuple/list

        # Add a batch dimension if it's not present to unify logic
        squeeze_batch = False
        squeeze_all = False
        if x.ndim == 2:  # [H, W] -> [1, H, W, 1]
            x = x.unsqueeze(0).unsqueeze(-1)
            squeeze_all = True
        elif x.ndim == 3:  # [H, W, T]
            x = x.unsqueeze(0)  # [1, H, W, T]
            squeeze_batch = True

        # x: [B, H, W, T] -> out: [B, C, T]
        out = x[:, self.row_idx, self.col_idx, ...]

        # If original input was 3D, remove batch dimension
        if squeeze_batch:
            out = out.squeeze(0)  # [C, T]
        if squeeze_all:
            out = out.squeeze(-1).squeeze(0)

        return out

    def __call__(self, *tensors):
        return tuple(map(self.reshape, tensors))


class ChunkDataset(Dataset):
    """Dataset for chunked numpy files with canonicalised channel layouts."""

    IndexEntry = Tuple[str, str, str, int]

    def __init__(
        self,
        root_dir: Union[str, Mapping[str, str]],
        indices: Sequence[Union[IndexEntry, Tuple[str, int, int], object]],
        length: int,
        ch_names: Sequence[str],
        pos_2d: Sequence[Tuple[float, float]],
        sfreq: Union[int, float],
        name: Optional[str] = None,
        image_size: int = 32,
        aug_cfg: Optional[dict] = None,
        tmp_dir: str = "tmp",
        fill_value: int = 0,
        *,
        ch_types: Optional[Sequence[Union[int, str]]] = None,
        session_channels: Optional[Mapping[Tuple[str, str], object]] = None,
    ) -> None:
        self.root_dirs = self._normalise_roots(root_dir)
        self.default_dataset_key = next(iter(self.root_dirs))
        self.root_dir = self.root_dirs[self.default_dataset_key]
        self.indices = self._normalise_indices(indices)
        self.length = int(length)
        self.sfreq = float(sfreq)
        self.image_size = image_size
        self.tmp_dir = tmp_dir
        self.fill_value = fill_value

        self.ch_names = [str(name) for name in ch_names]
        self.pos_2d = np.asarray(pos_2d, dtype=np.float32)
        self.num_channels = len(self.ch_names)

        self.augmentations = Augmentations(aug_cfg)
        Postprocessor(self.pos_2d, image_size, tmp_dir)

        self.ch_type_labels = self._resolve_channel_labels(ch_types, name)
        self.ch_type = self._encode_channel_types(self.ch_type_labels)

        self.session_indices = self._prepare_session_channels(
            session_channels, len(self.ch_names)
        )

        print(
            "Canonical channels:",
            len(self.ch_names),
            "| sessions:",
            len({(d, s) for d, s, _, _ in self.indices}),
        )

    @staticmethod
    def _normalise_roots(root_dir: Union[str, Mapping[str, str]]) -> Dict[str, str]:
        if isinstance(root_dir, Mapping):
            roots = {str(k): str(v) for k, v in root_dir.items()}
        else:
            roots = {"dataset0": str(root_dir)}

        for key, path in roots.items():
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Dataset root not found for {key}: {path}")
        return roots

    def _normalise_indices(
        self, indices: Sequence[Union[IndexEntry, Tuple[str, int, int], object]]
    ) -> List[IndexEntry]:
        normalised: List[ChunkDataset.IndexEntry] = []

        for item in indices:
            if hasattr(item, "dataset") and hasattr(item, "session"):
                dataset_key = str(getattr(item, "dataset"))
                session = str(getattr(item, "session"))
                chunk = str(getattr(item, "chunk"))
                start = int(getattr(item, "start"))
            else:
                session, chunk_idx, start = item  # type: ignore[misc]
                dataset_key = self.default_dataset_key
                chunk = f"{int(chunk_idx)}.npy"
                start = int(start)

            if dataset_key not in self.root_dirs:
                raise KeyError(
                    f"Unknown dataset key '{dataset_key}' for index (session={session})"
                )

            normalised.append((dataset_key, session, chunk, start))

        return normalised

    def _resolve_channel_labels(
        self,
        ch_types: Optional[Sequence[Union[int, str]]],
        name: Optional[str],
    ) -> List[Union[int, str]]:
        if ch_types is not None:
            return list(ch_types)

        if name and name in DATASET_NAMES:
            sensor_key = DATASET_NAMES[name]
            mapped = SENSOR_TYPES.get(sensor_key)
            if mapped is not None:
                return [mapped] * len(self.ch_names)

        return ["unknown"] * len(self.ch_names)

    @staticmethod
    def _encode_channel_types(labels: Sequence[Union[int, str]]) -> np.ndarray:
        arr = np.asarray(labels)
        if arr.dtype.kind in {"i", "u"}:
            return arr.astype(np.int64)

        unique = {label: idx for idx, label in enumerate(sorted(set(arr.tolist())))}
        return np.array([unique[label] for label in arr], dtype=np.int64)

    def _prepare_session_channels(
        self,
        session_channels: Optional[Mapping[Tuple[str, str], object]],
        canonical_size: int,
    ) -> Dict[Tuple[str, str], np.ndarray]:
        identity = np.arange(canonical_size, dtype=np.int64)
        mapping: Dict[Tuple[str, str], np.ndarray] = {}

        if session_channels:
            for key, value in session_channels.items():
                indices = getattr(value, "indices", value)
                arr = np.asarray(indices, dtype=np.int64)
                mapping[(str(key[0]), str(key[1]))] = arr

        self._identity_indices = identity
        self._session_present: Dict[Tuple[str, str], np.ndarray] = {}
        for key, arr in mapping.items():
            mask = np.zeros(canonical_size, dtype=bool)
            mask[arr] = True
            self._session_present[key] = mask

        return mapping

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.indices)

    def make_postprocessor(self):
        return Postprocessor(self.pos_2d, self.image_size, self.tmp_dir)

    def _get_session_indices(
        self, dataset_key: str, session: str, n_channels: int
    ) -> np.ndarray:
        key = (dataset_key, session)
        if key in self.session_indices:
            return self.session_indices[key]
        if n_channels > len(self._identity_indices):
            raise ValueError(
                f"Session {session} ({dataset_key}) "
                "has more channels than canonical layout"
            )
        return self._identity_indices[:n_channels]

    def _resolve_index(self, idx: int) -> IndexEntry:
        return self.indices[idx]

    def _load_data(self, idx: int):
        dataset_key, session, chunk, start = self._resolve_index(idx)
        root = self.root_dirs[dataset_key]
        file_path = os.path.join(root, session, chunk)
        data_dict = np.load(file_path, allow_pickle=True).item()

        data = data_dict["data"]
        window = data[:, start : start + self.length]
        if window.shape[1] < self.length:
            pad_width = self.length - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad_width)), mode="constant")

        mapped = np.ones((len(self.ch_names), self.length), dtype=window.dtype)
        mapped *= self.fill_value
        indices = self._get_session_indices(dataset_key, session, window.shape[0])

        if len(indices) != window.shape[0]:
            raise ValueError(
                f"Channel count mismatch for session {session} ({dataset_key}):"
                f" expected {len(indices)}, got {window.shape[0]}"
            )

        mapped[indices, :] = window
        x = torch.from_numpy(mapped)

        data_dict["indices"] = torch.from_numpy(indices)
        return x, data_dict

    def __getitem__(self, idx: int):  # type: ignore[override]
        x, _ = self._load_data(idx)
        x = x.long()

        inputs = x[:, :-1]
        targets = x[:, 1:]

        return inputs, targets


class ChunkDatasetCondition(ChunkDataset):
    def _load_data(self, idx: int):
        dataset_key, session, chunk, start = self._resolve_index(idx)
        root = self.root_dirs[dataset_key]
        file_path = os.path.join(root, session, chunk)
        data_dict = np.load(file_path, allow_pickle=True).item()

        data = data_dict["data"]
        condition = data[-1, start : start + self.length]

        # check if condition contains a few integers, or continuous values
        if "rest" in session:
            window = data[:, start : start + self.length]
            condition = np.zeros(self.length, dtype=window.dtype)

        else:
            window = data[:-1, start : start + self.length]

        if window.shape[1] < self.length:
            pad_width = self.length - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad_width)), mode="constant")

        mapped = np.ones((len(self.ch_names), self.length), dtype=window.dtype)
        mapped *= self.fill_value
        indices = self._get_session_indices(dataset_key, session, window.shape[0])

        if len(indices) != window.shape[0]:
            raise ValueError(
                f"Channel count mismatch for session {session} ({dataset_key}):"
                f" expected {len(indices)}, got {window.shape[0]}"
            )

        mapped[indices, :] = window
        x = torch.from_numpy(mapped)

        data_dict["indices"] = torch.from_numpy(indices)
        data_dict["condition"] = torch.from_numpy(condition).long()

        return x, data_dict

    def __getitem__(self, idx: int):  # type: ignore[override]
        x, data_dict = self._load_data(idx)
        x = x.long()

        inputs = x[:, :-1]
        targets = x[:, 1:]
        cond = data_dict["condition"][None, :-1]

        return (inputs, cond), targets


class ChunkDatasetMasked(ChunkDataset):
    def __getitem__(self, idx: int):  # type: ignore[override]
        x, data_dict = self._load_data(idx)
        x = x.long()

        inputs = x[:, :-1]
        targets = x[:, 1:]

        indices = data_dict["indices"]
        mask = torch.zeros(self.num_channels, dtype=torch.bool, device=indices.device)
        mask[indices] = True

        return inputs, (targets, mask)


class ChunkDatasetSubset(ChunkDataset):
    _PRESET_KEYS = ("visual",)

    def __init__(
        self,
        *args,
        channel_subset: Union[str, Sequence[str]] = "visual",
        **kwargs,
    ) -> None:
        if channel_subset is None:
            raise ValueError("channel_subset must be provided for ChunkDatasetSubset.")

        self._requested_subset = channel_subset

        super().__init__(*args, **kwargs)

        self._canonical_ch_names = list(self.ch_names)
        self._canonical_pos_2d = np.array(self.pos_2d, copy=True)
        self._canonical_ch_type = np.array(self.ch_type, copy=True)
        self._canonical_ch_type_labels = list(self.ch_type_labels)

        subset_names = self._normalise_subset(channel_subset)
        subset_indices = self._indices_from_names(subset_names)
        if subset_indices.size == 0:
            raise ValueError("Resolved channel subset is empty.")

        self._subset_indices = subset_indices
        self._subset_names = [self._canonical_ch_names[i] for i in subset_indices]

        self.ch_names = list(self._subset_names)
        self.pos_2d = self._canonical_pos_2d[subset_indices]
        self.ch_type_labels = [
            self._canonical_ch_type_labels[i] for i in subset_indices
        ]
        self.ch_type = self._canonical_ch_type[subset_indices]
        self.num_channels = len(self.ch_names)

        # Regenerate postprocessor artefacts for the active subset layout.
        Postprocessor(self.pos_2d, self.image_size, self.tmp_dir)

    # ------------------------------------------------------------------ #
    def _normalise_subset(self, channel_subset: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(channel_subset, str):
            option = channel_subset.strip().lower()
            if option in self._PRESET_KEYS:
                resolved = self._resolve_preset(option)
                if resolved:
                    return resolved
                raise ValueError(
                    f"Preset '{channel_subset}' did not match any channels."
                )
            # Treat as single explicit channel name.
            return [channel_subset]

        if not isinstance(channel_subset, Sequence):
            raise TypeError(
                "channel_subset must be a string preset or a sequence of channel names."
            )

        subset_list = [str(name) for name in channel_subset]
        if not subset_list:
            raise ValueError("channel_subset must contain at least one channel name.")
        return subset_list

    def _indices_from_names(self, subset_names: Sequence[str]) -> np.ndarray:
        name_to_index = {name: idx for idx, name in enumerate(self._canonical_ch_names)}
        indices: List[int] = []
        missing: List[str] = []
        seen = set()

        for name in subset_names:
            key = str(name)
            idx = name_to_index.get(key)
            if idx is None:
                missing.append(key)
                continue
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)

        if missing:
            raise KeyError(
                "Unknown channel names in subset: " + ", ".join(sorted(missing))
            )

        if not indices:
            raise ValueError("No valid channel names resolved for the subset.")

        return np.array(indices, dtype=np.int64)

    def _resolve_preset(self, option: str) -> List[str]:
        if option == "visual":
            return self._resolve_visual_channels()
        raise ValueError(f"Unknown channel subset preset '{option}'.")

    def _resolve_visual_channels(self) -> List[str]:
        prefixes = ("MLP", "MLO", "MRO", "MZP", "MZO")
        matches = [
            name
            for name in self._canonical_ch_names
            if any(name.startswith(prefix) for prefix in prefixes)
        ]
        if matches:
            return matches

        # Fallback: pattern-based matching for generic CTF posterior sensors.
        pattern = re.compile(r"^M[LRZ][OP].*")
        matches = [name for name in self._canonical_ch_names if pattern.match(name)]
        if matches:
            return matches

        # Last resort: pick channels in the posterior third based on 2D layout.
        if self._canonical_pos_2d.size == 0:
            return []
        y_coords = self._canonical_pos_2d[:, 1]
        threshold = np.quantile(y_coords, 0.33)
        posterior_mask = y_coords <= threshold
        return [
            name for name, keep in zip(self._canonical_ch_names, posterior_mask) if keep
        ]

    # ------------------------------------------------------------------ #
    def _load_data(self, idx: int):
        dataset_key, session, chunk, start = self._resolve_index(idx)
        root = self.root_dirs[dataset_key]
        file_path = os.path.join(root, session, chunk)
        data_dict = np.load(file_path, allow_pickle=True).item()

        data = data_dict["data"]
        window = data[:, start : start + self.length]
        if window.shape[1] < self.length:
            pad_width = self.length - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad_width)), mode="constant")

        canonical_size = len(self._canonical_ch_names)
        mapped = np.ones((canonical_size, self.length), dtype=window.dtype)
        mapped *= self.fill_value
        indices = self._get_session_indices(dataset_key, session, window.shape[0])

        if len(indices) != window.shape[0]:
            raise ValueError(
                f"Channel count mismatch for session {session} ({dataset_key}):"
                f" expected {len(indices)}, got {window.shape[0]}"
            )

        mapped[indices, :] = window

        subset_idx = self._subset_indices
        # missing_mask = ~np.isin(subset_idx, indices)
        # if np.any(missing_mask):
        # missing = [self._canonical_ch_names[i] for i in subset_idx[missing_mask]]

        subset_data = mapped[subset_idx, :]
        x = torch.from_numpy(subset_data)

        data_dict["canonical_indices"] = torch.from_numpy(indices)
        data_dict["indices"] = torch.arange(x.shape[0], dtype=torch.int64)
        return x, data_dict


class ChunkDatasetMous(ChunkDataset):
    def __getitem__(self, idx: int):  # type: ignore[override]
        inputs, targets = super().__getitem__(idx)
        return inputs[:272, :], targets[:272, :]


class ChunkDatasetJIT(ChunkDataset):
    def __init__(self, *args, quant_levels: int = 256, max_val: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_levels = quant_levels
        self.max_val = max_val

    def __getitem__(self, idx: int):  # type: ignore[override]
        x, _ = self._load_data(idx)

        # need to clip
        x = x / self.max_val
        x = mulaw_torch(x, self.quant_levels - 1)

        inputs = x[:, :-1]
        targets = x[:, 1:]

        return inputs, targets


class ChunkDatasetForecastCont(ChunkDataset):
    def __getitem__(self, idx: int):  # type: ignore[override]
        x, _ = self._load_data(idx)

        inputs = x[:, :-1].float()
        targets = x[:, 1:].float()

        return inputs, targets


class ChunkDatasetReconstruction(ChunkDataset):
    def __getitem__(self, idx: int):  # type: ignore[override]
        x, _ = self._load_data(idx)
        x = x.float()

        pos = torch.from_numpy(self.pos_2d).float()
        ch_type = torch.from_numpy(self.ch_type).long()

        inputs = (x, pos, ch_type)

        return inputs, x


class ChunkDatasetSensorPos(ChunkDataset):
    def __getitem__(self, idx):
        dataset_key, session, chunk, _ = self._resolve_index(idx)
        file_path = os.path.join(self.root_dirs[dataset_key], session, chunk)
        data_dict = np.load(file_path, allow_pickle=True).item()

        return data_dict["ch_names"], data_dict["pos_2d"]


class ChunkDatasetImage(ChunkDataset):
    """
    Dataset that maps sensor channels into a sparse H×W image (default 32×32).
    Each channel’s value is written to the pixel closest to its 2-D position
    in ``pos_2d``.  The spatial layout of the MEG helmet is thus roughly
    preserved inside an image that can be processed by vision models.

    Input  : x ∈ ℝ^{C×T}
    Output : img ∈ ℝ^{H×W×T}  (sparse – most pixels are zero)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postprocessor = self.make_postprocessor()
        self.row_idx = self.postprocessor.row_idx
        self.col_idx = self.postprocessor.col_idx

    def __getitem__(self, idx: int):
        x, _ = self._load_data(idx)  # x: [C, T]
        x = x.float()

        H = W = self.image_size
        _, T = x.shape
        img = torch.ones((H, W, T), dtype=x.dtype) * self.fill_value

        # Vectorised scatter – assign each channel to its pixel across time
        img[self.row_idx, self.col_idx, :] = x

        return img, img


class ChunkDatasetImageQuantized(ChunkDatasetImage):
    def __getitem__(self, idx: int):
        img, _ = super().__getitem__(idx)

        return img[..., :-1].long(), img[..., 1:].long()


class ChunkDatasetImageCondition(ChunkDatasetCondition):
    """
    Dataset that maps sensor channels into a sparse H×W image (default 32×32).
    Each channel’s value is written to the pixel closest to its 2-D position
    in ``pos_2d``.  The spatial layout of the MEG helmet is thus roughly
    preserved inside an image that can be processed by vision models.

    Input  : x ∈ ℝ^{C×T}
    Output : img ∈ ℝ^{H×W×T}  (sparse – most pixels are zero)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.postprocessor = self.make_postprocessor()
        self.row_idx = self.postprocessor.row_idx
        self.col_idx = self.postprocessor.col_idx

    def __getitem__(self, idx: int):
        x, data_dict = self._load_data(idx)  # x: [C, T]
        cond = data_dict["condition"][None, None, :-1]
        x = x.float()

        H = W = self.image_size
        _, T = x.shape
        img = torch.ones((H, W, T), dtype=x.dtype) * self.fill_value

        # Vectorised scatter – assign each channel to its pixel across time
        img[self.row_idx, self.col_idx, :] = x

        return (img, cond), img


class ChunkDatasetImageQuantizedCondition(ChunkDatasetImageCondition):
    def __getitem__(self, idx: int):
        img, _ = super().__getitem__(idx)
        img, cond = img

        return (img[..., :-1].long(), cond), img[..., 1:].long()


class ChunkDatasetImage01(ChunkDataset):
    """
    Dataset that maps sensor channels into a sparse H×W image (default 32×32).
    Each channel’s value is written to the pixel closest to its 2-D position
    in ``pos_2d``.  The spatial layout of the MEG helmet is thus roughly
    preserved inside an image that can be processed by vision models.

    Input  : x ∈ ℝ^{C×T}
    Output : img ∈ ℝ^{H×W×T}  (sparse – most pixels are zero)

    The dataset returns (*img*[:, :, :-1], *img*[:, :, 1:]) so that models can
    learn to predict the next timestep given the past.
    """

    def __init__(self, *args, quant_levels=256, **kwargs):
        super().__init__(*args, **kwargs)
        postprocessor = self.make_postprocessor()
        self.row_idx = postprocessor.row_idx
        self.col_idx = postprocessor.col_idx
        self.quant_levels = quant_levels

    def __getitem__(self, idx: int):
        x, _ = self._load_data(idx)  # x: [C, T]
        x = x.float().transpose(0, 1)  # T, C
        T, _ = x.shape

        # squash 1-256 to 0-1
        x = (x + 1) / (self.quant_levels + 1)

        H = W = self.image_size
        img = torch.ones((T, H, W), dtype=x.dtype) * self.fill_value

        # Vectorised scatter – assign each channel to its pixel across time
        img[..., self.row_idx, self.col_idx] = x

        img = img.unsqueeze(1)  # T, 1, H, W

        return img, img
