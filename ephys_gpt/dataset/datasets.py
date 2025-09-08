import os
from typing import List, Tuple, Dict, Optional
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset
from ephys_gpt.utils.quantizers import mulaw_torch
from .augmentations import Augmentations

SENSOR_TYPES = {
    "GRAD_CTF": 0,
    "MAG": 1,
    "GRAD_ELEKTA_X": 2,
    "GRAD_ELEKTA_Y": 3,
}

DATASET_NAMES = {
    "omega": "GRAD_CTF",
}


class ChunkDataset(Dataset):
    """Dataset for chunked numpy files."""

    def __init__(
        self,
        root_dir: str,
        indices: List[Tuple[str, int, int]],
        length: int,
        ch_names: List[str],
        pos_2d: List[Tuple[float, float]],
        sfreq: int,
        name: str = "omega",
        aug_cfg: dict = None,
    ) -> None:
        """
        Args:
            root_dir: Root directory of the dataset
            indices: List of tuples containing (session, chunk_idx, start)
            length: Length of the chunk
            ch_names: List of channel names
            pos_2d: List of tuples containing (x, y) positions
            sfreq: Sampling frequency
        """
        self.root_dir = root_dir
        self.indices = indices
        self.length = length
        self.ch_names = ch_names
        self.pos_2d = np.array(pos_2d)
        self.sfreq = sfreq

        print(f"Number of channels: {len(ch_names)}")
        print(f"Number of positions: {len(pos_2d)}")

        ch_type = SENSOR_TYPES[DATASET_NAMES[name]]
        self.ch_type = np.array([ch_type] * len(ch_names))

        # create a mapping from ch_names to indices
        self.ch_to_idx = {ch: i for i, ch in enumerate(self.ch_names)}

        self.augmentations = Augmentations(aug_cfg)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.indices)

    def _load_data(self, idx: int):
        session, chunk_idx, start = self.indices[idx]
        file_path = os.path.join(self.root_dir, session, f"{chunk_idx}.npy")
        data_dict = np.load(file_path, allow_pickle=True).item()
        x = data_dict["data"][:, start : start + self.length]
        current_inds = range(x.shape[0])

        # Sample len(self.ch_names) random indices with replacement
        sampled_inds = np.random.choice(
            current_inds, size=len(self.ch_names), replace=True
        )
        x_new = x[sampled_inds]

        # now place the data in the correct indices
        for i, ch_name in enumerate(data_dict["ch_names"]):
            x_new[self.ch_to_idx[ch_name]] = x[i]

        x = torch.from_numpy(x_new)

        return x, data_dict

    def __getitem__(self, idx: int):  # type: ignore[override]
        x, _ = self._load_data(idx)
        x = x.long()

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


class ChunkDatasetImage(ChunkDataset):
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

    def __init__(
        self,
        root_dir: str,
        indices: List[Tuple[str, int, int]],
        length: int,
        ch_names: List[str],
        pos_2d: List[Tuple[float, float]],
        sfreq: int,
        name: str = "omega",
        image_size: int = 32,
    ) -> None:
        super().__init__(root_dir, indices, length, ch_names, pos_2d, sfreq, name)
        self.image_size = image_size

        # --- Pre-compute the (row, col) pixel for each channel -----------------
        pos = self.pos_2d.astype(np.float32)
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

        # create a H x W mask based on the row_idx and col_idx
        self.mask = torch.zeros((image_size, image_size), dtype=torch.bool)
        self.mask[row_idx, col_idx] = True

    # --------------------------------------------------------------------- #
    def __getitem__(self, idx: int):
        x, _ = self._load_data(idx)  # x: [C, T]
        x = x.float()

        H = W = self.image_size
        _, T = x.shape
        img = torch.zeros((H, W, T), dtype=x.dtype)

        # Vectorised scatter – assign each channel to its pixel across time
        img[self.row_idx, self.col_idx, :] = x

        return img, img

    def reshape(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape an image into a channel-wise tensor. This is the opposite of
        the vectorised scatter operation in __getitem__. It should create a tensor
        with shape (C, T), where C is the number of channels, always lower
        than H * W.

        x: [H, W, T] -> [C, T]
        """
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

    def postprocess(self, *tensors):
        return tuple(map(self.reshape, tensors))


class ChunkDatasetImageQuantized(ChunkDatasetImage):
    def __getitem__(self, idx: int):
        x, _ = self._load_data(idx)  # x: [C, T]
        x = x.long()

        H = W = self.image_size
        _, T = x.shape
        img = torch.zeros((H, W, T), dtype=x.dtype)

        # Vectorised scatter – assign each channel to its pixel across time
        img[self.row_idx, self.col_idx, :] = x

        return img[..., :-1], img[..., 1:]


class ChunkDatasetSensorPos(ChunkDataset):
    def __getitem__(self, idx):
        session, chunk_idx, start = self.indices[idx]
        file_path = os.path.join(self.root_dir, session, f"{chunk_idx}.npy")
        data_dict = np.load(file_path, allow_pickle=True).item()

        return data_dict["ch_names"], data_dict["pos_2d"]


class RandomLabelGroupedDataset(Dataset):
    """
    Advanced grouping dataset that, at access time, randomly selects a label
    and then randomly samples ``grouped_samples`` examples with that label.

    This avoids precomputing static groups and supports label-balanced sampling
    even when labels are imbalanced.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        grouped_samples_range: Tuple[int, int] = (1, 200),
        grouped_samples_mean: int = 100,
        grouped_samples_std: int = 10,
        average_grouped_samples: bool = True,
        lazy_cache: bool = False,
        cache_max_gb: Optional[float] = None,
    ) -> None:
        self.original_dataset = original_dataset
        self.grouped_samples_range = grouped_samples_range
        self.grouped_samples_mean = int(grouped_samples_mean)
        self.grouped_samples_std = int(grouped_samples_std)
        self.average_grouped_samples = bool(average_grouped_samples)

        # Per-index lazy cache (process-local)
        self._lazy_cache_enabled: bool = bool(lazy_cache)
        self._item_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._cache_max_bytes: Optional[int] = (
            int(cache_max_gb * (1024**3)) if cache_max_gb is not None else None
        )
        self._cache_cur_bytes: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Build mapping label -> list of sample indices for that label
        label_to_indices: Dict[int, List[int]] = {}
        for i, sample in enumerate(original_dataset.samples):
            _, _, _, _, _, label = sample
            phoneme = label.split("_")[0]
            label_val = original_dataset.phoneme_to_id[phoneme]
            label_to_indices.setdefault(label_val, []).append(i)
        self.label_to_indices = label_to_indices
        self.labels = sorted(self.label_to_indices.keys())

        if len(self.labels) == 0:
            raise ValueError(
                "RandomLabelGroupedDataset: no labels found in original dataset."
            )

    def __len__(self) -> int:  # type: ignore[override]
        # Keep length tied to base dataset so epoch size remains familiar
        return len(self.original_dataset) // self.grouped_samples_mean

    def __getitem__(self, idx: int):  # type: ignore[override]
        # Randomly select a label,
        # then sample k indices from that label with replacement
        num_labels = len(self.labels)
        label_idx = int(torch.randint(low=0, high=num_labels, size=(1,)).item())
        chosen_label = self.labels[label_idx]
        pool = self.label_to_indices[chosen_label]

        if len(pool) == 0:
            raise RuntimeError(
                f"RandomLabelGroupedDataset: no samples for label {chosen_label}."
            )

        # Sample k ~ N(mean, std), clamp to [low, high]
        low, high = self.grouped_samples_range
        if self.grouped_samples_std > 0:
            k_sample = torch.normal(
                mean=torch.tensor(float(self.grouped_samples_mean)),
                std=torch.tensor(float(self.grouped_samples_std)),
            ).item()
        else:
            k_sample = float(self.grouped_samples_mean)
        k = max(low, min(int(k_sample), high))

        choice_ix = torch.randperm(len(pool))[:k]
        sampled_indices = [pool[j] for j in choice_ix]

        xs = [self._get_cached_item(i) for i in sampled_indices]

        if self.average_grouped_samples:
            x = torch.stack(xs, dim=0).mean(dim=0)
        else:
            x = torch.cat(xs, dim=0)

        y = torch.tensor(chosen_label, dtype=torch.long)
        return x, y

    # ------------------------------------------------------------------ #
    def _get_cached_item(self, idx: int) -> torch.Tensor:
        """Return x for base index ``idx``, caching on first access.

        Note: This cache is per-process. With multiple DataLoader workers,
        each worker maintains its own cache.
        """
        if self._lazy_cache_enabled:
            cached = self._item_cache.get(idx)
            if cached is not None:
                # LRU: mark as recently used
                self._item_cache.move_to_end(idx, last=True)
                self._cache_hits += 1
                return cached

        # Cache miss or caching disabled: load from underlying dataset
        x = self.original_dataset[idx][0]
        if self._lazy_cache_enabled:
            self._cache_misses += 1
            self._cache_put(idx, x)
        return x

    def _cache_put(self, idx: int, tensor: torch.Tensor) -> None:
        # If present, just refresh order
        if idx in self._item_cache:
            self._item_cache.move_to_end(idx, last=True)
            return
        size_bytes = tensor.element_size() * tensor.numel()
        # Evict LRU until within budget
        if self._cache_max_bytes is not None and self._cache_max_bytes > 0:
            while (
                self._cache_cur_bytes + size_bytes > self._cache_max_bytes
                and len(self._item_cache) > 0
            ):
                old_idx, old_tensor = self._item_cache.popitem(last=False)
                self._cache_cur_bytes -= old_tensor.element_size() * old_tensor.numel()
        self._item_cache[idx] = tensor
        self._cache_cur_bytes += size_bytes


class GroupedDatasetAugmented(RandomLabelGroupedDataset):
    def __init__(
        self,
        *args,
        max_val: float = 10.0,
        aug_cfg: dict = None,
        quantize: bool = False,
        training: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.max_val = max_val
        self.quantize = quantize
        self.training = training
        self.augmentations = Augmentations(aug_cfg)

    def mulaw_quantize(self, x: torch.Tensor, mu: int = 255) -> torch.Tensor:
        """
        Args:
            data: Dictionary containing raw data and metadata
            n_bits: Number of bits to use for quantization

        Returns:
            Dictionary containing quantized data
        """
        x = x / self.max_val
        x = mulaw_torch(x, mu)

        return x

    def __getitem__(self, idx: int):
        data, label = super().__getitem__(idx)
        data = self.augmentations(data, training=self.training)

        if self.quantize:
            data = self.mulaw_quantize(data)
        return data, label
