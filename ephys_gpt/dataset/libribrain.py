import h5py
import random
import numpy as np
import torch
import yaml
import math
import multiprocessing as mp

from numpy.random import choice
from typing import List, Tuple, Dict, Iterator, Optional
from collections import defaultdict
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from pnpl.datasets import GroupedDataset  # noqa: F401

from ..utils.quantizers import mulaw_torch
from .augmentations import Augmentations


class RandomLabelGroupedDataset(Dataset):
    """Advanced grouping dataset that, at access time, randomly selects a label and then
    randomly samples ``grouped_samples`` examples with that label.

    This avoids precomputing static groups and supports label-balanced sampling even
    when labels are imbalanced.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        grouped_samples_range: Tuple[int, int] = (1, 120),
        grouped_samples_mean: int = 100,
        grouped_samples_std: int = 0,
        average_grouped_samples: bool = True,
    ) -> None:
        self.original_dataset = original_dataset
        self.grouped_samples_range = grouped_samples_range
        self.grouped_samples_mean = int(grouped_samples_mean)
        self.grouped_samples_std = int(grouped_samples_std)
        self.average_grouped_samples = bool(average_grouped_samples)

        # Build mapping label -> list of sample indices for that label
        label_to_indices: Dict[int, List[int]] = {}
        for i, sample in enumerate(original_dataset.samples):
            _, _, _, _, _, label = sample
            phoneme = label.split("_")[0]
            label_val = original_dataset.phoneme_to_id[phoneme]
            label_to_indices.setdefault(label_val, []).append(i)
        self.label_to_indices = label_to_indices
        self.labels = sorted(self.label_to_indices.keys())

        # convert each list of indices to a tensor
        for label, indices in self.label_to_indices.items():
            self.label_to_indices[label] = torch.tensor(indices).long()

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
        sampled_indices = pool[choice_ix]

        xs = [self.original_dataset[i][0] for i in sampled_indices]

        if self.average_grouped_samples:
            x = torch.stack(xs, dim=0).mean(dim=0)
        else:
            x = torch.cat(xs, dim=0)

        y = torch.tensor(chosen_label, dtype=torch.long)
        return x, y, int(k)


class SessionGroupedDataset(Dataset):
    """Grouped dataset that keeps aggregated samples within the same session."""

    def __init__(
        self,
        original_dataset: Dataset,
        grouped_samples: int = 10,
        drop_remaining: bool = False,
        shuffle: bool = False,
        average_grouped_samples: bool = True,
    ) -> None:
        if not drop_remaining and not average_grouped_samples:
            raise ValueError(
                "drop_remaining and average_grouped_samples cannot both be False. "
                "Otherwise the dimension of the output will be inconsistent."
            )

        if not hasattr(original_dataset, "samples"):
            raise AttributeError(
                "SessionGroupedDataset"
                " expects the base dataset to expose a 'samples' attribute."
            )

        self.original_dataset = original_dataset
        self.average_grouped_samples = average_grouped_samples
        self.grouped_samples = grouped_samples

        # Precompute subject/session lookup aligned with dataset indices
        self._index_to_session: List[Tuple[str, str]] = []
        for sample in getattr(original_dataset, "samples", []):
            if len(sample) < 2:
                raise ValueError(
                    "Each sample metadata entry must contain at least subject and"
                    "session information."
                )
            session, task = sample[1], sample[2]
            self._index_to_session.append((task, session))

        if len(self._index_to_session) != len(original_dataset):
            raise ValueError(
                "Mismatch between metadata samples and dataset length in "
                "SessionGroupedDataset."
            )

        self.groups: List[List[int]] = []
        self.partial_groups: Dict[Tuple[int, Tuple[str, str]], List[int]] = {}
        self.label_session_groups: Dict[Tuple[int, Tuple[str, str]], List[int]] = {}

        if shuffle:
            indices = torch.randperm(len(original_dataset))
        else:
            indices = torch.arange(len(original_dataset))

        class_counts = defaultdict(int)
        phoneme_dict = {}

        for i in indices:
            idx = int(i)
            phoneme = original_dataset.samples[idx][-1]
            phoneme = phoneme.split("_")[0]
            label_tensor = original_dataset.phoneme_to_id[phoneme]
            label = (
                int(label_tensor.item())
                if torch.is_tensor(label_tensor)
                else int(label_tensor)
            )
            phoneme_dict[label] = phoneme
            session_key = self._index_to_session[idx]
            key = (label, session_key)
            group = self.partial_groups.get(key, [])
            group.append(idx)
            self.partial_groups[key] = group

            group_full = self.label_session_groups.get(key, [])
            group_full.append(idx)
            self.label_session_groups[key] = group_full
            if len(group) == grouped_samples:
                self.groups.append(group)
                self.partial_groups[key] = []
                class_counts[label] += 1

        if not drop_remaining:
            for key, group in self.partial_groups.items():
                # print(f"{key} has {len(group)} samples")
                if group:
                    self.groups.append(group)
                    class_counts[key[0]] += 1

        # sort dict by keys
        class_counts = dict(sorted(class_counts.items()))
        phoneme_dict = dict(sorted(phoneme_dict.items()))
        print(phoneme_dict)

        class_counts = torch.tensor(list(class_counts.values()))
        self.class_weights = torch.clamp(class_counts, min=1.0).log()

        # print class weights
        for lab, w in zip(phoneme_dict.values(), self.class_weights):
            print(f"{lab} class_weight: {float(w):.2f}")

        num_samples = []
        for key, group in self.label_session_groups.items():
            print(f"{key} has {len(group)} samples")
            num_samples.append(len(group))

        # create probability distribution
        num_samples = np.array(num_samples)
        self.label_session_keys = list(self.label_session_groups.keys())
        self.label_session_probs = num_samples / num_samples.sum()

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int):
        group = self.groups[idx]
        samples = [self.original_dataset[i] for i in group]
        samples_data = [sample[0] for sample in samples]
        if self.average_grouped_samples:
            data = torch.stack(samples_data)
            data = data.mean(dim=0)
        else:
            data = torch.concat(samples_data, dim=0)
        label = samples[0][1]

        return data, (label, self.class_weights), len(samples)


class SessionGroupedDatasetFull(SessionGroupedDataset):
    def __init__(self, *args, balanced: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.balanced = balanced
        print(f"SessionGroupedDatasetFull: balanced={self.balanced}")

    def __getitem__(self, idx: int):
        # choose a random key from self.label_session_groups
        num_keys = len(self.label_session_keys)
        if self.balanced:
            key_idx = random.randint(0, num_keys - 1)
        else:
            key_idx = choice(np.arange(num_keys), p=self.label_session_probs)

        key = self.label_session_keys[key_idx]
        inds = self.label_session_groups[key]
        random.shuffle(inds)

        # choose random inds of length=1..100
        # Make choosing 100 X times more likely (e.g., X=5)
        X = 300
        choices = list(range(1, 100)) + [100] * X
        num_samples = random.choice(choices)
        inds_sample = inds[:num_samples]

        samples = [self.original_dataset[i] for i in inds_sample]
        samples_data = [sample[0] for sample in samples]
        if self.average_grouped_samples:
            data = torch.stack(samples_data)
            data = data.mean(dim=0)
        else:
            data = torch.concat(samples_data, dim=0)
        label = samples[0][1]

        return data, (label, self.class_weights), num_samples


class IndexedLabelGroupedDataset(Dataset):
    """Map-style dataset that precomputes an index->(label, pool-slice) map.

    - Builds the same label->pool mapping as RandomLabelGroupedDataset. - Precomputes a
    list of groups (blocks) by slicing each label's pool into   consecutive chunks. Each
    chunk corresponds to exactly one dataset index.   Importantly, the map stores
    positions within each label's pool (not the   actual sample IDs) so that we can
    reshuffle pools each epoch without   rebuilding the index->slice map when
    ``grouped_samples_std == 0``. - When ``grouped_samples_std > 0`` and
    ``training=True``, the block sizes   are re-sampled at the start of each epoch so
    that the entire pool for each   label is covered by variable-sized consecutive
    chunks. - At each new epoch in training mode, the per-label pools are reshuffled,
    but the index->slice map remains unchanged unless std>0.

    This reduces per-__getitem__ randomness and makes sampling deterministic given the
    epoch and worker seeds while still shuffling data across epochs.
    """

    def __init__(
        self,
        original_dataset: Dataset,
        grouped_samples_range: Tuple[int, int] = (1, 120),
        grouped_samples_mean: int = 100,
        grouped_samples_std: int = 0,
        average_grouped_samples: bool = True,
        training: bool = True,
    ) -> None:
        self.original_dataset = original_dataset
        self.grouped_samples_range = grouped_samples_range
        self.grouped_samples_mean = int(grouped_samples_mean)
        self.grouped_samples_std = int(grouped_samples_std)
        self.average_grouped_samples = bool(average_grouped_samples)
        self.training = bool(training)

        # Build mapping label -> list of sample indices for that label
        label_to_indices: Dict[int, List[int]] = {}
        label_distribution: Dict[int, int] = {}
        phonemes: Dict[int, str] = {}
        for i, sample in enumerate(original_dataset.samples):
            _, _, _, _, _, label = sample
            phoneme = label.split("_")[0]
            label_val = original_dataset.phoneme_to_id[phoneme]
            label_to_indices.setdefault(label_val, []).append(i)

            # build label distribution
            label_distribution.setdefault(label_val, 0)
            label_distribution[label_val] += 1
            phonemes[label_val] = phoneme

        # Convert lists to tensors (we will shuffle these per epoch)
        self.label_to_indices: Dict[int, torch.Tensor] = {
            lab: torch.tensor(ixs, dtype=torch.long)
            for lab, ixs in label_to_indices.items()
        }
        self.labels = sorted(self.label_to_indices.keys())
        phonemes = [phonemes[lab] for lab in self.labels]

        # Build inverse-frequency class weights (normalized to mean=1)
        counts = torch.tensor(
            [label_distribution[lab] for lab in self.labels], dtype=torch.float
        )
        inv = 1.0 / torch.clamp(counts, min=1.0).sqrt()
        inv = inv * (inv.numel() / inv.sum())
        self.class_weights = inv

        # use log counts
        self.class_weights = torch.clamp(counts, min=1.0).log()

        for lab, w in zip(phonemes, self.class_weights):
            print(f"{lab} class_weight: {float(w):.4f}")

        if len(self.labels) == 0:
            raise ValueError(
                "IndexedLabelGroupedDataset: no labels found in original dataset."
            )

        # Shared epoch/seed state so that worker copies stay in sync
        self._shared_epoch = mp.Value("i", 0)
        self._seen_epoch_local = -1  # per-process cache
        self._block_seed_shared = mp.Value("q", random.randint(0, 2**31 - 1))
        self._shuffle_seed_shared = mp.Value("q", random.randint(0, 2**31 - 1))

        # Precompute the idx->(label, start, end) blocks
        # If std == 0, we can build once; otherwise build deterministically with seed
        self._build_idx_blocks(
            sample_random_sizes=(self.grouped_samples_std > 0 and self.training),
            seed=int(self._block_seed_shared.value),
        )

        self.groups_per_epoch = len(self._idx_to_block)

        # Initial shuffle (deterministic across workers)
        self._shuffle_pools(seed=int(self._shuffle_seed_shared.value))

    def on_epoch_start(self, epoch: Optional[int] = None) -> None:
        """Call at the start of each epoch (training loop responsibility).

        - Always reshuffles the per-label pools if ``training`` is True. - If
        ``grouped_samples_std > 0`` and ``training`` is True, rebuilds the   idx->pool-
        slice blocks with freshly sampled group sizes for this epoch.
        """
        if epoch is not None:
            self.set_epoch(epoch)
        # Apply in this process; workers will lazily update on first access
        self._maybe_update_for_epoch(force=True)

    def set_epoch(self, epoch: int) -> None:
        """Set the current epoch in shared state so worker copies can react lazily."""
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    def _draw_k(self) -> int:
        low, high = self.grouped_samples_range
        if self.grouped_samples_std > 0:
            kf = torch.normal(
                mean=torch.tensor(float(self.grouped_samples_mean)),
                std=torch.tensor(float(self.grouped_samples_std)),
            ).item()
        else:
            kf = float(self.grouped_samples_mean)
        k = max(low, min(int(round(kf)), high))
        return max(1, k)

    def _build_idx_blocks(
        self, sample_random_sizes: bool, seed: Optional[int] = None
    ) -> None:
        """Construct ``self._idx_to_block``: list of (label, start, end) slices.

        For each label's pool of length N, partition the range [0, N) into consecutive
        blocks. If ``sample_random_sizes`` is True, draw block sizes from Normal(mean,
        std) (clamped by ``grouped_samples_range``); otherwise use a fixed size of
        ``grouped_samples_mean``. The last block is trimmed to exactly cover the pool.
        """
        blocks: List[Tuple[int, int, int]] = []  # (label, start, end)
        low, high = self.grouped_samples_range
        fixed_k = max(low, min(int(self.grouped_samples_mean), high))
        rng: Optional[random.Random] = None
        if sample_random_sizes:
            rng = random.Random(int(seed) if seed is not None else None)

        # Build per-label consecutive chunks that cover each pool
        per_label_blocks: Dict[int, List[Tuple[int, int, int]]] = {}
        for lab in self.labels:
            pool_len = int(self.label_to_indices[lab].numel())
            lab_blocks: List[Tuple[int, int, int]] = []
            if pool_len > 0:
                start = 0
                while start < pool_len:
                    if sample_random_sizes and rng is not None:
                        kf = rng.normalvariate(
                            float(self.grouped_samples_mean),
                            float(max(1e-6, self.grouped_samples_std)),
                        )
                        k = max(low, min(int(round(kf)), high))
                    else:
                        k = fixed_k
                    end = min(pool_len, start + max(1, k))
                    lab_blocks.append((lab, start, end))
                    start = end
            per_label_blocks[lab] = lab_blocks

        # Interleave per-label blocks round-robin to balance labels
        idx = 0
        while True:
            progressed = False
            for lab in self.labels:
                lab_list = per_label_blocks.get(lab, [])
                if idx < len(lab_list):
                    blocks.append(lab_list[idx])
                    progressed = True
            if not progressed:
                break
            idx += 1

        self._idx_to_block: List[Tuple[int, int, int]] = blocks

    def _shuffle_pools(self, seed: Optional[int] = None) -> None:
        # Deterministic shuffle across workers derived from a shared seed
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        else:
            g.manual_seed(torch.initial_seed())
        for lab in self.labels:
            if len(self.label_to_indices[lab]) == 0:
                continue
            perm = torch.randperm(len(self.label_to_indices[lab]), generator=g)
            self.label_to_indices[lab] = self.label_to_indices[lab][perm]

    def _maybe_update_for_epoch(self, force: bool = False) -> None:
        """Check shared epoch and rebuild/shuffle once per epoch per process."""
        cur = int(self._shared_epoch.value)
        if not force and cur == self._seen_epoch_local:
            return
        if self.training and self.grouped_samples_std > 0:
            block_seed = int(self._block_seed_shared.value) + cur * 1_000_003
            self._build_idx_blocks(sample_random_sizes=True, seed=block_seed)
        if self.training:
            shuffle_seed = int(self._shuffle_seed_shared.value) + cur * 2_000_033
            self._shuffle_pools(seed=shuffle_seed)
        self._seen_epoch_local = cur

    # ---- Dataset protocol ---------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        # Constant length for sampler stability across epochs
        return int(self.groups_per_epoch)

    def __getitem__(self, idx: int):  # type: ignore[override]
        # Lazily apply epoch changes within each worker process
        self._maybe_update_for_epoch()
        if len(self._idx_to_block) == 0:
            raise IndexError("Empty block map")
        lab, s, e = self._idx_to_block[idx % len(self._idx_to_block)]

        pool = self.label_to_indices[lab]
        if e > len(pool):
            # Should not happen, but guard in case of external mutation
            e = len(pool)
        slice_ix = pool[s:e]

        xs = [self.original_dataset[int(i)][0] for i in slice_ix]
        if self.average_grouped_samples:
            x = torch.stack(xs, dim=0).mean(dim=0)
        else:
            x = torch.cat(xs, dim=0)

        y = torch.tensor(int(lab), dtype=torch.long)
        k = int(e - s)
        return x, (y, self.class_weights), k

    # ---- Debugging aids -----------------------------------------------
    def epoch_fingerprint(self) -> str:
        """Stable digest of current block layout and pool heads (for debugging).

        Returns an opaque hex string that should be identical across workers within the
        same epoch when seeds are coordinated correctly.
        """
        import hashlib

        h = hashlib.md5()
        h.update(str(len(self._idx_to_block)).encode())
        for lab in self.labels:
            pool = self.label_to_indices[lab]
            head = pool[: min(16, len(pool))].cpu().numpy().tobytes()
            h.update(head)
        if self._idx_to_block:
            for i in (0, len(self._idx_to_block) // 2, len(self._idx_to_block) - 1):
                lab, s, e = self._idx_to_block[i]
                h.update(f"{lab}:{s}:{e}".encode())
        return h.hexdigest()


class GroupedDatasetAugmented(Dataset):
    def __init__(
        self,
        *args,
        base_class: str = "IndexedLabelGroupedDataset",
        max_val: float = 10.0,
        augmentations: dict = None,
        quantize: bool = False,
        quant_levels: int = 256,
        training: bool = False,
        normalize: bool = False,
        **kwargs,
    ):
        dataset_class = globals()[base_class]
        self.dataset = dataset_class(*args, **kwargs)

        self.max_val = max_val
        self.quantize = quantize
        self.training = training
        self.quant_levels = quant_levels
        self.eps = 1e-8
        # k-conditioning controls
        self.k_conditioning = True
        self.k_method = "sigma"  # 'sigma' -> 1/sqrt(k), 'log' -> log(k)
        # Capture fixed-k if provided to the base dataset (e.g., validation/test)
        self.fixed_k = kwargs.get("grouped_samples", None)
        self.normalize = normalize

        if augmentations is not None:
            with open(augmentations) as f:
                augmentations = yaml.safe_load(f)
        self.augmentations = Augmentations(augmentations)

    def per_example_channel_norm(self, x: torch.Tensor) -> torch.Tensor:
        # Expect x of shape (C, T)
        if x.dim() != 2:
            raise ValueError(
                f"PerExampleChannelNorm1D expects (B,C,T), got {tuple(x.shape)}"
            )

        var = x.var(dim=(-1, -2), unbiased=False, keepdim=True)
        x = x / torch.sqrt(var + self.eps)
        return x

    def mulaw_quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Args: data: Dictionary containing raw data and metadata n_bits: Number of
        bits to use for quantization.

        Returns:     Dictionary containing quantized data
        """
        max_val = 5.0 if self.normalize else self.max_val
        x = x / max_val
        x = mulaw_torch(x, self.quant_levels - 1)

        return x

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        # Base may return (x, y), (x, (y, weights)), (x, y, k) or (x, (y, weights), k)
        if isinstance(sample, tuple) and len(sample) == 2:
            data, label = sample
            k = self.fixed_k
        elif isinstance(sample, tuple) and len(sample) == 3:
            data, label, k = sample
        else:
            raise ValueError("Unexpected sample structure from base dataset")

        if self.normalize:
            data = self.per_example_channel_norm(data)
        data = self.augmentations(data, training=self.training)

        if self.quantize:
            data = self.mulaw_quantize(data)

        # Build conditioning scalar if enabled
        if self.k_conditioning and k is not None:
            if self.k_method == "sigma":
                sk = 1.0 / float(k) ** 0.5
            elif self.k_method == "log":
                sk = math.log(max(1, int(k)))
            else:
                sk = float(k)
            sigma_k = torch.tensor([sk], dtype=data.dtype)
            inputs = (data, sigma_k)
        else:
            inputs = data

        return inputs, label


class FusedGroupedIterable(IterableDataset):
    """Yields one (x, y) per iteration, where x is the aggregation (e.g., mean) of k
    samples of the same label, chosen to be file-local to enable a *single* contiguous
    HDF5 read per group.

    No per-item cache is used.
    """

    def __init__(
        self,
        base_ds,  # your LibriBrainBase (or compatible)
        grouped_samples_range: Tuple[int, int] = (1, 120),
        grouped_samples_mean: int = 100,
        grouped_samples_std: int = 0,
        average_grouped_samples: bool = True,
        prefer_file_local: bool = True,
        max_span_seconds: Optional[float] = 1.0,
        # HDF5 low-level tunables (affect only internal chunk cache, not user caching)
        h5_rdcc_nbytes: int = 512 * 1024**2,  # 512MB raw chunk cache
        h5_rdcc_nslots: int = 1_000_003,
        h5_rdcc_w0: float = 0.75,
        training: bool = False,
    ):
        self.base = base_ds
        self.groups_per_epoch = len(base_ds) // grouped_samples_mean
        self.low, self.high = grouped_samples_range
        self.k_mean = grouped_samples_mean
        self.k_std = grouped_samples_std
        self.avg = bool(average_grouped_samples)
        self.prefer_file_local = prefer_file_local
        self.max_span_seconds = max_span_seconds
        self.h5_rdcc_nbytes = h5_rdcc_nbytes
        self.h5_rdcc_nslots = h5_rdcc_nslots
        self.h5_rdcc_w0 = h5_rdcc_w0

        # Build label/run indexes once (read-only, light)
        # base.samples: (subject, session, task, run, onset, label)
        self.label_to_run_to_indices: Dict[
            int, Dict[Tuple[str, str, str, str], List[int]]
        ] = defaultdict(lambda: defaultdict(list))
        for idx, sample in enumerate(self.base.samples):
            _, _, _, run, _, label = sample
            key = (
                sample[0],
                sample[1],
                sample[2],
                sample[3],
            )  # (subject, session, task, run)

            phoneme = label.split("_")[0]
            label_val = self.base.phoneme_to_id[phoneme]
            self.label_to_run_to_indices[label_val][key].append(idx)

        # Sort indices within a run by onset (start sample) to maximize locality
        def start_ix(i: int) -> int:
            subj, sess, task, run, onset, _ = self.base.samples[i]
            return max(0, int((onset + self.base.tmin) * self.base.sfreq))

        for lab, run_map in self.label_to_run_to_indices.items():
            for run_key, idxs in run_map.items():
                idxs.sort(key=start_ix)

        # precompute seconds->samples
        self.points_per_sample = self.base.points_per_sample
        self.sfreq = self.base.sfreq
        self.max_span_pts = (
            None
            if self.max_span_seconds is None
            else int(self.max_span_seconds * self.sfreq)
        )

        # Worker-local state (set in __iter__)
        self._open: Dict[Tuple[str, str, str, str], h5py.Dataset] = {}

    def _open_h5(self, run_key: Tuple[str, str, str, str]) -> h5py.Dataset:
        if run_key not in self._open:
            subj, sess, task, run = run_key
            h5_path = self.base._ids_to_h5_path(subj, sess, task, run)
            # Open with a *bigger* raw chunk cache (internal HDF5 read buffer).
            f = h5py.File(
                h5_path,
                "r",
                rdcc_nbytes=self.h5_rdcc_nbytes,
                rdcc_nslots=self.h5_rdcc_nslots,
                rdcc_w0=self.h5_rdcc_w0,
            )
            self._open[run_key] = f["data"]
        return self._open[run_key]

    def _draw_k(self, rng: random.Random) -> int:
        if self.k_std > 0:
            k = int(round(rng.normalvariate(self.k_mean, self.k_std)))
        else:
            k = int(self.k_mean)
        return max(self.low, min(k, self.high))

    def _group_from_run(
        self, rng: random.Random, lab: int, run_key: Tuple[str, str, str, str], k: int
    ) -> List[int]:
        """Pick k near-by indices (by onset) from a specific (label, run), to keep group
        windows within a small span."""
        idxs = self.label_to_run_to_indices[lab][run_key]
        if not idxs:
            return []
        if len(idxs) <= k:
            return idxs

        # pick a random start pointer, then take k consecutive (keeps locality)
        start = rng.randrange(0, len(idxs) - k)
        cand = idxs[start: start + k]

        if self.max_span_pts is None:
            return cand

        # enforce span limit (optional)
        def start_ix(i: int) -> int:
            subj, sess, task, run, onset, _ = self.base.samples[i]
            return max(0, int((onset + self.base.tmin) * self.base.sfreq))

        s0 = start_ix(cand[0])
        sk = start_ix(cand[-1])
        if (sk - s0) <= self.max_span_pts:
            return cand

        # fallback: shrink to the largest prefix within span
        out = [cand[0]]
        for i in cand[1:]:
            if start_ix(i) - s0 <= self.max_span_pts:
                out.append(i)
            else:
                break
        return out

    def _fused_read_and_aggregate(self, idxs: List[int]) -> Tuple[torch.Tensor, int]:
        """Do ONE contiguous read per run (here all idxs share run) and aggregate."""
        assert idxs, "empty group"
        # All share run_key because we picked from a single (label, run)
        subj, sess, task, run, _, lab = self.base.samples[idxs[0]]
        run_key = (subj, sess, task, run)
        ds = self._open_h5(run_key)

        # Compute starts
        starts = []
        for i in idxs:
            _, _, _, _, onset, _ = self.base.samples[i]
            starts.append(max(0, int((onset + self.base.tmin) * self.base.sfreq)))

        smin = min(starts)
        smax = max(starts) + self.points_per_sample
        # ONE contiguous read:
        block = ds[:, smin:smax]  # (C, L)

        # Slice views, aggregate
        if self.avg:
            acc = None
            for s in starts:
                w = block[:, (s - smin): (s - smin + self.points_per_sample)]
                acc = w if acc is None else acc + w
            x = acc / len(starts)
        else:
            chunks = []
            for s in starts:
                w = block[:, (s - smin): (s - smin + self.points_per_sample)]
                chunks.append(w)
            x = np.concatenate(chunks, axis=1)  # (C, k*points)

        # Standardize/clipping if base is configured that way
        if getattr(self.base, "standardize", False):
            bm = self.base.broadcasted_means
            bs = self.base.broadcasted_stds
            # handle edge case near end-of-file
            if x.shape[1] < bm.shape[1]:
                bm = bm[:, : x.shape[1]]
                bs = bs[:, : x.shape[1]]
            x = (x - bm) / bs

        if getattr(self.base, "clipping_boundary", None) is not None:
            b = self.base.clipping_boundary
            np.clip(x, -b, b, out=x)

        # torch tensor
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))

        phoneme = lab.split("_")[0]
        label_val = self.base.phoneme_to_id[phoneme]
        y = torch.tensor(label_val, dtype=torch.long)
        return x, y

    def __len__(self):
        return self.groups_per_epoch

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        # Worker-local RNG so groups differ across workers but are deterministic-ish
        wi = get_worker_info()
        seed = (torch.initial_seed() if wi is None else wi.seed) & 0xFFFFFFFF
        rng = random.Random(seed)

        labels = list(self.label_to_run_to_indices.keys())

        for _ in range(self.groups_per_epoch):
            lab = rng.choice(labels)

            # choose a run for this label (weighted by available items)
            run_map = self.label_to_run_to_indices[lab]
            runs, weights = zip(*[(rk, len(ix)) for rk, ix in run_map.items() if ix])
            rk = rng.choices(runs, weights=weights, k=1)[0]

            k = self._draw_k(rng)
            idxs = self._group_from_run(rng, lab, rk, k)
            if not idxs:
                continue  # rare; skip
            x, y = self._fused_read_and_aggregate(idxs)
            yield x, (y, k)
