import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from pnpl.datasets import LibriBrainPhoneme


from .datasets import (
    ChunkDataset,
    ChunkDatasetReconstruction,
    ChunkDatasetForecastCont,
    ChunkDatasetImage,
    ChunkDatasetImageQuantized,
    GroupedDatasetAugmented,
    RandomLabelGroupedDataset,
)

DATASET_CLASSES = {
    "ChunkDataset": ChunkDataset,
    "ChunkDatasetReconstruction": ChunkDatasetReconstruction,
    "ChunkDatasetForecastCont": ChunkDatasetForecastCont,
    "ChunkDatasetImage": ChunkDatasetImage,
    "ChunkDatasetImageQuantized": ChunkDatasetImageQuantized,
    "GroupedDatasetAugmented": GroupedDatasetAugmented,
    "RandomLabelGroupedDataset": RandomLabelGroupedDataset,
}


@dataclass
class Split:
    train: ChunkDataset
    val: ChunkDataset
    test: ChunkDataset


def build_indices(
    session_dir: str, example_len: int, overlap: int
) -> List[Tuple[str, int, int]]:
    """
    Build indices for the dataset.

    Args:
        session_dir: Directory containing the sessions
        example_len: Length of the example
        overlap: Overlap between examples

    Returns:
        List of tuples containing the session, chunk index,
        and start index of the example
        example_len_samples: Length of the example in samples
        overlap_samples: Overlap between examples in samples
    """
    indices: List[Tuple[str, int, int]] = []
    sessions = [
        d
        for d in os.listdir(session_dir)
        if os.path.isdir(os.path.join(session_dir, d))
    ]

    ch_names = []
    pos_2d = []
    for session in sessions:
        chunk_files = [
            f
            for f in os.listdir(os.path.join(session_dir, session))
            if f.endswith(".npy")
        ]
        for f in chunk_files:
            chunk_idx = int(os.path.splitext(f)[0])
            data_dict = np.load(
                os.path.join(session_dir, session, f), allow_pickle=True
            ).item()
            arr = data_dict["data"]
            length = arr.shape[1]
            sfreq = data_dict["sfreq"]

            if len(data_dict["ch_names"]) > len(ch_names):
                ch_names = data_dict["ch_names"]
            if len(data_dict["pos_2d"]) > len(pos_2d):
                pos_2d = data_dict["pos_2d"]

            example_len_samples = int(example_len * sfreq)
            overlap_samples = int(overlap * sfreq)

            for start in range(
                0,
                length - example_len_samples + 1,
                example_len_samples - overlap_samples,
            ):
                indices.append((session, chunk_idx, start))
    return indices, example_len_samples, overlap_samples, ch_names, pos_2d, sfreq


def split_datasets(
    dataset_root: str,
    example_seconds: int,
    overlap_seconds: int,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    dataset_class: str = "ChunkDataset",
) -> Split:
    """
    Args:
        root: Root directory of the dataset
        example_len: Length of the example
        overlap: Overlap between examples
        val_ratio: Ratio of the validation set
        test_ratio: Ratio of the test set
        seed: Random seed

    Returns:
        Split object containing train, validation, and test datasets
    """
    dataset_class = DATASET_CLASSES[dataset_class]
    sessions = []
    for d in os.listdir(dataset_root):
        if os.path.isdir(os.path.join(dataset_root, d)) and d.startswith("sub-"):
            sessions.append(d)

    rng = random.Random(seed)
    rng.shuffle(sessions)

    n_train = int(len(sessions) * (1 - val_ratio - test_ratio))
    n_val = int(len(sessions) * val_ratio)
    train_sessions = sessions[:n_train]
    val_sessions = sessions[n_train : n_train + n_val]
    test_sessions = sessions[n_train + n_val :]
    print(f"Train sessions: {len(train_sessions)}")
    print(f"Val sessions: {len(val_sessions)}")
    print(f"Test sessions: {len(test_sessions)}")

    all_indices, example_len_samples, overlap_samples, ch_names, pos_2d, sfreq = (
        build_indices(dataset_root, example_seconds, overlap_seconds)
    )

    def collect(sess_list: List[str]) -> List[Tuple[str, int, int]]:
        return [i for i in all_indices if i[0] in sess_list]

    train_idx = collect(train_sessions)
    val_idx = collect(val_sessions)
    test_idx = collect(test_sessions)

    train_ds = dataset_class(
        dataset_root, train_idx, example_len_samples, ch_names, pos_2d, sfreq
    )
    val_ds = dataset_class(
        dataset_root, val_idx, example_len_samples, ch_names, pos_2d, sfreq
    )
    test_ds = dataset_class(
        dataset_root, test_idx, example_len_samples, ch_names, pos_2d, sfreq
    )
    return Split(train_ds, val_ds, test_ds)


@dataclass
class SplitLibriBrain:
    train: RandomLabelGroupedDataset
    val: RandomLabelGroupedDataset
    test: RandomLabelGroupedDataset


def split_datasets_libribrain(
    dataset_root: str,
    tmin: float = 0.0,
    tmax: float = 0.5,
    grouped_samples_std: int = 0,
    dataset_class: str = "RandomLabelGroupedDataset",
    lazy_cache: bool = False,
    cache_max_gb: Optional[float] = None,
):
    train_dataset = LibriBrainPhoneme(
        data_path=f"{dataset_root}/data/",
        partition="train",
        # include_run_keys=[("0", str(i), "Sherlock1", "1") for i in range(1, 10)],
        tmin=tmin,
        tmax=tmax,
    )

    val_dataset = LibriBrainPhoneme(
        data_path=f"{dataset_root}/data/",
        partition="validation",
        tmin=tmin,
        tmax=tmax,
    )

    test_dataset = LibriBrainPhoneme(
        data_path=f"{dataset_root}/data/",
        partition="test",
        tmin=tmin,
        tmax=tmax,
    )

    # compute maximum absolute value of training data
    # max_val = 10.0
    # for example, _ in train_dataset:
    #    max_val = max(max_val, torch.max(torch.abs(example)))

    common_kwargs = {"lazy_cache": lazy_cache, "cache_max_gb": cache_max_gb}
    kwargs_train = {
        "grouped_samples_std": grouped_samples_std,
        "training": True,
        **common_kwargs,
    }
    kwargs_val = {"grouped_samples_std": 0, **common_kwargs}
    kwargs_test = {"grouped_samples_std": 0, **common_kwargs}

    dataset_class = DATASET_CLASSES[dataset_class]

    averaged_train = dataset_class(train_dataset, **kwargs_train)
    averaged_val = dataset_class(val_dataset, **kwargs_val)
    averaged_test = dataset_class(test_dataset, **kwargs_test)

    return Split(averaged_train, averaged_val, averaged_test)
