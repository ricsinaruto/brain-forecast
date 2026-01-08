import numpy as np
import numpy.testing as npt
import pytest

from ephys_gpt.mdl.file_utils import (
    concatenate_session_chunks,
    consolidate_dataset_chunks,
)


def _make_chunk(data: np.ndarray, pos_2d: np.ndarray, ch_names: list[str]) -> dict:
    return {
        "data": data,
        "pos_2d": pos_2d,
        "ch_names": ch_names,
        "sfreq": 100.0,
    }


def test_concatenate_session_chunks_merges_time_arrays(tmp_path):
    session_dir = tmp_path / "session_a"
    session_dir.mkdir()

    ch_names = ["ch0", "ch1"]
    pos_2d = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    data0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    events0 = np.array([0, 1, 0], dtype=np.int64)
    chunk0 = _make_chunk(data0, pos_2d, ch_names)
    chunk0["events"] = events0

    data1 = np.array([[7, 8], [9, 10]], dtype=np.float32)
    events1 = np.array([1, 1], dtype=np.int64)
    chunk1 = _make_chunk(data1, pos_2d, ch_names)
    chunk1["events"] = events1

    np.save(session_dir / "0.npy", chunk0)
    np.save(session_dir / "1.npy", chunk1)

    merged = concatenate_session_chunks(session_dir)

    npt.assert_array_equal(merged["data"], np.concatenate([data0, data1], axis=-1))
    npt.assert_array_equal(
        merged["events"], np.concatenate([events0, events1], axis=-1)
    )
    assert merged["ch_names"] == ch_names
    npt.assert_array_equal(merged["pos_2d"], pos_2d)
    assert merged["sfreq"] == 100.0


def test_concatenate_session_chunks_requires_consecutive_indices(tmp_path):
    session_dir = tmp_path / "session_b"
    session_dir.mkdir()

    ch_names = ["ch0"]
    pos_2d = np.array([[0.0, 0.0]], dtype=np.float32)
    chunk = _make_chunk(np.array([[1, 2]], dtype=np.float32), pos_2d, ch_names)

    np.save(session_dir / "0.npy", chunk)
    np.save(session_dir / "2.npy", chunk)

    with pytest.raises(ValueError):
        concatenate_session_chunks(session_dir)


def test_consolidate_dataset_chunks_writes_merged_session(tmp_path):
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    session_dir = source_root / "session_c"
    session_dir.mkdir(parents=True)

    ch_names = ["ch0", "ch1"]
    pos_2d = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

    data0 = np.array([[1, 2], [3, 4]], dtype=np.float32)
    data1 = np.array([[5], [6]], dtype=np.float32)

    np.save(session_dir / "0.npy", _make_chunk(data0, pos_2d, ch_names))
    np.save(session_dir / "1.npy", _make_chunk(data1, pos_2d, ch_names))

    saved_paths = consolidate_dataset_chunks(source_root, target_root)

    expected_path = target_root / "session_c" / "0.npy"
    assert saved_paths == [expected_path]

    merged = np.load(expected_path, allow_pickle=True).item()
    npt.assert_array_equal(merged["data"], np.concatenate([data0, data1], axis=-1))
    npt.assert_array_equal(merged["pos_2d"], pos_2d)
    assert merged["ch_names"] == ch_names
