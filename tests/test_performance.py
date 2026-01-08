"""Performance benchmarks for identified hotspots.

This module benchmarks:
1. ChunkDataset._load_data - data loading from disk
2. mulaw_torch - mu-law quantization

Run with:
    pytest tests/test_performance.py -v --benchmark-only
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from ephys_gpt.utils.quantizers import mulaw_torch


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def mock_dataset_dir():
    """Create a temporary directory with mock MEG data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        session_dir = root / "session_001"
        session_dir.mkdir()

        # Create multiple chunks with realistic dimensions
        n_channels = 274  # Typical MEG channel count
        chunk_length = 6000  # ~10 seconds at 600 Hz
        sfreq = 600.0
        n_chunks = 3

        pos_2d = np.random.randn(n_channels, 2).astype(np.float32)
        ch_names = [f"MEG{i:03d}" for i in range(n_channels)]

        for chunk_idx in range(n_chunks):
            data = np.random.randn(n_channels, chunk_length).astype(np.float32)
            chunk_dict = {
                "data": data,
                "ch_names": ch_names,
                "pos_2d": pos_2d,
                "sfreq": sfreq,
            }
            np.save(session_dir / f"{chunk_idx}.npy", chunk_dict)

        yield str(root), ch_names, pos_2d, sfreq


@pytest.fixture
def sample_tensor_small():
    """Small tensor for micro-benchmarks."""
    return torch.randn(32, 500)  # 32 channels, 500 timesteps


@pytest.fixture
def sample_tensor_medium():
    """Medium tensor matching typical batch."""
    return torch.randn(8, 274, 1000)  # batch=8, 274 channels, 1000 timesteps


@pytest.fixture
def sample_tensor_large():
    """Large tensor for stress testing."""
    return torch.randn(16, 274, 2000)  # batch=16, 274 channels, 2000 timesteps


# =============================================================================
# mulaw_torch Benchmarks
# =============================================================================


class TestMulawTorchPerformance:
    """Benchmarks for mu-law quantization."""

    def test_mulaw_torch_small(self, benchmark, sample_tensor_small):
        """Benchmark mulaw_torch with small tensor."""
        x = sample_tensor_small.clone()
        x = x / x.abs().max()  # Normalize to [-1, 1]

        result = benchmark(mulaw_torch, x, 255)
        assert result is not None

    def test_mulaw_torch_medium(self, benchmark, sample_tensor_medium):
        """Benchmark mulaw_torch with medium tensor (typical batch)."""
        x = sample_tensor_medium.clone()
        x = x / x.abs().max()

        result = benchmark(mulaw_torch, x, 255)
        assert result is not None

    def test_mulaw_torch_large(self, benchmark, sample_tensor_large):
        """Benchmark mulaw_torch with large tensor."""
        x = sample_tensor_large.clone()
        x = x / x.abs().max()

        result = benchmark(mulaw_torch, x, 255)
        assert result is not None


# =============================================================================
# ChunkDataset._load_data Benchmarks
# =============================================================================


class TestDataLoadingPerformance:
    """Benchmarks for dataset loading operations."""

    def test_np_load_single_chunk(self, benchmark, mock_dataset_dir):
        """Benchmark raw np.load for a single chunk."""
        root, _, _, _ = mock_dataset_dir
        chunk_path = Path(root) / "session_001" / "0.npy"

        def load_chunk():
            return np.load(chunk_path, allow_pickle=True).item()

        result = benchmark(load_chunk)
        assert "data" in result

    def test_chunk_dataset_getitem(self, benchmark, mock_dataset_dir):
        """Benchmark ChunkDataset.__getitem__ for a single sample."""
        from ephys_gpt.dataset.datasets import ChunkDataset

        root, ch_names, pos_2d, sfreq = mock_dataset_dir

        # Build minimal indices
        indices = [("session_001", 0, 0)]  # session, chunk_idx, start

        dataset = ChunkDataset(
            root_dir=root,
            indices=indices,
            length=500,
            ch_names=ch_names,
            pos_2d=pos_2d,
            sfreq=sfreq,
        )

        result = benchmark(dataset.__getitem__, 0)
        assert result is not None

    def test_chunk_dataset_multiple_getitem(self, benchmark, mock_dataset_dir):
        """Benchmark ChunkDataset iteration (simulating dataloader)."""
        from ephys_gpt.dataset.datasets import ChunkDataset, clear_chunk_cache

        root, ch_names, pos_2d, sfreq = mock_dataset_dir

        # Build indices for multiple windows
        indices = [
            ("session_001", 0, 0),
            ("session_001", 0, 200),
            ("session_001", 0, 400),
            ("session_001", 1, 0),
            ("session_001", 1, 200),
        ]

        dataset = ChunkDataset(
            root_dir=root,
            indices=indices,
            length=500,
            ch_names=ch_names,
            pos_2d=pos_2d,
            sfreq=sfreq,
        )

        # Clear cache before benchmarking to measure cold-start
        clear_chunk_cache()

        def iterate_samples():
            results = []
            for i in range(len(dataset)):
                results.append(dataset[i])
            return results

        result = benchmark(iterate_samples)
        assert len(result) == len(indices)

    def test_chunk_dataset_cache_benefit(self, benchmark, mock_dataset_dir):
        """Benchmark showing cache benefit for repeated access to same chunks."""
        from ephys_gpt.dataset.datasets import (
            ChunkDataset,
            get_chunk_cache_info,
        )

        root, ch_names, pos_2d, sfreq = mock_dataset_dir

        # Multiple windows from SAME chunk file - should benefit from cache
        indices = [
            ("session_001", 0, 0),
            ("session_001", 0, 100),
            ("session_001", 0, 200),
            ("session_001", 0, 300),
            ("session_001", 0, 400),
            ("session_001", 0, 500),
            ("session_001", 0, 600),
            ("session_001", 0, 700),
        ]

        dataset = ChunkDataset(
            root_dir=root,
            indices=indices,
            length=500,
            ch_names=ch_names,
            pos_2d=pos_2d,
            sfreq=sfreq,
        )

        # Warm the cache first
        _ = dataset[0]
        info_before = get_chunk_cache_info()

        def iterate_cached():
            results = []
            for i in range(len(dataset)):
                results.append(dataset[i])
            return results

        result = benchmark(iterate_cached)
        info_after = get_chunk_cache_info()

        assert len(result) == len(indices)
        # After iteration, most accesses should be cache hits
        # (all 8 windows come from the same chunk file)
        assert info_after.hits > info_before.hits


# =============================================================================
# Comparison Benchmarks (for optimization validation)
# =============================================================================


def _mulaw_torch_original(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """Original implementation for comparison (before optimization)."""
    shape = x.shape
    x_flat = x.reshape(-1)

    # clip to -1, 1
    x_flat = torch.clamp(x_flat, -0.999, 0.999)

    # Mu-law compression
    compressed = (
        torch.sign(x_flat)
        * torch.log1p(mu * torch.abs(x_flat))
        / torch.log1p(torch.tensor(float(mu), device=x.device))
    )

    # Quantize to integers in [0, mu]
    compressed = (compressed + 1) * 0.5 * mu + 0.5
    digitized = compressed.long()

    return digitized.reshape(shape)


class TestMulawOptimizationComparison:
    """Compare original vs optimized mulaw implementations."""

    def test_mulaw_original(self, benchmark, sample_tensor_medium):
        """Benchmark original implementation."""
        x = sample_tensor_medium.clone()
        x = x / x.abs().max()

        result = benchmark(_mulaw_torch_original, x, 255)
        assert result is not None

    def test_mulaw_optimized(self, benchmark, sample_tensor_medium):
        """Benchmark optimized implementation."""
        x = sample_tensor_medium.clone()
        x = x / x.abs().max()

        result = benchmark(mulaw_torch, x, 255)
        assert result is not None

    def test_mulaw_equivalence(self, sample_tensor_medium):
        """Verify that optimized implementation produces equivalent results.

        Due to floating-point precision differences in the order of operations,
        the quantized outputs may differ by at most 1 bin. This is acceptable
        for quantization since a 1-bin difference is negligible.
        """
        x = sample_tensor_medium.clone()
        x = x / x.abs().max()

        original = _mulaw_torch_original(x, 255)
        optimized = mulaw_torch(x, 255)

        max_diff = (original - optimized).abs().max().item()
        assert max_diff <= 1, f"Mismatch: max diff = {max_diff} (expected <= 1)"

        # Also verify the distributions are statistically similar
        assert original.float().mean().isclose(optimized.float().mean(), rtol=0.01)
        assert original.float().std().isclose(optimized.float().std(), rtol=0.01)


# =============================================================================
# Quick Profiling (non-benchmark tests for cProfile integration)
# =============================================================================


class TestProfilingHelpers:
    """Helper tests that can be run with cProfile."""

    @pytest.mark.skip(reason="Run manually with: python -m cProfile -s cumtime")
    def test_profile_mulaw_batch(self):
        """Profile mulaw_torch with realistic batch size."""
        x = torch.randn(32, 274, 1000)
        x = x / x.abs().max()

        for _ in range(100):
            _ = mulaw_torch(x, 255)

    @pytest.mark.skip(reason="Run manually with: python -m cProfile -s cumtime")
    def test_profile_dataset_iteration(self, mock_dataset_dir):
        """Profile dataset iteration."""
        from ephys_gpt.dataset.datasets import ChunkDataset

        root, ch_names, pos_2d, sfreq = mock_dataset_dir

        indices = [("session_001", i % 3, (i * 100) % 5000) for i in range(100)]

        dataset = ChunkDataset(
            root_dir=root,
            indices=indices,
            length=500,
            ch_names=ch_names,
            pos_2d=pos_2d,
            sfreq=sfreq,
        )

        for i in range(len(dataset)):
            _ = dataset[i]
