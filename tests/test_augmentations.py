"""
Unit tests for data augmentations in ephys_gpt.dataset.augmentations.

Tests verify determinism with seeded RNG and that augmentations preserve
expected invariants (shape, dtype, probability control).
"""

import torch

from ephys_gpt.dataset.augmentations import (
    Augmentations,
    RandomTimeWarp,
    RandomTimeMask,
    RandomTimeShift,
    AdditiveNoise,
    RandomSpatialWarp,
    RandomNeighborChannelSwap,
)


class TestRandomTimeWarp:
    """Tests for RandomTimeWarp augmentation."""

    def test_output_shape_preserved_2d(self):
        """Shape should be preserved for (C, T) inputs."""
        torch.manual_seed(42)
        x = torch.randn(10, 100)
        warp = RandomTimeWarp(num_anchors=4, strength=0.2, p=1.0)
        y = warp(x)
        assert y.shape == x.shape

    def test_output_shape_preserved_3d(self):
        """Shape should be preserved for (H, W, T) inputs."""
        torch.manual_seed(42)
        x = torch.randn(8, 8, 100)
        warp = RandomTimeWarp(num_anchors=4, strength=0.2, p=1.0)
        y = warp(x)
        assert y.shape == x.shape

    def test_probability_zero_returns_identity(self):
        """p=0 should never apply the transform."""
        x = torch.randn(10, 100)
        warp = RandomTimeWarp(p=0.0)
        y = warp(x)
        torch.testing.assert_close(y, x)

    def test_strength_zero_is_near_identity(self):
        """strength=0 should produce near-identity warping."""
        torch.manual_seed(42)
        x = torch.randn(10, 100)
        warp = RandomTimeWarp(strength=0.0, p=1.0)
        y = warp(x)
        # With zero strength, warp should be close to identity
        torch.testing.assert_close(y, x, atol=1e-4, rtol=0.0)

    def test_determinism_with_seed(self):
        """Same seed should produce same output."""
        x = torch.randn(10, 100)
        warp = RandomTimeWarp(p=1.0)

        torch.manual_seed(123)
        y1 = warp(x.clone())

        torch.manual_seed(123)
        y2 = warp(x.clone())

        torch.testing.assert_close(y1, y2)


class TestRandomTimeMask:
    """Tests for RandomTimeMask augmentation."""

    def test_output_shape_preserved(self):
        """Shape should be preserved."""
        x = torch.randn(10, 100)
        mask = RandomTimeMask(num_blocks=2, block_len=10, p=1.0)
        torch.manual_seed(42)
        y = mask(x)
        assert y.shape == x.shape

    def test_some_values_zeroed(self):
        """With p=1, some values should be zeroed."""
        torch.manual_seed(42)
        x = torch.ones(10, 100)
        mask = RandomTimeMask(num_blocks=2, block_len=10, value=0.0, p=1.0)
        y = mask(x)
        # Some values should be zero
        assert (y == 0).any()

    def test_probability_zero_returns_identity(self):
        """p=0 should never apply the transform."""
        x = torch.randn(10, 100)
        mask = RandomTimeMask(p=0.0)
        y = mask(x)
        torch.testing.assert_close(y, x)


class TestRandomTimeShift:
    """Tests for RandomTimeShift augmentation."""

    def test_output_shape_preserved(self):
        """Shape should be preserved."""
        torch.manual_seed(42)
        x = torch.randn(10, 100)
        shift = RandomTimeShift(max_shift=10, p=1.0)
        y = shift(x)
        assert y.shape == x.shape

    def test_wrap_mode_preserves_content(self):
        """With wrap=True, all values should be present (just rotated)."""
        torch.manual_seed(42)
        x = torch.arange(100).float().unsqueeze(0)  # (1, 100)
        shift = RandomTimeShift(max_shift=10, wrap=True, p=1.0)
        y = shift(x)
        # Same values, different order
        assert set(y.flatten().tolist()) == set(x.flatten().tolist())

    def test_probability_zero_returns_identity(self):
        """p=0 should never apply the transform."""
        x = torch.randn(10, 100)
        shift = RandomTimeShift(p=0.0)
        y = shift(x)
        torch.testing.assert_close(y, x)


class TestAdditiveNoise:
    """Tests for AdditiveNoise augmentation."""

    def test_output_shape_preserved(self):
        """Shape should be preserved."""
        torch.manual_seed(42)
        x = torch.randn(10, 100)
        noise = AdditiveNoise(sigma=0.1, p=1.0)
        y = noise(x)
        assert y.shape == x.shape

    def test_noise_added_changes_values(self):
        """With p=1 and sigma>0, output should differ from input."""
        torch.manual_seed(42)
        x = torch.randn(10, 100)
        noise = AdditiveNoise(sigma=0.1, p=1.0)
        y = noise(x)
        assert not torch.allclose(y, x)

    def test_sigma_zero_returns_identity(self):
        """sigma=0 should not change values."""
        x = torch.randn(10, 100)
        noise = AdditiveNoise(sigma=0.0, p=1.0)
        y = noise(x)
        torch.testing.assert_close(y, x)

    def test_probability_zero_returns_identity(self):
        """p=0 should never apply the transform."""
        x = torch.randn(10, 100)
        noise = AdditiveNoise(p=0.0)
        y = noise(x)
        torch.testing.assert_close(y, x)


class TestRandomSpatialWarp:
    """Tests for RandomSpatialWarp augmentation."""

    def test_only_applies_to_3d(self):
        """Should only apply to (H, W, T) inputs."""
        warp = RandomSpatialWarp(p=1.0)

        # 2D input should be unchanged
        x_2d = torch.randn(10, 100)
        y_2d = warp(x_2d)
        torch.testing.assert_close(y_2d, x_2d)

        # 3D input should be transformed
        torch.manual_seed(42)
        x_3d = torch.randn(8, 8, 50)
        y_3d = warp(x_3d)
        assert y_3d.shape == x_3d.shape

    def test_probability_zero_returns_identity(self):
        """p=0 should never apply the transform."""
        x = torch.randn(8, 8, 50)
        warp = RandomSpatialWarp(p=0.0)
        y = warp(x)
        torch.testing.assert_close(y, x)


class TestRandomNeighborChannelSwap:
    """Tests for RandomNeighborChannelSwap augmentation."""

    def test_only_applies_to_3d(self):
        """Should only apply to (H, W, T) inputs."""
        swap = RandomNeighborChannelSwap(num_swaps=5, p=1.0)

        # 2D input should be unchanged
        x_2d = torch.randn(10, 100)
        y_2d = swap(x_2d)
        torch.testing.assert_close(y_2d, x_2d)

    def test_output_shape_preserved(self):
        """Shape should be preserved."""
        torch.manual_seed(42)
        x = torch.randn(8, 8, 50)
        swap = RandomNeighborChannelSwap(num_swaps=5, p=1.0)
        y = swap(x)
        assert y.shape == x.shape

    def test_probability_zero_returns_identity(self):
        """p=0 should never apply the transform."""
        x = torch.randn(8, 8, 50)
        swap = RandomNeighborChannelSwap(p=0.0)
        y = swap(x)
        torch.testing.assert_close(y, x)


class TestAugmentationsContainer:
    """Tests for the Augmentations wrapper class."""

    def test_none_config_does_nothing(self):
        """None config should result in identity transform."""
        aug = Augmentations(None)
        x = torch.randn(10, 100)
        y = aug(x, training=True)
        torch.testing.assert_close(y, x)

    def test_augmentations_only_applied_during_training(self):
        """Augmentations should not be applied when training=False."""
        cfg = {"AdditiveNoise": {"sigma": 1.0, "p": 1.0}}
        aug = Augmentations(cfg)
        x = torch.randn(10, 100)

        # Not training -> no change
        y = aug(x, training=False)
        torch.testing.assert_close(y, x)

    def test_augmentations_applied_during_training(self):
        """Augmentations should be applied when training=True."""
        cfg = {"AdditiveNoise": {"sigma": 1.0, "p": 1.0}}
        aug = Augmentations(cfg)
        torch.manual_seed(42)
        x = torch.randn(10, 100)

        # Training -> changes expected
        y = aug(x, training=True)
        assert not torch.allclose(y, x)

    def test_multiple_augmentations_chained(self):
        """Multiple augmentations should be applied in sequence."""
        cfg = {
            "RandomTimeShift": {"max_shift": 5, "p": 1.0},
            "AdditiveNoise": {"sigma": 0.1, "p": 1.0},
        }
        aug = Augmentations(cfg)
        torch.manual_seed(42)
        x = torch.randn(10, 100)
        y = aug(x, training=True)
        # Output should differ from input
        assert not torch.allclose(y, x)
        assert y.shape == x.shape
