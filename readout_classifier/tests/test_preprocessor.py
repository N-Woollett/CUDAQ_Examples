"""Tests for readout_classifier.src.preprocessor."""

import numpy as np
import pytest

from readout_classifier.src.preprocessor import StandardScaler, standardise, angle_encode


# ─── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def iq_train():
    """Deterministic (N, 2) IQ training data with known statistics."""
    rng = np.random.default_rng(0)
    return rng.normal(loc=[3.0, -1.0], scale=[2.0, 0.5], size=(500, 2))


@pytest.fixture
def iq_test():
    """Small (N, 2) IQ test set drawn from a different seed."""
    rng = np.random.default_rng(99)
    return rng.normal(loc=[3.0, -1.0], scale=[2.0, 0.5], size=(100, 2))


# ─── basic behaviour ────────────────────────────────────────────────────────

class TestStandardiseFitTransform:
    """Training-mode (scaler=None) behaviour."""

    def test_output_shape_matches_input(self, iq_train):
        scaled, _ = standardise(iq_train)
        assert scaled.shape == iq_train.shape

    def test_output_has_zero_mean(self, iq_train):
        scaled, _ = standardise(iq_train)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)

    def test_output_has_unit_std(self, iq_train):
        scaled, _ = standardise(iq_train)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-12)

    def test_scaler_records_mu_and_sigma(self, iq_train):
        _, scaler = standardise(iq_train)
        np.testing.assert_allclose(scaler.mu, iq_train.mean(axis=0))
        np.testing.assert_allclose(scaler.sigma, iq_train.std(axis=0))

    def test_scaler_is_namedtuple(self, iq_train):
        _, scaler = standardise(iq_train)
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mu")
        assert hasattr(scaler, "sigma")


# ─── transform-only (re-use scaler) ─────────────────────────────────────────

class TestStandardiseTransformOnly:
    """Inference-mode — apply a pre-fitted scaler to new data."""

    def test_uses_training_statistics(self, iq_train, iq_test):
        _, scaler = standardise(iq_train)
        scaled_test, returned_scaler = standardise(iq_test, scaler=scaler)

        # The returned scaler must be the same object we passed in.
        assert returned_scaler is scaler

        # Manually verify the transform: (x - mu) / sigma
        expected = (iq_test - scaler.mu) / scaler.sigma
        np.testing.assert_allclose(scaled_test, expected)

    def test_test_set_not_zero_mean(self, iq_train, iq_test):
        """Test data should generally NOT have exactly zero mean — that
        would indicate it was re-fitted (data leakage)."""
        _, scaler = standardise(iq_train)
        scaled_test, _ = standardise(iq_test, scaler=scaler)

        # With different seeds, the test-set mean won't be exactly 0.
        # We just check it's *not* close to zero (within generous tol).
        mean_test = scaled_test.mean(axis=0)
        assert not np.allclose(mean_test, 0.0, atol=0.05)


# ─── no data leakage ────────────────────────────────────────────────────────

class TestNoDataLeakage:
    """Ensure the scaler computed on training data is independent of
    any val/test data passed later."""

    def test_scaler_unchanged_after_transform(self, iq_train, iq_test):
        _, scaler_before = standardise(iq_train)
        mu_copy = scaler_before.mu.copy()
        sigma_copy = scaler_before.sigma.copy()

        standardise(iq_test, scaler=scaler_before)

        np.testing.assert_array_equal(scaler_before.mu, mu_copy)
        np.testing.assert_array_equal(scaler_before.sigma, sigma_copy)


# ─── edge cases & error handling ─────────────────────────────────────────────

class TestEdgeCases:

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            standardise(np.array([1.0, 2.0, 3.0]))

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            standardise(np.zeros((2, 3, 4)))

    def test_rejects_constant_feature(self):
        # Column 1 is constant → std == 0 → should raise.
        data = np.column_stack([
            np.arange(10, dtype=float),
            np.ones(10),
        ])
        with pytest.raises(ValueError, match="zero standard deviation"):
            standardise(data)

    def test_single_feature(self):
        """Works for (N, 1) data (e.g. amplitude-only readout)."""
        data = np.arange(100, dtype=float).reshape(-1, 1)
        scaled, scaler = standardise(data)
        np.testing.assert_allclose(scaled.mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaled.std(), 1.0, atol=1e-12)

    def test_many_features(self):
        """Works for (N, k) with k > 2."""
        rng = np.random.default_rng(7)
        data = rng.normal(size=(200, 5))
        scaled, scaler = standardise(data)
        assert scaled.shape == (200, 5)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-12)


# ─── integration with the IQ simulator ──────────────────────────────────────

class TestIntegrationWithSimulator:
    """Round-trip: generate → standardise → check statistics."""

    def test_standardise_simulated_iq(self):
        from readout_classifier.src.iq_simulator import generate_iq_data
        import json
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent / "config" / "default_params.json"
        )
        with open(config_path) as f:
            params = json.load(f)

        iq_train, _ = generate_iq_data(n_samples=params["n_train"], params=params)
        iq_test, _ = generate_iq_data(n_samples=params["n_test"], params=params)

        scaled_train, scaler = standardise(iq_train)
        scaled_test, _ = standardise(iq_test, scaler=scaler)

        # Training set should be perfectly centred
        np.testing.assert_allclose(scaled_train.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaled_train.std(axis=0), 1.0, atol=1e-12)

        # Test set should be *approximately* centred (same distribution,
        # different seed → up to sampling noise)
        assert scaled_test.shape == iq_test.shape


# ─── angle_encode ────────────────────────────────────────────────────────────

class TestAngleEncodeBasic:
    """Core π·tanh mapping properties."""

    def test_output_shape_matches_input(self, iq_train):
        scaled, _ = standardise(iq_train)
        encoded = angle_encode(scaled)
        assert encoded.shape == scaled.shape

    def test_all_values_within_minus_pi_to_pi(self, iq_train):
        scaled, _ = standardise(iq_train)
        encoded = angle_encode(scaled)
        assert np.all(encoded > -np.pi)
        assert np.all(encoded < np.pi)

    def test_zero_maps_to_zero(self):
        data = np.zeros((5, 2))
        encoded = angle_encode(data)
        np.testing.assert_allclose(encoded, 0.0)

    def test_monotonically_increasing(self):
        """π·tanh is strictly increasing, so order must be preserved."""
        x = np.linspace(-5, 5, 100).reshape(-1, 1)
        encoded = angle_encode(x)
        assert np.all(np.diff(encoded, axis=0) > 0)

    def test_antisymmetric(self):
        """tanh is an odd function, so angle_encode(-x) == -angle_encode(x)."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=(50, 2))
        np.testing.assert_allclose(angle_encode(-x), -angle_encode(x))

    def test_large_values_saturate_near_pi(self):
        large = np.array([[100.0, -100.0]])
        encoded = angle_encode(large)
        np.testing.assert_allclose(encoded[0, 0],  np.pi, atol=1e-10)
        np.testing.assert_allclose(encoded[0, 1], -np.pi, atol=1e-10)

    def test_exact_value_at_one(self):
        """Verify π·tanh(1) ≈ 0.7616·π."""
        data = np.array([[1.0, -1.0]])
        encoded = angle_encode(data)
        expected = np.pi * np.tanh(np.array([[1.0, -1.0]]))
        np.testing.assert_allclose(encoded, expected)


class TestAngleEncodeEdgeCases:

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            angle_encode(np.array([1.0, 2.0]))

    def test_rejects_3d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            angle_encode(np.zeros((2, 3, 4)))

    def test_single_feature(self):
        data = np.array([[0.0], [1.0], [-1.0]])
        encoded = angle_encode(data)
        assert encoded.shape == (3, 1)
        assert np.all(np.abs(encoded) < np.pi)

    def test_many_features(self):
        rng = np.random.default_rng(7)
        data = rng.normal(size=(200, 5))
        encoded = angle_encode(data)
        assert encoded.shape == (200, 5)
        assert np.all(np.abs(encoded) < np.pi)


# ─── full pipeline integration: simulate → standardise → angle_encode ────────

class TestFullPipelineIntegration:
    """End-to-end: generate IQ → standardise → angle_encode."""

    def test_pipeline(self):
        from readout_classifier.src.iq_simulator import generate_iq_data
        import json
        from pathlib import Path

        config_path = (
            Path(__file__).resolve().parent.parent / "config" / "default_params.json"
        )
        with open(config_path) as f:
            params = json.load(f)

        iq_train, _ = generate_iq_data(n_samples=params["n_train"], params=params)
        iq_test, _ = generate_iq_data(n_samples=params["n_test"], params=params)

        # Standardise
        scaled_train, scaler = standardise(iq_train)
        scaled_test, _ = standardise(iq_test, scaler=scaler)

        # Angle encode
        angles_train = angle_encode(scaled_train)
        angles_test = angle_encode(scaled_test)

        # Both sets must be bounded in (-π, π)
        for angles in (angles_train, angles_test):
            assert angles.shape[1] == 2
            assert np.all(angles > -np.pi)
            assert np.all(angles < np.pi)
