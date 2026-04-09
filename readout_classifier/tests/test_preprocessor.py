"""Tests for readout_classifier.src.preprocessor."""

import numpy as np
import pytest

from sklearn.decomposition import PCA

from readout_classifier.src.preprocessor import (
    StandardScaler, PipelineParams, PipelineResult,
    standardise, angle_encode, pca_rotate, split_dataset, preprocess_pipeline,
    save_pipeline, load_pipeline,
)


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
        """Standardising should not change the array dimensions.

        Expected: output shape equals input shape (N, 2).
        """
        scaled, _ = standardise(iq_train)
        assert scaled.shape == iq_train.shape

    def test_output_has_zero_mean(self, iq_train):
        """Z-score standardisation should centre each feature to zero.

        Expected: per-feature mean of the scaled output is 0.0.
        """
        scaled, _ = standardise(iq_train)
        np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-12)

    def test_output_has_unit_std(self, iq_train):
        """Z-score standardisation should scale each feature to unit variance.

        Expected: per-feature standard deviation of the scaled output is 1.0.
        """
        scaled, _ = standardise(iq_train)
        np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-12)

    def test_scaler_records_mu_and_sigma(self, iq_train):
        """The returned scaler should store the training set statistics.

        Expected: scaler.mu equals np.mean(iq_train, axis=0) and
        scaler.sigma equals np.std(iq_train, axis=0).
        """
        _, scaler = standardise(iq_train)
        np.testing.assert_allclose(scaler.mu, iq_train.mean(axis=0))
        np.testing.assert_allclose(scaler.sigma, iq_train.std(axis=0))

    def test_scaler_is_namedtuple(self, iq_train):
        """The scaler should be a StandardScaler NamedTuple.

        Expected: scaler is an instance of StandardScaler with 'mu' and
        'sigma' attributes.
        """
        _, scaler = standardise(iq_train)
        assert isinstance(scaler, StandardScaler)
        assert hasattr(scaler, "mu")
        assert hasattr(scaler, "sigma")


# ─── transform-only (re-use scaler) ─────────────────────────────────────────

class TestStandardiseTransformOnly:
    """Inference-mode — apply a pre-fitted scaler to new data."""

    def test_uses_training_statistics(self, iq_train, iq_test):
        """Passing a pre-fitted scaler should apply the training-set
        statistics rather than recomputing them from the new data.

        Expected: the returned scaler is the same object passed in, and
        the output equals (iq_test - mu) / sigma computed from the
        training set.
        """
        _, scaler = standardise(iq_train)
        scaled_test, returned_scaler = standardise(iq_test, scaler=scaler)

        # The returned scaler must be the same object we passed in.
        assert returned_scaler is scaler

        # Manually verify the transform: (x - mu) / sigma
        expected = (iq_test - scaler.mu) / scaler.sigma
        np.testing.assert_allclose(scaled_test, expected)

    def test_test_set_not_zero_mean(self, iq_train, iq_test):
        """Applying the training scaler to a test set drawn from a
        different seed should not produce exactly zero mean.

        Expected: per-feature mean of the scaled test set is not close
        to 0.0 (atol=0.05), confirming that the statistics were not
        recomputed (which would indicate data leakage).
        """
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
        """Transforming new data with an existing scaler must not mutate
        the scaler's stored statistics.

        Expected: scaler.mu and scaler.sigma are identical before and
        after transforming the test set.
        """
        _, scaler_before = standardise(iq_train)
        mu_copy = scaler_before.mu.copy()
        sigma_copy = scaler_before.sigma.copy()

        standardise(iq_test, scaler=scaler_before)

        np.testing.assert_array_equal(scaler_before.mu, mu_copy)
        np.testing.assert_array_equal(scaler_before.sigma, sigma_copy)


# ─── edge cases & error handling ─────────────────────────────────────────────

class TestEdgeCases:

    def test_rejects_1d_input(self):
        """Standardise requires a 2-D array; a 1-D vector is invalid.

        Expected: raises ValueError mentioning '2-D'.
        """
        with pytest.raises(ValueError, match="2-D"):
            standardise(np.array([1.0, 2.0, 3.0]))

    def test_rejects_3d_input(self):
        """Standardise requires a 2-D array; a 3-D tensor is invalid.

        Expected: raises ValueError mentioning '2-D'.
        """
        with pytest.raises(ValueError, match="2-D"):
            standardise(np.zeros((2, 3, 4)))

    def test_rejects_constant_feature(self):
        """A feature with zero variance would cause division by zero.

        Expected: raises ValueError mentioning 'zero standard deviation'
        when one column is constant.
        """
        data = np.column_stack([
            np.arange(10, dtype=float),
            np.ones(10),
        ])
        with pytest.raises(ValueError, match="zero standard deviation"):
            standardise(data)

    def test_rejects_provided_scaler_with_zero_sigma(self):
        """A provided scaler with zero sigma should raise ValueError.

        Expected: raises ValueError mentioning 'zero sigma'.
        """
        bad_scaler = StandardScaler(
            mu=np.array([0.0, 0.0]),
            sigma=np.array([1.0, 0.0]),
        )
        data = np.ones((10, 2))
        with pytest.raises(ValueError, match="zero sigma"):
            standardise(data, scaler=bad_scaler)

    def test_single_feature(self):
        """Standardise should work for single-feature (N, 1) data.

        Expected: output has zero mean and unit standard deviation.
        """
        data = np.arange(100, dtype=float).reshape(-1, 1)
        scaled, scaler = standardise(data)
        np.testing.assert_allclose(scaled.mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(scaled.std(), 1.0, atol=1e-12)

    def test_many_features(self):
        """Standardise should generalise to (N, k) data with k > 2.

        Expected: output shape is (200, 5) with per-feature zero mean
        and unit standard deviation.
        """
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
        """Generate IQ data with the simulator, standardise, and verify
        that the training set is perfectly centred and the test set
        retains the correct shape when transformed with the training
        scaler.

        Expected: training set has per-feature mean 0.0 and std 1.0;
        test set shape is unchanged.
        """
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
        """Angle encoding is element-wise and should not alter dimensions.

        Expected: output shape equals input shape (N, 2).
        """
        scaled, _ = standardise(iq_train)
        encoded = angle_encode(scaled)
        assert encoded.shape == scaled.shape

    def test_all_values_within_minus_pi_to_pi(self, iq_train):
        """π·tanh maps all real values into the open interval (-π, π).

        Expected: every element of the encoded output is strictly
        between -π and π.
        """
        scaled, _ = standardise(iq_train)
        encoded = angle_encode(scaled)
        assert np.all(encoded > -np.pi)
        assert np.all(encoded < np.pi)

    def test_zero_maps_to_zero(self):
        """tanh(0) = 0, so zero-valued inputs should encode to exactly 0.

        Expected: all output elements are 0.0.
        """
        data = np.zeros((5, 2))
        encoded = angle_encode(data)
        np.testing.assert_allclose(encoded, 0.0)

    def test_monotonically_increasing(self):
        """π·tanh is a strictly increasing function, so the relative
        ordering of input values must be preserved after encoding.

        Expected: consecutive differences along a sorted column are
        all positive.
        """
        x = np.linspace(-5, 5, 100).reshape(-1, 1)
        encoded = angle_encode(x)
        assert np.all(np.diff(encoded, axis=0) > 0)

    def test_antisymmetric(self):
        """tanh is an odd function, so negating the input should negate
        the output.

        Expected: angle_encode(-x) equals -angle_encode(x) element-wise.
        """
        rng = np.random.default_rng(42)
        x = rng.normal(size=(50, 2))
        np.testing.assert_allclose(angle_encode(-x), -angle_encode(x))

    def test_large_values_saturate_near_pi(self):
        """For very large inputs tanh saturates at ±1, so the encoded
        values should approach ±π.

        Expected: encoding ±100 yields values within 1e-10 of ±π.
        """
        large = np.array([[100.0, -100.0]])
        encoded = angle_encode(large)
        np.testing.assert_allclose(encoded[0, 0],  np.pi, atol=1e-10)
        np.testing.assert_allclose(encoded[0, 1], -np.pi, atol=1e-10)

    def test_exact_value_at_one(self):
        """Verify the encoding at ±1 against the analytic formula.

        Expected: angle_encode([[1, -1]]) equals π·tanh([[1, -1]])
        (≈ [+0.7616π, −0.7616π]).
        """
        data = np.array([[1.0, -1.0]])
        encoded = angle_encode(data)
        expected = np.pi * np.tanh(np.array([[1.0, -1.0]]))
        np.testing.assert_allclose(encoded, expected)


class TestAngleEncodeEdgeCases:

    def test_rejects_1d_input(self):
        """angle_encode requires a 2-D array; a 1-D vector is invalid.

        Expected: raises ValueError mentioning '2-D'.
        """
        with pytest.raises(ValueError, match="2-D"):
            angle_encode(np.array([1.0, 2.0]))

    def test_rejects_3d_input(self):
        """angle_encode requires a 2-D array; a 3-D tensor is invalid.

        Expected: raises ValueError mentioning '2-D'.
        """
        with pytest.raises(ValueError, match="2-D"):
            angle_encode(np.zeros((2, 3, 4)))

    def test_single_feature(self):
        """angle_encode should handle (N, 1) single-feature data.

        Expected: output shape is (3, 1) and all values lie in (-π, π).
        """
        data = np.array([[0.0], [1.0], [-1.0]])
        encoded = angle_encode(data)
        assert encoded.shape == (3, 1)
        assert np.all(np.abs(encoded) < np.pi)

    def test_many_features(self):
        """angle_encode should generalise to (N, k) data with k > 2.

        Expected: output shape is (200, 5) and all values lie in (-π, π).
        """
        rng = np.random.default_rng(7)
        data = rng.normal(size=(200, 5))
        encoded = angle_encode(data)
        assert encoded.shape == (200, 5)
        assert np.all(np.abs(encoded) < np.pi)


# ─── full pipeline integration: simulate → standardise → angle_encode ────────

class TestFullPipelineIntegration:
    """End-to-end: generate IQ → standardise → angle_encode."""

    def test_pipeline(self):
        """Run the full preprocessing pipeline: generate synthetic IQ
        data, standardise using training statistics, then angle-encode.

        Expected: both training and test angle arrays have 2 features
        and every value is strictly within (-π, π).
        """
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


# ─── pca_rotate ──────────────────────────────────────────────────────────────

class TestPCARotate:
    """Tests for the pca_rotate function."""

    def test_fit_transform(self, iq_train):
        """Test that passing pca=None fits a new PCA and transforms the data.

        Expected: returns a fitted PCA object and the transformed data
        with the same shape as the input. The variance of the first
        component should be greater than or equal to the second.
        """
        rotated, pca = pca_rotate(iq_train)
        
        assert isinstance(pca, PCA)
        assert rotated.shape == iq_train.shape
        # The components should be orthogonal and variances sorted (PCA properties)
        variances = np.var(rotated, axis=0)
        assert variances[0] >= variances[1]

    def test_transform_only(self, iq_train, iq_test):
        """Test that passing an existing PCA transforms without refitting.

        Expected: the returned PCA object is the exact same instance passed
        in, and the output matches a direct transform() call on that PCA.
        """
        _, pca_train = pca_rotate(iq_train)
        
        rotated_test, pca_returned = pca_rotate(iq_test, pca=pca_train)
        
        assert pca_returned is pca_train
        assert rotated_test.shape == iq_test.shape
        
        # Manually transform to ensure no refitting happened
        expected_rotated = pca_train.transform(iq_test)
        np.testing.assert_allclose(rotated_test, expected_rotated)

    def test_rejects_n_components_mismatch(self, iq_train, iq_test):
        """Passing a pre-fitted PCA with a different n_components should
        raise ValueError.

        Expected: raises ValueError mentioning 'components'.
        """
        _, pca_fitted = pca_rotate(iq_train, n_components=2)
        with pytest.raises(ValueError, match="components"):
            pca_rotate(iq_test, pca=pca_fitted, n_components=1)

    def test_rejects_1d_input(self):
        """pca_rotate requires a 2-D array; a 1-D vector is invalid.

        Expected: raises ValueError mentioning '2-D'.
        """
        with pytest.raises(ValueError, match="2-D"):
            pca_rotate(np.array([1.0, 2.0, 3.0]))


# ─── split_dataset ───────────────────────────────────────────────────────────

class TestSplitDataset:
    """Tests for the split_dataset function."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset of 1000 samples with 30% class 1 and 70% class 0."""
        rng = np.random.default_rng(42)
        n_samples = 1000
        iq = rng.normal(size=(n_samples, 2))
        labels = np.zeros(n_samples, dtype=int)
        labels[:300] = 1  # 30% are 1s
        rng.shuffle(labels)
        return iq, labels

    def test_split_ratios(self, mock_dataset):
        """Test dataset is split according to the given ratios.

        Expected: shapes of returned sets match ratios 0.7, 0.15, 0.15.
        """
        iq, labels = mock_dataset
        train, val, test = split_dataset(iq, labels, ratios=(0.7, 0.15, 0.15))
        
        train_iq, train_labels = train
        val_iq, val_labels = val
        test_iq, test_labels = test

        assert len(train_iq) == 700
        assert len(val_iq) == 150
        assert len(test_iq) == 150

        assert len(train_labels) == 700
        assert len(val_labels) == 150
        assert len(test_labels) == 150

    def test_stratification(self, mock_dataset):
        """Test the class ratio is preserved across all splits.

        Expected: ratio of class 1 is ~0.3 in all splits.
        """
        iq, labels = mock_dataset
        train, val, test = split_dataset(iq, labels, ratios=(0.8, 0.1, 0.1))
        
        for split_iq, split_labels in (train, val, test):
            class_1_ratio = np.sum(split_labels == 1) / len(split_labels)
            assert np.isclose(class_1_ratio, 0.3)

    def test_rejects_invalid_ratios(self, mock_dataset):
        """Test it raises an error when ratios don't sum to 1.0.

        Expected: raises ValueError.
        """
        iq, labels = mock_dataset
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_dataset(iq, labels, ratios=(0.5, 0.2, 0.2))

    def test_rejects_zero_ratio(self, mock_dataset):
        """Test it raises an error when any ratio is zero.

        Expected: raises ValueError mentioning 'positive'.
        """
        iq, labels = mock_dataset
        with pytest.raises(ValueError, match="positive"):
            split_dataset(iq, labels, ratios=(1.0, 0.0, 0.0))

    def test_rejects_negative_ratio(self, mock_dataset):
        """Test it raises an error when any ratio is negative.

        Expected: raises ValueError mentioning 'positive'.
        """
        iq, labels = mock_dataset
        with pytest.raises(ValueError, match="positive"):
            split_dataset(iq, labels, ratios=(0.8, 0.3, -0.1))


# ─── preprocess_pipeline ─────────────────────────────────────────────────────

class TestPreprocessPipeline:
    """Tests for the preprocess_pipeline function."""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock dataset."""
        rng = np.random.default_rng(42)
        n_samples = 1000
        iq = rng.normal(size=(n_samples, 2))
        labels = np.zeros(n_samples, dtype=int)
        labels[:300] = 1
        rng.shuffle(labels)
        return iq, labels

    def test_pipeline_without_pca(self, mock_dataset):
        """Test pipeline when PCA is not requested."""
        iq, labels = mock_dataset
        params = {"split_ratios": (0.7, 0.15, 0.15), "seed": 42, "use_pca": False}
        
        result = preprocess_pipeline(iq, labels, params)

        assert isinstance(result, PipelineResult)
        assert result.pca is None
        assert isinstance(result.scaler, StandardScaler)

        train_iq, train_labels = result.train
        val_iq, val_labels = result.val
        test_iq, test_labels = result.test
        
        assert len(train_iq) == 700
        assert len(val_iq) == 150
        assert len(test_iq) == 150
        
        # Check angle encoding bounds
        for subset_iq in (train_iq, val_iq, test_iq):
            assert np.all(subset_iq > -np.pi)
            assert np.all(subset_iq < np.pi)

    def test_pipeline_with_pca(self, mock_dataset):
        """Test pipeline when PCA is requested."""
        iq, labels = mock_dataset
        params = {"split_ratios": (0.7, 0.15, 0.15), "seed": 42, "use_pca": True, "pca_components": 2}

        result = preprocess_pipeline(iq, labels, params)

        assert isinstance(result.pca, PCA)

        train_iq, train_labels = result.train
        # Check angle encoding bounds
        assert np.all(train_iq > -np.pi)
        assert np.all(train_iq < np.pi)

    def test_returns_pipeline_result(self, mock_dataset):
        """Test that the return value is a PipelineResult named tuple.

        Expected: result is a PipelineResult with named field access.
        """
        iq, labels = mock_dataset
        result = preprocess_pipeline(iq, labels, PipelineParams())

        assert isinstance(result, PipelineResult)
        assert isinstance(result.scaler, StandardScaler)

    def test_accepts_pipeline_params(self, mock_dataset):
        """Test that PipelineParams dataclass is accepted directly.

        Expected: same result as passing an equivalent dict.
        """
        iq, labels = mock_dataset
        params = PipelineParams(
            split_ratios=(0.7, 0.15, 0.15), seed=42, use_pca=False
        )
        result = preprocess_pipeline(iq, labels, params)
        assert len(result.train[0]) == 700

    def test_rejects_dict_with_typo(self, mock_dataset):
        """A dict with an unrecognised key should raise TypeError.

        Expected: raises TypeError (from PipelineParams.__init__).
        """
        iq, labels = mock_dataset
        with pytest.raises(TypeError):
            preprocess_pipeline(iq, labels, {"split_ratio": (0.7, 0.15, 0.15)})


# ─── save / load pipeline ──────────────────────────────────────────────────────

class TestSaveLoadPipeline:
    """Tests for save_pipeline and load_pipeline."""

    @pytest.fixture
    def fitted_result(self):
        """Run the full pipeline and return the PipelineResult."""
        rng = np.random.default_rng(42)
        iq = rng.normal(size=(1000, 2))
        labels = np.zeros(1000, dtype=int)
        labels[:300] = 1
        rng.shuffle(labels)
        return preprocess_pipeline(iq, labels, PipelineParams())

    def test_roundtrip_without_pca(self, tmp_path, fitted_result):
        """Save and reload a pipeline that has no PCA.

        Expected: loaded scaler matches the original; loaded PCA is None.
        """
        path = tmp_path / "pipe.joblib"
        save_pipeline(path, fitted_result.scaler, pca=None)

        scaler, pca = load_pipeline(path)

        assert pca is None
        np.testing.assert_array_equal(scaler.mu, fitted_result.scaler.mu)
        np.testing.assert_array_equal(scaler.sigma, fitted_result.scaler.sigma)

    def test_roundtrip_with_pca(self, tmp_path):
        """Save and reload a pipeline that includes PCA.

        Expected: loaded scaler and PCA reproduce the same transform.
        """
        rng = np.random.default_rng(42)
        iq = rng.normal(size=(1000, 2))
        labels = np.zeros(1000, dtype=int)
        labels[:300] = 1
        rng.shuffle(labels)
        result = preprocess_pipeline(
            iq, labels, PipelineParams(use_pca=True, pca_components=2)
        )

        path = tmp_path / "pipe_pca.joblib"
        save_pipeline(path, result.scaler, pca=result.pca)

        scaler, pca = load_pipeline(path)

        np.testing.assert_array_equal(scaler.mu, result.scaler.mu)
        np.testing.assert_array_equal(scaler.sigma, result.scaler.sigma)
        assert pca is not None
        np.testing.assert_array_equal(
            pca.components_, result.pca.components_
        )

    def test_loaded_scaler_reproduces_transform(self, tmp_path, fitted_result):
        """A loaded scaler must produce identical output to the original.

        Expected: standardising the same data with the original and
        loaded scalers yields bit-identical results.
        """
        path = tmp_path / "pipe.joblib"
        save_pipeline(path, fitted_result.scaler)

        scaler, _ = load_pipeline(path)

        rng = np.random.default_rng(99)
        new_data = rng.normal(size=(50, 2))
        expected, _ = standardise(new_data, scaler=fitted_result.scaler)
        actual, _ = standardise(new_data, scaler=scaler)

        np.testing.assert_array_equal(actual, expected)

    def test_load_missing_file(self, tmp_path):
        """Loading from a non-existent path should raise FileNotFoundError.

        Expected: raises FileNotFoundError.
        """
        with pytest.raises(FileNotFoundError):
            load_pipeline(tmp_path / "nonexistent.joblib")

    def test_load_corrupt_file(self, tmp_path):
        """Loading a file without a 'scaler' key should raise KeyError.

        Expected: raises KeyError mentioning 'scaler'.
        """
        import joblib
        path = tmp_path / "bad.joblib"
        joblib.dump({"foo": 1}, path)

        with pytest.raises(KeyError, match="scaler"):
            load_pipeline(path)

    def test_string_path_accepted(self, tmp_path, fitted_result):
        """Both save and load should accept plain string paths.

        Expected: round-trip succeeds with str instead of Path.
        """
        path = str(tmp_path / "pipe_str.joblib")
        save_pipeline(path, fitted_result.scaler)

        scaler, pca = load_pipeline(path)

        assert pca is None
        np.testing.assert_array_equal(scaler.mu, fitted_result.scaler.mu)
