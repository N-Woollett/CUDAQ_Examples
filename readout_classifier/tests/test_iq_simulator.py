"""Tests for readout_classifier.src.iq_simulator."""

import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from readout_classifier.src.iq_simulator import generate_iq_data, generate_iq_data_gpu


# ─── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def default_params():
    """Load the default parameter set from the project config file."""
    config_path = (
        Path(__file__).resolve().parent.parent / "config" / "default_params.json"
    )
    with open(config_path) as f:
        return json.load(f)


@pytest.fixture
def simple_params():
    """Minimal parameter set with no noise effects for cleaner tests.

    SNR=3, sigma=1 → blob separation of 6 along the I-axis.
    p_thermal=0, T1_over_tmeas=1e6 (negligible decay), blob_angle=0.
    """
    return {
        "SNR": 3.0,
        "sigma": 1.0,
        "p_thermal": 0.0,
        "T1_over_tmeas": 1e6,
        "blob_angle": 0.0,
        "seed": 42,
    }


# ─── output shape and types ─────────────────────────────────────────────────

class TestOutputShape:
    """Verify that generate_iq_data returns arrays of the correct shape
    and data type."""

    def test_iq_shape_is_n_by_2(self, simple_params):
        """The IQ output must be a 2-D array with shape (n_samples, 2),
        where columns correspond to the I and Q channels.

        Expected: iq.shape == (1000, 2).
        """
        iq, _ = generate_iq_data(1000, simple_params)
        assert iq.shape == (1000, 2)

    def test_labels_shape_matches_n_samples(self, simple_params):
        """The labels array must be 1-D with one entry per sample.

        Expected: labels.shape == (1000,).
        """
        _, labels = generate_iq_data(1000, simple_params)
        assert labels.shape == (1000,)

    def test_labels_are_binary(self, simple_params):
        """Labels should contain only the values 0 (ground) and
        1 (excited).

        Expected: the set of unique label values is {0, 1}.
        """
        _, labels = generate_iq_data(1000, simple_params)
        assert set(np.unique(labels)) == {0, 1}

    def test_iq_dtype_is_float(self, simple_params):
        """IQ data should be floating point.

        Expected: iq.dtype is a float sub-type.
        """
        iq, _ = generate_iq_data(100, simple_params)
        assert np.issubdtype(iq.dtype, np.floating)

    def test_labels_dtype_is_integer(self, simple_params):
        """Labels should be integer-valued.

        Expected: labels.dtype is an integer sub-type.
        """
        _, labels = generate_iq_data(100, simple_params)
        assert np.issubdtype(labels.dtype, np.integer)


# ─── reproducibility ────────────────────────────────────────────────────────

class TestReproducibility:
    """Verify that the random seed produces deterministic output."""

    def test_same_seed_gives_identical_output(self, simple_params):
        """Calling generate_iq_data twice with the same seed must
        produce bit-identical IQ data and labels.

        Expected: both IQ arrays and both label arrays are exactly equal.
        """
        iq1, labels1 = generate_iq_data(500, simple_params)
        iq2, labels2 = generate_iq_data(500, simple_params)
        np.testing.assert_array_equal(iq1, iq2)
        np.testing.assert_array_equal(labels1, labels2)

    def test_different_seed_gives_different_output(self, simple_params):
        """Changing the seed should produce different data.

        Expected: the IQ arrays from two different seeds are not equal.
        """
        iq1, _ = generate_iq_data(500, simple_params)
        params2 = {**simple_params, "seed": 99}
        iq2, _ = generate_iq_data(500, params2)
        assert not np.array_equal(iq1, iq2)


# ─── blob separation ────────────────────────────────────────────────────────

class TestBlobSeparation:
    """Verify that the ground and excited state blobs are positioned
    correctly in the IQ plane based on the SNR and sigma parameters."""

    def test_mean_separation_along_i_axis(self, simple_params):
        """With blob_angle=0 the two blobs should be separated along the
        I-axis by 2 · SNR · sigma.  For SNR=3, sigma=1 the expected
        separation is 6.0.

        Expected: |mean_I(|1⟩) - mean_I(|0⟩)| ≈ 6.0 (within 0.3 for
        10 000 samples).
        """
        n = 10_000
        iq, labels = generate_iq_data(n, simple_params)
        mean_0 = iq[labels == 0].mean(axis=0)
        mean_1 = iq[labels == 1].mean(axis=0)
        expected_sep = 2 * simple_params["SNR"] * simple_params["sigma"]
        actual_sep = mean_1[0] - mean_0[0]
        np.testing.assert_allclose(actual_sep, expected_sep, atol=0.3)

    def test_q_means_near_zero(self, simple_params):
        """With blob_angle=0, both blob centres should have Q ≈ 0.

        Expected: per-state Q-channel means are within 0.1 of zero.
        """
        n = 10_000
        iq, labels = generate_iq_data(n, simple_params)
        mean_q_0 = iq[labels == 0, 1].mean()
        mean_q_1 = iq[labels == 1, 1].mean()
        np.testing.assert_allclose(mean_q_0, 0.0, atol=0.1)
        np.testing.assert_allclose(mean_q_1, 0.0, atol=0.1)

    def test_higher_snr_increases_separation(self, simple_params):
        """Doubling the SNR should approximately double the distance
        between blob centres along the I-axis.

        Expected: separation at SNR=6 is roughly twice that at SNR=3.
        """
        n = 10_000
        iq_lo, labels_lo = generate_iq_data(n, simple_params)
        params_hi = {**simple_params, "SNR": 6.0}
        iq_hi, labels_hi = generate_iq_data(n, params_hi)

        sep_lo = iq_lo[labels_lo == 1, 0].mean() - iq_lo[labels_lo == 0, 0].mean()
        sep_hi = iq_hi[labels_hi == 1, 0].mean() - iq_hi[labels_hi == 0, 0].mean()
        np.testing.assert_allclose(sep_hi / sep_lo, 2.0, atol=0.15)


# ─── isotropic covariance (default) ─────────────────────────────────────────

class TestIsotropicCovariance:
    """When no explicit covariance is provided, the blobs should be
    isotropic with variance sigma²."""

    def test_per_state_std_matches_sigma(self, simple_params):
        """The per-state standard deviation along each axis should be
        close to the configured sigma value.

        Expected: std of I and Q channels for each state ≈ sigma (1.0),
        within 0.1 for 10 000 samples.
        """
        n = 10_000
        iq, labels = generate_iq_data(n, simple_params)
        for state in (0, 1):
            state_iq = iq[labels == state]
            std_i = state_iq[:, 0].std()
            std_q = state_iq[:, 1].std()
            np.testing.assert_allclose(std_i, simple_params["sigma"], atol=0.1)
            np.testing.assert_allclose(std_q, simple_params["sigma"], atol=0.1)


# ─── anisotropic covariance ─────────────────────────────────────────────────

class TestAnisotropicCovariance:
    """Verify that explicit covariance matrices control the blob shape."""

    def test_shared_covariance(self, simple_params):
        """Passing a single 2×2 covariance matrix should be used by
        both states, producing elongated blobs.

        Expected: the sample covariance of each state blob is close to
        the specified matrix.
        """
        cov_shared = [[2.0, 0.5], [0.5, 0.5]]
        params = {**simple_params, "cov": cov_shared}
        n = 20_000
        iq, labels = generate_iq_data(n, params)
        for state in (0, 1):
            sample_cov = np.cov(iq[labels == state].T)
            np.testing.assert_allclose(sample_cov, cov_shared, atol=0.15)

    def test_per_state_covariance(self, simple_params):
        """Passing a dict with per-state covariance matrices should give
        each blob a different shape.

        Expected: sample covariance of state 0 matches cov["0"] and
        state 1 matches cov["1"], within tolerance.
        """
        cov_dict = {
            "0": [[1.0, 0.0], [0.0, 2.0]],
            "1": [[2.0, 0.0], [0.0, 1.0]],
        }
        params = {**simple_params, "cov": cov_dict}
        n = 20_000
        iq, labels = generate_iq_data(n, params)
        for state, key in [(0, "0"), (1, "1")]:
            sample_cov = np.cov(iq[labels == state].T)
            np.testing.assert_allclose(sample_cov, cov_dict[key], atol=0.15)


# ─── T1 decay ───────────────────────────────────────────────────────────────

class TestT1Decay:
    """Verify that T1 relaxation moves a fraction of |1⟩ samples toward
    the |0⟩ blob centre."""

    def test_decay_shifts_excited_state_mean(self):
        """With a short T1 relative to measurement time, a significant
        fraction of |1⟩ samples decay to |0⟩, pulling the mean of the
        nominally-excited samples toward the ground state.

        Expected: the I-channel mean of |1⟩ samples is lower than the
        ideal blob centre (+3.0) because decayed points cluster near
        the |0⟩ centre (−3.0).
        """
        params = {
            "SNR": 3.0,
            "sigma": 1.0,
            "p_thermal": 0.0,
            "T1_over_tmeas": 0.5,     # aggressive decay
            "blob_angle": 0.0,
            "seed": 42,
        }
        n = 20_000
        iq, labels = generate_iq_data(n, params)
        mean_1_i = iq[labels == 1, 0].mean()
        ideal_centre = params["SNR"] * params["sigma"]  # +3.0
        assert mean_1_i < ideal_centre

    def test_no_decay_when_t1_is_large(self, simple_params):
        """With T1/t_meas → ∞ the decay probability approaches zero
        and the |1⟩ mean should sit close to the ideal centre.

        Expected: |1⟩ I-channel mean ≈ +SNR·sigma (3.0), within 0.15.
        """
        n = 10_000
        iq, labels = generate_iq_data(n, simple_params)
        mean_1_i = iq[labels == 1, 0].mean()
        ideal = simple_params["SNR"] * simple_params["sigma"]
        np.testing.assert_allclose(mean_1_i, ideal, atol=0.15)


# ─── thermal excitation ─────────────────────────────────────────────────────

class TestThermalExcitation:
    """Verify that thermal excitation moves a fraction of |0⟩ samples
    toward the |1⟩ blob centre."""

    def test_thermal_shifts_ground_state_mean(self):
        """With a high thermal population, some |0⟩ samples are redrawn
        from the |1⟩ distribution, pulling the ground-state mean toward
        the excited-state centre.

        Expected: the I-channel mean of |0⟩ samples is higher (less
        negative) than the ideal |0⟩ centre (−3.0).
        """
        params = {
            "SNR": 3.0,
            "sigma": 1.0,
            "p_thermal": 0.3,         # exaggerated thermal population
            "T1_over_tmeas": 1e6,
            "blob_angle": 0.0,
            "seed": 42,
        }
        n = 20_000
        iq, labels = generate_iq_data(n, params)
        mean_0_i = iq[labels == 0, 0].mean()
        ideal_centre = -params["SNR"] * params["sigma"]  # -3.0
        assert mean_0_i > ideal_centre

    def test_no_thermal_when_p_is_zero(self, simple_params):
        """With p_thermal=0 no ground-state samples should be redrawn.
        The |0⟩ mean should sit close to the ideal centre.

        Expected: |0⟩ I-channel mean ≈ −SNR·sigma (−3.0), within 0.15.
        """
        n = 10_000
        iq, labels = generate_iq_data(n, simple_params)
        mean_0_i = iq[labels == 0, 0].mean()
        ideal = -simple_params["SNR"] * simple_params["sigma"]
        np.testing.assert_allclose(mean_0_i, ideal, atol=0.15)


# ─── blob rotation ──────────────────────────────────────────────────────────

class TestBlobRotation:
    """Verify that the blob_angle parameter rotates all IQ points."""

    def test_90_degree_rotation_swaps_axes(self, simple_params):
        """A π/2 rotation should swap the I and Q axes: what was a
        horizontal separation becomes vertical.

        Expected: the Q-channel separation ≈ 6.0 (the original
        I-channel separation) and the I-channel separation ≈ 0.
        """
        params = {**simple_params, "blob_angle": np.pi / 2}
        n = 10_000
        iq, labels = generate_iq_data(n, params)
        mean_0 = iq[labels == 0].mean(axis=0)
        mean_1 = iq[labels == 1].mean(axis=0)

        expected_sep = 2 * params["SNR"] * params["sigma"]
        # After 90° rotation, separation should be along Q
        np.testing.assert_allclose(
            mean_1[1] - mean_0[1], expected_sep, atol=0.3
        )
        # I-channel separation should be near zero
        np.testing.assert_allclose(mean_1[0] - mean_0[0], 0.0, atol=0.3)

    def test_zero_angle_is_identity(self, simple_params):
        """With blob_angle=0 the rotation is the identity — blobs are
        separated along the I-axis only.

        Expected: identical output to the unrotated case (verified by
        checking Q-channel means near zero).
        """
        n = 10_000
        iq, labels = generate_iq_data(n, simple_params)
        mean_q_0 = iq[labels == 0, 1].mean()
        mean_q_1 = iq[labels == 1, 1].mean()
        np.testing.assert_allclose(mean_q_0, 0.0, atol=0.1)
        np.testing.assert_allclose(mean_q_1, 0.0, atol=0.1)

    def test_rotation_preserves_separation_magnitude(self, simple_params):
        """Rotation should not change the Euclidean distance between
        blob centres, only the direction.

        Expected: ‖mean_1 − mean_0‖ ≈ 2·SNR·sigma for any angle.
        """
        for angle in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            params = {**simple_params, "blob_angle": angle}
            n = 10_000
            iq, labels = generate_iq_data(n, params)
            mean_0 = iq[labels == 0].mean(axis=0)
            mean_1 = iq[labels == 1].mean(axis=0)
            dist = np.linalg.norm(mean_1 - mean_0)
            expected = 2 * params["SNR"] * params["sigma"]
            np.testing.assert_allclose(dist, expected, atol=0.3)


# ─── integration with default config ────────────────────────────────────────

class TestDefaultConfig:
    """Smoke tests using the project's default_params.json."""

    def test_generates_correct_number_of_train_samples(self, default_params):
        """Using the default config, the function should produce
        n_train IQ samples.

        Expected: IQ shape is (10000, 2) and labels shape is (10000,).
        """
        iq, labels = generate_iq_data(
            n_samples=default_params["n_train"], params=default_params
        )
        assert iq.shape == (default_params["n_train"], 2)
        assert labels.shape == (default_params["n_train"],)

    def test_generates_correct_number_of_test_samples(self, default_params):
        """Using the default config, the function should produce
        n_test IQ samples.

        Expected: IQ shape is (2000, 2) and labels shape is (2000,).
        """
        iq, labels = generate_iq_data(
            n_samples=default_params["n_test"], params=default_params
        )
        assert iq.shape == (default_params["n_test"], 2)
        assert labels.shape == (default_params["n_test"],)


# ─── GPU fallback ────────────────────────────────────────────────────────────

class TestGPUFallback:
    """Verify that generate_iq_data_gpu falls back gracefully when CuPy
    is not available."""

    def test_fallback_emits_warning(self, simple_params):
        """When CuPy is unavailable the GPU function should emit a
        RuntimeWarning and transparently delegate to the CPU path.

        Expected: a RuntimeWarning containing 'CuPy is not available'
        is raised.
        """
        with patch(
            "readout_classifier.src.iq_simulator._CUPY_AVAILABLE", False
        ):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                generate_iq_data_gpu(100, simple_params)
                assert len(w) == 1
                assert issubclass(w[0].category, RuntimeWarning)
                assert "CuPy is not available" in str(w[0].message)

    def test_fallback_returns_numpy_arrays(self, simple_params):
        """When falling back to CPU, the returned arrays should be
        ordinary NumPy arrays (not CuPy device arrays).

        Expected: both iq and labels are np.ndarray instances.
        """
        with patch(
            "readout_classifier.src.iq_simulator._CUPY_AVAILABLE", False
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                iq, labels = generate_iq_data_gpu(100, simple_params)
        assert isinstance(iq, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_fallback_output_matches_cpu(self, simple_params):
        """The fallback path should produce identical output to calling
        generate_iq_data directly with the same parameters.

        Expected: IQ and label arrays from the GPU fallback are
        bit-identical to the CPU function output.
        """
        iq_cpu, labels_cpu = generate_iq_data(100, simple_params)
        with patch(
            "readout_classifier.src.iq_simulator._CUPY_AVAILABLE", False
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                iq_gpu, labels_gpu = generate_iq_data_gpu(100, simple_params)
        np.testing.assert_array_equal(iq_cpu, iq_gpu)
        np.testing.assert_array_equal(labels_cpu, labels_gpu)


# ─── edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions and small-sample behaviour."""

    def test_single_sample(self, simple_params):
        """The generator should work for n_samples=1 without error.

        Expected: IQ shape is (1, 2) and labels shape is (1,).
        """
        iq, labels = generate_iq_data(1, simple_params)
        assert iq.shape == (1, 2)
        assert labels.shape == (1,)

    def test_zero_snr_overlapping_blobs(self, simple_params):
        """With SNR=0 both blobs share the same centre at the origin,
        so the overall mean should be near (0, 0).

        Expected: overall I and Q means are within 0.1 of zero.
        """
        params = {**simple_params, "SNR": 0.0}
        n = 10_000
        iq, _ = generate_iq_data(n, params)
        np.testing.assert_allclose(iq.mean(axis=0), [0.0, 0.0], atol=0.1)

    def test_very_large_sample_count(self, simple_params):
        """Generating a large dataset should complete without error
        and return the correct shape.

        Expected: IQ shape is (100_000, 2).
        """
        iq, labels = generate_iq_data(100_000, simple_params)
        assert iq.shape == (100_000, 2)
        assert labels.shape == (100_000,)
