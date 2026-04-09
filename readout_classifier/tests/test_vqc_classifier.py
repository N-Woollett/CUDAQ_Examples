"""Tests for readout_classifier.src.vqc_classifier."""

import numpy as np
import pytest

import cudaq

from readout_classifier.src.vqc_classifier import (
    angle_encoding_feature_map,
    classifier_kernel,
    HAMILTONIAN,
    N_PARAMS,
    N_QUBITS,
)

# CUDA-Q's default simulator uses complex64 (single precision, ~7 decimal
# digits).  All amplitude / probability tolerances must reflect this.
_AMP_TOL = 1e-6
_PROB_TOL = 1e-6


# ─── wrapper kernel ────────────────────────────────────────────────────────

@cudaq.kernel
def _wrapper(features: list[float]):
    """Allocate 2 qubits and apply the feature map under test."""
    q = cudaq.qvector(2)
    angle_encoding_feature_map(q, features)


# ─── analytic helper ───────────────────────────────────────────────────────

def _expected_statevector(f0: float, f1: float) -> np.ndarray:
    """Return the 4-element statevector for Ry(f0)Rz(f0)|0> x Ry(f1)Rz(f1)|0>.

    Uses CUDA-Q little-endian ordering (qubit 0 = LSB):
      index 0 → |00⟩, index 1 → |10⟩, index 2 → |01⟩, index 3 → |11⟩.
    """
    # Single-qubit amplitudes: Rz(θ)·Ry(θ)|0⟩
    a0_0 = np.exp(-1j * f0 / 2) * np.cos(f0 / 2)  # qubit 0, |0⟩
    a0_1 = np.exp(1j * f0 / 2) * np.sin(f0 / 2)   # qubit 0, |1⟩
    a1_0 = np.exp(-1j * f1 / 2) * np.cos(f1 / 2)  # qubit 1, |0⟩
    a1_1 = np.exp(1j * f1 / 2) * np.sin(f1 / 2)   # qubit 1, |1⟩

    return np.array([
        a0_0 * a1_0,  # |00⟩
        a0_1 * a1_0,  # |10⟩
        a0_0 * a1_1,  # |01⟩
        a0_1 * a1_1,  # |11⟩
    ])


def _get_state_array(features: list[float]) -> np.ndarray:
    """Run the wrapper kernel and return the statevector as a numpy array."""
    state = cudaq.get_state(_wrapper, features)
    return np.array(state)


def _get_probabilities(features: list[float]) -> np.ndarray:
    """Return measurement probabilities from the statevector."""
    sv = _get_state_array(features)
    return np.abs(sv) ** 2


# ─── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def zero_features():
    """Feature vector of all zeros — qubits should remain in |0⟩."""
    return [0.0, 0.0]


@pytest.fixture
def pi_features():
    """Both features set to pi for symmetric rotation tests."""
    return [np.pi, np.pi]


# ─── qubit ordering ───────────────────────────────────────────────────────

class TestQubitOrdering:
    """Canary test: confirm CUDA-Q's little-endian statevector index
    convention so that all subsequent amplitude tests are valid."""

    def test_pi_on_qubit0_identifies_index_convention(self):
        """Ry(pi)|0⟩ = |1⟩, so features [pi, 0] should excite only qubit 0.

        Expected: in little-endian ordering, |10⟩ maps to index 1 and
        should carry all the probability.
        """
        probs = _get_probabilities([np.pi, 0.0])
        assert np.argmax(probs) == 1, (
            f"Expected index 1 (little-endian |10⟩) to have max probability, "
            f"got index {np.argmax(probs)}. Probabilities: {probs}"
        )
        np.testing.assert_allclose(probs[1], 1.0, atol=_PROB_TOL)


# ─── zero features ─────────────────────────────────────────────────────────

class TestZeroFeatures:
    """Verify that zero-valued features leave the system in |00⟩."""

    def test_all_probability_in_ground_state(self, zero_features):
        """Ry(0) and Rz(0) are both identity, so the state should be |00⟩.

        Expected: statevector is [1, 0, 0, 0].
        """
        sv = _get_state_array(zero_features)
        np.testing.assert_allclose(sv, [1, 0, 0, 0], atol=_AMP_TOL)

    def test_ground_state_probability_is_unity(self, zero_features):
        """The |00⟩ probability should be 1.0 and all others 0.0.

        Expected: probabilities are [1, 0, 0, 0].
        """
        probs = _get_probabilities(zero_features)
        np.testing.assert_allclose(probs, [1, 0, 0, 0], atol=_PROB_TOL)


# ─── known rotations ──────────────────────────────────────────────────────

class TestKnownRotations:
    """Feature values that produce deterministic computational basis states."""

    def test_pi_zero_gives_qubit0_excited(self):
        """Features [pi, 0] should put qubit 0 in |1⟩ and leave qubit 1 in |0⟩.

        Expected: all probability in |10⟩ (index 1).
        """
        probs = _get_probabilities([np.pi, 0.0])
        np.testing.assert_allclose(probs, [0, 1, 0, 0], atol=_AMP_TOL)

    def test_zero_pi_gives_qubit1_excited(self):
        """Features [0, pi] should put qubit 1 in |1⟩ and leave qubit 0 in |0⟩.

        Expected: all probability in |01⟩ (index 2).
        """
        probs = _get_probabilities([0.0, np.pi])
        np.testing.assert_allclose(probs, [0, 0, 1, 0], atol=_AMP_TOL)

    def test_pi_pi_gives_both_excited(self, pi_features):
        """Features [pi, pi] should excite both qubits.

        Expected: all probability in |11⟩ (index 3).
        """
        probs = _get_probabilities(pi_features)
        np.testing.assert_allclose(probs, [0, 0, 0, 1], atol=_AMP_TOL)


# ─── statevector amplitudes ───────────────────────────────────────────────

class TestStatevectorAmplitudes:
    """Verify full complex amplitudes against the analytic Ry-Rz formula."""

    def test_half_pi_amplitudes(self):
        """Features [pi/2, pi/2] should produce an equal superposition.

        Expected: statevector [-i/2, 1/2, 1/2, i/2].
        """
        sv = _get_state_array([np.pi / 2, np.pi / 2])
        expected = np.array([-0.5j, 0.5, 0.5, 0.5j])
        np.testing.assert_allclose(sv, expected, atol=_AMP_TOL)

    def test_asymmetric_features(self):
        """Features [pi/3, pi/4] — asymmetric case verified against formula.

        Expected: statevector matches _expected_statevector(pi/3, pi/4).
        """
        f0, f1 = np.pi / 3, np.pi / 4
        sv = _get_state_array([f0, f1])
        expected = _expected_statevector(f0, f1)
        np.testing.assert_allclose(sv, expected, atol=_AMP_TOL)

    def test_negative_features(self):
        """Negative feature values should be handled correctly by Ry and Rz.

        Expected: statevector matches _expected_statevector(-pi/2, -pi/3).
        """
        f0, f1 = -np.pi / 2, -np.pi / 3
        sv = _get_state_array([f0, f1])
        expected = _expected_statevector(f0, f1)
        np.testing.assert_allclose(sv, expected, atol=_AMP_TOL)

    def test_large_angle_features(self):
        """Angles outside [-pi, pi] should wrap correctly via rotation algebra.

        Expected: statevector matches _expected_statevector(3*pi, -2.5*pi).
        """
        f0, f1 = 3 * np.pi, -2.5 * np.pi
        sv = _get_state_array([f0, f1])
        expected = _expected_statevector(f0, f1)
        np.testing.assert_allclose(sv, expected, atol=_AMP_TOL)


# ─── qubit independence ───────────────────────────────────────────────────

class TestQubitIndependence:
    """Verify that qubits are independent: changing one feature does not
    affect the other qubit's reduced state."""

    def test_qubit1_unaffected_by_f0(self):
        """Varying f0 while holding f1 constant should not change qubit 1's
        marginal probabilities.

        Expected: P(q1=0) and P(q1=1) are identical for both feature vectors.
        """
        probs_a = _get_probabilities([np.pi / 3, np.pi / 4])
        probs_b = _get_probabilities([2 * np.pi / 3, np.pi / 4])

        # Qubit 1 marginal: sum over qubit 0 states (little-endian)
        q1_marginal_a = [probs_a[0] + probs_a[1], probs_a[2] + probs_a[3]]
        q1_marginal_b = [probs_b[0] + probs_b[1], probs_b[2] + probs_b[3]]

        np.testing.assert_allclose(q1_marginal_a, q1_marginal_b, atol=_AMP_TOL)

    def test_qubit0_unaffected_by_f1(self):
        """Varying f1 while holding f0 constant should not change qubit 0's
        marginal probabilities.

        Expected: P(q0=0) and P(q0=1) are identical for both feature vectors.
        """
        probs_a = _get_probabilities([np.pi / 4, np.pi / 3])
        probs_b = _get_probabilities([np.pi / 4, 2 * np.pi / 3])

        # Qubit 0 marginal: sum over qubit 1 states (little-endian)
        q0_marginal_a = [probs_a[0] + probs_a[2], probs_a[1] + probs_a[3]]
        q0_marginal_b = [probs_b[0] + probs_b[2], probs_b[1] + probs_b[3]]

        np.testing.assert_allclose(q0_marginal_a, q0_marginal_b, atol=_AMP_TOL)


# ─── periodicity ──────────────────────────────────────────────────────────

class TestPeriodicity:
    """Verify that Rz(θ+2π)·Ry(θ+2π) = Rz(θ)·Ry(θ) — the minus signs
    from each gate's 2π half-period cancel in the product."""

    def test_2pi_shift_gives_identical_state(self):
        """Shifting one feature by 2π should produce the same statevector.

        Expected: state([pi/3, pi/4]) == state([pi/3 + 2pi, pi/4]).
        """
        base = [np.pi / 3, np.pi / 4]
        shifted = [np.pi / 3 + 2 * np.pi, np.pi / 4]

        sv_base = _get_state_array(base)
        sv_shifted = _get_state_array(shifted)

        np.testing.assert_allclose(sv_shifted, sv_base, atol=_AMP_TOL)

    def test_both_features_shift_by_4pi(self):
        """Shifting both features by 4π should produce the same statevector.

        Expected: state([pi/3, pi/4]) == state([pi/3 + 4pi, pi/4 + 4pi]).
        """
        base = [np.pi / 3, np.pi / 4]
        shifted = [np.pi / 3 + 4 * np.pi, np.pi / 4 + 4 * np.pi]

        sv_base = _get_state_array(base)
        sv_shifted = _get_state_array(shifted)

        np.testing.assert_allclose(sv_shifted, sv_base, atol=_AMP_TOL)


# ─── probabilities ─────────────────────────────────────────────────────────

class TestProbabilities:
    """Verify measurement probabilities match squared amplitudes."""

    def test_equal_superposition_at_half_pi(self):
        """Features [pi/2, pi/2] should give equal probability across all
        four basis states.

        Expected: each probability is 0.25.
        """
        probs = _get_probabilities([np.pi / 2, np.pi / 2])
        np.testing.assert_allclose(probs, [0.25, 0.25, 0.25, 0.25], atol=_AMP_TOL)

    def test_probabilities_match_squared_amplitudes(self):
        """Probabilities should equal |amplitude|^2 from the analytic formula.

        Expected: probs([pi/3, pi/4]) == |_expected_statevector(pi/3, pi/4)|^2.
        """
        f0, f1 = np.pi / 3, np.pi / 4
        probs = _get_probabilities([f0, f1])
        expected = np.abs(_expected_statevector(f0, f1)) ** 2
        np.testing.assert_allclose(probs, expected, atol=_AMP_TOL)

    def test_probabilities_sum_to_one(self):
        """Probabilities must sum to 1.0 for any feature values.

        Expected: sum of all probabilities is 1.0.
        """
        probs = _get_probabilities([1.234, -0.567])
        np.testing.assert_allclose(np.sum(probs), 1.0, atol=_PROB_TOL)


# ─── hamiltonian observable ──────────────────────────────────────────────


@cudaq.kernel
def _qubit2_ground():
    """All qubits in |0⟩ — qubit 2 is in |0⟩ (eigenvalue +1 for Z)."""
    q = cudaq.qvector(N_QUBITS)


@cudaq.kernel
def _qubit2_excited():
    """Flip qubit 2 to |1⟩ (eigenvalue -1 for Z)."""
    q = cudaq.qvector(N_QUBITS)
    x(q[N_QUBITS - 1])


class TestHamiltonian:
    """Verify the HAMILTONIAN observable measures Z on the readout qubit."""

    def test_qubit_count(self):
        """HAMILTONIAN acts on exactly one qubit (the readout qubit).

        Expected: qubit_count is 1 (only qubit N_QUBITS-1 has a non-identity term).
        """
        assert HAMILTONIAN.qubit_count == 1

    def test_z_expectation_ground_state(self):
        """All qubits in |0⟩ — Z expectation on qubit 2 should be +1.

        Expected: <Z_2> = +1.0.
        """
        result = cudaq.observe(_qubit2_ground, HAMILTONIAN)
        np.testing.assert_allclose(result.expectation(), 1.0, atol=_AMP_TOL)

    def test_z_expectation_excited_state(self):
        """Qubit 2 flipped to |1⟩ — Z expectation should be -1.

        Expected: <Z_2> = -1.0.
        """
        result = cudaq.observe(_qubit2_excited, HAMILTONIAN)
        np.testing.assert_allclose(result.expectation(), -1.0, atol=_AMP_TOL)


# ─── full classifier circuit ────────────────────────────────────────────────


class TestClassifierCircuit:
    """Verify the full classifier_kernel circuit structure and observable."""

    def test_circuit_diagram_gate_counts(self):
        """Use cudaq.draw() to verify the circuit contains the expected gates.

        Encoding: 2 Ry + 2 Rz = 4 gates
        2 variational layers: 2 * 3 Ry = 6 Ry, 2 * 2 CNOT = 4 CNOT
        Total: 8 Ry + 2 Rz + 4 CNOT = 14 gates
        """
        thetas = [0.0] * N_PARAMS
        features = [0.0, 0.0]

        diagram = cudaq.draw(classifier_kernel, thetas, features)
        print(diagram)

        # Count gate occurrences in the diagram string (case-insensitive)
        diagram_lower = diagram.lower()
        ry_count = diagram_lower.count("ry")
        rz_count = diagram_lower.count("rz")

        assert ry_count == 8, (
            f"Expected 8 Ry gates (2 encoding + 6 variational), got {ry_count}"
        )
        assert rz_count == 2, (
            f"Expected 2 Rz gates (encoding only), got {rz_count}"
        )

    def test_observe_expectation_bounded(self):
        """cudaq.observe with random thetas must yield expectation in [-1, +1].

        The Z operator has eigenvalues +/-1, so the expectation value of
        spin.z(2) is bounded by [-1, +1] for any state.
        """
        rng = np.random.default_rng(seed=42)
        thetas = rng.uniform(-np.pi, np.pi, size=N_PARAMS).tolist()
        features = rng.uniform(-np.pi, np.pi, size=2).tolist()

        result = cudaq.observe(classifier_kernel, HAMILTONIAN, thetas, features)
        exp_val = result.expectation()

        assert -1.0 <= exp_val <= 1.0, (
            f"Expectation value {exp_val} is outside [-1, +1]"
        )
