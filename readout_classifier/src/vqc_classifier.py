from dataclasses import dataclass

import numpy as np

import cudaq
from cudaq import spin


@dataclass(frozen=True)
class ClassifierConfig:
    """Circuit geometry for the VQC readout classifier.

    Attributes:
        n_qubits: Number of qubits in the circuit.
        n_layers: Number of variational layers.
    """
    n_qubits: int = 3
    n_layers: int = 2

    @property
    def n_params(self) -> int:
        return self.n_qubits * self.n_layers

    @property
    def hamiltonian(self) -> spin.SpinOperator:
        return spin.z(self.n_qubits - 1)


DEFAULT_CONFIG = ClassifierConfig()

N_QUBITS = DEFAULT_CONFIG.n_qubits
N_LAYERS = DEFAULT_CONFIG.n_layers
N_PARAMS = DEFAULT_CONFIG.n_params


@cudaq.kernel
def angle_encoding_feature_map(q: cudaq.qview, features: list[float]):
    """
    Angle-encoding feature map for the IQ readout classifier.
    Applies Ry and Rz rotations to qubits based on the input features.

    Args:
        q (cudaq.qview): The quantum register view.
        features (list[float]): The encoded features (length 2), usually the standardized
                                and angle-encoded I and Q values.
    """
    ry(features[0], q[0])
    rz(features[0], q[0])
    ry(features[1], q[1])
    rz(features[1], q[1])


def make_classifier(
    config: ClassifierConfig = DEFAULT_CONFIG,
) -> tuple:
    """Build a classifier kernel and Hamiltonian for the given config.

    Returns:
        (kernel, hamiltonian) where *kernel* is a ``@cudaq.kernel`` accepting
        ``(thetas: list[float], features: list[float])`` and *hamiltonian* is
        the Z observable on the readout qubit.
    """
    n_qubits = config.n_qubits
    n_layers = config.n_layers

    @cudaq.kernel
    def _variational_layer(q: cudaq.qview, thetas: list[float], layer_offset: int):
        for i in range(n_qubits):
            ry(thetas[layer_offset + i], q[i])
        for i in range(n_qubits - 1):
            x.ctrl(q[i], q[i + 1])

    @cudaq.kernel
    def kernel(thetas: list[float], features: list[float]):
        q = cudaq.qvector(n_qubits)
        angle_encoding_feature_map(q, features)
        for layer in range(n_layers):
            _variational_layer(q, thetas, layer * n_qubits)

    return kernel, config.hamiltonian


classifier_kernel, HAMILTONIAN = make_classifier()


def predict_score(thetas: list[float], features: list[float]) -> float:
    """Return the raw <Z> expectation value from the classifier circuit.

    Args:
        thetas: Variational parameters (length N_PARAMS).
        features: Input features (length 2).

    Returns:
        Expectation value in [-1, +1]. Useful for ROC curves and
        threshold tuning.
    """
    result = cudaq.observe(classifier_kernel, HAMILTONIAN, thetas, features)
    return result.expectation()


def predict(thetas: list[float], features: list[float]) -> int:
    """Classify a single input as 0 or 1.

    Args:
        thetas: Variational parameters (length N_PARAMS).
        features: Input features (length 2).

    Returns:
        0 if <Z> >= 0, else 1.
    """
    return 0 if predict_score(thetas, features) >= 0 else 1


def predict_batch(
    thetas: list[float], features_array: list[list[float]]
) -> list[int]:
    """Classify a batch of inputs as 0 or 1.

    Uses cudaq.observe broadcasting to evaluate all feature vectors in a
    single call when supported, falling back to a sequential loop otherwise.

    Args:
        thetas: Variational parameters (length N_PARAMS).
        features_array: List of feature vectors, each of length 2.

    Returns:
        List of labels (0 or 1), one per feature vector.
    """
    scores = predict_score_batch(thetas, features_array)
    return [0 if s >= 0 else 1 for s in scores]


def predict_score_batch(
    thetas: list[float], features_array: list[list[float]]
) -> list[float]:
    """Return raw <Z> expectation values for a batch of inputs.

    Uses cudaq.observe broadcasting to evaluate all feature vectors in a
    single call when supported, falling back to a sequential loop otherwise.

    Args:
        thetas: Variational parameters (length N_PARAMS).
        features_array: List of feature vectors, each of length 2.

    Returns:
        List of expectation values in [-1, +1], one per feature vector.
    """
    n = len(features_array)
    if n == 0:
        return []

    try:
        thetas_broadcast = np.tile(thetas, (n, 1))
        features_broadcast = np.array(features_array)
        results = cudaq.observe(
            classifier_kernel,
            HAMILTONIAN,
            thetas_broadcast,
            features_broadcast,
        )
        return [r.expectation() for r in results]
    except (TypeError, RuntimeError):
        return [predict_score(thetas, f) for f in features_array]


def cost_function(
    thetas: list[float],
    features_batch: list[list[float]],
    labels_batch: list[int],
) -> float:
    """Compute MSE loss between observed <Z> and target values.

    Target mapping: label 0 -> +1, label 1 -> -1.

    Args:
        thetas: Variational parameters (length N_PARAMS).
        features_batch: List of feature vectors, each of length 2.
        labels_batch: List of binary labels (0 or 1), one per feature vector.

    Returns:
        Mean squared error averaged over the batch.
    """
    scores = predict_score_batch(thetas, features_batch)
    targets = [1.0 - 2.0 * label for label in labels_batch]
    mse = sum((s - t) ** 2 for s, t in zip(scores, targets)) / len(scores)
    return mse
