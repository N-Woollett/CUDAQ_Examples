import numpy as np

import cudaq
from cudaq import spin

N_QUBITS = 3
N_LAYERS = 2
N_PARAMS = N_QUBITS * N_LAYERS  # 6
HAMILTONIAN = spin.z(N_QUBITS - 1)


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


@cudaq.kernel
def variational_layer(q: cudaq.qview, thetas: list[float], layer_offset: int):
    """Single variational layer: Ry rotation on each qubit followed by a CNOT ladder.

    Applies Ry(thetas[layer_offset + i]) to qubit i for i in 0..N_QUBITS-1,
    then cascading CNOT gates from qubit i to qubit i+1.

    Args:
        q: The quantum register view (must have at least N_QUBITS qubits).
        thetas: Full parameter vector shared across all layers.
        layer_offset: Starting index into thetas for this layer's parameters.
    """
    for i in range(N_QUBITS):
        ry(thetas[layer_offset + i], q[i])

    for i in range(N_QUBITS - 1):
        x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def classifier_kernel(thetas: list[float], features: list[float]):
    """Full VQC classifier circuit: feature map followed by variational ansatz layers.

    Allocates N_QUBITS qubits, applies the angle-encoding feature map, then
    applies N_LAYERS variational layers. No explicit measurement is performed;
    measurement is handled by cudaq.observe.

    Args:
        thetas: Variational parameters (length N_PARAMS = N_QUBITS * N_LAYERS).
        features: Input features (length 2) for the angle-encoding map.
    """
    q = cudaq.qvector(N_QUBITS)

    angle_encoding_feature_map(q, features)

    for layer in range(N_LAYERS):
        variational_layer(q, thetas, layer * N_QUBITS)


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
