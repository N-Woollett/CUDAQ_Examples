import cudaq

N_QUBITS = 3
N_LAYERS = 2
N_PARAMS = N_QUBITS * N_LAYERS  # 6


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
