import cudaq

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
