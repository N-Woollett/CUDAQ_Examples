import json
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Tuple

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    _CUPY_AVAILABLE = False

import matplotlib.pyplot as plt 

def generate_iq_data(
    n_samples: int,
    params: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The IQ data simulator generates synthetic single-shot readout data by forward-modelling the dispersive readout process. Rather than solving the full Lindblad master equation (computationally expensive for millions of training samples), we use an analytic Gaussian mixture model calibrated to match the key features of experimental IQ data.

    Args:
        n_samples (int): Number of single-shot samples to generate.
        params (dict[str, float]): Dictionary of physical parameters with the following keys:
            - SNR (float): Signal-to-noise ratio defining the separation of the ground and excited state Gaussian blobs.
            - sigma (float): Standard deviation of the Gaussian noise.
            - p_thermal (float): Probability of the qubit spontaneously transitioning to the excited state due to thermal excitation.
            - T1_over_tmeas (float): Ratio of the qubit's relaxation time (T1) to the measurement time, representing decay events during measurement.
            - blob_angle (float): The angle (in radians) by which the IQ blobs are rotated in the IQ plane.
            - seed (int): The random seed for reproducibility of the data generation process.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - iq_data (np.ndarray): The synthetic single-shot readings in the IQ plane, typically an array of complex numbers or an (N, 2) shaped array.
            - labels (np.ndarray): The ground truth states (e.g., 0 for ground, 1 for excited) corresponding to each generated IQ data point.
    """
    snr = params["SNR"]
    sigma = params["sigma"]
    p_thermal = params["p_thermal"]
    t1_over_tmeas = params["T1_over_tmeas"]
    blob_angle = params["blob_angle"]
    seed = int(params["seed"])

    rng = np.random.default_rng(seed)
    
    # Draw balanced random labels {0, 1}
    labels = rng.integers(0, 2, size=n_samples)
    
    # Compute blob centres from SNR and blob_angle
    separation = 2 * snr * sigma
    mu_0_unrot = np.array([-separation / 2.0, 0.0])
    mu_1_unrot = np.array([separation / 2.0, 0.0])
    
    rot_matrix = np.array([
        [np.cos(blob_angle), -np.sin(blob_angle)],
        [np.sin(blob_angle),  np.cos(blob_angle)]
    ])
    
    mu_0 = rot_matrix.dot(mu_0_unrot)
    mu_1 = rot_matrix.dot(mu_1_unrot)
    
    # Sample from N(mu_s, Sigma_s) for each state
    iq = np.zeros((n_samples, 2))
    
    mask_0 = (labels == 0)
    mask_1 = (labels == 1)
    
    n_0 = np.sum(mask_0)
    n_1 = np.sum(mask_1)
    
    cov = np.diag([sigma**2, sigma**2])
    
    if n_0 > 0:
        iq[mask_0] = rng.multivariate_normal(mu_0, cov, size=n_0)
    if n_1 > 0:
        iq[mask_1] = rng.multivariate_normal(mu_1, cov, size=n_1)
        
    # Apply T1 decay by re-drawing a fraction of |1> points from the |0> distribution
    p_t1 = 1.0 - np.exp(-1.0 / t1_over_tmeas)
    decay_mask = mask_1 & (rng.random(n_samples) < p_t1)
    n_decay = np.sum(decay_mask)
    if n_decay > 0:
        iq[decay_mask] = rng.multivariate_normal(mu_0, cov, size=n_decay)
        
    # Apply thermal excitation by re-drawing a fraction of |0> from |1> distribution
    thermal_mask = mask_0 & (rng.random(n_samples) < p_thermal)
    n_thermal = np.sum(thermal_mask)
    if n_thermal > 0:
        iq[thermal_mask] = rng.multivariate_normal(mu_1, cov, size=n_thermal)
        
    return iq, labels


def generate_iq_data_gpu(
    n_samples: int,
    params: Dict[str, float]
) -> Tuple["cp.ndarray", "cp.ndarray"]:
    """
    GPU-accelerated version of generate_iq_data using CuPy.

    Mirrors the CPU implementation exactly but runs all random sampling
    and array operations on the GPU.  If CuPy is not installed, a warning
    is printed and the call is transparently forwarded to the CPU path.

    Args:
        n_samples (int): Number of single-shot samples to generate.
        params (dict[str, float]): Same parameter dictionary as
            ``generate_iq_data``.

    Returns:
        tuple[cp.ndarray, cp.ndarray]: (iq_data, labels) as CuPy device
        arrays.  Use ``.get()`` to transfer back to the host if needed.
        Falls back to NumPy arrays when CuPy is unavailable.
    """
    if not _CUPY_AVAILABLE:
        warnings.warn(
            "CuPy is not available — falling back to CPU implementation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return generate_iq_data(n_samples, params)

    snr = params["SNR"]
    sigma = params["sigma"]
    p_thermal = params["p_thermal"]
    t1_over_tmeas = params["T1_over_tmeas"]
    blob_angle = params["blob_angle"]
    seed = int(params["seed"])

    rng = cp.random.default_rng(seed)

    # Draw balanced random labels {0, 1}
    labels = rng.integers(0, 2, size=n_samples)

    # Compute blob centres from SNR and blob_angle
    separation = 2 * snr * sigma
    mu_0_unrot = cp.array([-separation / 2.0, 0.0])
    mu_1_unrot = cp.array([separation / 2.0, 0.0])

    rot_matrix = cp.array([
        [cp.cos(blob_angle), -cp.sin(blob_angle)],
        [cp.sin(blob_angle),  cp.cos(blob_angle)]
    ])

    mu_0 = rot_matrix.dot(mu_0_unrot)
    mu_1 = rot_matrix.dot(mu_1_unrot)

    # Sample IQ points — CuPy Generator lacks multivariate_normal,
    # so we draw independent normals and shift/rotate manually.
    iq = cp.zeros((n_samples, 2))

    mask_0 = (labels == 0)
    mask_1 = (labels == 1)

    n_0 = int(cp.sum(mask_0))
    n_1 = int(cp.sum(mask_1))

    if n_0 > 0:
        noise_0 = rng.standard_normal((n_0, 2)) * sigma
        iq[mask_0] = noise_0 + mu_0
    if n_1 > 0:
        noise_1 = rng.standard_normal((n_1, 2)) * sigma
        iq[mask_1] = noise_1 + mu_1

    # Apply T1 decay by re-drawing a fraction of |1⟩ points from the |0⟩ distribution
    p_t1 = 1.0 - cp.exp(-1.0 / t1_over_tmeas)
    decay_mask = mask_1 & (rng.random(n_samples) < p_t1)
    n_decay = int(cp.sum(decay_mask))
    if n_decay > 0:
        noise_decay = rng.standard_normal((n_decay, 2)) * sigma
        iq[decay_mask] = noise_decay + mu_0

    # Apply thermal excitation by re-drawing a fraction of |0⟩ from |1⟩ distribution
    thermal_mask = mask_0 & (rng.random(n_samples) < p_thermal)
    n_thermal = int(cp.sum(thermal_mask))
    if n_thermal > 0:
        noise_thermal = rng.standard_normal((n_thermal, 2)) * sigma
        iq[thermal_mask] = noise_thermal + mu_1

    return iq, labels


def main():
    config_path = Path(__file__).resolve().parent.parent / "config" / "default_params.json"
    with open(config_path) as f:
        params = json.load(f)

    results = generate_iq_data(n_samples=params["n_train"], params=params)
    print(results)

    plt.scatter(results[0][:, 0], results[0][:, 1], c=results[1])
    plt.show()

    results = generate_iq_data_gpu(n_samples=10*params["n_train"], params=params)
    print(results)

    plt.scatter(results[0].get()[:, 0], results[0].get()[:, 1], c=results[1].get())
    plt.show()

if __name__ == "__main__":
    main()