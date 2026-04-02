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
    The IQ data simulator generates synthetic single-shot readout data by
    forward-modelling the dispersive readout process. Rather than solving the
    full Lindblad master equation (computationally expensive for millions of
    training samples), we use an analytic Gaussian mixture model calibrated
    to match the key features of experimental IQ data.

    Args:
        n_samples (int): Number of single-shot samples to generate.
        params (dict[str, float]): Dictionary of physical parameters with
            the following keys:
            - SNR (float): Signal-to-noise ratio defining the separation
              of the ground and excited state Gaussian blobs.
            - sigma (float): Standard deviation of the Gaussian noise
              (used only when ``cov`` is not provided).
            - cov (optional): Covariance specification for the Gaussian
              blobs.  Accepts three formats:
                  * A 2×2 list-of-lists shared by both states.
                  * A dict ``{"0": [[...]], "1": [[...]]}`` giving a
                    separate 2×2 matrix per state.
                  * Omitted — defaults to ``sigma² · I``.
            - p_thermal (float): Probability of the qubit spontaneously
              transitioning to the excited state due to thermal excitation.
            - T1_over_tmeas (float): Ratio of the qubit's relaxation time
              (T1) to the measurement time, representing decay events
              during measurement.
            - blob_angle (float): The angle (in radians) by which the IQ
              blobs are rotated in the IQ plane.
            - seed (int): The random seed for reproducibility of the data
              generation process.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - iq_data (np.ndarray): Synthetic single-shot readings in the
              IQ plane as an (N, 2) shaped array.
            - labels (np.ndarray): Ground truth states (0 for ground,
              1 for excited) for each IQ data point.
    """
    snr = params["SNR"]
    sigma = params["sigma"]
    p_thermal = params["p_thermal"]
    t1_over_tmeas = params["T1_over_tmeas"]
    blob_angle = params["blob_angle"]
    seed = int(params["seed"])

    # Build per-state covariance matrices
    raw_cov = params.get("cov", None)
    if raw_cov is None:
        # Isotropic default: sigma² · I
        cov_0 = np.diag([sigma ** 2, sigma ** 2])
        cov_1 = cov_0
    elif isinstance(raw_cov, dict):
        # Per-state covariance: {"0": [[...]], "1": [[...]]}
        cov_0 = np.array(raw_cov["0"], dtype=float)
        cov_1 = np.array(raw_cov["1"], dtype=float)
    else:
        # Shared 2×2 covariance for both states
        cov_0 = np.array(raw_cov, dtype=float)
        cov_1 = cov_0.copy()

    rng = np.random.default_rng(seed)
    
    # Draw balanced random labels {0, 1}
    labels = rng.integers(0, 2, size=n_samples)
    
    # Compute blob centres along I-axis (rotation applied globally later)
    separation = 2 * snr * sigma
    mu_0 = np.array([-separation / 2.0, 0.0])
    mu_1 = np.array([ separation / 2.0, 0.0])
    
    # Sample from N(mu_s, Sigma_s) for each state
    iq = np.zeros((n_samples, 2))
    
    mask_0 = (labels == 0)
    mask_1 = (labels == 1)
    
    n_0 = np.sum(mask_0)
    n_1 = np.sum(mask_1)
    
    if n_0 > 0:
        iq[mask_0] = rng.multivariate_normal(mu_0, cov_0, size=n_0)
    if n_1 > 0:
        iq[mask_1] = rng.multivariate_normal(mu_1, cov_1, size=n_1)
        
    # --- T1 decay during measurement ---
    # Flip probability: p_t1 = 1 - exp(-t_meas / T1).
    # Since the parameter stored in config is T1_over_tmeas = T1 / t_meas,
    # this is equivalent to 1 - exp(-1 / T1_over_tmeas).
    # For each |1⟩-labelled point, draw a uniform random number; if it falls
    # below p_t1, replace the IQ point with a new sample from the |0⟩ blob.
    # The label is kept as 1 (the ground-truth prepared state) but the IQ
    # coordinates now look like |0⟩.  This models a qubit that decayed from
    # |1⟩ to |0⟩ during the measurement integration window.
    p_t1 = 1.0 - np.exp(-1.0 / t1_over_tmeas)
    decay_mask = mask_1 & (rng.random(n_samples) < p_t1)
    n_decay = np.sum(decay_mask)
    if n_decay > 0:
        iq[decay_mask] = rng.multivariate_normal(mu_0, cov_0, size=n_decay)
        
    # Apply thermal excitation by re-drawing a fraction of |0> from |1> distribution
    thermal_mask = mask_0 & (rng.random(n_samples) < p_thermal)
    n_thermal = np.sum(thermal_mask)
    if n_thermal > 0:
        iq[thermal_mask] = rng.multivariate_normal(mu_1, cov_1, size=n_thermal)

    # Apply global rotation R(blob_angle) to all IQ points.
    # This simulates a non-axis-aligned readout, which occurs when the LO
    # frequency is not perfectly centred between the two dispersed resonator
    # frequencies.  Rotating *after* generation ensures both the blob centres
    # and the covariance ellipses are rotated together.
    if blob_angle != 0.0:
        rot = np.array([
            [np.cos(blob_angle), -np.sin(blob_angle)],
            [np.sin(blob_angle),  np.cos(blob_angle)]
        ])
        iq = iq @ rot.T   # (N,2) @ (2,2) -> (N,2)

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

    # Build per-state covariance matrices and their Cholesky factors
    raw_cov = params.get("cov", None)
    if raw_cov is None:
        cov_0 = cp.diag(cp.array([sigma ** 2, sigma ** 2]))
        cov_1 = cov_0
    elif isinstance(raw_cov, dict):
        cov_0 = cp.array(raw_cov["0"], dtype=cp.float64)
        cov_1 = cp.array(raw_cov["1"], dtype=cp.float64)
    else:
        cov_0 = cp.array(raw_cov, dtype=cp.float64)
        cov_1 = cov_0.copy()

    L_0 = cp.linalg.cholesky(cov_0)  # cov = L @ L.T
    L_1 = cp.linalg.cholesky(cov_1)

    rng = cp.random.default_rng(seed)

    # Draw balanced random labels {0, 1}
    labels = rng.integers(0, 2, size=n_samples)

    # Compute blob centres along I-axis (rotation applied globally later)
    separation = 2 * snr * sigma
    mu_0 = cp.array([-separation / 2.0, 0.0])
    mu_1 = cp.array([ separation / 2.0, 0.0])

    # Sample IQ points — CuPy Generator lacks multivariate_normal,
    # so we draw standard normals and transform via Cholesky: x = mu + L @ z.
    iq = cp.zeros((n_samples, 2))

    mask_0 = (labels == 0)
    mask_1 = (labels == 1)

    n_0 = int(cp.sum(mask_0))
    n_1 = int(cp.sum(mask_1))

    if n_0 > 0:
        z_0 = rng.standard_normal((n_0, 2))
        iq[mask_0] = z_0 @ L_0.T + mu_0
    if n_1 > 0:
        z_1 = rng.standard_normal((n_1, 2))
        iq[mask_1] = z_1 @ L_1.T + mu_1

    # --- T1 decay during measurement ---
    # Flip probability: p_t1 = 1 - exp(-t_meas / T1).
    # Since the parameter stored in config is T1_over_tmeas = T1 / t_meas,
    # this is equivalent to 1 - exp(-1 / T1_over_tmeas).
    # For each |1⟩-labelled point, draw a uniform random number; if it falls
    # below p_t1, replace the IQ point with a new sample from the |0⟩ blob.
    # The label is kept as 1 (the ground-truth prepared state) but the IQ
    # coordinates now look like |0⟩.  This models a qubit that decayed from
    # |1⟩ to |0⟩ during the measurement integration window.
    p_t1 = 1.0 - cp.exp(-1.0 / t1_over_tmeas)
    decay_mask = mask_1 & (rng.random(n_samples) < p_t1)
    n_decay = int(cp.sum(decay_mask))
    if n_decay > 0:
        z_decay = rng.standard_normal((n_decay, 2))
        iq[decay_mask] = z_decay @ L_0.T + mu_0

    # Apply thermal excitation by re-drawing a fraction of |0⟩ from |1⟩ distribution
    thermal_mask = mask_0 & (rng.random(n_samples) < p_thermal)
    n_thermal = int(cp.sum(thermal_mask))
    if n_thermal > 0:
        z_thermal = rng.standard_normal((n_thermal, 2))
        iq[thermal_mask] = z_thermal @ L_1.T + mu_1

    # Apply global rotation R(blob_angle) to all IQ points (see CPU docstring).
    if blob_angle != 0.0:
        cos_a, sin_a = float(cp.cos(blob_angle)), float(cp.sin(blob_angle))
        rot = cp.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        iq = iq @ rot.T

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