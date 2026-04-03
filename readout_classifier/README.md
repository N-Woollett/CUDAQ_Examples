# Readout Classifier

**A hands-on training project for learning NVIDIA CUDA-Q through a physically motivated, solvable toy problem.**

## What This Project Is

This project builds a variational quantum circuit (VQC) that classifies synthetic qubit readout data — distinguishing the |0⟩ ground state from the |1⟩ excited state using simulated IQ (in-phase and quadrature) voltage measurements from a dispersive readout system. It is implemented entirely within the NVIDIA CUDA-Q framework, with GPU acceleration via CuPy for data generation and cuQuantum-backed simulation for the quantum classifier.

The project is **deliberately built around a problem that classical methods already solve optimally.** For two symmetric Gaussian blobs in a 2D plane, Linear Discriminant Analysis (LDA) is provably Bayes-optimal and runs in microseconds. The variational quantum circuit will, at best, match LDA's accuracy. This is by design.

## Why a Solvable Toy Model?

The educational value of working against a known-optimal baseline is that it gives you a built-in sanity check at every stage of development. If your VQC cannot match LDA on clean Gaussian blobs, something is wrong with your kernel construction, cost function, or training loop — not with the data. The physics provides the ground truth, and the classical baseline tells you exactly how well you should be doing.

This makes the project ideal as a training exercise because you can focus on learning the CUDA-Q programming model without simultaneously debugging an unsolved research problem. Specifically, by working through this project you will gain practical experience with:

- **`@cudaq.kernel`** — defining parameterised quantum circuits as JIT-compiled Python functions
- **`cudaq.observe()`** — computing expectation values of spin operators with respect to quantum states
- **`cudaq.optimizers`** — using built-in gradient-free (COBYLA) and gradient-based optimisation
- **`cudaq.gradients`** — implementing the parameter-shift rule for exact quantum gradients
- **`cudaq.NoiseModel`** — composing Kraus channels (depolarisation, amplitude damping, custom readout error) into realistic noise models
- **Backend switching** — moving between CPU (`qpp-cpu`), single-GPU (`nvidia`), and multi-GPU (`nvidia` with MQPU) targets
- **CuPy integration** — GPU-accelerated classical data generation alongside quantum simulation

These are the same primitives required for any serious CUDA-Q application. Here, you learn them in a context where you can validate every intermediate result.

## Where It Gets Interesting

While the base problem is intentionally simple, the architecture is designed to scale into regimes where the quantum classifier may offer genuine advantages:

- **Multi-qubit readout with crosstalk** — simultaneous measurement of multiple qubits through a shared feedline creates a 2ⁿ-class problem in 2n-dimensional IQ space with non-linear class boundaries, breaking LDA's optimality assumptions.
- **Non-Gaussian blob shapes** — readout-induced state transitions, multi-level transmon effects, and strong T1 asymmetry produce distributions that no linear classifier can handle optimally.
- **Time-resolved trajectory classification** — feeding full IQ time traces (rather than single integrated points) into the classifier via data re-uploading creates a high-dimensional input where the quantum feature map's exponential Hilbert space embedding becomes expressive.

The CUDA-Q infrastructure you build here — the training loop, noise models, evaluation pipeline — carries directly into these harder problems without re-architecting.

## Project Structure

```
qubit-readout-classifier/
├── README.md
├── pyproject.toml
├── config/
│   ├── default_params.json          # Physical + training parameters
│   └── noise_profiles/
│       ├── ideal.json
│       ├── moderate_noise.json
│       └── high_noise.json
├── src/
│   ├── __init__.py
│   ├── iq_simulator.py              # Stage 1: Synthetic IQ data generation
│   ├── preprocessor.py              # Stage 2: Normalisation & angle encoding
│   ├── vqc_classifier.py            # Stage 3: CUDA-Q variational classifier
│   ├── classical_baselines.py       # LDA, threshold, MLP for comparison
│   ├── evaluator.py                 # Stage 4: Metrics & visualisation
│   └── utils.py
├── notebooks/
│   ├── 01_iq_data_exploration.ipynb  # Visualise IQ blobs and noise effects
│   ├── 02_vqc_training.ipynb        # Train the variational classifier
│   └── 03_evaluation.ipynb          # Compare VQC against classical baselines
├── tests/
│   ├── test_iq_simulator.py
│   ├── test_preprocessor.py
│   ├── test_vqc_classifier.py
│   ├── test_evaluator.py
│   └── test_integration.py          # End-to-end pipeline test
└── docs/
    ├── design_document.docx
    └── implementation_plan.docx
```

## Pipeline Overview

The system runs as a four-stage pipeline:

1. **IQ Data Synthesis** — generates realistic single-shot dispersive readout data from a Gaussian mixture model with configurable SNR, T1 decay, thermal excitation, and blob geometry. GPU-accelerated via CuPy for large datasets.

2. **Preprocessing** — standardises the raw IQ data, optionally applies PCA rotation to align the blob axis, and maps features to rotation angles via tanh squashing for quantum circuit encoding.

3. **Variational Quantum Classifier** — a 3-qubit, 2-layer CUDA-Q kernel that angle-encodes the IQ point into a quantum state and applies a trainable hardware-efficient ansatz. The readout qubit's ⟨Z⟩ expectation value determines the classification. Trained via mini-batch optimisation using COBYLA or L-BFGS-B with parameter-shift gradients.

4. **Evaluation** — produces confusion matrices, ROC curves, assignment fidelity, and a head-to-head comparison table against threshold, LDA, and MLP classifiers on the same test data.

## Environment Setup

This project uses Conda for environment management. An `environment.yml` file is provided to recreate the necessary environment.

### Prerequisites

Ensure you have Conda installed (e.g., Anaconda or Miniconda).

### Setup Instructions

1. **Navigate to the project directory:**
   ```bash
   cd readout_classifier
   ```

2. **Create the Conda environment:**
   Use the provided `environment.yml` to create a new environment named `readout_classifier_demo`.
   ```bash
   conda env create -f environment.yml -n readout_classifier_demo
   ```

3. **Activate the environment:**
   ```bash
   conda activate readout_classifier_demo
   ```

4. **Verify the installation (Optional):**
   You can verify the environment was created and activated successfully by running:
   ```bash
   conda info --envs
   ```
   The `readout_classifier_demo` environment should be listed and have an asterisk (`*`) next to it indicating it is currently active.


### Quickstart

```bash
# Clone and install
git clone <repo-url>
cd qubit-readout-classifier
pip install -e .

# Generate synthetic IQ data
python -m src.iq_simulator --config config/default_params.json --output data/

# Preprocess
python -m src.preprocessor --input data/iq_data.npz --output data/preprocessed.npz

# Train the VQC
python -m src.vqc_classifier --data data/preprocessed.npz --epochs 20 --optimizer cobyla

# Evaluate against classical baselines
python -m src.evaluator --model checkpoints/best.npz --data data/preprocessed.npz
```

Or work through the notebooks in order for a guided experience.

## Expected Results

On default parameters (SNR = 3.0, T1/t_meas = 10.0, 2% thermal excitation):

| Classifier | Accuracy | Assignment Fidelity | AUC  |
|-----------|----------|-------------------|------|
| Threshold | ~82%     | ~0.81              | ~0.89 |
| LDA       | ~85%     | ~0.84              | ~0.92 |
| MLP       | ~85%     | ~0.84              | ~0.92 |
| **VQC**   | ~83%     | ~0.82              | ~0.91 |

The VQC is expected to approach but not exceed LDA on this Gaussian data. The value is in the learning, not the leaderboard.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| cuda-quantum | ≥ 0.9 | Quantum kernels, simulation, variational algorithms |
| cupy-cuda12x | ≥ 13.0 | GPU-accelerated data generation |
| numpy | ≥ 1.24 | Array operations, CPU fallback |
| scipy | ≥ 1.11 | L-BFGS-B optimiser |
| scikit-learn | ≥ 1.3 | PCA, LDA, metrics, train/test splitting |
| matplotlib | ≥ 3.8 | Evaluation plots |
| pytest | ≥ 7.0 | Testing |

A GPU with CUDA 12.x and Compute Capability 7.0+ is recommended. The project falls back to CPU backends (`qpp-cpu` for CUDA-Q, NumPy for data generation) if no GPU is available.

## Further Reading

The `docs/` directory contains a full design document covering the dispersive readout physics, CUDA-Q architecture decisions, and a detailed implementation plan with per-task verification tests. Key references:

- Blais, A. et al., *Phys. Rev. A* 69, 062320 (2004) — Circuit QED architecture
- Lienhard, B. et al., *Phys. Rev. Applied* 17, 014024 (2022) — DNN qubit-state discrimination
- Cerezo, M. et al., *Nat. Rev. Phys.* 3, 625 (2021) — Variational quantum algorithms
- NVIDIA CUDA-Q Documentation: [nvidia.github.io/cuda-quantum](https://nvidia.github.io/cuda-quantum/latest/)


