# Readout Classifier

This project provides a readout classifier demonstration.

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
