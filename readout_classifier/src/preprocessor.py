"""Preprocessing utilities for IQ readout data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Optional, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class StandardScaler(NamedTuple):
    """Stores the per-feature mean and standard deviation used by
    :func:`standardise` so the same transform can be re-applied to
    validation / test data without data leakage.

    Attributes:
        mu (np.ndarray):    Per-feature means, shape ``(n_features,)``.
        sigma (np.ndarray): Per-feature standard deviations, shape
            ``(n_features,)``.
    """
    mu: np.ndarray
    sigma: np.ndarray


@dataclass(frozen=True)
class PipelineParams:
    """Typed configuration for :func:`preprocess_pipeline`.

    Using a dataclass instead of a plain dict prevents silent
    fallback to defaults when a key is misspelled.

    Attributes:
        split_ratios: Train/val/test split ratios.
        seed: Random seed for reproducibility.
        use_pca: Whether to apply PCA rotation.
        pca_components: Number of PCA components if used.
    """
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
    seed: int = 42
    use_pca: bool = False
    pca_components: int = 2


class PipelineResult(NamedTuple):
    """Typed return value for :func:`preprocess_pipeline`.

    Attributes:
        train: (encoded_train_iq, train_labels)
        val: (encoded_val_iq, val_labels)
        test: (encoded_test_iq, test_labels)
        scaler: The fitted StandardScaler.
        pca: The fitted PCA object (None if not used).
    """
    train: tuple[np.ndarray, np.ndarray]
    val: tuple[np.ndarray, np.ndarray]
    test: tuple[np.ndarray, np.ndarray]
    scaler: StandardScaler
    pca: Optional[PCA]


def standardise(
    iq_data: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Z-score standardise IQ data to zero mean and unit variance.

    When ``scaler`` is *None* (the default), the per-feature mean and
    standard deviation are computed from ``iq_data`` itself — this is the
    mode used for the **training** set.  When a ``scaler`` is provided
    (obtained from a previous call on the training set), those statistics
    are re-used so that validation / test data undergo the *same*
    transform, preventing data leakage.

    .. math::

        x_{\\text{scaled}} = \\frac{x - \\mu}{\\sigma}

    Args:
        iq_data (np.ndarray): IQ samples with shape ``(N, n_features)``
            (typically ``(N, 2)`` for I and Q channels).
        scaler (StandardScaler | None): If *None*, fit-and-transform
            (training mode).  If provided, transform only (inference
            mode).

    Returns:
        tuple[np.ndarray, StandardScaler]: A tuple of:
            - **scaled** — the standardised data, same shape as input.
            - **scaler** — the :class:`StandardScaler` holding ``(mu,
              sigma)`` used for the transform.

    Raises:
        ValueError: If ``iq_data`` is not a 2-D array.
        ValueError: If any per-feature standard deviation is zero (i.e.
            a constant feature), which would cause a division-by-zero.
    """
    if iq_data.ndim != 2:
        raise ValueError(
            f"iq_data must be 2-D (N, n_features), got shape {iq_data.shape}"
        )

    if scaler is None:
        mu = np.mean(iq_data, axis=0)
        sigma = np.std(iq_data, axis=0)

        if np.any(sigma == 0.0):
            raise ValueError(
                "At least one feature has zero standard deviation — "
                "cannot standardise a constant feature."
            )

        scaler = StandardScaler(mu=mu, sigma=sigma)
    elif np.any(scaler.sigma == 0.0):
        raise ValueError(
            "Provided scaler has zero sigma — cannot standardise."
        )

    scaled = (iq_data - scaler.mu) / scaler.sigma
    return scaled, scaler


def angle_encode(iq_scaled: np.ndarray) -> np.ndarray:
    """Map standardised IQ values to rotation angles in ``(-π, π)``.

    Applies the element-wise transform:

    .. math::

        \\theta = \\pi \\, \\tanh(x)

    The hyperbolic tangent saturates smoothly at ±1, so the resulting
    angles are bounded in the open interval ``(-π, π)`` regardless of
    input magnitude.  This avoids the aliasing problems of a simple
    linear map (where values outside ``[-π, π]`` would wrap) while
    preserving the sign and relative magnitude of the standardised
    features.

    This encoding is designed to feed directly into single-qubit
    rotation gates (e.g. ``Ry(θ)``) in a variational quantum circuit.

    Args:
        iq_scaled (np.ndarray): Standardised IQ data with shape
            ``(N, n_features)``.  Typically the output of
            :func:`standardise`.

    Returns:
        np.ndarray: Angle-encoded data with the same shape as the
        input, with all values in ``(-π, π)``.

    Raises:
        ValueError: If ``iq_scaled`` is not a 2-D array.
    """
    if iq_scaled.ndim != 2:
        raise ValueError(
            f"iq_scaled must be 2-D (N, n_features), got shape {iq_scaled.shape}"
        )

    return np.pi * np.tanh(iq_scaled)


def pca_rotate(
    iq_data: np.ndarray,
    pca: Optional[PCA] = None,
    n_components: int = 2
) -> tuple[np.ndarray, PCA]:
    """Apply PCA rotation to IQ data to align the maximum-variance axis with I.

    When ``pca`` is *None* (the default), a PCA object is fitted on the
    ``iq_data`` — this is the mode used for the **training** set. When a
    ``pca`` is provided (obtained from a previous call on the training set),
    that object is re-used to transform the validation / test data, preventing
    data leakage.

    Args:
        iq_data (np.ndarray): IQ samples with shape ``(N, n_features)``.
        pca (PCA | None): If *None*, fit-and-transform
            (training mode). If provided, transform only (inference
            mode).
        n_components (int): Number of components to keep. Defaults to 2.

    Returns:
        tuple[np.ndarray, PCA]: A tuple of:
            - **rotated** — the PCA-transformed data, shape ``(N, n_components)``.
            - **pca** — the fitted :class:`PCA` object.

    Raises:
        ValueError: If ``iq_data`` is not a 2-D array.
    """
    if iq_data.ndim != 2:
        raise ValueError(
            f"iq_data must be 2-D (N, n_features), got shape {iq_data.shape}"
        )

    if pca is None:
        pca = PCA(n_components=n_components)
        rotated = pca.fit_transform(iq_data)
    else:
        if pca.n_components_ != n_components:
            raise ValueError(
                f"Provided PCA has {pca.n_components_} components, "
                f"but n_components={n_components} was requested."
            )
        rotated = pca.transform(iq_data)

    return rotated, pca


def split_dataset(
    iq: np.ndarray,
    labels: np.ndarray,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Split dataset into train, validation and test sets.

    Args:
        iq (np.ndarray): The IQ data array.
        labels (np.ndarray): The ground truth labels.
        ratios (tuple[float, float, float]): The ratios for (train, val, test).
            Must sum to 1.0. Defaults to (0.7, 0.15, 0.15).
        seed (int): The random seed for the split.

    Returns:
        tuple: A tuple containing:
            - (train_iq, train_labels)
            - (val_iq, val_labels)
            - (test_iq, test_labels)

    Raises:
        ValueError: If ratios do not sum to 1.0.
    """
    train_r, val_r, test_r = ratios
    if any(r <= 0 for r in ratios):
        raise ValueError(f"All ratios must be positive, got {ratios}")
    if not np.isclose(train_r + val_r + test_r, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    # First split into train and (val + test)
    val_test_r = val_r + test_r
    train_iq, temp_iq, train_labels, temp_labels = train_test_split(
        iq, labels, test_size=val_test_r, random_state=seed, stratify=labels
    )

    # Then split temp into val and test
    test_rel_r = test_r / val_test_r
    val_iq, test_iq, val_labels, test_labels = train_test_split(
        temp_iq, temp_labels, test_size=test_rel_r, random_state=seed, stratify=temp_labels
    )

    return (train_iq, train_labels), (val_iq, val_labels), (test_iq, test_labels)


def preprocess_pipeline(
    raw_iq: np.ndarray,
    raw_labels: np.ndarray,
    params: PipelineParams | dict[str, Any],
) -> PipelineResult:
    """Execute the full preprocessing pipeline on raw IQ data.

    Chains the following steps:
      1. Split dataset into train, validation, and test sets.
      2. Standardise (fit on train, transform val/test).
      3. Optionally apply PCA rotation (fit on train, transform val/test).
      4. Angle encode mapped to (-π, π).

    Args:
        raw_iq (np.ndarray): Raw IQ samples.
        raw_labels (np.ndarray): Ground truth labels.
        params (PipelineParams | dict): Configuration. When a dict is
            passed it is converted to a :class:`PipelineParams` — any
            unrecognised keys will raise a ``TypeError``.

    Returns:
        PipelineResult: A named tuple containing train, val, test splits,
            the fitted scaler, and the fitted PCA object (None if unused).
    """
    if isinstance(params, dict):
        params = PipelineParams(**params)

    # 1. Split
    train_split, val_split, test_split = split_dataset(
        raw_iq, raw_labels, ratios=params.split_ratios, seed=params.seed
    )
    train_iq, train_labels = train_split
    val_iq, val_labels = val_split
    test_iq, test_labels = test_split

    # 2. Standardise
    scaled_train, scaler = standardise(train_iq)
    scaled_val, _ = standardise(val_iq, scaler=scaler)
    scaled_test, _ = standardise(test_iq, scaler=scaler)

    # 3. PCA (optional)
    pca = None
    if params.use_pca:
        scaled_train, pca = pca_rotate(
            scaled_train, n_components=params.pca_components
        )
        scaled_val, _ = pca_rotate(scaled_val, pca=pca)
        scaled_test, _ = pca_rotate(scaled_test, pca=pca)

    # 4. Angle Encode
    enc_train = angle_encode(scaled_train)
    enc_val = angle_encode(scaled_val)
    enc_test = angle_encode(scaled_test)

    return PipelineResult(
        train=(enc_train, train_labels),
        val=(enc_val, val_labels),
        test=(enc_test, test_labels),
        scaler=scaler,
        pca=pca,
    )


def save_pipeline(
    path: Union[str, Path],
    scaler: StandardScaler,
    pca: Optional[PCA] = None,
) -> None:
    """Persist the fitted preprocessing transforms to disk.

    Saves the scaler and (optionally) the PCA object so that
    :func:`standardise`, :func:`pca_rotate`, and :func:`angle_encode`
    can be re-applied to new data without re-fitting on the training
    set.

    Args:
        path (str | Path): Destination file path.  The ``.joblib``
            extension is recommended but not enforced.
        scaler (StandardScaler): The fitted scaler from the training
            run.
        pca (PCA | None): The fitted PCA object, or *None* if PCA was
            not used.
    """
    joblib.dump({"scaler": scaler, "pca": pca}, path)


def load_pipeline(
    path: Union[str, Path],
) -> tuple[StandardScaler, Optional[PCA]]:
    """Load previously saved preprocessing transforms from disk.

    Args:
        path (str | Path): Path to the file written by
            :func:`save_pipeline`.

    Returns:
        tuple[StandardScaler, PCA | None]: The fitted scaler and PCA
        object (or *None* if PCA was not used when the pipeline was
        saved).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If the file does not contain the expected keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: '{path}'")

    blob = joblib.load(path)

    if "scaler" not in blob:
        raise KeyError("File does not contain a 'scaler' entry")

    return blob["scaler"], blob.get("pca")
