"""Preprocessing utilities for IQ readout data."""

from typing import NamedTuple, Optional

import numpy as np


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

    scaled = (iq_data - scaler.mu) / scaler.sigma
    return scaled, scaler
