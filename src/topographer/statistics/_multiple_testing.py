"""Statistical functions and multiple testing adjustments."""

from __future__ import annotations

import numpy as np


def compute_bh_qvalues(pvalues: np.ndarray | list[float]) -> np.ndarray:
    """Compute Benjamini-Hochberg false discovery rate (FDR) q-values.

    Parameters
    ----------
    pvalues : np.ndarray or list of float
        1D array or sequence of p-values in the interval [0, 1].

    Returns
    -------
    np.ndarray
        1D array of Benjamini-Hochberg adjusted q-values in [0, 1].

    Raises
    ------
    ValueError
        If pvalues is not a 1D sequence or contains non-finite values or values outside [0, 1].

    Examples
    --------
    >>> from topographer.statistics import compute_bh_qvalues
    >>> [round(float(q), 4) for q in compute_bh_qvalues([0.01, 0.04, 0.03, 0.2])]
    [0.04, 0.0533, 0.0533, 0.2]

    """
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1:
        raise ValueError("P-values must be provided as a 1D array.")
    if values.size == 0:
        return np.asarray([], dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("P-values must be finite numbers in the interval [0, 1].")

    n = values.size
    sort_order = np.argsort(values)
    sorted_p = values[sort_order]

    # Rank indices 1..n
    ranks = np.arange(1, n + 1, dtype=float)

    # BH adjusted value = p * n / rank
    q_sorted = sorted_p * n / ranks

    # Enforce monotonicity running backwards: q_i = min(q_i, q_{i+1})
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]

    # Cap at 1.0
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    # Revert to original input ordering
    qvalues = np.empty_like(q_sorted)
    qvalues[sort_order] = q_sorted

    return qvalues
