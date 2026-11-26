"""Unit tests for topographer.statistics multiple testing adjustments."""

from __future__ import annotations

import numpy as np
import pytest

from topographer.statistics import compute_bh_qvalues


def test_compute_bh_qvalues_known_values() -> None:
    # Standard p-values
    pvals = [0.01, 0.04, 0.03, 0.2]
    # Sorted p: [0.01, 0.03, 0.04, 0.2]
    # Ranks: [1, 2, 3, 4]
    # Raw q: [0.01*4/1, 0.03*4/2, 0.04*4/3, 0.2*4/4] = [0.04, 0.06, 0.05333333..., 0.2]
    # Monotonicity adjustment (from right to left):
    # q[3] = 0.2
    # q[2] = min(0.05333333, 0.2) = 0.05333333...
    # q[1] = min(0.06, 0.05333333...) = 0.05333333...
    # q[0] = min(0.04, 0.05333333...) = 0.04
    # Expected in original order [0.01, 0.04, 0.03, 0.2]:
    expected = np.array([0.04, 0.05333333333333334, 0.05333333333333334, 0.2])
    result = compute_bh_qvalues(pvals)
    np.testing.assert_allclose(result, expected)


def test_compute_bh_qvalues_empty() -> None:
    res = compute_bh_qvalues([])
    assert isinstance(res, np.ndarray)
    assert res.size == 0


def test_compute_bh_qvalues_all_zeros_and_ones() -> None:
    pvals = [0.0, 0.0, 1.0, 1.0]
    expected = np.array([0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(compute_bh_qvalues(pvals), expected)


def test_compute_bh_qvalues_clipping_at_one() -> None:
    pvals = [0.8, 0.9]
    # Sorted p: 0.8, 0.9
    # Ranks: 1, 2
    # Raw q: 0.8 * 2 / 1 = 1.6 -> clipped to 1.0; 0.9 * 2 / 2 = 0.9 -> min(0.9, 1.0) = 0.9
    expected = np.array([0.9, 0.9])
    np.testing.assert_allclose(compute_bh_qvalues(pvals), expected)


def test_compute_bh_qvalues_validation_ndim() -> None:
    with pytest.raises(ValueError, match="1D array"):
        compute_bh_qvalues([[0.01, 0.05], [0.02, 0.04]])


def test_compute_bh_qvalues_validation_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="interval \\[0, 1\\]"):
        compute_bh_qvalues([-0.1, 0.05])

    with pytest.raises(ValueError, match="interval \\[0, 1\\]"):
        compute_bh_qvalues([0.05, 1.1])


def test_compute_bh_qvalues_validation_nan_inf() -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        compute_bh_qvalues([0.01, np.nan, 0.05])

    with pytest.raises(ValueError, match="finite numbers"):
        compute_bh_qvalues([0.01, np.inf, 0.05])
