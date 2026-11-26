"""Validation of per-vertex standard errors for the parametric bootstrap."""

from __future__ import annotations

import math

import numpy as np

from topographer.graph import ScalarNetwork

__all__ = ["validate_se_attr"]


def validate_se_attr(network: ScalarNetwork, se_attr: str = "se") -> np.ndarray:
    """Validate and extract vertex standard error attributes from a ScalarNetwork.

    Parameters
    ----------
    network : ScalarNetwork
        Validated scalar network instance.
    se_attr : str, default "se"
        Name of the vertex attribute containing standard errors.

    Returns
    -------
    np.ndarray
        1D NumPy float64 array of vertex standard errors of shape (num_nodes,).

    Raises
    ------
    TypeError
        If network is not a ScalarNetwork instance or se_attr is not a string.
    ValueError
        If se_attr is empty, missing from any vertex payload, or contains non-numeric,
        non-finite, NaN, null, or negative (< 0.0) values.

    """
    if not isinstance(network, ScalarNetwork):
        raise TypeError(f"Expected ScalarNetwork instance, got {type(network).__name__}.")
    if not isinstance(se_attr, str) or not se_attr:
        raise ValueError("se_attr must be a non-empty string.")

    vertex_count = network.vertex_count
    se_values = np.empty(vertex_count, dtype=np.float64)

    for i, idx in enumerate(network._graph_snapshot.node_indices()):
        payload = network._graph_snapshot.get_node_data(idx)
        if payload is None or se_attr not in payload:
            raise ValueError(f"Standard error attribute {se_attr!r} not found on vertex payload.")
        val = payload[se_attr]
        if val is None or isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(
                f"Standard error value must be numeric and not None, got {type(val).__name__}."
            )
        if not math.isfinite(val):
            raise ValueError(f"Standard error value {val} must be finite.")
        f_val = float(val)
        if f_val < 0.0:
            raise ValueError(f"Standard error value {f_val} cannot be negative.")
        se_values[i] = f_val

    return se_values
