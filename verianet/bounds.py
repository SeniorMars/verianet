"""Interval bound propagation utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .activations import gelu, gelu_bounds

Array = np.ndarray


def ibp_affine(L: Array, U: Array, W: Array, b: Array) -> tuple[Array, Array]:
    """
    Exact IBP bounds for y = W @ x + b.

    W is expected in mathematical shape (out_dim, in_dim).
    """
    L_arr = np.asarray(L, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    Wpos = np.maximum(W_arr, 0.0)
    Wneg = np.minimum(W_arr, 0.0)
    return Wpos @ L_arr + Wneg @ U_arr + b_arr, Wpos @ U_arr + Wneg @ L_arr + b_arr


def ibp_affine_keras(L: Array, U: Array, W: Array, b: Array) -> tuple[Array, Array]:
    """
    Exact IBP bounds for y = x @ W + b.

    W is expected in Keras Dense shape (in_dim, out_dim).
    """
    L_arr = np.asarray(L, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    Wpos = np.maximum(W_arr, 0.0)
    Wneg = np.minimum(W_arr, 0.0)
    return L_arr @ Wpos + U_arr @ Wneg + b_arr, U_arr @ Wpos + L_arr @ Wneg + b_arr


def _is_gelu(f: Callable | str) -> bool:
    if f == "gelu":
        return True
    return callable(f) and getattr(f, "__name__", "") == "gelu"


def ibp_activation(
    L: Array,
    U: Array,
    f: Callable[[Array], Array] | str = gelu,
    *,
    grid_points: int = 1025,
) -> tuple[Array, Array]:
    """
    Activation bounds for interval propagation.

    GELU uses exact scalar bounds. Other activations fall back to dense sampling,
    which is useful for exploration but is not a formal certificate.
    """
    L_arr = np.asarray(L, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)
    if np.any(L_arr > U_arr):
        raise ValueError("activation lower bounds must be <= upper bounds")

    if _is_gelu(f):
        return gelu_bounds(L_arr, U_arr)

    if not callable(f):
        raise ValueError(f"unknown activation: {f!r}")

    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")

    t = np.linspace(0.0, 1.0, grid_points, dtype=np.float64)
    xs = L_arr[:, None] + (U_arr - L_arr)[:, None] * t[None, :]
    vals = np.asarray(f(xs), dtype=np.float64)
    return vals.min(axis=1), vals.max(axis=1)
