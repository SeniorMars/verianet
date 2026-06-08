"""Explicit H-representation LP builder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import linprog

Array = np.ndarray
Line = tuple[float, float, float, float]
Envelope = tuple[list[Line], list[Line]]
EnvelopeBuilder = Callable[[float, float], Envelope]


def envelope_to_constraints(lower_lines: list[Line], upper_lines: list[Line]) -> tuple[Array, Array]:
    """Convert z >= m*a+c and z <= m*a+c lines into A @ [a,z] <= b."""
    rows: list[list[float]] = []
    rhs: list[float] = []

    for m, c, _, _ in lower_lines:
        rows.append([float(m), -1.0])
        rhs.append(-float(c))

    for m, c, _, _ in upper_lines:
        rows.append([-float(m), 1.0])
        rhs.append(float(c))

    if not rows:
        return np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64)


def bounds_from_envelope(lower_lines: list[Line], upper_lines: list[Line], L: float, U: float) -> tuple[float, float]:
    """
    Coarse scalar z bounds implied by an activation envelope over a in [L, U].
    """
    if lower_lines:
        z_lb = max(min(m * L + c, m * U + c) for m, c, _, _ in lower_lines)
    else:
        z_lb = -np.inf

    if upper_lines:
        z_ub = min(max(m * L + c, m * U + c) for m, c, _, _ in upper_lines)
    else:
        z_ub = np.inf

    return float(z_lb), float(z_ub)


@dataclass
class OptimizeResult:
    value: float | None
    raw: object


class ScipyPolytopeAnalyzer:
    """
    Builds a single global linear system A @ x <= b.

    Dense-layer weights use Keras shape (in_dim, out_dim), matching the saved
    project weights and TensorFlow forward pass.
    """

    def __init__(self) -> None:
        self.A = np.zeros((0, 0), dtype=np.float64)
        self.b = np.zeros(0, dtype=np.float64)
        self.var_slices: dict[str, slice] = {}
        self.nvars = 0

    def _alloc(self, name: str, dim: int) -> slice:
        if name in self.var_slices:
            raise ValueError(f"variable block already exists: {name}")
        sl = slice(self.nvars, self.nvars + dim)
        self.var_slices[name] = sl

        if self.A.size == 0:
            self.A = np.zeros((0, self.nvars + dim), dtype=np.float64)
        else:
            self.A = np.hstack([self.A, np.zeros((self.A.shape[0], dim), dtype=np.float64)])
        self.nvars += dim
        return sl

    def add_input_box(self, name: str, lb: Array, ub: Array) -> slice:
        lb_arr = np.asarray(lb, dtype=np.float64)
        ub_arr = np.asarray(ub, dtype=np.float64)
        if np.any(lb_arr > ub_arr):
            raise ValueError("input lower bounds must be <= upper bounds")

        sl = self._alloc(name, lb_arr.size)
        A_local = np.vstack([np.eye(lb_arr.size), -np.eye(lb_arr.size)])
        b_local = np.concatenate([ub_arr, -lb_arr])
        self.add_block_constraints(sl, A_local, b_local)
        return sl

    def add_affine(self, in_name: str, W: Array, bias: Array, out_name: str) -> slice:
        W_arr = np.asarray(W, dtype=np.float64)
        bias_arr = np.asarray(bias, dtype=np.float64)
        in_sl = self.var_slices[in_name]
        out_sl = self._alloc(out_name, bias_arr.size)

        if W_arr.shape != (in_sl.stop - in_sl.start, bias_arr.size):
            raise ValueError(
                f"{out_name}: expected W shape {(in_sl.stop - in_sl.start, bias_arr.size)}, got {W_arr.shape}"
            )

        Aeq = np.zeros((bias_arr.size, self.nvars), dtype=np.float64)
        Aeq[:, out_sl] = np.eye(bias_arr.size)
        Aeq[:, in_sl] -= W_arr.T
        self.A = np.vstack([self.A, Aeq, -Aeq])
        self.b = np.concatenate([self.b, bias_arr, -bias_arr])
        return out_sl

    def add_activation(
        self,
        pre_name: str,
        post_name: str,
        bounds: tuple[Array, Array],
        builder: EnvelopeBuilder,
    ) -> slice:
        pre_sl = self.var_slices[pre_name]
        dim = pre_sl.stop - pre_sl.start
        post_sl = self._alloc(post_name, dim)
        L, U = (np.asarray(bounds[0], dtype=np.float64), np.asarray(bounds[1], dtype=np.float64))
        if L.size != dim or U.size != dim:
            raise ValueError(f"{post_name}: activation bounds must have length {dim}")

        # Enforce pre-activation bounds because envelopes are only valid there.
        A_bounds = np.zeros((2 * dim, self.nvars), dtype=np.float64)
        b_bounds = np.zeros(2 * dim, dtype=np.float64)
        for i in range(dim):
            A_bounds[i, pre_sl.start + i] = 1.0
            b_bounds[i] = U[i]
            A_bounds[dim + i, pre_sl.start + i] = -1.0
            b_bounds[dim + i] = -L[i]
        self.A = np.vstack([self.A, A_bounds])
        self.b = np.concatenate([self.b, b_bounds])

        rows: list[Array] = []
        rhs: list[Array] = []
        z_lb = np.zeros(dim, dtype=np.float64)
        z_ub = np.zeros(dim, dtype=np.float64)

        for i in range(dim):
            lower_lines, upper_lines = builder(float(L[i]), float(U[i]))
            z_lb[i], z_ub[i] = bounds_from_envelope(lower_lines, upper_lines, float(L[i]), float(U[i]))
            Ai, bi = envelope_to_constraints(lower_lines, upper_lines)
            if Ai.size == 0:
                continue
            block = np.zeros((Ai.shape[0], self.nvars), dtype=np.float64)
            block[:, pre_sl.start + i] = Ai[:, 0]
            block[:, post_sl.start + i] = Ai[:, 1]
            rows.append(block)
            rhs.append(bi)

        self.add_block_constraints(post_sl, np.vstack([np.eye(dim), -np.eye(dim)]), np.concatenate([z_ub, -z_lb]))

        if rows:
            self.A = np.vstack([self.A, *rows])
            self.b = np.concatenate([self.b, *rhs])
        return post_sl

    def add_scalar(self, name: str, lower: float | None = None, upper: float | None = None) -> int:
        sl = self._alloc(name, 1)
        rows: list[Array] = []
        rhs: list[float] = []
        if upper is not None:
            row = np.zeros(self.nvars, dtype=np.float64)
            row[sl.start] = 1.0
            rows.append(row)
            rhs.append(float(upper))
        if lower is not None:
            row = np.zeros(self.nvars, dtype=np.float64)
            row[sl.start] = -1.0
            rows.append(row)
            rhs.append(-float(lower))
        if rows:
            self.A = np.vstack([self.A, *rows])
            self.b = np.concatenate([self.b, np.asarray(rhs, dtype=np.float64)])
        return sl.start

    def add_linear_constraint(self, coeffs: Array, rhs: float) -> None:
        coeffs_arr = np.asarray(coeffs, dtype=np.float64)
        if coeffs_arr.size != self.nvars:
            raise ValueError(f"constraint length {coeffs_arr.size} does not match nvars {self.nvars}")
        self.A = np.vstack([self.A, coeffs_arr.reshape(1, -1)])
        self.b = np.concatenate([self.b, np.array([float(rhs)], dtype=np.float64)])

    def add_block_constraints(self, sl: slice, A_local: Array, b_local: Array) -> None:
        block = np.zeros((A_local.shape[0], self.nvars), dtype=np.float64)
        block[:, sl] = np.asarray(A_local, dtype=np.float64)
        self.A = np.vstack([self.A, block])
        self.b = np.concatenate([self.b, np.asarray(b_local, dtype=np.float64)])

    def objective(self, block_name: str, coeffs: Array) -> Array:
        c = np.zeros(self.nvars, dtype=np.float64)
        sl = self.var_slices[block_name]
        coeffs_arr = np.asarray(coeffs, dtype=np.float64)
        if coeffs_arr.size != sl.stop - sl.start:
            raise ValueError(f"objective length must match block {block_name}")
        c[sl] = coeffs_arr
        return c

    def optimize(self, c: Array, sense: str = "min") -> OptimizeResult:
        c_arr = np.asarray(c, dtype=np.float64)
        if c_arr.size != self.nvars:
            raise ValueError(f"objective length {c_arr.size} does not match nvars {self.nvars}")
        if sense not in {"min", "max"}:
            raise ValueError("sense must be 'min' or 'max'")
        lp_c = -c_arr if sense == "max" else c_arr
        res = linprog(lp_c, A_ub=self.A, b_ub=self.b, bounds=(None, None), method="highs")
        if not res.success:
            return OptimizeResult(None, res)
        value = -res.fun if sense == "max" else res.fun
        return OptimizeResult(float(value), res)

    def values(self, result: OptimizeResult, block_name: str) -> Array:
        if result.value is None or not hasattr(result.raw, "x"):
            raise ValueError("no optimal solution available")
        return np.asarray(result.raw.x[self.var_slices[block_name]], dtype=np.float64)
