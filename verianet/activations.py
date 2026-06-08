"""Activation functions and certified scalar envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import erf

Array = np.ndarray
Line = tuple[float, float, float, float]
Envelope = tuple[list[Line], list[Line]]

SQRT_2 = float(np.sqrt(2.0))
SQRT_2_NEG = -SQRT_2

# Unique stationary point of exact GELU x * Phi(x), where GELU reaches
# its global minimum. This is the root of Phi(x) + x * phi(x) = 0.
GELU_MIN_X = -0.7517915246935645

BoundKind = Literal["lower", "upper"]


@dataclass(frozen=True)
class GeluLineCertificate:
    """Certificate that one affine line bounds GELU on an interval."""

    kind: BoundKind
    slope: float
    intercept: float
    interval: tuple[float, float]
    min_residual: float
    witness_x: float


def gelu(x: Array | float) -> Array:
    """Exact GELU, x * Phi(x), vectorized over NumPy arrays."""
    x_arr = np.asarray(x, dtype=np.float64)
    return 0.5 * x_arr * (1.0 + erf(x_arr / SQRT_2))


def gelu_derivative(x: Array | float) -> Array:
    """Derivative of exact GELU."""
    x_arr = np.asarray(x, dtype=np.float64)
    phi = np.exp(-0.5 * x_arr**2) / np.sqrt(2.0 * np.pi)
    Phi = 0.5 * (1.0 + erf(x_arr / SQRT_2))
    return Phi + x_arr * phi


def gelu_second_derivative(x: Array | float) -> Array:
    """Second derivative of exact GELU."""
    x_arr = np.asarray(x, dtype=np.float64)
    phi = np.exp(-0.5 * x_arr**2) / np.sqrt(2.0 * np.pi)
    return phi * (2.0 - x_arr**2)


def gelu_bounds(L: Array | Iterable[float], U: Array | Iterable[float]) -> tuple[Array, Array]:
    """
    Exact interval bounds for GELU on each interval [L_i, U_i].

    GELU has one global minimum at GELU_MIN_X and no finite local maximum.
    Therefore the maximum over a bounded interval is attained at an endpoint,
    and the minimum is the smaller endpoint value unless GELU_MIN_X lies inside.
    """
    L_arr = np.asarray(L, dtype=np.float64)
    U_arr = np.asarray(U, dtype=np.float64)
    if np.any(L_arr > U_arr):
        raise ValueError("activation lower bounds must be <= upper bounds")

    y_L = gelu(L_arr)
    y_U = gelu(U_arr)
    lower = np.minimum(y_L, y_U)
    upper = np.maximum(y_L, y_U)

    has_min = (L_arr <= GELU_MIN_X) & (GELU_MIN_X <= U_arr)
    if np.any(has_min):
        lower = np.where(has_min, np.minimum(lower, float(gelu(GELU_MIN_X))), lower)

    return lower.astype(np.float64), upper.astype(np.float64)


def gelu_tangent(x0: float) -> Line:
    """Affine tangent line to exact GELU at x0."""
    slope = float(gelu_derivative(x0))
    intercept = float(gelu(x0)) - slope * float(x0)
    return slope, intercept, float(x0), float(x0)


def gelu_secant(x0: float, x1: float) -> Line:
    """Affine secant line through exact GELU at x0 and x1."""
    x0 = float(x0)
    x1 = float(x1)
    if x0 == x1:
        return gelu_tangent(x0)
    y0 = float(gelu(x0))
    y1 = float(gelu(x1))
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    return slope, intercept, x0, x1


def _critical_points_for_slope(slope: float, L: float, U: float) -> list[float]:
    """
    Find all stationary points of GELU(x) - slope*x on [L, U].

    GELU' is monotone on each of (-inf, -sqrt(2)), (-sqrt(2), sqrt(2)),
    and (sqrt(2), inf), so each segment has at most one root.
    """
    if L > U:
        raise ValueError("interval lower bound must be <= upper bound")

    cuts = [float(L)]
    for cut in (SQRT_2_NEG, SQRT_2):
        if L < cut < U:
            cuts.append(float(cut))
    cuts.append(float(U))
    cuts = sorted(set(cuts))

    roots: list[float] = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        ga = float(gelu_derivative(a) - slope)
        gb = float(gelu_derivative(b) - slope)
        if abs(ga) <= 1e-12:
            roots.append(a)
        if abs(gb) <= 1e-12:
            roots.append(b)
        if ga * gb < 0.0:
            roots.append(float(brentq(lambda x: float(gelu_derivative(x) - slope), a, b)))

    deduped: list[float] = []
    for root in sorted(roots):
        if L - 1e-12 <= root <= U + 1e-12 and (
            not deduped or abs(root - deduped[-1]) > 1e-9
        ):
            deduped.append(min(max(root, L), U))
    return deduped


def certify_gelu_line(
    slope: float,
    intercept: float,
    L: float,
    U: float,
    kind: BoundKind,
    *,
    tol: float = 1e-10,
) -> GeluLineCertificate | None:
    """
    Certify that an affine line bounds GELU over [L, U].

    For lower lines we certify GELU(x) - (slope*x + intercept) >= 0.
    For upper lines we certify (slope*x + intercept) - GELU(x) >= 0.
    The minimum of each residual can only occur at endpoints or where
    GELU'(x) = slope, so checking those finitely many points is enough.
    """
    if kind not in {"lower", "upper"}:
        raise ValueError("kind must be 'lower' or 'upper'")
    if L > U:
        raise ValueError("activation lower bound must be <= upper bound")

    slope = float(slope)
    intercept = float(intercept)
    candidates = [float(L), float(U), *_critical_points_for_slope(slope, float(L), float(U))]

    def residual(x: float) -> float:
        line = slope * x + intercept
        fx = float(gelu(x))
        return fx - line if kind == "lower" else line - fx

    values = [(residual(x), x) for x in candidates]
    min_residual, witness_x = min(values, key=lambda pair: pair[0])
    if min_residual < -tol:
        return None
    return GeluLineCertificate(
        kind=kind,
        slope=slope,
        intercept=intercept,
        interval=(float(L), float(U)),
        min_residual=float(min_residual),
        witness_x=float(witness_x),
    )


def certify_gelu_envelope(
    lower_lines: list[Line],
    upper_lines: list[Line],
    L: float,
    U: float,
    *,
    tol: float = 1e-10,
) -> list[GeluLineCertificate]:
    """Certify every line in a GELU envelope over [L, U]."""
    certificates: list[GeluLineCertificate] = []
    for slope, intercept, _, _ in lower_lines:
        certificate = certify_gelu_line(slope, intercept, L, U, "lower", tol=tol)
        if certificate is None:
            raise ValueError(f"invalid GELU lower line: z >= {slope} * x + {intercept}")
        certificates.append(certificate)
    for slope, intercept, _, _ in upper_lines:
        certificate = certify_gelu_line(slope, intercept, L, U, "upper", tol=tol)
        if certificate is None:
            raise ValueError(f"invalid GELU upper line: z <= {slope} * x + {intercept}")
        certificates.append(certificate)
    return certificates


def _candidate_points(L: float, U: float, max_points: int) -> list[float]:
    width = U - L
    adaptive_points = min(max_points, max(5, int(np.ceil(width * 3.0)) + 3))
    points = list(np.linspace(L, U, adaptive_points))
    for point in (GELU_MIN_X, SQRT_2_NEG, 0.0, SQRT_2, 0.5 * (L + U)):
        if L <= point <= U:
            points.append(float(point))
    return sorted({round(float(point), 12) for point in points})


def _add_verified_line(
    lines: list[Line],
    line: Line,
    L: float,
    U: float,
    kind: BoundKind,
    *,
    tol: float,
) -> None:
    slope, intercept, _, _ = line
    if certify_gelu_line(slope, intercept, L, U, kind, tol=tol) is None:
        return
    for old_slope, old_intercept, _, _ in lines:
        if abs(old_slope - slope) <= 1e-10 and abs(old_intercept - intercept) <= 1e-10:
            return
    lines.append((float(slope), float(intercept), float(L), float(U)))


def sound_gelu_envelope(
    L: float,
    U: float,
    *,
    max_candidate_points: int = 13,
    tol: float = 1e-10,
) -> Envelope:
    """
    Sound linear GELU relaxation over [L, U].

    The previous project implementation attempted a tangent/secant envelope, but
    GELU is not globally convex or concave over many intervals. This function
    always includes exact scalar output bounds, then adds curvature-aware
    tangent/secant candidates only after analytically certifying that each line
    bounds GELU on the whole interval.
    """
    if L > U:
        raise ValueError("activation lower bound must be <= upper bound")

    lb, ub = gelu_bounds(np.array([L]), np.array([U]))
    lower: list[Line] = [(0.0, float(lb[0]), L, U)]
    upper: list[Line] = [(0.0, float(ub[0]), L, U)]

    if U <= 0.0:
        # For x <= 0, x <= GELU(x) <= 0.
        lower.append((1.0, 0.0, L, U))
    elif L >= 0.0:
        # For x >= 0, 0 <= GELU(x) <= x.
        lower.append((0.0, 0.0, L, U))
        upper.append((1.0, 0.0, L, U))

    points = _candidate_points(float(L), float(U), max_candidate_points)
    for point in points:
        tangent = gelu_tangent(point)
        _add_verified_line(lower, tangent, L, U, "lower", tol=tol)
        _add_verified_line(upper, tangent, L, U, "upper", tol=tol)

    # Secants over curvature-aware spans sometimes provide useful global bounds
    # on mixed-curvature intervals. Each is certified before use, so subinterval
    # secants that would cut through GELU outside their span are rejected.
    for i, x0 in enumerate(points[:-1]):
        for x1 in points[i + 1 :]:
            if x1 - x0 <= 1e-12:
                continue
            secant = gelu_secant(x0, x1)
            _add_verified_line(lower, secant, L, U, "lower", tol=tol)
            _add_verified_line(upper, secant, L, U, "upper", tol=tol)

    certify_gelu_envelope(lower, upper, L, U, tol=tol)

    return lower, upper


# Backwards-compatible alias for older scripts that imported the "tight" name.
tight_gelu_envelope = sound_gelu_envelope
