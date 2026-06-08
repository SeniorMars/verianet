"""Reusable LP objectives for Verianet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .activations import Envelope, sound_gelu_envelope
from .bounds import ibp_activation, ibp_affine_keras
from .lp import OptimizeResult, ScipyPolytopeAnalyzer
from .network import NetworkWeights

Array = np.ndarray
ActivationEnvelopeExtras = dict[str, list[Envelope]]


@dataclass
class PolytopeBuild:
    analyzer: ScipyPolytopeAnalyzer
    bounds: dict[str, tuple[Array, Array]]


def input_box(x0: Array, epsilon: float) -> tuple[Array, Array]:
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    x_flat = np.asarray(x0, dtype=np.float64).reshape(-1)
    return np.maximum(x_flat - epsilon, 0.0), np.minimum(x_flat + epsilon, 1.0)


def build_network_polytope(
    weights: NetworkWeights,
    lb: Array,
    ub: Array,
    *,
    inherited_activation_envelopes: ActivationEnvelopeExtras | None = None,
) -> PolytopeBuild:
    """Build the global LP relaxation for the project MLP."""
    lb_arr = np.asarray(lb, dtype=np.float64)
    ub_arr = np.asarray(ub, dtype=np.float64)

    L1, U1 = ibp_affine_keras(lb_arr, ub_arr, weights.W1, weights.b1)
    Lz1, Uz1 = ibp_activation(L1, U1)
    L2, U2 = ibp_affine_keras(Lz1, Uz1, weights.W2, weights.b2)

    analyzer = ScipyPolytopeAnalyzer()
    analyzer.add_input_box("x0", lb_arr, ub_arr)
    analyzer.add_affine("x0", weights.W1, weights.b1, "a1")
    analyzer.add_activation(
        "a1",
        "z1",
        (L1, U1),
        _activation_builder_with_inherited_lines("z1", inherited_activation_envelopes),
    )
    analyzer.add_affine("z1", weights.W2, weights.b2, "a2")
    Lz2, Uz2 = ibp_activation(L2, U2)
    analyzer.add_activation(
        "a2",
        "z2",
        (L2, U2),
        _activation_builder_with_inherited_lines("z2", inherited_activation_envelopes),
    )
    analyzer.add_affine("z2", weights.W3, weights.b3, "a3")

    return PolytopeBuild(
        analyzer=analyzer,
        bounds={
            "x0": (lb_arr, ub_arr),
            "a1": (L1, U1),
            "z1": (Lz1, Uz1),
            "a2": (L2, U2),
            "z2": (Lz2, Uz2),
        },
    )


def activation_envelopes_from_bounds(bounds: dict[str, tuple[Array, Array]]) -> ActivationEnvelopeExtras:
    """Build per-neuron activation envelopes from a previous polytope build."""
    envelopes: ActivationEnvelopeExtras = {}
    for activation_name, pre_name in (("z1", "a1"), ("z2", "a2")):
        L, U = bounds[pre_name]
        envelopes[activation_name] = [
            sound_gelu_envelope(float(L[i]), float(U[i])) for i in range(L.size)
        ]
    return envelopes


def _activation_builder_with_inherited_lines(
    activation_name: str,
    inherited: ActivationEnvelopeExtras | None,
) -> Callable[[float, float], Envelope]:
    extras = [] if inherited is None else inherited.get(activation_name, [])
    idx = 0

    def builder(L: float, U: float) -> Envelope:
        nonlocal idx
        lower, upper = sound_gelu_envelope(L, U)
        if idx < len(extras):
            extra_lower, extra_upper = extras[idx]
            lower = _dedupe_lines([*extra_lower, *lower])
            upper = _dedupe_lines([*extra_upper, *upper])
        idx += 1
        return lower, upper

    return builder


def _dedupe_lines(lines: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    deduped: list[tuple[float, float, float, float]] = []
    for line in lines:
        slope, intercept, x0, x1 = line
        if any(abs(slope - old[0]) <= 1e-10 and abs(intercept - old[1]) <= 1e-10 for old in deduped):
            continue
        deduped.append((float(slope), float(intercept), float(x0), float(x1)))
    return deduped


def class_margin(
    weights: NetworkWeights,
    x0: Array,
    epsilon: float,
    label: int,
    competitor: int,
) -> OptimizeResult:
    """Maximize logit[competitor] - logit[label] over an epsilon box."""
    lb, ub = input_box(x0, epsilon)
    build = build_network_polytope(weights, lb, ub)
    c = np.zeros(build.analyzer.nvars, dtype=np.float64)
    a3 = build.analyzer.var_slices["a3"]
    c[a3.start + competitor] = 1.0
    c[a3.start + label] = -1.0
    return build.analyzer.optimize(c, sense="max")


def verify_robust(
    weights: NetworkWeights,
    x0: Array,
    epsilon: float,
    label: int,
    *,
    tolerance: float = 1e-8,
) -> tuple[bool, dict[int, float | None]]:
    """
    Verify whether label beats every competing class in the relaxation.

    Returns (is_robust, per_competitor_margin). A None margin means the LP did
    not solve to optimality, and the sample is treated as not robust.
    """
    lb, ub = input_box(x0, epsilon)
    build = build_network_polytope(weights, lb, ub)
    a3 = build.analyzer.var_slices["a3"]
    margins: dict[int, float | None] = {}

    for competitor in range(weights.num_classes):
        if competitor == label:
            continue
        c = np.zeros(build.analyzer.nvars, dtype=np.float64)
        c[a3.start + competitor] = 1.0
        c[a3.start + label] = -1.0
        result = build.analyzer.optimize(c, sense="max")
        margins[competitor] = result.value
        if result.value is None or result.value > tolerance:
            return False, margins

    return True, margins


def maximize_margin_pattern(
    weights: NetworkWeights,
    x_center: Array,
    target_class: int,
    epsilon: float,
    *,
    l1_penalty: float = 0.0,
    zero_perimeter: bool = False,
) -> tuple[OptimizeResult, ScipyPolytopeAnalyzer]:
    """
    Maximize logit[target] - max_other_logit - l1_penalty * sum(x).
    """
    lb, ub = input_box(x_center, epsilon)
    if zero_perimeter:
        side = int(np.sqrt(weights.input_dim))
        if side * side != weights.input_dim:
            raise ValueError("zero_perimeter requires a square input grid")
        perimeter = [
            r * side + c
            for r in range(side)
            for c in range(side)
            if r in (0, side - 1) or c in (0, side - 1)
        ]
        lb[perimeter] = 0.0
        ub[perimeter] = 0.0

    build = build_network_polytope(weights, lb, ub)
    analyzer = build.analyzer
    t_idx = analyzer.add_scalar("max_other_logit")
    a3 = analyzer.var_slices["a3"]

    for k in range(weights.num_classes):
        if k == target_class:
            continue
        row = np.zeros(analyzer.nvars, dtype=np.float64)
        row[a3.start + k] = 1.0
        row[t_idx] = -1.0
        analyzer.add_linear_constraint(row, 0.0)

    objective = np.zeros(analyzer.nvars, dtype=np.float64)
    objective[a3.start + target_class] = 1.0
    objective[t_idx] = -1.0
    objective[analyzer.var_slices["x0"]] -= float(l1_penalty)
    return analyzer.optimize(objective, sense="max"), analyzer
