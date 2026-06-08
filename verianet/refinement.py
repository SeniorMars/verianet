"""LP refinement by splitting input boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .lp import ScipyPolytopeAnalyzer
from .network import NetworkWeights
from .objectives import (
    activation_envelopes_from_bounds,
    build_network_polytope,
    input_box,
)

Array = np.ndarray
ObjectiveFactory = Callable[[ScipyPolytopeAnalyzer], Array]


@dataclass(frozen=True)
class InputBox:
    lb: Array
    ub: Array

    @property
    def widths(self) -> Array:
        return self.ub - self.lb

    @property
    def volume(self) -> float:
        return float(np.prod(self.widths))


@dataclass(frozen=True)
class SplitOptimizationResult:
    value: float | None
    sense: str
    boxes_solved: int
    solver_failures: int
    leaf_values: tuple[float | None, ...]
    leaf_boxes: tuple[InputBox, ...]


@dataclass(frozen=True)
class SplitRobustnessResult:
    robust: bool
    margins: dict[int, float | None]
    boxes_solved: int
    solver_failures: int


def split_input_box(lb: Array, ub: Array, *, max_leaves: int) -> list[InputBox]:
    """
    Recursively split the widest input dimension until max_leaves is reached.

    The returned boxes are disjoint and cover the original box.
    """
    if max_leaves < 1:
        raise ValueError("max_leaves must be at least 1")

    root = InputBox(np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64))
    if np.any(root.lb > root.ub):
        raise ValueError("input lower bounds must be <= upper bounds")

    leaves = [root]
    while len(leaves) < max_leaves:
        widths = [box.widths for box in leaves]
        split_idx = int(np.argmax([float(np.max(w)) for w in widths]))
        box = leaves[split_idx]
        dim = int(np.argmax(box.widths))
        if box.widths[dim] <= 0.0:
            break

        midpoint = 0.5 * (box.lb[dim] + box.ub[dim])
        left_lb, left_ub = box.lb.copy(), box.ub.copy()
        right_lb, right_ub = box.lb.copy(), box.ub.copy()
        left_ub[dim] = midpoint
        right_lb[dim] = midpoint
        leaves[split_idx : split_idx + 1] = [
            InputBox(left_lb, left_ub),
            InputBox(right_lb, right_ub),
        ]

    return leaves


def optimize_over_input_splits(
    weights: NetworkWeights,
    lb: Array,
    ub: Array,
    objective_factory: ObjectiveFactory,
    *,
    sense: str = "max",
    max_leaves: int = 8,
) -> SplitOptimizationResult:
    """
    Optimize an LP objective over a split input box cover.

    Each leaf LP inherits the activation cuts from the unsplit root LP, then adds
    cuts from its narrower IBP intervals. This makes each leaf relaxation a
    refinement of the root relaxation while preserving LP-only solving.
    """
    if sense not in {"min", "max"}:
        raise ValueError("sense must be 'min' or 'max'")

    root_build = build_network_polytope(weights, lb, ub)
    inherited = activation_envelopes_from_bounds(root_build.bounds)
    leaves = split_input_box(lb, ub, max_leaves=max_leaves)

    values: list[float | None] = []
    failures = 0
    for box in leaves:
        build = build_network_polytope(
            weights,
            box.lb,
            box.ub,
            inherited_activation_envelopes=inherited,
        )
        objective = objective_factory(build.analyzer)
        result = build.analyzer.optimize(objective, sense=sense)
        values.append(result.value)
        if result.value is None:
            failures += 1

    solved_values = [value for value in values if value is not None]
    if failures or not solved_values:
        combined: float | None = None
    elif sense == "max":
        combined = float(max(solved_values))
    else:
        combined = float(min(solved_values))

    return SplitOptimizationResult(
        value=combined,
        sense=sense,
        boxes_solved=len(leaves) - failures,
        solver_failures=failures,
        leaf_values=tuple(values),
        leaf_boxes=tuple(leaves),
    )


def class_margin_with_splits(
    weights: NetworkWeights,
    x0: Array,
    epsilon: float,
    label: int,
    competitor: int,
    *,
    max_leaves: int = 8,
) -> SplitOptimizationResult:
    """Maximize logit[competitor] - logit[label] over input-box splits."""
    lb, ub = input_box(x0, epsilon)

    def objective_factory(analyzer: ScipyPolytopeAnalyzer) -> Array:
        c = np.zeros(analyzer.nvars, dtype=np.float64)
        a3 = analyzer.var_slices["a3"]
        c[a3.start + competitor] = 1.0
        c[a3.start + label] = -1.0
        return c

    return optimize_over_input_splits(
        weights,
        lb,
        ub,
        objective_factory,
        sense="max",
        max_leaves=max_leaves,
    )


def verify_robust_with_splits(
    weights: NetworkWeights,
    x0: Array,
    epsilon: float,
    label: int,
    *,
    max_leaves: int = 8,
    tolerance: float = 1e-8,
) -> SplitRobustnessResult:
    """
    Verify robustness with a split-box LP refinement.

    If this returns robust=True, every split LP certified every competing
    margin <= tolerance, so the original input box is certified robust.
    """
    margins: dict[int, float | None] = {}
    boxes_solved = 0
    solver_failures = 0

    for competitor in range(weights.num_classes):
        if competitor == label:
            continue
        result = class_margin_with_splits(
            weights,
            x0,
            epsilon,
            label,
            competitor,
            max_leaves=max_leaves,
        )
        margins[competitor] = result.value
        boxes_solved += result.boxes_solved
        solver_failures += result.solver_failures
        if result.value is None or result.value > tolerance:
            return SplitRobustnessResult(False, margins, boxes_solved, solver_failures)

    return SplitRobustnessResult(True, margins, boxes_solved, solver_failures)
