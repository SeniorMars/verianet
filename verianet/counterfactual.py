"""Counterfactual search over the LP relaxation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .network import NetworkWeights
from .objectives import build_network_polytope, input_box

Array = np.ndarray


@dataclass(frozen=True)
class CounterfactualCandidate:
    epsilon: float
    image: Array
    delta: Array
    relaxed_margin: float
    concrete_margin: float
    original_logits: Array
    candidate_logits: Array
    predicted: int
    relaxed_feasible: bool
    concrete_valid: bool

    @property
    def accepted(self) -> bool:
        """Backward-compatible alias for a concrete-network counterfactual."""
        return self.concrete_valid


def solve_target_margin(
    weights: NetworkWeights,
    x0: Array,
    epsilon: float,
    original_class: int,
    target_class: int,
    *,
    margin: float = 0.1,
    require_target_prediction: bool = True,
) -> CounterfactualCandidate | None:
    """
    Maximize logit[target] - logit[original] over an epsilon box.

    `relaxed_feasible` means the LP relaxation can achieve the requested target
    margin. This is a certificate for the relaxed model only. `concrete_valid`
    means the LP-selected input also satisfies the margin in the true GELU
    network and, by default, predicts the target class.
    """
    if original_class == target_class:
        raise ValueError("original_class and target_class must differ")
    if not 0 <= original_class < weights.num_classes:
        raise ValueError("original_class is out of range")
    if not 0 <= target_class < weights.num_classes:
        raise ValueError("target_class is out of range")

    x_flat = np.asarray(x0, dtype=np.float64).reshape(-1)
    lb, ub = input_box(x_flat, epsilon)
    build = build_network_polytope(weights, lb, ub)
    a3 = build.analyzer.var_slices["a3"]

    objective = np.zeros(build.analyzer.nvars, dtype=np.float64)
    objective[a3.start + target_class] = 1.0
    objective[a3.start + original_class] = -1.0
    relaxed = build.analyzer.optimize(objective, sense="max")
    if relaxed.value is None:
        return None

    candidate_flat = build.analyzer.values(relaxed, "x0")
    original_logits = weights.forward_logits(x_flat)
    candidate_logits = weights.forward_logits(candidate_flat)
    concrete_margin = float(candidate_logits[target_class] - candidate_logits[original_class])
    predicted = int(np.argmax(candidate_logits))
    relaxed_feasible = float(relaxed.value) >= margin
    concrete_valid = concrete_margin >= margin and (
        not require_target_prediction or predicted == target_class
    )

    return CounterfactualCandidate(
        epsilon=float(epsilon),
        image=candidate_flat,
        delta=candidate_flat - x_flat,
        relaxed_margin=float(relaxed.value),
        concrete_margin=concrete_margin,
        original_logits=original_logits,
        candidate_logits=candidate_logits,
        predicted=predicted,
        relaxed_feasible=relaxed_feasible,
        concrete_valid=concrete_valid,
    )


def find_minimal_relaxed_counterfactual(
    weights: NetworkWeights,
    x0: Array,
    original_class: int,
    target_class: int,
    *,
    margin: float = 0.1,
    max_epsilon: float = 0.5,
    coarse_epsilons: Iterable[float] | None = None,
    binary_steps: int = 10,
    require_target_prediction: bool = True,
) -> CounterfactualCandidate | None:
    """
    Binary-search the smallest epsilon certified feasible in the LP relaxation.

    This matches the paper's relaxed feasibility query:
    max_{x in P_epsilon} logit[target] - logit[original] >= margin.
    Because P_epsilon grows monotonically with epsilon, the relaxed predicate is
    suitable for binary search. The returned input is still separately validated
    against the concrete GELU network via `concrete_valid`.
    """
    if max_epsilon < 0:
        raise ValueError("max_epsilon must be non-negative")
    if binary_steps < 0:
        raise ValueError("binary_steps must be non-negative")

    x_flat = np.asarray(x0, dtype=np.float64).reshape(-1)

    at_zero = solve_target_margin(
        weights,
        x_flat,
        0.0,
        original_class,
        target_class,
        margin=margin,
        require_target_prediction=require_target_prediction,
    )
    if at_zero is not None and at_zero.relaxed_feasible:
        return at_zero

    if coarse_epsilons is None:
        coarse = np.linspace(0.01, max_epsilon, 8)
    else:
        coarse = np.asarray(list(coarse_epsilons), dtype=np.float64)
    coarse = np.unique(coarse[(coarse > 0.0) & (coarse <= max_epsilon)])
    if coarse.size == 0 and max_epsilon > 0.0:
        coarse = np.array([max_epsilon], dtype=np.float64)

    low = 0.0
    high: float | None = None
    best: CounterfactualCandidate | None = None

    for eps in coarse:
        candidate = solve_target_margin(
            weights,
            x_flat,
            float(eps),
            original_class,
            target_class,
            margin=margin,
            require_target_prediction=require_target_prediction,
        )
        if candidate is not None and candidate.relaxed_feasible:
            high = float(eps)
            best = candidate
            break
        low = float(eps)

    if high is None:
        return None

    for _ in range(binary_steps):
        mid = 0.5 * (low + high)
        candidate = solve_target_margin(
            weights,
            x_flat,
            mid,
            original_class,
            target_class,
            margin=margin,
            require_target_prediction=require_target_prediction,
        )
        if candidate is not None and candidate.relaxed_feasible:
            high = mid
            best = candidate
        else:
            low = mid

    return best


def find_minimal_counterfactual(
    weights: NetworkWeights,
    x0: Array,
    original_class: int,
    target_class: int,
    *,
    margin: float = 0.1,
    max_epsilon: float = 0.5,
    coarse_epsilons: Iterable[float] | None = None,
    binary_steps: int = 10,
    require_target_prediction: bool = True,
) -> CounterfactualCandidate | None:
    """
    Heuristically search for a concrete counterfactual from LP-optimal witnesses.

    Unlike `find_minimal_relaxed_counterfactual`, this validates the selected
    LP input on the true GELU network. This is useful for generating examples,
    but it is not a completeness guarantee for true-network counterfactuals:
    the LP optimum can be a spurious relaxed witness even when another input in
    the same box would validate concretely.
    """
    if max_epsilon < 0:
        raise ValueError("max_epsilon must be non-negative")
    if binary_steps < 0:
        raise ValueError("binary_steps must be non-negative")

    x_flat = np.asarray(x0, dtype=np.float64).reshape(-1)
    orig_logits = weights.forward_logits(x_flat)
    current_margin = float(orig_logits[target_class] - orig_logits[original_class])
    current_pred = int(np.argmax(orig_logits))
    if current_margin >= margin and (
        not require_target_prediction or current_pred == target_class
    ):
        return CounterfactualCandidate(
            epsilon=0.0,
            image=x_flat.copy(),
            delta=np.zeros_like(x_flat),
            relaxed_margin=current_margin,
            concrete_margin=current_margin,
            original_logits=orig_logits,
            candidate_logits=orig_logits,
            predicted=current_pred,
            relaxed_feasible=True,
            concrete_valid=True,
        )

    if coarse_epsilons is None:
        coarse = np.linspace(0.01, max_epsilon, 8)
    else:
        coarse = np.asarray(list(coarse_epsilons), dtype=np.float64)
    coarse = np.unique(coarse[(coarse > 0.0) & (coarse <= max_epsilon)])
    if coarse.size == 0 and max_epsilon > 0.0:
        coarse = np.array([max_epsilon], dtype=np.float64)

    low = 0.0
    high: float | None = None
    best: CounterfactualCandidate | None = None

    for eps in coarse:
        candidate = solve_target_margin(
            weights,
            x_flat,
            float(eps),
            original_class,
            target_class,
            margin=margin,
            require_target_prediction=require_target_prediction,
        )
        if candidate is not None and candidate.concrete_valid:
            high = float(eps)
            best = candidate
            break
        low = float(eps)

    if high is None:
        return None

    for _ in range(binary_steps):
        mid = 0.5 * (low + high)
        candidate = solve_target_margin(
            weights,
            x_flat,
            mid,
            original_class,
            target_class,
            margin=margin,
            require_target_prediction=require_target_prediction,
        )
        if candidate is not None and candidate.concrete_valid:
            high = mid
            best = candidate
        else:
            low = mid

    return best
