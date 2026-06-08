"""Core utilities for Verianet experiments."""

from .activations import gelu, gelu_bounds, sound_gelu_envelope
from .bounds import ibp_activation, ibp_affine, ibp_affine_keras
from .counterfactual import (
    CounterfactualCandidate,
    find_minimal_counterfactual,
    find_minimal_relaxed_counterfactual,
    solve_target_margin,
)
from .network import NetworkWeights
from .objectives import build_network_polytope, class_margin, maximize_margin_pattern, verify_robust
from .paths import PROJECT_ROOT, RESULTS_DIR, WEIGHTS_PATH
from .refinement import (
    SplitOptimizationResult,
    SplitRobustnessResult,
    class_margin_with_splits,
    split_input_box,
    verify_robust_with_splits,
)
from .stats import clipped_error_interval, hoeffding_samples

__all__ = [
    "CounterfactualCandidate",
    "NetworkWeights",
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "SplitOptimizationResult",
    "SplitRobustnessResult",
    "WEIGHTS_PATH",
    "build_network_polytope",
    "class_margin",
    "class_margin_with_splits",
    "clipped_error_interval",
    "find_minimal_counterfactual",
    "find_minimal_relaxed_counterfactual",
    "gelu",
    "gelu_bounds",
    "hoeffding_samples",
    "ibp_activation",
    "ibp_affine",
    "ibp_affine_keras",
    "maximize_margin_pattern",
    "solve_target_margin",
    "split_input_box",
    "sound_gelu_envelope",
    "verify_robust",
    "verify_robust_with_splits",
]
