"""Statistical helpers for experiment scripts."""

from __future__ import annotations

import math


def hoeffding_samples(
    confidence: float = 0.95,
    max_error: float = 0.05,
    *,
    two_sided: bool = True,
) -> int:
    """
    Number of Bernoulli samples needed for a Hoeffding error bound.

    With `two_sided=True`, returns n such that
    P(|p_hat - p| > max_error) <= 1 - confidence.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if max_error <= 0.0:
        raise ValueError("max_error must be positive")

    alpha = 1.0 - confidence
    tail_alpha = alpha / 2.0 if two_sided else alpha
    return int(math.ceil(math.log(1.0 / tail_alpha) / (2.0 * max_error**2)))


def clipped_error_interval(rate: float, max_error: float) -> tuple[float, float]:
    """Return [rate - max_error, rate + max_error] clipped to [0, 1]."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    if max_error < 0.0:
        raise ValueError("max_error must be non-negative")
    return max(0.0, rate - max_error), min(1.0, rate + max_error)
