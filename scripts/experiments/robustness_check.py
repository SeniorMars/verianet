"""Small robustness smoke check for digit 0."""

from __future__ import annotations

import numpy as np

from verianet.network import NetworkWeights
from verianet.objectives import verify_robust
from verianet.stats import clipped_error_interval, hoeffding_samples


def load_mnist_7x7() -> tuple[np.ndarray, np.ndarray]:
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255.0
    return x_test, y_test


def run_digit_zero_smoke(
    weights: NetworkWeights,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epsilon: float = 0.01,
    confidence: float = 0.95,
    max_error: float = 0.05,
) -> float:
    zeros = x_test[y_test == 0]
    n_samples = hoeffding_samples(confidence, max_error, two_sided=True)
    n_samples = min(n_samples, len(zeros))
    if n_samples == 0:
        raise ValueError("no digit-0 samples available")
    rng = np.random.default_rng(0)
    sample_indices = rng.permutation(len(zeros))[:n_samples]

    print(f"\n=== MONTE CARLO ROBUSTNESS (epsilon={epsilon}, n={n_samples}) ===")
    print(f"Confidence: {confidence}, Max error: +/-{max_error}")

    robust_count = 0
    solver_failures = 0
    for i, sample_idx in enumerate(sample_indices):
        ok, margins = verify_robust(weights, zeros[sample_idx], epsilon, label=0)
        if any(v is None for v in margins.values()):
            solver_failures += 1
        if ok:
            robust_count += 1

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{n_samples} ({100 * robust_count / (i + 1):.1f}% robust)")

    robust_rate = robust_count / n_samples
    lower, upper = clipped_error_interval(robust_rate, max_error)
    print(f"\nResult: {robust_count}/{n_samples} robust ({100 * robust_rate:.1f}%)")
    print(
        f"With {confidence} confidence: true rate in "
        f"[{100 * lower:.1f}%, {100 * upper:.1f}%]"
    )
    if solver_failures:
        print(f"Solver failures counted as non-robust: {solver_failures}")
    return robust_rate


def main() -> None:
    weights = NetworkWeights.load()
    x_test, y_test = load_mnist_7x7()
    run_digit_zero_smoke(weights, x_test, y_test)


if __name__ == "__main__":
    main()
