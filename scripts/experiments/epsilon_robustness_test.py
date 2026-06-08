"""Run epsilon-vs-robustness experiments with the core SciPy LP path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from verianet.network import NetworkWeights
from verianet.objectives import verify_robust
from verianet.paths import RESULTS_DIR, ensure_dir
from verianet.stats import clipped_error_interval, hoeffding_samples


@dataclass(frozen=True)
class RobustnessRate:
    rate: float
    lower: float
    upper: float
    robust_count: int
    total_count: int
    solver_failures: int


def load_mnist_7x7() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST test data resized to 7x7. Requires TensorFlow."""
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255.0
    return x_test, y_test


def samples_for_confidence(confidence: float = 0.95, max_error: float = 0.05) -> int:
    """Two-sided Hoeffding sample count for a Bernoulli rate estimate."""
    return hoeffding_samples(confidence, max_error, two_sided=True)


def evaluate_digit_epsilon(
    weights: NetworkWeights,
    digit_samples: np.ndarray,
    digit: int,
    epsilon: float,
    *,
    max_samples: int,
    max_error: float,
    progress_every: int = 50,
    rng: np.random.Generator | None = None,
) -> RobustnessRate:
    n_samples = min(max_samples, len(digit_samples))
    if n_samples == 0:
        raise ValueError(f"no samples available for digit {digit}")
    indices = np.arange(len(digit_samples))
    if rng is not None:
        indices = rng.permutation(indices)
    indices = indices[:n_samples]
    robust_count = 0
    solver_failures = 0

    for i, sample_idx in enumerate(indices):
        ok, margins = verify_robust(weights, digit_samples[sample_idx], epsilon, digit)
        if any(v is None for v in margins.values()):
            solver_failures += 1
        if ok:
            robust_count += 1

        if progress_every and ((i + 1) % progress_every == 0 or i + 1 == n_samples):
            current_rate = robust_count / (i + 1)
            extra = f", {solver_failures} solver failures" if solver_failures else ""
            print(f"  Progress: {i + 1}/{n_samples} ({100 * current_rate:.1f}% robust{extra})")

    rate = robust_count / n_samples
    lower, upper = clipped_error_interval(rate, max_error)
    return RobustnessRate(
        rate=rate,
        lower=lower,
        upper=upper,
        robust_count=robust_count,
        total_count=n_samples,
        solver_failures=solver_failures,
    )


def run_sweep(
    weights: NetworkWeights,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epsilons: Iterable[float],
    digits: Iterable[int],
    n_samples_per_config: int,
    max_error: float,
    rng_seed: int | None = 0,
) -> dict[int, list[RobustnessRate]]:
    results: dict[int, list[RobustnessRate]] = {}
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else None

    for digit in digits:
        print(f"\n{'=' * 60}")
        print(f"DIGIT {digit}")
        print("=" * 60)

        digit_samples = x_test[y_test == digit]
        results[digit] = []

        for epsilon in epsilons:
            print(f"\n--- epsilon={epsilon} ---")
            rate = evaluate_digit_epsilon(
                weights,
                digit_samples,
                digit,
                float(epsilon),
                max_samples=n_samples_per_config,
                max_error=max_error,
                rng=rng,
            )
            results[digit].append(rate)
            print(
                f"\n  RESULT: {rate.robust_count}/{rate.total_count} robust = {100 * rate.rate:.1f}%"
            )
            print(f"  95% CI: [{100 * rate.lower:.1f}%, {100 * rate.upper:.1f}%]")
            if rate.solver_failures:
                print(f"  ({rate.solver_failures} solver failures counted as non-robust)")

    return results


def save_results(
    results: dict[int, list[RobustnessRate]],
    epsilons: list[float],
    *,
    n_samples: int,
    confidence: float,
    max_error: float,
    save_path: str | None = None,
) -> None:
    output_path = RESULTS_DIR / "epsilon_robustness_results.npz" if save_path is None else Path(save_path)
    ensure_dir(output_path.parent)
    serializable = {
        digit: [
            (rate.rate, rate.lower, rate.upper, rate.robust_count, rate.total_count, rate.solver_failures)
            for rate in rates
        ]
        for digit, rates in results.items()
    }
    np.savez(
        output_path,
        results=serializable,
        epsilons=np.asarray(epsilons, dtype=np.float64),
        n_samples=n_samples,
        confidence=confidence,
        max_error=max_error,
    )
    print(f"Saved to: {output_path}")


def plot_results(
    results: dict[int, list[RobustnessRate]],
    epsilons: list[float],
    *,
    n_samples: int,
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output_path = RESULTS_DIR / "epsilon_robustness_test.png" if save_path is None else Path(save_path)
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, digit in enumerate(sorted(results)):
        ax = axes[idx]
        rates = [100 * rate.rate for rate in results[digit]]
        lower = [100 * rate.lower for rate in results[digit]]
        upper = [100 * rate.upper for rate in results[digit]]
        errors_lower = [rates[i] - lower[i] for i in range(len(rates))]
        errors_upper = [upper[i] - rates[i] for i in range(len(rates))]

        x_pos = range(len(epsilons))
        ax.bar(
            x_pos,
            rates,
            color=f"C{digit}",
            alpha=0.7,
            edgecolor="black",
            yerr=[errors_lower, errors_upper],
            capsize=5,
            error_kw={"linewidth": 2},
        )

        for i, val in enumerate(rates):
            ax.text(i, val + errors_upper[i] + 3, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([f"{eps:.2f}" for eps in epsilons])
        ax.set_ylim([0, 105])
        ax.set_title(f"Digit {digit}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        if idx >= 5:
            ax.set_xlabel("Epsilon", fontsize=10)
        if idx % 5 == 0:
            ax.set_ylabel("Robustness Rate (%)", fontsize=10)

    fig.suptitle(f"Robustness vs Perturbation Size (n={n_samples}, 95% CI)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to: {output_path}")


def print_summary(results: dict[int, list[RobustnessRate]], epsilons: list[float]) -> None:
    print("\nSUMMARY OF RESULTS:")
    print("=" * 60)
    for digit in sorted(results):
        print(f"\nDigit {digit}:")
        for eps, rate in zip(epsilons, results[digit]):
            print(f"  epsilon={eps}: {100 * rate.rate:.1f}% [{100 * rate.lower:.1f}%, {100 * rate.upper:.1f}%]")


def main() -> None:
    epsilons = [0.01, 0.02, 0.05, 0.1]
    digits = list(range(10))
    confidence = 0.95
    max_error = 0.05
    n_samples_per_config = samples_for_confidence(confidence, max_error)

    print("=== FULL-SCALE RUN: Epsilon vs Robustness with Confidence Intervals ===")
    print(f"Confidence: {confidence}, Max error: +/-{max_error}")
    print(f"Epsilons: {epsilons}")
    print(f"Digits: {digits}")
    print(f"Samples per config: {n_samples_per_config}")
    print(f"Total verifications: {len(epsilons) * len(digits) * n_samples_per_config}")

    weights = NetworkWeights.load()
    x_test, y_test = load_mnist_7x7()
    results = run_sweep(
        weights,
        x_test,
        y_test,
        epsilons=epsilons,
        digits=digits,
        n_samples_per_config=n_samples_per_config,
        max_error=max_error,
    )

    save_results(
        results,
        epsilons,
        n_samples=n_samples_per_config,
        confidence=confidence,
        max_error=max_error,
    )
    plot_results(results, epsilons, n_samples=n_samples_per_config)
    print_summary(results, epsilons)
    print("\nFull-scale test complete!")


if __name__ == "__main__":
    main()
