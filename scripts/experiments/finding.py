"""Generate counterfactual MNIST examples using the core SciPy LP path."""

from __future__ import annotations

from typing import Optional

import numpy as np

from verianet.counterfactual import (
    CounterfactualCandidate,
    find_minimal_counterfactual,
    find_minimal_relaxed_counterfactual,
)
from verianet.network import NetworkWeights
from verianet.paths import COUNTERFACTUALS_DIR, ensure_dir


def load_mnist_7x7() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST test data resized to 7x7. Requires TensorFlow."""
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255.0
    return x_test, y_test


def sample_digit_with_correct_pred(
    weights: NetworkWeights,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label: int,
    *,
    max_tries: int = 200,
    rng: np.random.Generator | None = None,
) -> Optional[np.ndarray]:
    """Pick a real MNIST test image with the requested true and predicted label."""
    idxs = np.where(y_test == label)[0]
    if len(idxs) == 0:
        print(f"[warn] no test examples for label {label}")
        return None

    rng = rng or np.random.default_rng()
    idxs = rng.permutation(idxs)
    for idx in idxs[:max_tries]:
        x = x_test[idx]
        if weights.predict(x) == label:
            return x

    print(f"[warn] Could not find test {label} with pred={label} in {min(max_tries, len(idxs))} tries.")
    return None


def result_to_dict(candidate: CounterfactualCandidate, x0: np.ndarray) -> dict:
    """Compatibility shim for the plotting code."""
    side = int(np.sqrt(candidate.image.size))
    image = candidate.image.reshape(side, side)
    return {
        "epsilon": candidate.epsilon,
        "x0": x0,
        "image": image,
        "delta": image - x0,
        "original_logits": candidate.original_logits,
        "cf_logits": candidate.candidate_logits,
        "predicted": candidate.predicted,
        "margin": candidate.concrete_margin,
        "relaxed_margin": candidate.relaxed_margin,
        "relaxed_feasible": candidate.relaxed_feasible,
        "concrete_valid": candidate.concrete_valid,
        "accepted": candidate.accepted,
    }


def find_cf_for_pair(
    weights: NetworkWeights,
    x_test: np.ndarray,
    y_test: np.ndarray,
    orig: int,
    target: int,
    *,
    margin: float,
    max_epsilon: float,
    n_binary_steps: int,
    max_starts: int = 10,
    require_target_prediction: bool = True,
    rng: np.random.Generator | None = None,
) -> Optional[dict]:
    """Try starts and return the first LP-guided candidate that validates concretely."""
    rng = rng or np.random.default_rng(0)
    for attempt in range(1, max_starts + 1):
        x0 = sample_digit_with_correct_pred(weights, x_test, y_test, orig, rng=rng)
        if x0 is None:
            print(f"  [attempt {attempt}] No suitable starting example; stopping.")
            return None

        print(f"  [attempt {attempt}] starting from a pred={orig} example")
        candidate = find_minimal_counterfactual(
            weights,
            x0,
            original_class=orig,
            target_class=target,
            margin=margin,
            max_epsilon=max_epsilon,
            binary_steps=n_binary_steps,
            require_target_prediction=require_target_prediction,
        )
        if candidate is not None and candidate.concrete_valid:
            return result_to_dict(candidate, x0)

    print(f"  No concrete-valid LP-guided candidate found for {orig}->{target} after {max_starts} starts.")
    return None


def find_relaxed_cf_for_pair(
    weights: NetworkWeights,
    x_test: np.ndarray,
    y_test: np.ndarray,
    orig: int,
    target: int,
    *,
    margin: float,
    max_epsilon: float,
    n_binary_steps: int,
    rng: np.random.Generator | None = None,
) -> Optional[dict]:
    """Return the relaxed-model minimal epsilon witness for one starting image."""
    rng = rng or np.random.default_rng(0)
    x0 = sample_digit_with_correct_pred(weights, x_test, y_test, orig, rng=rng)
    if x0 is None:
        return None
    candidate = find_minimal_relaxed_counterfactual(
        weights,
        x0,
        original_class=orig,
        target_class=target,
        margin=margin,
        max_epsilon=max_epsilon,
        binary_steps=n_binary_steps,
    )
    return result_to_dict(candidate, x0) if candidate is not None else None


def visualize_counterfactual(
    result: dict,
    orig: int,
    target: int,
    *,
    margin: float = 0.1,
    save_path: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    axes[0].imshow(result["x0"], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original\npred={np.argmax(result['original_logits'])}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(result["image"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Counterfactual\npred={result['predicted']}", fontsize=11)
    axes[1].axis("off")

    diff = result["delta"]
    limit = max(abs(float(diff.min())), abs(float(diff.max())), 0.01)
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title(f"Delta (epsilon={result['epsilon']:.4f})\nRed=+ink, Blue=-ink", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    x = np.arange(10)
    width = 0.35
    axes[3].bar(x - width / 2, result["original_logits"], width, label="Original", alpha=0.7)
    axes[3].bar(x + width / 2, result["cf_logits"], width, label="Counterfactual", alpha=0.7)
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_xticks(x)
    axes[3].set_xlabel("Digit")
    axes[3].set_ylabel("Logit")
    axes[3].legend(fontsize=8)
    axes[3].set_title("Logits", fontsize=11)

    plt.suptitle(
        f"LP-guided counterfactual candidate: logit[{target}] >= logit[{orig}] + {margin:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def main() -> None:
    weights = NetworkWeights.load()
    x_test, y_test = load_mnist_7x7()

    pairs = [(1, 2), (0, 6), (4, 9), (3, 8), (7, 1)]
    margin = 0.1
    max_epsilon = 0.5

    for orig, target in pairs:
        print(f"\n{'=' * 60}")
        print(f"Finding: {orig} -> {target}")
        print("=" * 60)

        result = find_cf_for_pair(
            weights,
            x_test,
            y_test,
            orig,
            target,
            margin=margin,
            max_epsilon=max_epsilon,
            n_binary_steps=8,
            max_starts=10,
        )

        if result is None:
            print("  No valid counterfactual found within search range.")
            continue

        print(
            f"  Found at epsilon={result['epsilon']:.4f}, "
            f"concrete margin={result['margin']:.2f}, "
            f"relaxed margin={result['relaxed_margin']:.2f}, "
            f"pred_cf={result['predicted']}, "
            f"relaxed_feasible={result['relaxed_feasible']}, "
            f"concrete_valid={result['concrete_valid']}"
        )
        visualize_counterfactual(
            result,
            orig,
            target,
            margin=margin,
            save_path=str(ensure_dir(COUNTERFACTUALS_DIR) / f"counterfactual_{orig}_to_{target}.png"),
        )

    print("\nDONE")


if __name__ == "__main__":
    main()
