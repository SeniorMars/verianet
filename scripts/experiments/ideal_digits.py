"""Generate margin-optimized digit prototypes using the core SciPy LP path."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from verianet.network import NetworkWeights
from verianet.objectives import maximize_margin_pattern
from verianet.paths import RESULTS_DIR, ensure_dir


def load_mnist_7x7() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST test data resized to 7x7. Requires TensorFlow."""
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255.0
    return x_test, y_test


def margin_for_class(logits: np.ndarray, target_class: int) -> float:
    others = np.delete(np.asarray(logits, dtype=np.float64), target_class)
    return float(logits[target_class] - np.max(others))


def find_best_examples(
    weights: NetworkWeights,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Find the highest-logit test example for each class."""
    best_images: list[np.ndarray] = []
    best_logits: list[np.ndarray] = []
    best_indices: list[int] = []

    for target_class in range(weights.num_classes):
        class_indices = np.where(y_test == target_class)[0]
        if len(class_indices) == 0:
            raise ValueError(f"no examples found for digit {target_class}")

        best_idx = int(class_indices[0])
        best_img = x_test[best_idx]
        best_logit_value = -np.inf
        found_correct = False

        for idx in class_indices:
            img = x_test[idx]
            logits = weights.forward_logits(img)
            predicted = int(np.argmax(logits))
            if predicted != target_class:
                continue
            found_correct = True
            if logits[target_class] > best_logit_value:
                best_logit_value = float(logits[target_class])
                best_idx = int(idx)
                best_img = img

        if not found_correct:
            # Fall back to the strongest true-class logit when the tiny model
            # has no correctly classified example for a class.
            for idx in class_indices:
                img = x_test[idx]
                logits = weights.forward_logits(img)
                if logits[target_class] > best_logit_value:
                    best_logit_value = float(logits[target_class])
                    best_idx = int(idx)
                    best_img = img

        logits = weights.forward_logits(best_img)
        best_images.append(best_img.reshape(-1))
        best_logits.append(logits)
        best_indices.append(best_idx)

        print(
            f"Digit {target_class}: idx={best_idx}, "
            f"logit={logits[target_class]:.2f}, pred={np.argmax(logits)}, "
            f"margin={margin_for_class(logits, target_class):.2f}"
        )

    return best_images, best_logits, best_indices


def optimize_digit_patterns(
    weights: NetworkWeights,
    best_images: list[np.ndarray],
    *,
    epsilon: float = 0.3,
    penalty_weight: float = 0.8,
    zero_perimeter: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Optimize one prototype per digit."""
    optimal_inputs: list[np.ndarray] = []
    optimized_logits: list[np.ndarray] = []

    for target_class, center in enumerate(best_images):
        result, analyzer = maximize_margin_pattern(
            weights,
            center,
            target_class,
            epsilon,
            l1_penalty=penalty_weight,
            zero_perimeter=zero_perimeter,
        )

        if result.value is None:
            print(f"Class {target_class}: optimization failed; using seed image")
            optimal_input = center.copy()
        else:
            optimal_input = analyzer.values(result, "x0")

        logits = weights.forward_logits(optimal_input)
        optimal_inputs.append(optimal_input)
        optimized_logits.append(logits)
        print(f"Class {target_class}: optimized concrete margin = {margin_for_class(logits, target_class):.2f}")

    return optimal_inputs, optimized_logits


def plot_optimized_digits(
    weights: NetworkWeights,
    best_images: list[np.ndarray],
    optimal_inputs: list[np.ndarray],
    optimized_logits: list[np.ndarray],
    *,
    epsilon: float,
    penalty_weight: float,
    save_path: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    output_path = RESULTS_DIR / "optimized_digits.png" if save_path is None else Path(save_path)
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(3, weights.num_classes, figsize=(20, 9))

    for k in range(weights.num_classes):
        axes[0, k].imshow(best_images[k].reshape((7, 7)), cmap="gray", vmin=0, vmax=1)
        orig_logits = weights.forward_logits(best_images[k])
        axes[0, k].set_title(f"Best {k}\n(m={margin_for_class(orig_logits, k):.1f})", fontsize=9)
        axes[0, k].axis("off")

        axes[1, k].imshow(optimal_inputs[k].reshape((7, 7)), cmap="gray", vmin=0, vmax=1)
        axes[1, k].set_title(f"Optimized\n(m={margin_for_class(optimized_logits[k], k):.1f})", fontsize=9)
        axes[1, k].axis("off")

        diff = optimal_inputs[k] - best_images[k]
        axes[2, k].imshow(diff.reshape((7, 7)), cmap="RdBu", vmin=-epsilon, vmax=epsilon)
        axes[2, k].set_title("Delta", fontsize=9)
        axes[2, k].axis("off")

    axes[0, 0].set_ylabel("Best MNIST", fontsize=11)
    axes[1, 0].set_ylabel("Optimized", fontsize=11)
    axes[2, 0].set_ylabel("Perturbation", fontsize=11)

    plt.suptitle(
        f"Margin-Optimized Digits (epsilon={epsilon}, penalty={penalty_weight}, perimeter=0)\n"
        "Starting from highest-logit MNIST examples",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


def main() -> None:
    weights = NetworkWeights.load()
    x_test, y_test = load_mnist_7x7()

    epsilon = 0.3
    penalty_weight = 0.8

    best_images, _best_logits, _best_indices = find_best_examples(weights, x_test, y_test)
    optimal_inputs, optimized_logits = optimize_digit_patterns(
        weights,
        best_images,
        epsilon=epsilon,
        penalty_weight=penalty_weight,
        zero_perimeter=True,
    )
    plot_optimized_digits(
        weights,
        best_images,
        optimal_inputs,
        optimized_logits,
        epsilon=epsilon,
        penalty_weight=penalty_weight,
    )


if __name__ == "__main__":
    main()
