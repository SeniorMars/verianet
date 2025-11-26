"""
Pixel-Level Investigation Using LP

For a given digit, find:
1. Which pixels are most important for classification
2. What happens when we maximize/minimize specific pixels
3. Adversarial perturbations that flip the classification
"""

import sys
sys.path.append('..')

from basic import *
from helper import *
from basic import tight_gelu_envelope
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt

# Load data
(x_test, y_test), _ = mnist.load_data()
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

# Load weights
data = np.load("../verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]


def find_critical_pixels(digit_idx, epsilon=0.3, target_class=None):
    """
    Find which pixels are critical for classifying a digit.

    Strategy: For each pixel, try to minimize it and see how much the
    target class logit decreases.
    """
    # Get the digit
    idx = np.where(y_test == digit_idx)[0][0]
    x0 = x_test[idx]
    x0_flat = x0.flatten()

    if target_class is None:
        target_class = digit_idx

    # Build polytope
    lb_in = np.maximum(x0_flat - epsilon, 0)
    ub_in = np.minimum(x0_flat + epsilon, 1)

    L1, U1 = ibp_affine_keras(lb_in, ub_in, W1, b1)
    Lz1, Uz1 = ibp_activation(L1, U1, gelu)
    L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)

    analyzer = PyomoPolyAnalyzer()
    x0_var = analyzer.add_input_box("x0", lb_in, ub_in)
    a1 = analyzer.add_affine(x0_var, W1, b1, "a1")
    z1 = analyzer.add_activation(a1, "z1", (L1, U1), tight_gelu_envelope)
    a2 = analyzer.add_affine(z1, W2, b2, "a2")
    z2 = analyzer.add_activation(a2, "z2", (L2, U2), tight_gelu_envelope)
    a3 = analyzer.add_affine(z2, W3, b3, "a3")

    # Get output logit for target class
    a3_vars = getattr(analyzer.model, 'a3')
    target_logit = a3_vars[target_class]

    # Find max and min values for the target class
    max_logit, _ = analyzer.optimize(target_logit, sense='max')
    min_logit, _ = analyzer.optimize(target_logit, sense='min')

    if max_logit is None or min_logit is None:
        print(f"Digit {digit_idx}, Target class {target_class}:")
        print(f"  ERROR: Optimization failed")
        return {
            'max_logit': None,
            'min_logit': None,
            'range': None
        }

    print(f"Digit {digit_idx}, Target class {target_class}:")
    print(f"  Logit range: [{min_logit:.3f}, {max_logit:.3f}]")
    print(f"  Range width: {max_logit - min_logit:.3f}")

    return {
        'max_logit': max_logit,
        'min_logit': min_logit,
        'range': max_logit - min_logit
    }


def find_adversarial_perturbation(digit_idx, target_class, epsilon=0.3):
    """
    Find an adversarial perturbation that makes the network classify
    digit_idx as target_class.

    Strategy: Maximize (target_class_logit - true_class_logit) to find
    the perturbation that most increases the target class score.
    """
    # Get the digit
    idx = np.where(y_test == digit_idx)[0][0]
    x0 = x_test[idx]
    x0_flat = x0.flatten()

    # Build polytope
    lb_in = np.maximum(x0_flat - epsilon, 0)
    ub_in = np.minimum(x0_flat + epsilon, 1)

    L1, U1 = ibp_affine_keras(lb_in, ub_in, W1, b1)
    Lz1, Uz1 = ibp_activation(L1, U1, gelu)
    L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)

    analyzer = PyomoPolyAnalyzer()
    x0_var = analyzer.add_input_box("x0", lb_in, ub_in)
    a1 = analyzer.add_affine(x0_var, W1, b1, "a1")
    z1 = analyzer.add_activation(a1, "z1", (L1, U1), tight_gelu_envelope)
    a2 = analyzer.add_affine(z1, W2, b2, "a2")
    z2 = analyzer.add_activation(a2, "z2", (L2, U2), tight_gelu_envelope)
    a3 = analyzer.add_affine(z2, W3, b3, "a3")

    a3_vars = getattr(analyzer.model, 'a3')
    x0_vars = getattr(analyzer.model, 'x0')

    # Objective: maximize (target_class_logit - true_class_logit)
    margin = a3_vars[target_class] - a3_vars[digit_idx]
    max_margin, _ = analyzer.optimize(margin, sense='max')

    if max_margin is None:
        print(f"\nAdversarial Attack: {digit_idx} → {target_class}")
        print(f"  ERROR: Optimization failed")
        return {
            'original': x0,
            'adversarial': None,
            'perturbation': None,
            'margin': None,
            'predicted_class': None,
            'logits': None
        }

    # Get the adversarial input
    adv_input = np.array([pyo.value(x0_vars[i]) for i in range(49)]).reshape(7, 7)

    # Get all logits for this adversarial input
    a3_vals = np.array([pyo.value(a3_vars[i]) for i in range(10)])
    adv_class = np.argmax(a3_vals)

    print(f"\nAdversarial Attack: {digit_idx} → {target_class}")
    print(f"  Max margin achieved: {max_margin:.3f}")
    print(f"  Adversarial class: {adv_class}")
    print(f"  Logit[{digit_idx}] = {a3_vals[digit_idx]:.3f}")
    print(f"  Logit[{target_class}] = {a3_vals[target_class]:.3f}")

    return {
        'original': x0,
        'adversarial': adv_input,
        'perturbation': adv_input - x0,
        'margin': max_margin,
        'predicted_class': adv_class,
        'logits': a3_vals
    }


def visualize_pixel_importance(digit_idx, epsilon=0.3):
    """
    For each pixel, measure how much the correct class logit can change.
    """
    idx = np.where(y_test == digit_idx)[0][0]
    x0 = x_test[idx]
    x0_flat = x0.flatten()

    pixel_importance = np.zeros(49)

    print(f"\nAnalyzing pixel importance for digit {digit_idx}...")

    for pixel_idx in range(49):
        # Create epsilon ball, but only allow this one pixel to vary
        lb_in = x0_flat.copy()
        ub_in = x0_flat.copy()
        lb_in[pixel_idx] = max(0, x0_flat[pixel_idx] - epsilon)
        ub_in[pixel_idx] = min(1, x0_flat[pixel_idx] + epsilon)

        # Build polytope
        L1, U1 = ibp_affine_keras(lb_in, ub_in, W1, b1)
        Lz1, Uz1 = ibp_activation(L1, U1, gelu)
        L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)

        analyzer = PyomoPolyAnalyzer()
        x0_var = analyzer.add_input_box("x0", lb_in, ub_in)
        a1 = analyzer.add_affine(x0_var, W1, b1, "a1")
        z1 = analyzer.add_activation(a1, "z1", (L1, U1), tight_gelu_envelope)
        a2 = analyzer.add_affine(z1, W2, b2, "a2")
        z2 = analyzer.add_activation(a2, "z2", (L2, U2), tight_gelu_envelope)
        a3 = analyzer.add_affine(z2, W3, b3, "a3")

        a3_vars = getattr(analyzer.model, 'a3')
        target_logit = a3_vars[digit_idx]

        # Find range
        max_val, _ = analyzer.optimize(target_logit, sense='max')
        min_val, _ = analyzer.optimize(target_logit, sense='min')

        # Handle optimization failure
        if max_val is None or min_val is None:
            print(f"  Warning: Optimization failed for pixel {pixel_idx}, setting importance to 0")
            pixel_importance[pixel_idx] = 0
        else:
            pixel_importance[pixel_idx] = max_val - min_val

        if pixel_idx % 10 == 0:
            print(f"  Processed {pixel_idx}/49 pixels...")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(x0, cmap='gray')
    axes[0].set_title(f'Original Digit {digit_idx}')
    axes[0].axis('off')

    im1 = axes[1].imshow(pixel_importance.reshape(7, 7), cmap='hot')
    axes[1].set_title('Pixel Importance\n(Logit Range When Varied)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Show overlay
    axes[2].imshow(x0, cmap='gray', alpha=0.5)
    axes[2].imshow(pixel_importance.reshape(7, 7), cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay: Image + Importance')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(f'pixel_importance_digit_{digit_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Saved: pixel_importance_digit_{digit_idx}.png")


def visualize_adversarial_attack(digit_idx, target_class, epsilon=0.3):
    """
    Visualize an adversarial attack.
    """
    result = find_adversarial_perturbation(digit_idx, target_class, epsilon)

    if result['adversarial'] is None:
        print(f"Skipping visualization due to optimization failure")
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(result['original'], cmap='gray')
    axes[0].set_title(f'Original (Digit {digit_idx})')
    axes[0].axis('off')

    axes[1].imshow(result['perturbation'], cmap='RdBu_r',
                   vmin=-epsilon, vmax=epsilon)
    axes[1].set_title(f'Perturbation (ε={epsilon})')
    axes[1].axis('off')
    plt.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046)

    axes[2].imshow(result['adversarial'], cmap='gray')
    axes[2].set_title(f'Adversarial\n(Classified as {result["predicted_class"]})')
    axes[2].axis('off')

    # Bar chart of logits
    colors = ['green' if i == digit_idx else ('red' if i == target_class else 'gray')
              for i in range(10)]
    axes[3].bar(range(10), result['logits'], color=colors)
    axes[3].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[3].set_xlabel('Digit Class')
    axes[3].set_ylabel('Logit Value')
    axes[3].set_title('Output Logits')
    axes[3].set_xticks(range(10))
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'adversarial_{digit_idx}_to_{target_class}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Saved: adversarial_{digit_idx}_to_{target_class}.png")


if __name__ == "__main__":
    print("=" * 60)
    print("PIXEL-LEVEL INVESTIGATION")
    print("=" * 60)

    # Example 1: Find critical pixels for digit 0
    print("\n--- Example 1: Pixel importance for digit 0 ---")
    visualize_pixel_importance(0, epsilon=0.3)

    # Example 2: Adversarial attack - make digit 0 classified as 8
    print("\n--- Example 2: Adversarial attack 0 → 8 ---")
    visualize_adversarial_attack(0, 8, epsilon=0.3)

    # Example 3: Check robustness for each digit
    print("\n--- Example 3: Logit ranges for all digits ---")
    for digit in range(10):
        find_critical_pixels(digit, epsilon=0.1)

    print("\n" + "=" * 60)
    print("PIXEL INVESTIGATION COMPLETE!")
    print("=" * 60)
