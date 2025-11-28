"""
1. Which digits activate each neuron the most
2. What the neuron's activation range is
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
from matplotlib.gridspec import GridSpecFromSubplotSpec

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.image.resize(x_train[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

# Load weights
data = np.load("../verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]


def forward_pass(x0):
    """Compute all activations for an input"""
    x0_flat = x0.flatten()
    a1 = x0_flat @ W1 + b1
    z1 = gelu(a1)
    a2 = z1 @ W2 + b2
    z2 = gelu(a2)
    a3 = z2 @ W3 + b3
    return {'a1': a1, 'z1': z1, 'a2': a2, 'z2': z2, 'a3': a3}


def analyze_neuron_on_real_data(layer, neuron_idx, n_samples=1000, show_plot=True):
    """
    Run real data through the network and see when this neuron activates.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing Layer {layer}, Neuron {neuron_idx}")
    print(f"{'='*60}")

    # Collect activations for each digit class
    activations_by_digit = {d: [] for d in range(10)}

    for i in range(min(n_samples, len(x_test))):
        x = x_test[i]
        y = y_test[i]

        result = forward_pass(x)

        if layer == 1:
            activation = result['z1'][neuron_idx]
        elif layer == 2:
            activation = result['z2'][neuron_idx]
        else:
            raise ValueError("layer must be 1 or 2")

        activations_by_digit[y].append(activation)

    # Compute statistics
    print("\nActivation Statistics by Digit Class:")
    print(f"{'Digit':<6} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 50)

    for digit in range(10):
        acts = np.array(activations_by_digit[digit])
        print(f"{digit:<6} {acts.mean():<10.3f} {acts.std():<10.3f} {acts.min():<10.3f} {acts.max():<10.3f}")

    # Find which digit activates this neuron most
    mean_activations = [np.mean(activations_by_digit[d]) for d in range(10)]
    max_digit = np.argmax(mean_activations)
    min_digit = np.argmin(mean_activations)

    print(f"\n🔥 Most activated by digit: {max_digit} (mean={mean_activations[max_digit]:.3f})")
    print(f"❄️  Least activated by digit: {min_digit} (mean={mean_activations[min_digit]:.3f})")

    # Visualize
    if not show_plot:
        # Just return the statistics without plotting
        return {
            'mean_activations': mean_activations,
            'max_digit': max_digit,
            'min_digit': min_digit,
            'activations_by_digit': activations_by_digit
        }

    # Continue with visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: Small multiples - 6 histograms (top 3 and bottom 3)
    ax = axes[0]
    ax.axis('off')

    top3_digits = np.argsort(mean_activations)[-3:]  # 3 highest
    bottom3_digits = np.argsort(mean_activations)[:3]  # 3 lowest

    # Create 2x3 grid of small histograms within this subplot
    inner_grid = GridSpecFromSubplotSpec(2, 3, subplot_spec=ax.get_subplotspec(), wspace=0.3, hspace=0.4)

    colors_hist = plt.cm.tab10(np.linspace(0, 1, 10))

    # Top row: top 3 digits (highest activation)
    for i, digit in enumerate(sorted(top3_digits)):
        inner_ax = plt.subplot(inner_grid[0, i])
        color = colors_hist[digit]
        inner_ax.hist(activations_by_digit[digit], bins=20, color=color,
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        inner_ax.set_title(f'Digit {digit}\nμ={mean_activations[digit]:.2f}',
                          fontsize=8, color='green', fontweight='bold')
        inner_ax.tick_params(labelsize=6)
        inner_ax.grid(True, alpha=0.2)

    # Bottom row: bottom 3 digits (lowest activation)
    for i, digit in enumerate(sorted(bottom3_digits)):
        inner_ax = plt.subplot(inner_grid[1, i])
        color = colors_hist[digit]
        inner_ax.hist(activations_by_digit[digit], bins=20, color=color,
                     edgecolor='black', linewidth=0.5, alpha=0.8)
        inner_ax.set_title(f'Digit {digit}\nμ={mean_activations[digit]:.2f}',
                          fontsize=8, color='red', fontweight='bold')
        inner_ax.tick_params(labelsize=6)
        inner_ax.grid(True, alpha=0.2)

    # Middle: Mean activation bar chart
    ax = axes[1]
    colors = ['green' if d == max_digit else ('red' if d == min_digit else 'lightgray')
              for d in range(10)]
    bars = ax.bar(range(10), mean_activations, color=colors, edgecolor='black', linewidth=1)
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mean_activations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Digit Class', fontsize=10)
    ax.set_ylabel('Mean Activation', fontsize=10)
    ax.set_title('Mean Activation by Digit', fontsize=11, fontweight='bold')
    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Weight pattern (from Layer 1 only)
    ax = axes[2]
    if layer == 1:
        weights = W1[:, neuron_idx].reshape(7, 7)
        im = ax.imshow(weights, cmap='RdBu', vmin=-abs(weights).max(), vmax=abs(weights).max())
        ax.set_title(f'Input Weight Pattern\n(b={b1[neuron_idx]:.2f})', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046)
    else:
        # For layer 2, show weights from layer 1
        weights = W2[:, neuron_idx]
        ax.bar(range(3), weights, color=['red' if w < 0 else 'blue' for w in weights],
               edgecolor='black', linewidth=1)
        ax.set_xlabel('Layer 1 Neuron', fontsize=10)
        ax.set_ylabel('Weight', fontsize=10)
        ax.set_title(f'Weights from Layer 1\n(b={b2[neuron_idx]:.2f})', fontsize=11, fontweight='bold')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xticks(range(3))
        ax.grid(True, alpha=0.3)
    ax.axis('off') if layer == 1 else None

    plt.tight_layout()
    plt.savefig(f'neuron_L{layer}_N{neuron_idx}.png', dpi=150, bbox_inches='tight')

    print(f"\nSaved: neuron_L{layer}_N{neuron_idx}.png")

    return {
        'mean_activations': mean_activations,
        'max_digit': max_digit,
        'min_digit': min_digit,
        'activations_by_digit': activations_by_digit
    }


if __name__ == "__main__":
    print("Running neuron analysis on real MNIST data...")
    print("This will show which digits activate each neuron.\n")

    # Analyze each individual neuron (generates 6 plots)
    for layer in [1, 2]:
        for neuron_idx in range(3):
            analyze_neuron_on_real_data(layer, neuron_idx, n_samples=1000, show_plot=True)
