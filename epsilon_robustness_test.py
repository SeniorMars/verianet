from basic import *
from helper import *
from basic import tight_gelu_envelope
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import signal
from contextlib import contextmanager

# Timeout context manager
@contextmanager
def timeout_context(seconds):
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# Load test data and resize to 7x7
(x_test, y_test), _ = mnist.load_data()
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

# Load weights
data = np.load("verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]

# Test parameters
epsilons = [0.01, 0.02, 0.05, 0.1]
digits = range(10)

# Set to True for full-scale run with confidence intervals, False for quick test
FULL_SCALE = True

if FULL_SCALE:
    # Hoeffding bound: n = -ln(1-conf)/(2*err^2)
    confidence = 0.95
    max_error = 0.05  # ±5% error
    n_samples_per_config = int(math.ceil(-math.log(1 - confidence) / (2 * max_error**2)))
    print(f"=== FULL-SCALE RUN: Epsilon vs Robustness with Confidence Intervals ===")
    print(f"Confidence: {confidence}, Max error: ±{max_error}")
else:
    n_samples_per_config = 20  # Small sample for test run
    confidence = None
    max_error = None
    print(f"=== TEST RUN: Epsilon vs Robustness ===")

print(f"Epsilons: {epsilons}")
print(f"Digits: {list(digits)}")
print(f"Samples per config: {n_samples_per_config}")
if FULL_SCALE:
    print(f"Total verifications: {len(epsilons) * len(digits) * n_samples_per_config}")
print()

# Store results: for full scale, each entry is (rate, lower_bound, upper_bound)
# For test run, each entry is just rate
results = {digit: [] for digit in digits}

for digit in digits:
    if FULL_SCALE:
        print(f"\n{'='*60}")
        print(f"DIGIT {digit}")
        print(f"{'='*60}")
    else:
        print(f"\n--- Digit {digit} ---")

    digit_samples = x_test[y_test == digit]
    n_samples_actual = min(n_samples_per_config, len(digit_samples))
    np.random.seed(42 + digit)  # Consistent but different per digit

    for epsilon in epsilons:
        if FULL_SCALE:
            print(f"\n--- ε={epsilon} ---")
        robust_count = 0
        skipped_count = 0

        for i in range(n_samples_actual):
            x0 = digit_samples[i].flatten()
            lb_in = np.maximum(x0 - epsilon, 0)
            ub_in = np.minimum(x0 + epsilon, 1)

            # IBP bounds
            L1, U1 = ibp_affine_keras(lb_in, ub_in, W1, b1)
            Lz1, Uz1 = ibp_activation(L1, U1, gelu)
            L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)

            # Build polytope
            analyzer = PyomoPolyAnalyzer()
            x0_var = analyzer.add_input_box("x0", lb_in, ub_in)
            a1 = analyzer.add_affine(x0_var, W1, b1, "a1")
            z1 = analyzer.add_activation(a1, "z1", (L1, U1), tight_gelu_envelope)
            a2 = analyzer.add_affine(z1, W2, b2, "a2")
            z2 = analyzer.add_activation(a2, "z2", (L2, U2), tight_gelu_envelope)
            a3 = analyzer.add_affine(z2, W3, b3, "a3")

            # Check robustness with timeout: verify digit beats all others
            is_robust = True
            try:
                with timeout_context(30):  # 30 second timeout per sample
                    for k in range(10):
                        if k == digit:
                            continue
                        val, _ = analyzer.optimize(analyzer.model.a3[k] - analyzer.model.a3[digit], sense="max")
                        if val and val > 0:
                            is_robust = False
                            break

                if is_robust:
                    robust_count += 1
            except TimeoutError:
                # Skip difficult LPs - count as non-robust (conservative)
                skipped_count += 1
                if FULL_SCALE:
                    print(f"  TIMEOUT on sample {i+1} (skipped, counted as non-robust)")
                is_robust = False
            except Exception as e:
                # Skip on any other error
                skipped_count += 1
                if FULL_SCALE:
                    print(f"  ERROR on sample {i+1}: {str(e)[:50]} (skipped, counted as non-robust)")
                is_robust = False

            # Progress updates for full scale
            if FULL_SCALE and ((i + 1) % 50 == 0 or i + 1 == n_samples_actual):
                current_rate = robust_count / (i + 1)
                if skipped_count > 0:
                    print(f"  Progress: {i+1}/{n_samples_actual} ({100*current_rate:.1f}% robust, {skipped_count} skipped)")
                else:
                    print(f"  Progress: {i+1}/{n_samples_actual} ({100*current_rate:.1f}% robust)")

        robust_rate = robust_count / n_samples_actual

        if FULL_SCALE:
            # Store with confidence intervals
            lower_bound = max(0, robust_rate - max_error)
            upper_bound = min(1, robust_rate + max_error)
            results[digit].append((robust_rate, lower_bound, upper_bound))
            print(f"\n  RESULT: {robust_count}/{n_samples_actual} robust = {100*robust_rate:.1f}%")
            print(f"  95% CI: [{100*lower_bound:.1f}%, {100*upper_bound:.1f}%]")
            if skipped_count > 0:
                print(f"  (Skipped {skipped_count} difficult samples, counted as non-robust)")
        else:
            # Store just the rate
            results[digit].append(robust_rate)
            print(f"  ε={epsilon}: {robust_count}/{n_samples_actual} robust ({100*robust_rate:.1f}%)")

# Save results if full scale
if FULL_SCALE:
    print(f"\n{'='*60}")
    print("Saving results...")
    np.savez('epsilon_robustness_results.npz',
             results=results,
             epsilons=epsilons,
             n_samples=n_samples_per_config,
             confidence=confidence,
             max_error=max_error)
    print("Saved to: epsilon_robustness_results.npz")

# Create plot with separate subplots for each digit
print("\nGenerating plot...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, digit in enumerate(digits):
    ax = axes[idx]

    if FULL_SCALE:
        # Extract rates and confidence intervals
        rates = [100 * r[0] for r in results[digit]]
        lower = [100 * r[1] for r in results[digit]]
        upper = [100 * r[2] for r in results[digit]]
        errors_lower = [rates[i] - lower[i] for i in range(len(rates))]
        errors_upper = [upper[i] - rates[i] for i in range(len(rates))]

        # Bar plot with error bars
        x_pos = range(len(epsilons))
        ax.bar(x_pos, rates, color=f'C{digit}', alpha=0.7, edgecolor='black',
               yerr=[errors_lower, errors_upper], capsize=5, error_kw={'linewidth': 2})

        # Add percentage labels above error bars
        for i, val in enumerate(rates):
            ax.text(i, val + errors_upper[i] + 3, f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=8)
    else:
        # Simple bar plot for test run
        robustness_pct = [100 * r for r in results[digit]]
        ax.bar(range(len(epsilons)), robustness_pct, color=f'C{digit}', alpha=0.7, edgecolor='black')

        # Add percentage labels on bars
        for i, val in enumerate(robustness_pct):
            ax.text(i, val + 2, f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f'{eps:.2f}' for eps in epsilons])
    ax.set_ylim([0, 105])
    ax.set_title(f'Digit {digit}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Only add x-label to bottom row
    if idx >= 5:
        ax.set_xlabel('Epsilon (ε)', fontsize=10)

    # Only add y-label to leftmost column
    if idx % 5 == 0:
        ax.set_ylabel('Robustness Rate (%)', fontsize=10)

# Title based on run type
if FULL_SCALE:
    fig.suptitle(f'Robustness vs Perturbation Size (n={n_samples_per_config}, 95% CI)',
                 fontsize=16, fontweight='bold')
else:
    fig.suptitle(f'Robustness vs Perturbation Size (Test Run: n={n_samples_per_config})',
                 fontsize=16, fontweight='bold')

plt.tight_layout()

plt.savefig('epsilon_robustness_test.png', dpi=150, bbox_inches='tight')
print("Saved plot to: epsilon_robustness_test.png")
plt.close()

print("\n" + "="*60)
if FULL_SCALE:
    print("Full-scale test complete!")
else:
    print("Test run complete!")
print("="*60)
