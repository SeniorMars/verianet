import sys
sys.path.append('..')

from basic import *
from helper import *
from basic import tight_gelu_envelope
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
import signal
from contextlib import contextmanager
import pyomo.environ as pyo

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

# SETUP & DATA
print("INITIALIZING LP SOLVER FOR HYPOTHESIS TESTING")

# Load weights
data = np.load("../verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]

# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

# Helper to run the solver robustly
def solve_lp(lb, ub, target_expr, sense="max", timeout=30):
    """
    Robustly solves the LP using IBP pre-calculation and timeouts.
    """
    L1, U1 = ibp_affine_keras(lb, ub, W1, b1)
    Lz1, Uz1 = ibp_activation(L1, U1, gelu)
    L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)
    
    analyzer = PyomoPolyAnalyzer()
    x0_var = analyzer.add_input_box("x0", lb, ub)
    a1 = analyzer.add_affine(x0_var, W1, b1, "a1")
    z1 = analyzer.add_activation(a1, "z1", (L1, U1), tight_gelu_envelope)
    a2 = analyzer.add_affine(z1, W2, b2, "a2")
    z2 = analyzer.add_activation(a2, "z2", (L2, U2), tight_gelu_envelope)
    a3 = analyzer.add_affine(z2, W3, b3, "a3")
    
    try:
        with timeout_context(timeout):
            # We must evaluate the target expression in the context of the new analyzer
            # The passed 'target_expr' is a lambda that takes the analyzer and returns the pyomo expression
            expr = target_expr(analyzer)
            print(f"  > Solving LP with {len(analyzer.model.x0)} variables...")
            print(f"  > Input bounds: [{lb.min():.3f}, {ub.max():.3f}]")
            val, _ = analyzer.optimize(expr, sense=sense)

            if val is None:
                print(f"  [!] Solver returned None (infeasible or error)")
                return None, None
            
            # Extract image
            img = np.zeros(49)
            for i in range(49):
                img[i] = pyo.value(analyzer.model.x0[i])
            return val, img.reshape(7, 7)
            
    except TimeoutError:
        print(f"  [!] Solver Timed Out ({timeout}s)")
        return None, None
    except Exception as e:
        print(f"  [!] Solver Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# EXPERIMENT: PLATONIC IDEALS
print("\nGENERATING PLATONIC IDEALS (Layer 1)")

# Global bounds [0, 1]
lb_global = np.zeros(49)
ub_global = np.ones(49)

ideals = [] # Stores (MaxImg, MinImg, MaxVal, MinVal)

for i in range(3):
    print(f"Optimizing Neuron L1_N{i}...")
    
    # Maximize pre-activation a1 (Linear and robust)
    # Note: Maximizing a1 is strictly correlated with maximizing z1
    target_lambda = lambda a: a.model.a1[i]
    
    val_max, img_max = solve_lp(lb_global, ub_global, target_lambda, "max")
    val_min, img_min = solve_lp(lb_global, ub_global, target_lambda, "min")
    
    if img_max is None: img_max = np.zeros((7,7)) # Fallback visual
    if img_min is None: img_min = np.zeros((7,7))
    
    ideals.append((img_max, img_min, val_max, val_min))

# Plotting
fig = plt.figure(figsize=(15, 10), layout="constrained")
gs = fig.add_gridspec(3, 4)
row_labels = ["L1_N0\n(Frame?)", "L1_N1\n(Spine?)", "L1_N2\n(Belt?)"]

for i in range(3):
    img_max, img_min, v_max, v_min = ideals[i]
    w = W1[:, i].reshape(7, 7)
    
    # Max
    ax = fig.add_subplot(gs[i, 0])
    ax.imshow(img_max, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Max Stimulus\nVal={v_max:.1f}" if v_max else "Solver Failed", color='green', fontweight='bold')
    ax.set_ylabel(row_labels[i], fontweight='bold', fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    
    # Min
    ax = fig.add_subplot(gs[i, 1])
    ax.imshow(img_min, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Min Stimulus\nVal={v_min:.1f}" if v_min else "Solver Failed", color='red', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

    # Weights
    ax = fig.add_subplot(gs[i, 2])
    limit = np.max(np.abs(w))
    ax.imshow(w, cmap='RdBu', vmin=-limit, vmax=limit) # Correct RdBu (Blue=+, Red=-)
    ax.set_title("Actual Weights\n(Blue=+, Red=-)")
    ax.set_xticks([]); ax.set_yticks([])

    # Net Preference
    ax = fig.add_subplot(gs[i, 3])
    diff = img_max - img_min
    ax.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
    ax.set_title("Net Preference\n(Max - Min)")
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle("Platonic Ideals (LP Generated)", fontsize=16, fontweight='bold')
plt.savefig("platonic_ideals.png", dpi=150)
print("Saved to 'platonic_ideals.png'")

# CONSTRAINED OPTIMIZATION
print("\nCONSTRAINED OPTIMIZATION")

EPSILON = 0.1  # Epsilon for perturbation ball

# Helper to compute activations
def get_acts(x):
    a1 = x.flatten() @ W1 + b1
    z1 = gelu(a1)
    return z1

# Store results for all digits
digit_results = []

for digit in range(10):
    print(f"\n>>> Processing Digit {digit}")

    # Find all examples of this digit
    indices = np.where(y_test == digit)[0]
    if len(indices) == 0:
        print(f"  No examples found for digit {digit}, skipping...")
        digit_results.append(None)
        continue

    # Collect correctly classified examples
    correct_indices = []
    for idx in indices:
        x_candidate = x_test[idx].flatten()
        # Quick forward pass to check prediction
        a1_check = x_candidate @ W1 + b1
        z1_check = gelu(a1_check)
        a2_check = z1_check @ W2 + b2
        z2_check = gelu(a2_check)
        logits_check = z2_check @ W3 + b3
        pred = np.argmax(logits_check)

        if pred == digit:
            correct_indices.append(idx)

    if len(correct_indices) == 0:
        print(f"  Warning: No correctly classified examples found for digit {digit}, skipping...")
        digit_results.append(None)
        continue

    print(f"  Found {len(correct_indices)} correctly classified examples")

    # Try to optimize each correctly classified example until one succeeds
    optimization_successful = False
    attempts = 0
    max_attempts = min(10, len(correct_indices))  # Try up to 10 examples

    for idx in correct_indices[:max_attempts]:
        attempts += 1
        x0_img = x_test[idx]
        x0_flat = x0_img.flatten()

        # Get original logit
        a1_check = x0_flat @ W1 + b1
        z1_check = gelu(a1_check)
        a2_check = z1_check @ W2 + b2
        z2_check = gelu(a2_check)
        logits_check = z2_check @ W3 + b3

        print(f"  Attempt {attempts}/{max_attempts}: example index {idx}, original logit[{digit}] = {logits_check[digit]:.3f}")

        # Define tight bounds
        lb_local = np.maximum(x0_flat - EPSILON, 0.0)
        ub_local = np.minimum(x0_flat + EPSILON, 1.0)

        target_lambda_d = lambda a, d=digit: a.model.a3[d]  # Capture digit in closure
        val_opt, x_opt_img = solve_lp(lb_local, ub_local, target_lambda_d, "max", timeout=60)

        if val_opt is not None:
            # Compute actual logits from the real network
            x_opt_flat = x_opt_img.flatten()
            a1_opt = x_opt_flat @ W1 + b1
            z1_opt = gelu(a1_opt)
            a2_opt = z1_opt @ W2 + b2
            z2_opt = gelu(a2_opt)
            logits_opt = z2_opt @ W3 + b3

            # Get logit values
            logit_old = logits_check[digit]
            logit_new = logits_opt[digit]

            # Success! LP returned a solution
            delta = x_opt_img - x0_img

            z1_old = get_acts(x0_img)
            z1_new = get_acts(x_opt_img)

            digit_results.append({
                'digit': digit,
                'x0': x0_img,
                'x_opt': x_opt_img,
                'delta': delta,
                'z1_old': z1_old,
                'z1_new': z1_new,
                'logit_old': logit_old,
                'logit_new': logit_new,
                'logit_improvement': logit_new - logit_old
            })
            print(f"Success: {logit_old:.3f} → {logit_new:.3f} (Δ={logit_new-logit_old:+.3f})")
            optimization_successful = True
            break
        else:
            print(f"Optimization failed, trying next example...")

    if not optimization_successful:
        print(f"All {attempts} attempts failed for digit {digit}")
        digit_results.append(None)

# Create individual visualizations for each digit
print("\n>>> Generating individual plots for each digit...")

for i, result in enumerate(digit_results):
    if result is None:
        print(f"  Skipping digit {i} (optimization failed)")
        continue

    # Create figure for this digit
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    gs = fig.add_gridspec(2, 3)

    # Original
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(result['x0'], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Original Digit {result['digit']}\nLogit={result['logit_old']:.2f}")
    ax.axis('off')

    # Delta
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(result['delta'], cmap='RdBu', vmin=-EPSILON, vmax=EPSILON)
    ax.set_title(f"Optimization Δ={result['logit_improvement']:+.2f}\n(Blue=+Ink, Red=-Ink)")
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Optimized
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(result['x_opt'], cmap='gray', vmin=0, vmax=1)
    ax.set_title(f"Optimized Digit {result['digit']}\nLogit={result['logit_new']:.2f}")
    ax.axis('off')

    # Activation Change
    ax = fig.add_subplot(gs[1, :])
    labels = ['N0 (Frame)', 'N1 (Spine)', 'N2 (Belt)']
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, result['z1_old'], width, label='Original', color='gray')
    ax.bar(x + width/2, result['z1_new'], width, label='Optimized', color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Mechanism Change: Which features did the LP boost?")
    ax.set_ylabel('Activation')
    ax.grid(True, alpha=0.3, axis='y')

    # Save individual plot
    filename = f"digit_{result['digit']}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close(fig)

print(f"\n>>> All visualizations complete!")
plt.show()