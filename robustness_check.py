from basic import *
from helper import *
from basic import tight_gelu_envelope
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import math

# Load test data and resize to 7x7
(x_test, y_test), _ = mnist.load_data()
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

# Load weights
data = np.load("verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]

# Monte Carlo sampling of MNIST zeros
epsilon = 0.01
zeros = x_test[y_test == 0]
np.random.seed(42)

# Hoeffding: n = -ln(1-conf)/(2*err^2) for confidence bound
confidence = 0.95
max_error = 0.05  # estimate within 5%
n_samples = int(math.ceil(-math.log(1 - confidence) / (2 * max_error**2)))
n_samples = min(n_samples, len(zeros))  # don't exceed available zeros

print(f"\n=== MONTE CARLO ROBUSTNESS (ε={epsilon}, n={n_samples}) ===")
print(f"Confidence: {confidence}, Max error: ±{max_error}")

robust_count = 0
for i in range(n_samples):
    x0 = zeros[i].flatten()
    lb_in = np.maximum(x0 - epsilon, 0)
    ub_in = np.minimum(x0 + epsilon, 1)

    # IBP bounds for this sample
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

    # Check robustness
    is_robust = True
    for k in range(1, 10):
        val, _ = analyzer.optimize(analyzer.model.a3[k] - analyzer.model.a3[0], sense="max")
        if val and val > 0:
            is_robust = False
            break

    if is_robust:
        robust_count += 1

    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{n_samples} ({100*robust_count/(i+1):.1f}% robust)")

robust_rate = robust_count / n_samples
print(f"\nResult: {robust_count}/{n_samples} robust ({100*robust_rate:.1f}%)")
print(f"With {confidence} confidence: true rate in [{100*max(0,robust_rate-max_error):.1f}%, {100*min(1,robust_rate+max_error):.1f}%]")
