from helper import PyomoPolyAnalyzer, ibp_affine_keras, relu_envelope
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

path = "verysmallnn_weights.npz"
data = np.load(path)
W1, b1 = data["W1"], data["b1"]
W2, b2 = data["W2"], data["b2"]
W3, b3 = data["W3"], data["b3"]
weights = [W1, W2, W3]
biases  = [b1, b2, b3]

(x_test, y_test), _ = mnist.load_data()
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

digit = 0
epsilon = 0.01
zeros = x_test[y_test == 0]

constraint_freq = Counter()

for i in range(50): 
    x0 = zeros[i].flatten()
    lb_in = np.maximum(x0 - epsilon, 0)
    ub_in = np.minimum(x0 + epsilon, 1)

    analyzer = PyomoPolyAnalyzer()
    x0_name = analyzer.add_input_box("x0", lb_in, ub_in)
    a1_name = analyzer.add_affine(x0_name, W1, b1, "a1")
    # IBP for bounds
    L1, U1 = ibp_affine_keras(lb_in, ub_in, W1, b1)
    z1_name = analyzer.add_activation(a1_name, "z1", (L1, U1), relu_envelope)  # or tight_gelu_envelope
    a2_name = analyzer.add_affine(z1_name, W2, b2, "a2")
    L2, U2 = ibp_affine_keras(L1, U1, W2, b2)
    z2_name = analyzer.add_activation(a2_name, "z2", (L2, U2), relu_envelope)
    a3_name = analyzer.add_affine(z2_name, W3, b3, "a3")

    # Check class-0 vs others and log active constraints
    for k in range(1, 10):
        obj = analyzer.model.a3[k] - analyzer.model.a3[0]
        val, _, active = analyzer.optimize_with_active(obj, sense="max")
        if val is None:
            continue  # infeasible or solver fail

        # just count names; you could separate by robust/non-robust
        for name, side in active:
            constraint_freq[name] += 1

for name, count in constraint_freq.most_common(20):
    print(name, count)