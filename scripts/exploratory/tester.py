"""Legacy active-constraint smoke script.

This file is intentionally import-safe because `python -m unittest` discovers
files named `test*.py`, and `tester.py` matches that pattern.
"""

from __future__ import annotations


def main() -> None:
    from collections import Counter

    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist

    from verianet.activations import gelu, tight_gelu_envelope
    from verianet.bounds import ibp_activation, ibp_affine_keras
    from verianet.legacy.pyomo import PyomoPolyAnalyzer
    from verianet.paths import WEIGHTS_PATH

    data = np.load(WEIGHTS_PATH)
    W1, b1 = data["W1"], data["b1"]
    W2, b2 = data["W2"], data["b2"]
    W3, b3 = data["W3"], data["b3"]

    (x_test, y_test), _ = mnist.load_data()
    x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

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

        L1, U1 = ibp_affine_keras(lb_in, ub_in, W1, b1)
        Lz1, Uz1 = ibp_activation(L1, U1, gelu)
        z1_name = analyzer.add_activation(a1_name, "z1", (L1, U1), tight_gelu_envelope)
        a2_name = analyzer.add_affine(z1_name, W2, b2, "a2")
        L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)
        z2_name = analyzer.add_activation(a2_name, "z2", (L2, U2), tight_gelu_envelope)
        analyzer.add_affine(z2_name, W3, b3, "a3")

        for k in range(1, 10):
            obj = analyzer.model.a3[k] - analyzer.model.a3[0]
            val, _, active = analyzer.optimize_with_active(obj, sense="max")
            if val is None:
                continue
            for name, _side in active:
                constraint_freq[name] += 1

    for name, count in constraint_freq.most_common(20):
        print(name, count)


if __name__ == "__main__":
    main()
