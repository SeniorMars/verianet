import numpy as np
import pyomo.environ as pyo
from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
import tensorflow as tf

from basic import gelu, tight_gelu_envelope, ibp_activation
from helper import PyomoPolyAnalyzer, ibp_affine_keras

data = np.load("verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]

# ---------- Load data (real MNIST, downsampled to 7×7) ----------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255.0


def forward_logits(x: np.ndarray) -> np.ndarray:
    """Forward pass through the network, return logits (shape (10,))."""
    x_flat = x.flatten()
    a1 = x_flat @ W1 + b1
    z1 = gelu(a1)
    a2 = z1 @ W2 + b2
    z2 = gelu(a2)
    return z2 @ W3 + b3


def build_polytope(x0_flat: np.ndarray, epsilon: float) -> Tuple[PyomoPolyAnalyzer, np.ndarray, np.ndarray]:
    """Build polyhedral relaxation for an ε-box around x0."""
    lb = np.maximum(x0_flat - epsilon, 0.0)
    ub = np.minimum(x0_flat + epsilon, 1.0)

    # IBP bounds
    L1, U1 = ibp_affine_keras(lb, ub, W1, b1)
    Lz1, Uz1 = ibp_activation(L1, U1, gelu)
    L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)

    analyzer = PyomoPolyAnalyzer()
    analyzer.add_input_box("x0", lb, ub)
    analyzer.add_affine("x0", W1, b1, "a1")
    analyzer.add_activation("a1", "z1", (L1, U1), tight_gelu_envelope)
    analyzer.add_affine("z1", W2, b2, "a2")
    analyzer.add_activation("a2", "z2", (L2, U2), tight_gelu_envelope)
    analyzer.add_affine("z2", W3, b3, "a3")

    return analyzer, lb, ub


def solve_margin(
    x0_flat: np.ndarray,
    epsilon: float,
    original_digit: int,
    target_digit: int,
    margin_target: float,
    step_desc: str = "",
    timelimit: Optional[float] = None,
) -> Tuple[bool, Optional[float], Optional[PyomoPolyAnalyzer]]:
    """
    Solve max (logit[target] - logit[original]) over the ε-box.

    Returns:
      (success, best_val, analyzer_if_success_else_None)
    """
    analyzer, lb, ub = build_polytope(x0_flat, epsilon)

    model = analyzer.model
    obj = model.a3[target_digit] - model.a3[original_digit]

    print(f"    [{step_desc}] ε={epsilon:.4f}: solving LP...", end="", flush=True)
    val, _ = analyzer.optimize_with_timelimit(obj, sense="max", time_limit=timelimit)
    print(f" val={val}", flush=True)

    if val is None:
        return False, None, None

    success = val >= margin_target
    if success:
        return True, val, analyzer
    else:
        return False, val, None

def find_minimal_counterfactual(
    x0: np.ndarray,
    original_digit: int,
    target_digit: int,
    margin: float = 0.1,
    max_epsilon: float = 0.5,
    n_binary_steps: int = 10,
    timelimit: Optional[float] = None,
) -> Optional[Dict]:
    """
    Two-phase search for smallest ε where we can achieve, in the TRUE network:
        logit[target] >= logit[original] + margin.

    We use the LP on the abstract polytope to propose a candidate point x,
    then check that inequality on the real GELU network. We DO NOT require
    that the argmax label is target_digit (we just show what it is).
    """
    x0_flat = x0.flatten()
    orig_logits = forward_logits(x0)
    pred_orig = int(np.argmax(orig_logits))

    print(
        f"  Original pred={pred_orig}, "
        f"logit[{original_digit}]={orig_logits[original_digit]:.2f}, "
        f"logit[{target_digit}]={orig_logits[target_digit]:.2f}"
    )

    # If the true network already satisfies the margin at ε = 0, we're done.
    current_margin = orig_logits[target_digit] - orig_logits[original_digit]
    if current_margin >= margin:
        print("  Already satisfies margin at ε=0.")
        return {
            "epsilon": 0.0,
            "image": x0,
            "delta": np.zeros_like(x0),
            "original_logits": orig_logits,
            "cf_logits": orig_logits,
            "predicted": pred_orig,
            "margin": current_margin,
        }

    # Phase 1: coarse scan to get a bracket [ε_low, ε_high]
    coarse_epsilons = np.linspace(0.01, max_epsilon, 8)
    bracket_low = 0.0
    bracket_high = None
    best_analyzer: Optional[PyomoPolyAnalyzer] = None

    for eps in coarse_epsilons:
        ok, val, analyzer = solve_margin(
            x0_flat,
            eps,
            original_digit,
            target_digit,
            margin_target=margin,
            step_desc="coarse",
            timelimit=timelimit,
        )
        if ok and analyzer is not None:
            # Check this candidate in the true network
            model = analyzer.model
            cf_flat = np.array([pyo.value(model.x0[i]) for i in range(49)])
            cf = cf_flat.reshape(7, 7)
            cf_logits = forward_logits(cf)
            margin_cf = cf_logits[target_digit] - cf_logits[original_digit]
            pred_cf = int(np.argmax(cf_logits))

            print(
                f"    [coarse] ε={eps:.4f}: LP ok (val={val:.3f}), "
                f"true margin={margin_cf:.3f}, pred_cf={pred_cf}"
            )

            if margin_cf >= margin:
                bracket_low = bracket_low
                bracket_high = eps
                best_analyzer = analyzer
                break

        bracket_low = eps

    if bracket_high is None:
        print(f"  No ε ≤ {max_epsilon} where TRUE margin ≥ {margin}.")
        return None

    print(f"Found TRUE bracket: low={bracket_low:.4f}, high={bracket_high:.4f}")

    # Phase 2: binary search using the TRUE margin as success criterion
    best_eps = bracket_high
    best_cf = None
    best_cf_logits = None
    best_margin = None
    best_pred = None

    for step in range(n_binary_steps):
        eps_mid = 0.5 * (bracket_low + bracket_high)
        ok, val, analyzer = solve_margin(
            x0_flat,
            eps_mid,
            original_digit,
            target_digit,
            margin_target=margin,
            step_desc=f"binary step {step}",
            timelimit=timelimit,
        )

        if not ok or analyzer is None:
            # LP couldn't even get the relaxed margin; need bigger ε
            bracket_low = eps_mid
            continue

        # Pull out concrete candidate and check in the true network
        model = analyzer.model
        cf_flat = np.array([pyo.value(model.x0[i]) for i in range(49)])
        cf = cf_flat.reshape(7, 7)
        cf_logits = forward_logits(cf)
        margin_cf = cf_logits[target_digit] - cf_logits[original_digit]
        pred_cf = int(np.argmax(cf_logits))

        print(
            f"    [binary] ε={eps_mid:.4f}: LP val={val:.3f}, "
            f"true margin={margin_cf:.3f}, pred_cf={pred_cf}"
        )

        if margin_cf >= margin:
            # TRUE success → try smaller ε
            best_eps = eps_mid
            best_cf = cf
            best_cf_logits = cf_logits
            best_margin = margin_cf
            best_pred = pred_cf
            bracket_high = eps_mid
        else:
            # TRUE failure → need bigger ε
            bracket_low = eps_mid

    if best_cf is None:
        print("  LP often said 'ok' but TRUE network never reached the margin.")
        return None

    return {
        "epsilon": best_eps,
        "image": best_cf,
        "delta": best_cf - x0,
        "original_logits": orig_logits,
        "cf_logits": best_cf_logits,
        "predicted": best_pred,
        "margin": best_margin,
    }

def find_cf_for_pair(
    orig: int,
    target: int,
    margin: float,
    max_epsilon: float,
    n_binary_steps: int,
    timelimit: Optional[float],
    max_starts: int = 10,
) -> Optional[Dict]:
    """
    Try up to max_starts different correctly-classified MNIST examples
    of class 'orig' and return the first successful counterfactual (if any).
    """
    for attempt in range(1, max_starts + 1):
        x0 = sample_digit_with_correct_pred(orig)
        if x0 is None:
            print(f"  [attempt {attempt}] No suitable starting example; stopping.")
            return None

        print(f"  [attempt {attempt}] starting from a pred={orig} example")
        result = find_minimal_counterfactual(
            x0,
            original_digit=orig,
            target_digit=target,
            margin=margin,
            max_epsilon=max_epsilon,
            n_binary_steps=n_binary_steps,
            timelimit=timelimit,
        )
        if result is not None:
            result["x0"] = x0   # keep original image around for plotting
            return result

    print(f"  No valid counterfactual found for {orig}→{target} after {max_starts} starts.")
    return None

def visualize_counterfactual(
    x0: np.ndarray,
    result: Dict,
    orig: int,
    target: int,
    margin: float = 0.1,
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

    # Original
    axes[0].imshow(x0, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(
        f"Original\npred={np.argmax(result['original_logits'])}", fontsize=11
    )
    axes[0].axis("off")

    # Counterfactual
    axes[1].imshow(result["image"], cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(
        f"Counterfactual\npred={result['predicted']}", fontsize=11
    )
    axes[1].axis("off")

    # Difference
    diff = result["delta"]
    limit = max(abs(diff.min()), abs(diff.max()), 0.01)
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[2].set_title(f"Δ (ε={result['epsilon']:.4f})\nRed=+ink, Blue=-ink", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    # Logit comparison
    x = np.arange(10)
    w = 0.35
    axes[3].bar(x - w / 2, result["original_logits"], w, label="Original", alpha=0.7)
    axes[3].bar(x + w / 2, result["cf_logits"], w, label="Counterfactual", alpha=0.7)
    axes[3].axhline(0, color="k", lw=0.5)
    axes[3].set_xticks(x)
    axes[3].set_xlabel("Digit")
    axes[3].set_ylabel("Logit")
    axes[3].legend(fontsize=8)
    axes[3].set_title("Logits", fontsize=11)

    plt.suptitle(
        f"Minimal logit counterfactual: "
        f"logit[{target}] ≥ logit[{orig}] + {margin:.2f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def sample_digit_with_correct_pred(label: int, max_tries: int = 200) -> Optional[np.ndarray]:
    """Pick a real MNIST test image with given label that the NN classifies correctly."""
    idxs = np.where(y_test == label)[0]
    if len(idxs) == 0:
        print(f"[warn] no test examples for label {label}")
        return None

    np.random.shuffle(idxs)
    tried = 0
    for idx in idxs:
        x = x_test[idx]
        pred = int(np.argmax(forward_logits(x)))
        tried += 1
        if pred == label:
            return x
        if tried >= max_tries:
            break

    print(f"[warn] Could not find test {label} with pred={label} in {tried} tries.")
    return None

if __name__ == "__main__":

    pairs = [
        (1, 2),
        (0, 6),
        (4, 9),
        (3, 8),
        (7, 1),
    ]

    LP_TIME_LIMIT = 30
    MARGIN = 0.1
    MAX_EPS = 0.5

    for orig, target in pairs:
        print(f"\n{'=' * 60}")
        print(f"Finding: {orig} → {target}")
        print("=" * 60)

        result = find_cf_for_pair(
            orig,
            target,
            margin=MARGIN,
            max_epsilon=MAX_EPS,
            n_binary_steps=8,
            timelimit=LP_TIME_LIMIT,
            max_starts=10,
        )

        if result is None:
            print("  ✗ No valid counterfactual found within search range.")
        else:
            print(
                f"  ✓ Found at ε={result['epsilon']:.4f}, "
                f"concrete margin={result['margin']:.2f}, "
                f"pred_cf={result['predicted']}"
            )
            # use stored x0 for plotting
            visualize_counterfactual(
                result["x0"],
                result,
                orig,
                target,
                margin=MARGIN,
                save_path=f"counterfactual_{orig}_to_{target}.png",
            )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
