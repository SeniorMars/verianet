import numpy as np
import pyomo.environ as pyo
from typing import Optional, Dict, Tuple
import matplotlib.pyplot as plt  # <- fixed

from basic import gelu, tight_gelu_envelope, ibp_activation
from helper import PyomoPolyAnalyzer, ibp_affine_keras

data = np.load("verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]


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

    success = val > margin_target
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
    Two-phase search for smallest ε where we can achieve:
        logit[target] ≥ logit[original] + margin.

    1) Coarse scan over a grid of ε values to find [ε_low, ε_high] bracket.
    2) Binary search inside that bracket.
    """
    x0_flat = x0.flatten()
    orig_logits = forward_logits(x0)
    pred = int(np.argmax(orig_logits))

    print(
        f"  Original pred={pred}, "
        f"logit[{original_digit}]={orig_logits[original_digit]:.2f}, "
        f"logit[{target_digit}]={orig_logits[target_digit]:.2f}"
    )

    # If the network already prefers target with margin, ε = 0 is enough.
    current_margin = orig_logits[target_digit] - orig_logits[original_digit]
    if current_margin > margin:
        print("  Already flipped at ε=0 (network already prefers target).")
        return {
            "epsilon": 0.0,
            "image": x0,
            "delta": np.zeros_like(x0),
            "original_logits": orig_logits,
            "cf_logits": orig_logits,
            "predicted": pred,
            "margin": current_margin,
        }

    # Use a small set of ε candidates to find first success.
    coarse_epsilons = np.linspace(0.01, max_epsilon, 6)
    bracket_low = 0.0
    bracket_high = None

    print("Phase 1: coarse scan")
    for eps in coarse_epsilons:
        ok, val, _ = solve_margin(
            x0_flat, eps, original_digit, target_digit,
            margin_target=margin,
            step_desc=f"coarse",
            timelimit=timelimit,
        )
        if ok:
            bracket_high = eps
            break
        else:
            bracket_low = eps

    if bracket_high is None:
        print(f"No ε≤{max_epsilon} where LP margin> {margin}.")
        return None

    print(f"Found bracket: low={bracket_low:.4f}, high={bracket_high:.4f}")

    best_eps = bracket_high
    best_analyzer: Optional[PyomoPolyAnalyzer] = None

    print("Phase 2: binary search")
    for step in range(n_binary_steps):
        eps_mid = 0.5 * (bracket_low + bracket_high)
        ok, val, analyzer = solve_margin(
            x0_flat, eps_mid, original_digit, target_digit,
            margin_target=margin,
            step_desc=f"binary step {step}",
            timelimit=timelimit,
        )

        if ok:
            # Success → try smaller epsilon
            best_eps = eps_mid
            best_analyzer = analyzer
            bracket_high = eps_mid
        else:
            # Failure → need larger epsilon
            bracket_low = eps_mid

    # If we never kept an analyzer from success, re-solve once at best_eps to get one.
    if best_analyzer is None:
        ok, val, best_analyzer = solve_margin(
            x0_flat, best_eps, original_digit, target_digit,
            margin_target=margin,
            step_desc="final",
            timelimit=timelimit,
        )
        if not ok or best_analyzer is None:
            print("  Unexpected failure when re-solving at best ε.")
            return None

    # Extract counterfactual from best_analyzer
    model = best_analyzer.model
    cf_flat = np.array([pyo.value(model.x0[i]) for i in range(49)])
    cf = cf_flat.reshape(7, 7)
    cf_logits = forward_logits(cf)
    pred_cf = int(np.argmax(cf_logits))
    margin_cf = cf_logits[target_digit] - cf_logits[original_digit]

    return {
        "epsilon": best_eps,
        "image": cf,
        "delta": cf - x0,
        "original_logits": orig_logits,
        "cf_logits": cf_logits,
        "predicted": pred_cf,
        "margin": margin_cf,
    }

def visualize_counterfactual(
    x0: np.ndarray,
    result: Dict,
    orig: int,
    target: int,
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
    color = "green" if result["predicted"] == target else "orange"
    axes[1].set_title(
        f"Counterfactual\npred={result['predicted']}", fontsize=11, color=color
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

    plt.suptitle(f"Minimal Counterfactual: {orig} → {target}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()

def generate_digit_pattern(digit: int, n: int = 1) -> np.ndarray:
    # np.random.seed(42 + digit)
    samples = []

    for _ in range(n):
        img = np.random.uniform(0, 0.2, (7, 7))
        if digit == 0:
            img[1:6, 0] += 0.5; img[1:6, 6] += 0.5
            img[0, 1:6] += 0.5; img[6, 1:6] += 0.5
        elif digit == 1:
            img[:, 3] += 0.7
        elif digit == 2:
            img[0, :] += 0.5
            for i in range(7): img[i, 6-i] += 0.3
            img[6, :] += 0.5
        elif digit == 3:
            img[0, :] += 0.5; img[3, :] += 0.5; img[6, :] += 0.5
        elif digit == 4:
            img[:4, 0] += 0.5; img[3, :] += 0.5; img[:, 4] += 0.6
        elif digit == 5:
            img[0, :] += 0.5; img[3, :] += 0.5; img[6, :] += 0.5
            img[0:4, 0] += 0.3; img[3:, 6] += 0.3
        elif digit == 6:
            img[:, 0] += 0.5; img[3, :] += 0.4
            img[6, 1:6] += 0.4; img[3:, 6] += 0.4
        elif digit == 7:
            img[0, :] += 0.6
            for i in range(7): img[i, 6-i] += 0.4
        elif digit == 8:
            img[0, 1:6] += 0.4; img[3, 1:6] += 0.4; img[6, 1:6] += 0.4
            img[0:4, 0] += 0.3; img[0:4, 6] += 0.3
            img[3:, 0] += 0.3; img[3:, 6] += 0.3
        elif digit == 9:
            img[0, 1:6] += 0.5; img[0:4, 0] += 0.4; img[0:4, 6] += 0.4
            img[3, 1:6] += 0.4; img[3:, 6] += 0.5
        img += np.random.uniform(-0.1, 0.1, (7, 7))
        samples.append(np.clip(img, 0, 1))

    return np.array(samples)

if __name__ == "__main__":

    pairs = [
        (1, 2),
        (0, 6),
        (4, 9),
        (3, 8),
        (7, 1),
    ]

    # Global timelimit per LP in seconds (tune or set to None)
    LP_TIME_LIMIT = 30

    for orig, target in pairs:
        print(f"\n{'=' * 60}")
        print(f"Finding: {orig} → {target}")
        print("=" * 60)

        x0 = generate_digit_pattern(orig, 1)[0]
        result = find_minimal_counterfactual(
            x0,
            original_digit=orig,
            target_digit=target,
            margin=0.1,
            max_epsilon=0.5,
            n_binary_steps=8,
            timelimit=LP_TIME_LIMIT,
        )

        if result is None:
            print("  ✗ No counterfactual found within search range.")
        else:
            print(f"  ✓ Found at ε={result['epsilon']:.4f}, "
                  f"margin={result['margin']:.2f}, "
                  f"pred={result['predicted']}")
            visualize_counterfactual(
                x0,
                result,
                orig,
                target,
                save_path=f"counterfactual_{orig}_to_{target}.png",
            )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
