"""
When we optimize over the polytope, the LP solution "presses against"
certain constraints. 

- Different digit classes should have characteristic "constraint signatures"
- These signatures correspond to interpretable features of the digit
"""

import numpy as np
import pyomo.environ as pyo
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from scipy import stats

from basic import gelu, tight_gelu_envelope, ibp_activation
from helper import PyomoPolyAnalyzer, ibp_affine_keras

# Load weights
data = np.load("verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]

# Generate synthetic MNIST-like data (7x7 images)
# Each "digit" class has a characteristic pattern
print("Generating synthetic digit data...")

def generate_digit_pattern(digit: int, n_samples: int = 100) -> np.ndarray:
    """Generate synthetic 7x7 patterns for each digit class."""
    np.random.seed(42 + digit)
    samples = []
    
    for _ in range(n_samples):
        # Start with noise
        img = np.random.uniform(0, 0.2, (7, 7))
        
        # Add digit-specific patterns
        if digit == 0:  # Circle-ish
            img[1:6, 0] += 0.5
            img[1:6, 6] += 0.5
            img[0, 1:6] += 0.5
            img[6, 1:6] += 0.5
        elif digit == 1:  # Vertical line
            img[:, 3] += 0.7
        elif digit == 2:  # Top + diagonal + bottom
            img[0, :] += 0.5
            for i in range(7): img[i, 6-i] += 0.3
            img[6, :] += 0.5
        elif digit == 3:  # Three horizontal bars
            img[0, :] += 0.5
            img[3, :] += 0.5
            img[6, :] += 0.5
        elif digit == 4:  # L + vertical
            img[:4, 0] += 0.5
            img[3, :] += 0.5
            img[:, 4] += 0.6
        elif digit == 5:  # S-like
            img[0, :] += 0.5
            img[3, :] += 0.5
            img[6, :] += 0.5
            img[0:4, 0] += 0.3
            img[3:, 6] += 0.3
        elif digit == 6:  # 6-like
            img[:, 0] += 0.5
            img[3, :] += 0.4
            img[6, 1:6] += 0.4
            img[3:, 6] += 0.4
        elif digit == 7:  # Top + diagonal
            img[0, :] += 0.6
            for i in range(7): img[i, 6-i] += 0.4
        elif digit == 8:  # Two stacked circles
            img[0, 1:6] += 0.4
            img[3, 1:6] += 0.4
            img[6, 1:6] += 0.4
            img[0:4, 0] += 0.3
            img[0:4, 6] += 0.3
            img[3:, 0] += 0.3
            img[3:, 6] += 0.3
        elif digit == 9:  # 9-like
            img[0, 1:6] += 0.5
            img[0:4, 0] += 0.4
            img[0:4, 6] += 0.4
            img[3, 1:6] += 0.4
            img[3:, 6] += 0.5
        
        # Add noise and clip
        img += np.random.uniform(-0.1, 0.1, (7, 7))
        img = np.clip(img, 0, 1)
        samples.append(img)
    
    return np.array(samples)

# Generate data for all digits
x_test = []
y_test = []
for d in range(10):
    patterns = generate_digit_pattern(d, n_samples=100)
    x_test.append(patterns)
    y_test.extend([d] * len(patterns))

x_test = np.concatenate(x_test, axis=0)
y_test = np.array(y_test)
print(f"Generated {len(x_test)} synthetic digit images (7x7)")


@dataclass
class ConstraintProfile:
    """Profile of active constraints for one optimization"""
    digit: int
    sample_idx: int
    
    # Active input pixel indices
    input_lower: List[int] = field(default_factory=list)  # pixels pushed DOWN
    input_upper: List[int] = field(default_factory=list)  # pixels pushed UP
    
    # Active GELU constraints (neuron indices)
    L1_gelu_lower: List[int] = field(default_factory=list)
    L1_gelu_upper: List[int] = field(default_factory=list)
    L2_gelu_lower: List[int] = field(default_factory=list)
    L2_gelu_upper: List[int] = field(default_factory=list)
    
    # Optimal value achieved
    optimal_value: Optional[float] = None


def build_polytope_and_solve(x0: np.ndarray, epsilon: float, 
                              target_digit: int) -> tuple:
    """
    Build polytope, maximize target logit, return (analyzer, value, profile).
    """
    x0_flat = x0.flatten()
    lb = np.maximum(x0_flat - epsilon, 0)
    ub = np.minimum(x0_flat + epsilon, 1)
    
    # IBP for bounds
    L1, U1 = ibp_affine_keras(lb, ub, W1, b1)
    Lz1, Uz1 = ibp_activation(L1, U1, gelu)
    L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)
    
    # Build polytope
    analyzer = PyomoPolyAnalyzer()
    analyzer.add_input_box("x0", lb, ub)
    analyzer.add_affine("x0", W1, b1, "a1")
    analyzer.add_activation("a1", "z1", (L1, U1), tight_gelu_envelope)
    analyzer.add_affine("z1", W2, b2, "a2")
    analyzer.add_activation("a2", "z2", (L2, U2), tight_gelu_envelope)
    analyzer.add_affine("z2", W3, b3, "a3")
    
    # Optimize
    val, _ = analyzer.optimize(analyzer.model.a3[target_digit], sense="max")
    
    return analyzer, val


def extract_active_constraints(analyzer: PyomoPolyAnalyzer, 
                                threshold: float = 1e-3) -> ConstraintProfile:
    """Extract which constraints are binding by checking if body ≈ 0."""
    profile = ConstraintProfile(digit=-1, sample_idx=-1)
    
    model = analyzer.model
    
    # Check input constraints
    for i in range(49):
        x_val = pyo.value(model.x0[i])
        lb = pyo.value(model.x0_lb[i].lower)
        ub = pyo.value(model.x0_ub[i].upper)
        
        if abs(x_val - lb) < threshold:
            profile.input_lower.append(i)
        if abs(ub - x_val) < threshold:
            profile.input_upper.append(i)
    
    # GELU constraints are formulated as body <= 0
    # Active constraint means body ≈ 0
    for comp in model.component_objects(pyo.Constraint, active=True):
        name = str(comp.name)
        
        # Only look at GELU envelope constraints
        if not (('z1_lower' in name) or ('z1_upper' in name) or 
                ('z2_lower' in name) or ('z2_upper' in name)):
            continue
        
        try:
            body_val = pyo.value(comp.body)
            
            # Constraint is active if body ≈ 0 (since constraint is body <= 0)
            if abs(body_val) < threshold:
                parts = name.split('_')
                neuron = int(parts[2])
                
                if 'z1_lower' in name and neuron not in profile.L1_gelu_lower:
                    profile.L1_gelu_lower.append(neuron)
                elif 'z1_upper' in name and neuron not in profile.L1_gelu_upper:
                    profile.L1_gelu_upper.append(neuron)
                elif 'z2_lower' in name and neuron not in profile.L2_gelu_lower:
                    profile.L2_gelu_lower.append(neuron)
                elif 'z2_upper' in name and neuron not in profile.L2_gelu_upper:
                    profile.L2_gelu_upper.append(neuron)
        except:
            continue
    
    return profile

def analyze_digit_class(digit: int, n_samples: int = 30, 
                        epsilon: float = 0.05) -> Dict:
    """Analyze constraint patterns for a digit class."""
    
    digit_mask = y_test == digit
    digit_samples = x_test[digit_mask][:n_samples]
    
    # Accumulators
    input_lower_counts = np.zeros(49)
    input_upper_counts = np.zeros(49)
    L1_lower_counts = np.zeros(3)
    L1_upper_counts = np.zeros(3)
    L2_lower_counts = np.zeros(3)
    L2_upper_counts = np.zeros(3)
    
    successful = 0
    profiles = []
    
    for i, x0 in enumerate(digit_samples):
        try:
            analyzer, val = build_polytope_and_solve(x0, epsilon, digit)
            
            if val is None:
                continue
            
            profile = extract_active_constraints(analyzer)
            profile.digit = digit
            profile.sample_idx = i
            profile.optimal_value = val
            profiles.append(profile)
            
            # Accumulate
            for idx in profile.input_lower:
                input_lower_counts[idx] += 1
            for idx in profile.input_upper:
                input_upper_counts[idx] += 1
            for idx in profile.L1_gelu_lower:
                L1_lower_counts[idx] += 1
            for idx in profile.L1_gelu_upper:
                L1_upper_counts[idx] += 1
            for idx in profile.L2_gelu_lower:
                L2_lower_counts[idx] += 1
            for idx in profile.L2_gelu_upper:
                L2_upper_counts[idx] += 1
            
            successful += 1
            
        except Exception as e:
            continue
    
    # Normalize
    if successful > 0:
        return {
            'digit': digit,
            'n_successful': successful,
            'input_lower_freq': input_lower_counts / successful,
            'input_upper_freq': input_upper_counts / successful,
            'L1_lower_freq': L1_lower_counts / successful,
            'L1_upper_freq': L1_upper_counts / successful,
            'L2_lower_freq': L2_lower_counts / successful,
            'L2_upper_freq': L2_upper_counts / successful,
            'profiles': profiles
        }
    else:
        return {
            'digit': digit,
            'n_successful': 0,
            'input_lower_freq': np.zeros(49),
            'input_upper_freq': np.zeros(49),
            'L1_lower_freq': np.zeros(3),
            'L1_upper_freq': np.zeros(3),
            'L2_lower_freq': np.zeros(3),
            'L2_upper_freq': np.zeros(3),
            'profiles': []
        }


def generate_hypotheses(results: Dict[int, Dict]) -> List[Dict]:
    """
    Generate testable hypotheses from constraint patterns.
    
    Returns list of hypothesis dicts with:
    - statement: natural language hypothesis
    - test_type: 'proportion', 'comparison', 'correlation'
    - digits_involved: list of digits
    - metric: what to measure
    """
    hypotheses = []
    
    # Type 1: Neuron dominance hypotheses
    for digit, res in results.items():
        for i in range(3):
            # Check if L1 neuron i has consistently active upper bound
            if res['L1_upper_freq'][i] > 0.5:
                hypotheses.append({
                    'statement': f"Digit {digit} consistently activates L1_N{i} to its upper envelope "
                                f"(freq={res['L1_upper_freq'][i]:.2f}). This neuron may detect a feature "
                                f"characteristic of {digit}.",
                    'test_type': 'proportion',
                    'digits_involved': [digit],
                    'layer': 'L1',
                    'neuron': i,
                    'constraint_type': 'upper',
                    'observed_freq': res['L1_upper_freq'][i]
                })
            
            # Check for suppressed neurons
            if res['L1_lower_freq'][i] > 0.5:
                hypotheses.append({
                    'statement': f"Digit {digit} consistently hits L1_N{i}'s lower envelope "
                                f"(freq={res['L1_lower_freq'][i]:.2f}). This neuron may detect a feature "
                                f"that {digit} LACKS.",
                    'test_type': 'proportion',
                    'digits_involved': [digit],
                    'layer': 'L1',
                    'neuron': i,
                    'constraint_type': 'lower',
                    'observed_freq': res['L1_lower_freq'][i]
                })
    
    # Type 2: Cross-digit comparison hypotheses
    for i in range(3):
        # Find digits where neuron i is most/least active
        upper_freqs = [(d, results[d]['L1_upper_freq'][i]) for d in results]
        upper_freqs.sort(key=lambda x: -x[1])
        
        if upper_freqs[0][1] > 0.3 and upper_freqs[-1][1] < 0.1:
            high_digit = upper_freqs[0][0]
            low_digit = upper_freqs[-1][0]
            hypotheses.append({
                'statement': f"L1_N{i} discriminates between digits: highly active for {high_digit} "
                            f"(freq={upper_freqs[0][1]:.2f}) but rarely for {low_digit} "
                            f"(freq={upper_freqs[-1][1]:.2f}).",
                'test_type': 'comparison',
                'digits_involved': [high_digit, low_digit],
                'layer': 'L1',
                'neuron': i,
                'constraint_type': 'upper'
            })
    
    # Type 3: Pixel pattern hypotheses
    for digit, res in results.items():
        # Find pixels that frequently hit upper bound (want more ink)
        hot_pixels = np.where(res['input_upper_freq'] > 0.3)[0]
        if len(hot_pixels) > 0:
            positions = [(p // 7, p % 7) for p in hot_pixels[:5]]
            hypotheses.append({
                'statement': f"Digit {digit} consistently wants MORE ink at pixels {positions}. "
                            f"These may be characteristic stroke locations.",
                'test_type': 'pixel_pattern',
                'digits_involved': [digit],
                'hot_pixels': hot_pixels.tolist()
            })
        
        # Find pixels that frequently hit lower bound (want less ink)
        cold_pixels = np.where(res['input_lower_freq'] > 0.3)[0]
        if len(cold_pixels) > 0:
            positions = [(p // 7, p % 7) for p in cold_pixels[:5]]
            hypotheses.append({
                'statement': f"Digit {digit} consistently wants LESS ink at pixels {positions}. "
                            f"These may need to be blank for this digit.",
                'test_type': 'pixel_pattern',
                'digits_involved': [digit],
                'cold_pixels': cold_pixels.tolist()
            })
    
    return hypotheses


def test_hypothesis_proportion(hypothesis: Dict, results: Dict, 
                               null_prob: float = 0.1) -> Dict:
    """
    Test a proportion hypothesis using binomial test.
    
    H0: The constraint is active with probability null_prob (by chance)
    H1: The constraint is active more often than chance
    """
    digit = hypothesis['digits_involved'][0]
    layer = hypothesis['layer']
    neuron = hypothesis['neuron']
    ctype = hypothesis['constraint_type']
    
    res = results[digit]
    n = res['n_successful']
    
    if ctype == 'upper':
        freq = res[f'{layer}_upper_freq'][neuron]
    else:
        freq = res[f'{layer}_lower_freq'][neuron]
    
    k = int(freq * n)  # number of "successes"
    
    # One-sided binomial test
    p_value = stats.binomtest(k, n, null_prob, alternative='greater').pvalue
    
    return {
        'hypothesis': hypothesis['statement'],
        'n_samples': n,
        'observed_freq': freq,
        'null_prob': null_prob,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'interpretation': f"{'SUPPORTED' if p_value < 0.05 else 'NOT SUPPORTED'}: "
                         f"Observed {freq:.2f} vs null {null_prob:.2f} (p={p_value:.4f})"
    }


def visualize_signatures(results: Dict[int, Dict], save_path: str = None):
    """Visualize constraint signatures for all digits."""
    
    fig = plt.figure(figsize=(20, 14))
    
    # Create layout: 4 rows x 10 columns
    # Row 0: Input pixels → LOWER bound (want less ink)
    # Row 1: Input pixels → UPPER bound (want more ink)
    # Row 2: L1 GELU constraints
    # Row 3: L2 GELU constraints
    
    for digit in range(10):
        if digit not in results:
            continue
        res = results[digit]
        
        # Row 0: Pixels pushing lower
        ax = plt.subplot(4, 10, digit + 1)
        img = res['input_lower_freq'].reshape(7, 7)
        im = ax.imshow(img, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'{digit}', fontsize=12, fontweight='bold')
        ax.axis('off')
        if digit == 0:
            ax.set_ylabel('Input→Lower\n(want LESS ink)', fontsize=9)
        
        # Row 1: Pixels pushing upper
        ax = plt.subplot(4, 10, 10 + digit + 1)
        img = res['input_upper_freq'].reshape(7, 7)
        ax.imshow(img, cmap='Reds', vmin=0, vmax=1)
        ax.axis('off')
        if digit == 0:
            ax.set_ylabel('Input→Upper\n(want MORE ink)', fontsize=9)
        
        # Row 2: L1 GELU
        ax = plt.subplot(4, 10, 20 + digit + 1)
        x = np.arange(3)
        width = 0.35
        ax.bar(x - width/2, res['L1_lower_freq'], width, color='steelblue', label='Lower')
        ax.bar(x + width/2, res['L1_upper_freq'], width, color='coral', label='Upper')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(['N0', 'N1', 'N2'], fontsize=7)
        if digit == 0:
            ax.set_ylabel('L1 GELU', fontsize=9)
            ax.legend(fontsize=6, loc='upper right')
        
        # Row 3: L2 GELU
        ax = plt.subplot(4, 10, 30 + digit + 1)
        ax.bar(x - width/2, res['L2_lower_freq'], width, color='steelblue')
        ax.bar(x + width/2, res['L2_upper_freq'], width, color='coral')
        ax.set_ylim(0, 1.1)
        ax.set_xticks(x)
        ax.set_xticklabels(['N0', 'N1', 'N2'], fontsize=7)
        if digit == 0:
            ax.set_ylabel('L2 GELU', fontsize=9)
    
    plt.suptitle('ACTIVE CONSTRAINT SIGNATURES BY DIGIT\n'
                 '(Frequency of constraint being binding when maximizing that digit\'s logit)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


# MAIN
if __name__ == "__main__":
    print("="*70)
    print("ACTIVE CONSTRAINT ANALYSIS")
    print("="*70)
    
    EPSILON = 0.05
    N_SAMPLES = 25
    
    print(f"\nParameters: ε={EPSILON}, n_samples={N_SAMPLES}")
    print("Analyzing constraint signatures for each digit class...\n")
    
    results = {}
    
    for digit in range(10):
        print(f"Processing digit {digit}...", end=" ")
        results[digit] = analyze_digit_class(digit, n_samples=N_SAMPLES, epsilon=EPSILON)
        print(f"Success: {results[digit]['n_successful']}/{N_SAMPLES}")
        
        # Quick signature summary
        l1_sig = results[digit]['L1_upper_freq'] - results[digit]['L1_lower_freq']
        print(f"    L1 signature (upper-lower): [{l1_sig[0]:.2f}, {l1_sig[1]:.2f}, {l1_sig[2]:.2f}]")
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION...")
    visualize_signatures(results, 'constraint_signatures.png')
    
    # Generate hypotheses
    print("\n" + "="*70)
    print("AUTO-GENERATED HYPOTHESES")
    print("="*70)
    
    hypotheses = generate_hypotheses(results)
    
    # Test proportion hypotheses
    print("\n--- STATISTICAL TESTS ---")
    tested = 0
    supported = 0
    
    for h in hypotheses:
        if h['test_type'] == 'proportion':
            test_result = test_hypothesis_proportion(h, results)
            tested += 1
            if test_result['significant']:
                supported += 1
            print(f"\n{test_result['interpretation']}")
            print(f"  → {h['statement'][:100]}...")
    
    print(f"\n\nSummary: {supported}/{tested} hypotheses statistically supported (p < 0.05)")
    
    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS...")
    np.savez('constraint_analysis.npz',
             results=results,
             hypotheses=hypotheses,
             epsilon=EPSILON,
             n_samples=N_SAMPLES)
    print("Saved to constraint_analysis.npz")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)