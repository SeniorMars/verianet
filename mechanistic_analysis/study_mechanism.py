import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from dataclasses import dataclass
from typing import List

# ==========================================
# 0. SETUP & HELPER FUNCTIONS
# ==========================================
sys.path.append('..')

# Robust GELU definition in case helper.py is missing
def gelu_custom(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

try:
    from basic import *
    from helper import *
    # Ensure gelu is available if imported from basic/helper
    if 'gelu' not in globals(): gelu = gelu_custom
except ImportError:
    gelu = gelu_custom
    print("Warning: 'basic' or 'helper' modules not found. Using local GELU.")

# ==========================================
# 1. STATE MANAGEMENT (Data Classes)
# ==========================================

@dataclass
class NeuronState:
    idx: int
    layer_name: str
    weights: np.ndarray         # The filter (7x7) or Effective Pattern
    input_signal: np.ndarray    # What the neuron 'sees' (7x7)
    match_map: np.ndarray       # Element-wise product (The "Evidence")
    activation: float           # Scalar output (post-GELU)
    pre_act: float              # Scalar pre-activation

@dataclass
class NetworkTrace:
    input_image: np.ndarray
    true_label: int
    pred_probs: np.ndarray
    logits: np.ndarray
    l1_neurons: List[NeuronState]
    l2_neurons: List[NeuronState]
    
    @property
    def predicted_label(self):
        return np.argmax(self.logits)

# ==========================================
# 2. VISUALIZATION ENGINE (The "Forensic" Dashboard)
# ==========================================

class NeuroVis:
    def __init__(self, weights_path="../verysmallnn_weights.npz"):
        # Load weights
        try:
            data = np.load(weights_path)
            self.W1, self.W2, self.W3 = data["W1"], data["W2"], data["W3"]
            self.b1, self.b2, self.b3 = data["b1"], data["b2"], data["b3"]
            print(f"Weights loaded from {weights_path}")
        except Exception as e:
            print(f"Error loading weights ({e}). Initializing random weights for demo.")
            self.W1 = np.random.randn(49, 3)
            self.b1 = np.zeros(3)
            self.W2 = np.random.randn(3, 3)
            self.b2 = np.zeros(3)
            self.W3 = np.random.randn(3, 10)
            self.b3 = np.zeros(10)

        # Aesthetic Configurations
        self.cmap_w = 'RdBu'     # Weights: Red (Neg) to Blue (Pos)
        self.cmap_act = 'viridis'  # Activations
        # PathEffect: Thick black outline for white text readability
        self.pe_txt = [pe.withStroke(linewidth=2.5, foreground='black')]
        
    def capture_trace(self, input_image, true_label) -> NetworkTrace:
        """Runs a forward pass and captures the mechanistic state of every neuron."""
        x_flat = input_image.flatten()
        
        # --- Layer 1 Processing ---
        a1 = x_flat @ self.W1 + self.b1
        z1 = gelu(a1)
        
        l1_states = []
        for i in range(3):
            w = self.W1[:, i].reshape(7, 7)
            # The "Match": Element-wise product of Input and Weights
            match = input_image * w 
            l1_states.append(NeuronState(i, "L1", w, input_image, match, z1[i], a1[i]))

        # --- Layer 2 Processing ---
        a2 = z1 @ self.W2 + self.b2
        z2 = gelu(a2)
        
        l2_states = []
        for i in range(3):
            # Effective Pattern: Project L2 weights back through W1 to input space
            # This answers: "What pattern of pixels does this L2 neuron like?"
            w_eff_flat = self.W1 @ self.W2[:, i]
            w_eff_img = w_eff_flat.reshape(7, 7)
            
            # The "Match": Compare Input to the Effective Pattern
            match = input_image * w_eff_img
            
            l2_states.append(NeuronState(i, "L2", w_eff_img, input_image, match, z2[i], a2[i]))

        # --- Output Processing ---
        logits = z2 @ self.W3 + self.b3
        probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
        
        return NetworkTrace(input_image, true_label, probs, logits, l1_states, l2_states)

    def plot_dashboard(self, trace: NetworkTrace, save_path=None):
        """Generates the v3 'Circuit Dashboard'."""
        fig = plt.figure(figsize=(20, 18), layout="constrained")
        
        # Master Grid: 4 Rows
        # Row 1: Context (Input + Predictions)
        # Row 2: L1 Mechanism (Triplets)
        # Row 3: L2 Mechanism (Triplets)
        # Row 4: Attribution (Voting)
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 2.8, 2.8, 1.2])

        # --- ROW 1: Context ---
        gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0])
        ax_in = fig.add_subplot(gs_row1[0])
        self._plot_pixel(ax_in, trace.input_image, "INPUT", cmap='gray', vmin=0, vmax=1)
        
        ax_pred = fig.add_subplot(gs_row1[1:])
        self._plot_predictions(ax_pred, trace)

        # --- ROW 2: Layer 1 (Feature Detectors) ---
        self._plot_layer_row(fig, gs[1], trace.l1_neurons, "LAYER 1: Local Feature Detectors")

        # --- ROW 3: Layer 2 (Feature Combiners) ---
        self._plot_layer_row(fig, gs[2], trace.l2_neurons, "LAYER 2: Global Feature Combiners (Effective Patterns)")

        # --- ROW 4: Voting ---
        gs_row4 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3])
        ax_vote = fig.add_subplot(gs_row4[1]) # Center the voting chart
        self._plot_voting(ax_vote, trace)

        # Global Title
        status = "CORRECT" if trace.predicted_label == trace.true_label else "MISCLASSIFIED"
        status_color = "forestgreen" if status == "CORRECT" else "crimson"
        fig.suptitle(f"Mechanistic Trace: Digit {trace.true_label} $\\rightarrow$ Predicted {trace.predicted_label} ({status})", 
                     fontsize=24, fontweight='bold', color=status_color, y=1.03)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            # Clear memory to prevent leak in loop
            plt.close(fig)
        else:
            plt.show()

    def _plot_layer_row(self, fig, slot, neurons, section_title):
        """Plots a row of 3 neurons, each with the 'Triplet' view."""
        # Header for the section
        # (We can't easily add a text title to a GridSpec, so we rely on the figure structure)
        
        # 3 Columns (one per neuron)
        gs_neur = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=slot)
        
        for i, n in enumerate(neurons):
            # Inner grid: Top (Triple Images) + Bottom (Activation Thermometer)
            gs_inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_neur[i], height_ratios=[3.5, 1])
            
            # The Triple Image Strip: [Input | Weights | Match]
            gs_imgs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_inner[0], wspace=0.05)
            
            # 1. Reference (Input)
            ax1 = fig.add_subplot(gs_imgs[0])
            self._plot_pixel(ax1, n.input_signal, "Input", 'gray', 0, 1, box=True)
            
            # 2. Filter (Weights/Pattern)
            ax2 = fig.add_subplot(gs_imgs[1])
            limit = np.max(np.abs(n.weights))
            # Safe limit for constant 0 weights
            limit = limit if limit > 1e-5 else 1.0 
            t_str = "Pattern" if n.layer_name=="L1" else "Eff. Pattern"
            self._plot_pixel(ax2, n.weights, t_str, self.cmap_w, -limit, limit, box=True)
            
            # 3. The Match (Hadamard Product)
            ax3 = fig.add_subplot(gs_imgs[2])
            match_limit = np.max(np.abs(n.match_map))
            match_limit = match_limit if match_limit > 1e-5 else 1.0
            self._plot_pixel(ax3, n.match_map, "Match", self.cmap_w, -match_limit, match_limit, box=True)

            # Activation Indicator (Bottom)
            ax_act = fig.add_subplot(gs_inner[1])
            self._plot_activation_bar(ax_act, n)

    def _plot_pixel(self, ax, data, title, cmap, vmin, vmax, box=False):
        """High-contrast pixel plotter."""
        if box:
            ax.set_facecolor('black') # Dark background makes pixels pop
            
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1) # FORCE SQUARE PIXELS
        
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
        
        # Border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_edgecolor('black' if not box else 'gray')

    def _plot_activation_bar(self, ax, n: NeuronState):
        """Visualizes activation as a horizontal thermometer."""
        val = n.activation
        c = 'forestgreen' if val > 0 else 'firebrick'
        
        # Bar (centered at 0)
        ax.barh([0], [val], color=c, edgecolor='black', height=0.6)
        ax.axvline(0, color='black', linewidth=1)
        
        # Range settings (Fixed range -3 to 3 ensures visual comparability across neurons)
        ax.set_xlim(-3, 3) 
        ax.set_yticks([])
        
        # Text Label
        ax.set_xlabel(f"Act: {val:.2f}", fontsize=12, fontweight='bold', color=c)
        ax.set_title(f"{n.layer_name}_N{n.idx}", fontsize=10, fontweight='bold', color='gray')

    def _plot_predictions(self, ax, trace):
        digits = np.arange(10)
        colors = ['lightgray'] * 10
        colors[trace.true_label] = 'forestgreen'
        if trace.predicted_label != trace.true_label:
            colors[trace.predicted_label] = 'crimson'
            
        bars = ax.bar(digits, trace.pred_probs, color=colors, edgecolor='black')
        ax.set_xticks(digits)
        ax.set_title("Network Prediction Confidence", fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(0, 1.1)

    def _plot_voting(self, ax, trace):
        """Shows contribution of L2 neurons to the PREDICTED class logit."""
        contribs = []
        labels = []
        target = trace.predicted_label # Explain the prediction
        
        for i, n in enumerate(trace.l2_neurons):
            w = self.W3[i, target]
            c = n.activation * w
            contribs.append(c)
            labels.append(f"L2_N{i}")
            
        bias = self.b3[target]
        contribs.append(bias)
        labels.append("Bias")
        
        colors = ['steelblue' if c>0 else 'crimson' for c in contribs]
        bars = ax.bar(labels, contribs, color=colors, edgecolor='black')
        ax.axhline(0, color='k')
        
        # Add values on bars
        for bar, v in zip(bars, contribs):
            h = bar.get_height()
            y_pos = h if h > 0 else 0
            va = 'bottom' if h > 0 else 'top'
            # Use path effects for readability
            ax.text(bar.get_x()+bar.get_width()/2, y_pos, f"{v:.2f}", 
                   ha='center', va=va, fontweight='bold', color='white',
                   path_effects=self.pe_txt)
            
        ax.set_title(f"Why did we predict {target}? (Contribution to Logit)", fontweight='bold')


# ==========================================
# 3. MAIN EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    print("Initializing NeuroVis v3 Engine...")
    
    # Load Data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Preprocessing: Resize to 7x7 and Normalize
    x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255
    
    # Initialize Visualizer
    viz = NeuroVis("../verysmallnn_weights.npz")
    
    print(f"{'='*50}")
    print(f"Generating Dashboards for Digits 0-9")
    print(f"{'='*50}")

    for digit in range(10):
        # Find all examples of this digit in the test set
        indices = np.where(y_test == digit)[0]
        if len(indices) == 0:
            print(f"No examples found for digit {digit}. Skipping.")
            continue

        # Find the first CORRECTLY CLASSIFIED example
        found_correct = False
        for idx in indices:
            example_img = x_test[idx]
            true_lbl = y_test[idx]

            # Quick forward pass to check prediction
            x_flat = example_img.flatten()
            a1 = x_flat @ viz.W1 + viz.b1
            z1 = gelu(a1)
            a2 = z1 @ viz.W2 + viz.b2
            z2 = gelu(a2)
            logits = z2 @ viz.W3 + viz.b3
            pred = np.argmax(logits)

            # Use this example if correctly classified
            if pred == true_lbl:
                found_correct = True
                break

        if not found_correct:
            print(f"Warning: No correctly classified examples found for digit {digit}. Using first example anyway.")
            idx = indices[0]
            example_img = x_test[idx]
            true_lbl = y_test[idx]

        print(f"Processing Digit {digit} (Index {idx})...")

        # 1. Capture the trace (Forward Pass)
        trace = viz.capture_trace(example_img, true_lbl)

        # 2. Render and Save
        filename = f"dashboard_digit_{digit}.png"
        viz.plot_dashboard(trace, save_path=filename)
        print(f"  -> Saved {filename}")
        
    print(f"\n{'='*50}")
    print("Analysis Complete.")