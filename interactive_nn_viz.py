"""
Interactive Neural Network Visualization with Polytope Correspondence

Features:
A. Displays NN architecture: nodes, edges, weights, activation functions
B. Shows LP variables/constraints for each node
C. "Lights up" the network when processing a specific digit (activation values)
"""

from basic import *
from helper import *
from basic import tight_gelu_envelope
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.patches as patches

# Load test data and resize to 7x7
(x_test, y_test), _ = mnist.load_data()
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

# Load weights
data = np.load("verysmallnn_weights.npz")
W1, W2, W3 = data["W1"], data["W2"], data["W3"]
b1, b2, b3 = data["b1"], data["b2"], data["b3"]

print("Network Architecture:")
print(f"  Input: {W1.shape[0]} neurons (7x7 flattened)")
print(f"  Hidden Layer 1: {W1.shape[1]} neurons (GELU)")
print(f"  Hidden Layer 2: {W2.shape[1]} neurons (GELU)")
print(f"  Output: {W3.shape[1]} neurons (logits)")
print(f"\nWeight shapes:")
print(f"  W1: {W1.shape}")
print(f"  W2: {W2.shape}")
print(f"  W3: {W3.shape}")

# Get sample digits (one per class)
sample_indices = []
sample_images = []
for digit in range(10):
    idx = np.where(y_test == digit)[0][0]
    sample_indices.append(idx)
    sample_images.append(x_test[idx])

# Global state
current_digit_idx = 0
epsilon = 0.01

class NNViz:
    def __init__(self):
        self.fig = plt.figure(figsize=(20, 11))

        # Create subplots with more space
        self.ax_nn = plt.subplot2grid((5, 5), (0, 0), colspan=3, rowspan=4)
        self.ax_digit = plt.subplot2grid((5, 5), (0, 3), colspan=2)
        self.ax_info = plt.subplot2grid((5, 5), (1, 3), colspan=2, rowspan=3)
        self.ax_legend = plt.subplot2grid((5, 5), (4, 0), colspan=5)

        # Create visual color legend
        self.ax_legend.axis('off')

        # Node color spectrum
        node_gradient = np.linspace(0, 1, 256).reshape(1, -1)
        self.ax_legend.imshow(node_gradient, aspect='auto', cmap='YlOrRd',
                             extent=[0.05, 0.35, 0.3, 0.7])
        self.ax_legend.text(0.2, 0.85, 'Node Color (Activation Intensity)',
                           ha='center', fontsize=10, fontweight='bold')
        self.ax_legend.text(0.05, 0.15, 'Low', ha='center', fontsize=9)
        self.ax_legend.text(0.35, 0.15, 'High', ha='center', fontsize=9)

        # Edge color examples
        self.ax_legend.plot([0.45, 0.50], [0.5, 0.5], 'b-', linewidth=3, alpha=0.8)
        self.ax_legend.text(0.475, 0.7, 'Positive Weight', ha='center', fontsize=9, color='blue')

        self.ax_legend.plot([0.55, 0.60], [0.5, 0.5], 'r-', linewidth=3, alpha=0.8)
        self.ax_legend.text(0.575, 0.7, 'Negative Weight', ha='center', fontsize=9, color='red')

        # Hover instruction
        self.ax_legend.text(0.8, 0.5, 'HOVER over neurons for details:\n' +
                           '• Symbolic name (e.g., x0[i], a1[j])\n' +
                           '• Activation function\n' +
                           '• Current value\n' +
                           '• Polytope constraints',
                           ha='center', va='center', fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        self.ax_legend.set_xlim(0, 1)
        self.ax_legend.set_ylim(0, 1)

        # Buttons
        self.ax_prev = plt.axes([0.75, 0.05, 0.1, 0.04])
        self.ax_next = plt.axes([0.86, 0.05, 0.1, 0.04])
        self.btn_prev = Button(self.ax_prev, 'Previous')
        self.btn_next = Button(self.ax_next, 'Next')

        self.btn_prev.on_clicked(self.prev_digit)
        self.btn_next.on_clicked(self.next_digit)

        # Network layout positions
        self.layer_x = [0.08, 0.36, 0.64, 0.92]
        self.neuron_positions = {}
        self.neuron_artists = {}
        self.edge_artists = []
        self.activation_values = {}
        self.hover_annotation = None
        self.pixel_highlight = None

        self.setup_layout()
        self.current_digit_idx = 0
        self.update_visualization()

        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

    def setup_layout(self):
        """Calculate neuron positions for each layer"""
        # Layer sizes: input, hidden1, hidden2, output
        # W1 shape is (input_dim, hidden1_dim), so W1.shape = (49, h1)
        layer_sizes = [W1.shape[0], W1.shape[1], W2.shape[1], W3.shape[1]]

        for layer_idx, size in enumerate(layer_sizes):
            x = self.layer_x[layer_idx]
            # Spread neurons vertically with padding to avoid title overlap
            y_positions = np.linspace(0.12, 0.90, size)
            for neuron_idx in range(size):
                self.neuron_positions[(layer_idx, neuron_idx)] = (x, y_positions[neuron_idx])

    def compute_activations(self, x0):
        """Forward pass through network, return all layer activations"""
        # Flatten input
        x0_flat = x0.flatten()

        # Layer 1: x @ W1 + b1
        a1 = x0_flat @ W1 + b1
        z1 = gelu(a1)

        # Layer 2: z1 @ W2 + b2
        a2 = z1 @ W2 + b2
        z2 = gelu(a2)

        # Output: z2 @ W3 + b3
        a3 = z2 @ W3 + b3

        return {
            'input': x0_flat,
            'a1': a1,
            'z1': z1,
            'a2': a2,
            'z2': z2,
            'a3': a3
        }

    def build_polytope(self, x0, epsilon):
        """Build polytope for the input region, return analyzer"""
        x0_flat = x0.flatten()
        lb_in = np.maximum(x0_flat - epsilon, 0)
        ub_in = np.minimum(x0_flat + epsilon, 1)

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

        return analyzer, (L1, U1), (L2, U2)

    def draw_network(self, activations):
        """Draw the neural network with activation-based coloring"""
        self.ax_nn.clear()
        self.hover_annotation = None
        self.ax_nn.set_xlim(0, 1)
        self.ax_nn.set_ylim(0, 1)
        self.ax_nn.axis('off')
        self.ax_nn.set_title('Neural Network Architecture\n(Node brightness = activation strength)',
                             fontsize=14, fontweight='bold')

        # Normalize activations for visualization
        def normalize_activation(val, layer_name):
            """Normalize to [0, 1] for color intensity"""
            if layer_name == 'input':
                return val  # Already [0, 1]
            else:
                # Use tanh-like scaling for activations
                return 1 / (1 + np.exp(-val))  # Sigmoid

        # Draw edges first (so they're behind nodes)
        # Layer 0 -> Layer 1: W1[input_i, hidden1_j] is weight from input_i to hidden1_j
        for i in range(W1.shape[0]):  # input neurons
            for j in range(W1.shape[1]):  # hidden1 neurons
                weight = W1[i, j]
                pos1 = self.neuron_positions[(0, i)]
                pos2 = self.neuron_positions[(1, j)]

                # Edge color based on weight (red=negative, blue=positive)
                color = 'red' if weight < 0 else 'blue'
                alpha = min(abs(weight) * 0.5, 1.0)  # Weight magnitude affects transparency
                linewidth = 0.3

                self.ax_nn.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                               color=color, alpha=alpha, linewidth=linewidth, zorder=1)

        # Layer 1 -> Layer 2
        for i in range(W2.shape[0]):  # hidden1 neurons
            for j in range(W2.shape[1]):  # hidden2 neurons
                weight = W2[i, j]
                pos1 = self.neuron_positions[(1, i)]
                pos2 = self.neuron_positions[(2, j)]

                color = 'red' if weight < 0 else 'blue'
                alpha = min(abs(weight) * 0.5, 1.0)
                linewidth = 0.5

                self.ax_nn.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                               color=color, alpha=alpha, linewidth=linewidth, zorder=1)

        # Layer 2 -> Output
        for i in range(W3.shape[0]):  # hidden2 neurons
            for j in range(W3.shape[1]):  # output neurons
                weight = W3[i, j]
                pos1 = self.neuron_positions[(2, i)]
                pos2 = self.neuron_positions[(3, j)]

                color = 'red' if weight < 0 else 'blue'
                alpha = min(abs(weight) * 0.5, 1.0)
                linewidth = 0.8

                self.ax_nn.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                               color=color, alpha=alpha, linewidth=linewidth, zorder=1)

        # Draw nodes
        layer_names = ['Input\n(7×7)', 'Hidden 1\n(GELU)', 'Hidden 2\n(GELU)', 'Output\n(Logits)']

        for layer_idx, layer_name in enumerate(layer_names):
            if layer_idx == 0:
                values = activations['input']
                size = len(values)
            elif layer_idx == 1:
                values = activations['z1']
                size = len(values)
            elif layer_idx == 2:
                values = activations['z2']
                size = len(values)
            else:
                values = activations['a3']
                size = len(values)

            for neuron_idx in range(size):
                pos = self.neuron_positions[(layer_idx, neuron_idx)]
                val = values[neuron_idx]

                # Color intensity based on activation
                intensity = normalize_activation(val, 'input' if layer_idx == 0 else 'hidden')
                color = plt.cm.YlOrRd(intensity)  # Yellow to Red colormap

                # Larger nodes for fewer neurons
                if size <= 10:
                    radius = 0.015
                elif size <= 20:
                    radius = 0.01
                else:
                    radius = 0.005

                circle = plt.Circle(pos, radius, color=color, ec='black', linewidth=1.5, zorder=2)
                self.ax_nn.add_patch(circle)
                self.neuron_artists[(layer_idx, neuron_idx)] = circle

            # Layer label - position at bottom to avoid overlap with top neurons
            self.ax_nn.text(self.layer_x[layer_idx], 0.02, layer_name,
                           ha='center', va='bottom', fontsize=10, fontweight='bold')

    def update_visualization(self):
        """Update the entire visualization for current digit"""
        digit = current_digit_idx
        x0 = sample_images[digit]
        true_label = digit

        # Compute activations
        activations = self.compute_activations(x0)
        self.current_activations = activations  # Store for hover access

        # Store activation values for easy lookup
        for i, val in enumerate(activations['input']):
            self.activation_values[(0, i)] = val
        for i, val in enumerate(activations['z1']):
            self.activation_values[(1, i)] = val
        for i, val in enumerate(activations['z2']):
            self.activation_values[(2, i)] = val
        for i, val in enumerate(activations['a3']):
            self.activation_values[(3, i)] = val

        # Build polytope
        analyzer, bounds1, bounds2 = self.build_polytope(x0, epsilon)

        # Draw network
        self.draw_network(activations)

        # Display digit
        self.ax_digit.clear()
        self.ax_digit.imshow(x0, cmap='gray')
        self.ax_digit.set_title(f'Digit: {true_label}', fontsize=12, fontweight='bold')
        self.ax_digit.axis('off')

        # Display info
        self.ax_info.clear()
        self.ax_info.axis('off')

        # Predicted class
        predicted = np.argmax(activations['a3'])
        confidence = activations['a3'][predicted]

        info_text = f"=== CURRENT DIGIT ===\n"
        info_text += f"True Label: {true_label}\n"
        info_text += f"Predicted: {predicted}\n"
        info_text += f"Confidence: {confidence:.3f}\n\n"

        info_text += f"=== OUTPUT LOGITS ===\n"
        for i in range(10):
            logit = activations['a3'][i]
            marker = "→ " if i == predicted else "  "
            info_text += f"{marker}{i}: {logit:6.3f}\n"

        info_text += f"\n=== POLYTOPE INFO ===\n"
        info_text += f"ε-ball: {epsilon}\n"
        info_text += f"Variables: x0, a1, z1, a2, z2, a3\n"
        info_text += f"Constraints:\n"
        info_text += f"  - Input box: {W1.shape[0]} vars\n"
        info_text += f"  - GELU env L1: {W1.shape[1]*2}\n"
        info_text += f"  - GELU env L2: {W2.shape[1]*2}\n"

        self.ax_info.text(0.05, 0.95, info_text,
                         fontsize=9, family='monospace',
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.draw()

    def next_digit(self, event):
        global current_digit_idx
        current_digit_idx = (current_digit_idx + 1) % 10
        self.current_digit_idx = current_digit_idx
        self.update_visualization()

    def on_hover(self, event):
        """Handle mouse hover over neurons"""
        if event.inaxes != self.ax_nn:
            if self.hover_annotation:
                self.hover_annotation.set_visible(False)
            if self.pixel_highlight:
                self.pixel_highlight.remove()
                self.pixel_highlight = None
            self.fig.canvas.draw_idle()
            return

        # Check if mouse is near any neuron
        for (layer_idx, neuron_idx), pos in self.neuron_positions.items():
            # Get neuron radius
            layer_sizes = [W1.shape[0], W1.shape[1], W2.shape[1], W3.shape[1]]
            size = layer_sizes[layer_idx]
            if size <= 10:
                radius = 0.015
            elif size <= 20:
                radius = 0.01
            else:
                radius = 0.005

            # Check if mouse is within neuron circle
            dx = event.xdata - pos[0] if event.xdata else float('inf')
            dy = event.ydata - pos[1] if event.ydata else float('inf')
            dist = np.sqrt(dx**2 + dy**2)

            if dist < radius * 1.5:  # Slightly larger hit area
                # Show neuron info
                self.show_neuron_info(layer_idx, neuron_idx, event.xdata, event.ydata)
                return

        # No neuron found, hide annotation and pixel highlight
        if self.hover_annotation:
            self.hover_annotation.set_visible(False)
        if self.pixel_highlight:
            self.pixel_highlight.remove()
            self.pixel_highlight = None
        self.fig.canvas.draw_idle()

    def show_neuron_info(self, layer_idx, neuron_idx, x, y):
        """Display detailed information about a neuron"""
        info_lines = []

        # Clear previous pixel highlight if exists
        if self.pixel_highlight:
            self.pixel_highlight.remove()
            self.pixel_highlight = None

        layer_names = ['Input', 'Hidden1', 'Hidden2', 'Output']
        info_lines.append(f"=== {layer_names[layer_idx]} Layer ===")
        info_lines.append(f"Neuron Index: {neuron_idx}")

        if layer_idx == 0:
            # Highlight corresponding pixel in the digit image
            pixel_row = neuron_idx // 7
            pixel_col = neuron_idx % 7
            self.pixel_highlight = self.ax_digit.add_patch(
                patches.Rectangle((pixel_col - 0.5, pixel_row - 0.5), 1, 1,
                                linewidth=3, edgecolor='red', facecolor='none')
            )
            info_lines.append(f"Variable: x0[{neuron_idx}]")
            info_lines.append(f"Meaning: Input pixel {neuron_idx} (position {neuron_idx//7},{neuron_idx%7})")
            info_lines.append(f"Value: {self.activation_values.get((layer_idx, neuron_idx), 0):.4f}")
            info_lines.append("")
            info_lines.append("Polytope Constraints:")
            lb = max(0, self.activation_values.get((layer_idx, neuron_idx), 0) - epsilon)
            ub = min(1, self.activation_values.get((layer_idx, neuron_idx), 0) + epsilon)
            info_lines.append(f"  {lb:.4f} ≤ x0[{neuron_idx}] ≤ {ub:.4f}")

        elif layer_idx == 1:
            info_lines.append(f"Pre-activation: a1[{neuron_idx}]")
            info_lines.append(f"Post-activation: z1[{neuron_idx}]")
            info_lines.append(f"Activation: GELU(a1[{neuron_idx}])")
            info_lines.append("")
            info_lines.append("Math:")
            info_lines.append(f"  a1[{neuron_idx}] = Σ W1[i,{neuron_idx}]·x0[i] + b1[{neuron_idx}]")
            info_lines.append(f"  z1[{neuron_idx}] = GELU(a1[{neuron_idx}])")
            info_lines.append("")
            # Get current values if available
            if hasattr(self, 'current_activations'):
                a_val = self.current_activations['a1'][neuron_idx]
                z_val = self.current_activations['z1'][neuron_idx]
                info_lines.append(f"Current Values:")
                info_lines.append(f"  a1[{neuron_idx}] = {a_val:.4f}")
                info_lines.append(f"  z1[{neuron_idx}] = {z_val:.4f}")
                info_lines.append("")
            info_lines.append("Polytope Constraints:")
            info_lines.append(f"  GELU lower envelope: z1[{neuron_idx}] ≥ L_slope·a1[{neuron_idx}] + L_intercept")
            info_lines.append(f"  GELU upper envelope: z1[{neuron_idx}] ≤ U_slope·a1[{neuron_idx}] + U_intercept")

        elif layer_idx == 2:
            info_lines.append(f"Pre-activation: a2[{neuron_idx}]")
            info_lines.append(f"Post-activation: z2[{neuron_idx}]")
            info_lines.append(f"Activation: GELU(a2[{neuron_idx}])")
            info_lines.append("")
            info_lines.append("Math:")
            info_lines.append(f"  a2[{neuron_idx}] = Σ W2[i,{neuron_idx}]·z1[i] + b2[{neuron_idx}]")
            info_lines.append(f"  z2[{neuron_idx}] = GELU(a2[{neuron_idx}])")
            info_lines.append("")
            if hasattr(self, 'current_activations'):
                a_val = self.current_activations['a2'][neuron_idx]
                z_val = self.current_activations['z2'][neuron_idx]
                info_lines.append(f"Current Values:")
                info_lines.append(f"  a2[{neuron_idx}] = {a_val:.4f}")
                info_lines.append(f"  z2[{neuron_idx}] = {z_val:.4f}")
                info_lines.append("")
            info_lines.append("Polytope Constraints:")
            info_lines.append(f"  GELU lower envelope: z2[{neuron_idx}] ≥ L_slope·a2[{neuron_idx}] + L_intercept")
            info_lines.append(f"  GELU upper envelope: z2[{neuron_idx}] ≤ U_slope·a2[{neuron_idx}] + U_intercept")

        elif layer_idx == 3:
            info_lines.append(f"Variable: a3[{neuron_idx}]")
            info_lines.append(f"Meaning: Output logit for digit {neuron_idx}")
            info_lines.append("")
            info_lines.append("Math:")
            info_lines.append(f"  a3[{neuron_idx}] = Σ W3[i,{neuron_idx}]·z2[i] + b3[{neuron_idx}]")
            info_lines.append("")
            if hasattr(self, 'current_activations'):
                val = self.current_activations['a3'][neuron_idx]
                info_lines.append(f"Current Value: {val:.4f}")
                info_lines.append("")
            info_lines.append("Polytope: No constraints")
            info_lines.append("(Output logits are unbounded)")

        # Create or update annotation with smart positioning
        info_text = '\n'.join(info_lines)

        # Calculate alignment based on mouse position relative to plot center
        ha_align = 'left' if x < 0.5 else 'right'
        va_align = 'top' if y > 0.5 else 'bottom'

        # Set offset direction based on alignment
        offset_x = 20 if ha_align == 'left' else -20
        offset_y = -20 if va_align == 'top' else 20

        if self.hover_annotation is None:
            self.hover_annotation = self.ax_nn.annotate(
                info_text,
                xy=(x, y),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='black'),
                fontsize=8,
                family='monospace',
                zorder=100,
                annotation_clip=False
            )
        else:
            self.hover_annotation.set_text(info_text)
            self.hover_annotation.xy = (x, y)
            self.hover_annotation.xytext = (offset_x, offset_y)
            self.hover_annotation.set_horizontalalignment(ha_align)
            self.hover_annotation.set_verticalalignment(va_align)
            self.hover_annotation.set_visible(True)

        self.fig.canvas.draw_idle()

    def prev_digit(self, event):
        global current_digit_idx
        current_digit_idx = (current_digit_idx - 1) % 10
        self.current_digit_idx = current_digit_idx
        self.update_visualization()

# Create and show visualization
print("\nLaunching interactive visualization...")
print("Use 'Previous' and 'Next' buttons to cycle through digits")
viz = NNViz()
plt.show()