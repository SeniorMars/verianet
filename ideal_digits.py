from helper import PyomoPolyAnalyzer, ibp_affine_keras
from basic import tight_gelu_envelope, ibp_activation, gelu
import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# Load data
(x_test, y_test), _ = mnist.load_data()
x_test = tf.image.resize(x_test[..., tf.newaxis], [7, 7]).numpy().squeeze() / 255

# Load weights
data = np.load("verysmallnn_weights.npz")
W1, b1 = data["W1"], data["b1"]
W2, b2 = data["W2"], data["b2"]
W3, b3 = data["W3"], data["b3"]


def forward(x):
    """
    Forward pass of our nn
    Input: x -- a np array representing an input for our nn
    Output: a3 -- final layer of nn
    """
    a1 = x @ W1 + b1
    z1 = gelu(a1)
    a2 = z1 @ W2 + b2
    z2 = gelu(a2)
    a3 = z2 @ W3 + b3
    return a3

#Want to find sample with highest logit to start
#Will shift from there to optimize and find ideal version
#This will prevent abstract patterns that technically guve ideal version
#of the number but in reality look nothing like it

#For each dugit, want to store the best images, logits, and indices
best_images = []
best_logits = []
best_indices = []

for target_class in range(10):
    class_mask = (y_test == target_class)
    class_images = x_test[class_mask]
    class_indices = np.where(class_mask)[0]
    
    # Compute logits for all images of this class
    max_logit = -np.inf
    best_idx = 0
    best_img = None
    
    for i, img in enumerate(class_images):
        logits = forward(img.flatten())
        predicted = np.argmax(logits)
        
        # Only consider correctly classified images
        if predicted != target_class:
            continue
        
        if logits[target_class] > max_logit:
            max_logit = logits[target_class]
            best_idx = class_indices[i]
            best_img = img
    
    best_images.append(best_img.flatten())
    best_logits.append(max_logit)
    best_indices.append(best_idx)
    
    # Verify it's correctly classified
    all_logits = forward(best_img.flatten())
    predicted = np.argmax(all_logits)
    margin = all_logits[target_class] - np.max([all_logits[j] for j in range(10) if j != target_class])
    
    print(f"Digit {target_class}: idx={best_idx}, logit={max_logit:.2f}, pred={predicted}, margin={margin:.2f}")

# We will optimize from best images with perturbation and penalty

input_dim = W1.shape[0]
num_classes = 10
# These choices of epsilon and penalty seem to work fairly well
# However they are hyperparameters that can be tweaked
epsilon = 0.3
penalty_weight = 0.8

# Get perimeter indices -- want to make perimeter black to prevent noise
perimeter_idx = [r*7 + c for r in range(7) for c in range(7)
            if r in (0, 6) or c in (0, 6)]


optimal_inputs = []
all_logits_optimized = []

for target_class in range(num_classes):
    # Start from the best MNIST example for this class
    x_center = best_images[target_class]
    
    lb_in = np.maximum(x_center - epsilon, 0.0)
    ub_in = np.minimum(x_center + epsilon, 1.0)
    
    # Force perimeter to 0
    for idx in perimeter_idx:
        lb_in[idx] = 0.0
        ub_in[idx] = 0.0
    
    #Bounds for polytope
    L1, U1 = ibp_affine_keras(lb_in, ub_in, W1, b1)
    Lz1, Uz1 = ibp_activation(L1, U1, gelu)
    L2, U2 = ibp_affine_keras(Lz1, Uz1, W2, b2)
    
    analyzer = PyomoPolyAnalyzer()
    
    x0_name = analyzer.add_input_box("x0", lb_in, ub_in)
    a1_name = analyzer.add_affine(x0_name, W1, b1, "a1")
    z1_name = analyzer.add_activation(a1_name, "z1", (L1, U1), tight_gelu_envelope)
    a2_name = analyzer.add_affine(z1_name, W2, b2, "a2")
    z2_name = analyzer.add_activation(a2_name, "z2", (L2, U2), tight_gelu_envelope)
    a3_name = analyzer.add_affine(z2_name, W3, b3, "a3")
    
    # Add margin constraint
    analyzer.model.t = pyo.Var(domain=pyo.Reals)
    
    def t_bound_rule(model, k):
        """
        Helper function for adding constraints to model
        """
        if k == target_class:
            return pyo.Constraint.Skip
        return model.t >= model.a3[k]
    
    analyzer.model.t_bounds = pyo.Constraint(range(10), rule=t_bound_rule)
    
    # Objective: maximize margin - penalty * sum(pixels)
    objective_expr = (analyzer.model.a3[target_class] - analyzer.model.t 
                      - penalty_weight * sum(analyzer.model.x0[i] for i in range(input_dim)))
    
    val, results = analyzer.optimize(objective_expr, sense='max')
    
    optimal_input = np.array([pyo.value(analyzer.model.x0[i]) for i in range(input_dim)])
    logits = np.array([pyo.value(analyzer.model.a3[k]) for k in range(10)])
    
    optimal_inputs.append(optimal_input)
    all_logits_optimized.append(logits)
    
    margin = logits[target_class] - np.max([logits[j] for j in range(10) if j != target_class])
    print(f"Class {target_class}: optimized margin = {margin:.2f}")


#Visualization

fig, axes = plt.subplots(3, 10, figsize=(20, 9))

for k in range(num_classes):
    # Row 0: Best version of each MNIST digit
    axes[0, k].imshow(best_images[k].reshape((7, 7)), cmap='gray', vmin=0, vmax=1)
    orig_logits = forward(best_images[k])
    orig_margin = orig_logits[k] - np.max([orig_logits[j] for j in range(10) if j != k])
    axes[0, k].set_title(f'Best {k}\n(m={orig_margin:.1f})', fontsize=9)
    axes[0, k].axis('off')
    
    # Row 1: Optimized version
    axes[1, k].imshow(optimal_inputs[k].reshape((7, 7)), cmap='gray', vmin=0, vmax=1)
    opt_margin = all_logits_optimized[k][k] - np.max([all_logits_optimized[k][j] for j in range(10) if j != k])
    axes[1, k].set_title(f'Optimized\n(m={opt_margin:.1f})', fontsize=9)
    axes[1, k].axis('off')
    
    # Row 2: Difference between best version and optimized version
    diff = optimal_inputs[k] - best_images[k]
    axes[2, k].imshow(diff.reshape((7, 7)), cmap='RdBu', vmin=-epsilon, vmax=epsilon)
    axes[2, k].set_title(f'Δ', fontsize=9)
    axes[2, k].axis('off')
    

# Add row labels
axes[0, 0].set_ylabel('Best MNIST', fontsize=11)
axes[1, 0].set_ylabel('Optimized', fontsize=11)
axes[2, 0].set_ylabel('Perturbation', fontsize=11)

plt.suptitle(f'Margin-Optimized Digits (ε={epsilon}, penalty={penalty_weight}, perimeter=0)\nStarting from highest-logit MNIST examples', fontsize=13)
plt.tight_layout()
plt.savefig('optimized_digits.png', dpi=150, bbox_inches='tight')
plt.show()
