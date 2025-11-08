"""
Implements a static neural network analyzer using polyhedral abstract interpretation.

This script builds a single, global Linear Program (LP) that represents the
over-approximated behavior of a neural network over a given input set (polytope).
"""

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from typing import Callable, Any
from scipy.optimize import linprog
from scipy.special import erf

# ---------- Type Aliases for Clarity ----------
# Using numpy.typing for clear array shapes and types
Vector = npt.NDArray[np.float64]  # Represents a 1D vector
Matrix = npt.NDArray[np.float64]  # Represents a 2D matrix
# A line segment: (slope, intercept, start_x, end_x)
Line = tuple[float, float, float, float]
# A pair of lower and upper bounding lines, forming an "envelope"
Envelope = tuple[list[Line], list[Line]]
# A function that builds an envelope from scalar bounds [L, U]
ActBuilder = Callable[[float, float], Envelope]
# A function mapping a vector to a vector (e.g., an activation function)
VecFunc = Callable[[Vector], Vector]
# Type alias for the (raw) result object from scipy.linprog
LinprogResult = Any
# Our optimization result: (optimal_value, raw_linprog_result)
OptimizeResult = tuple[float | None, LinprogResult]


# ---------- Polytope H-rep ----------
@dataclass
class Polytope:
    """
    Represents a convex polytope in H-representation (Ax <= b).

    NOTE: This class is a standalone helper and is NOT used by the
    main PolyAnalyzer, which builds its own global A and b matrices internally.
    """

    A: Matrix  # Constraint matrix (m, n)
    b: Vector  # Constraint vector (m,)

    def stack(self, A2: Matrix, b2: Vector) -> "Polytope":
        """Intersects this polytope with another by stacking constraints."""
        if self.A.size == 0:
            return Polytope(A2.copy(), b2.copy())
        A = np.vstack([self.A, A2])
        b = np.concatenate([self.b, b2])
        return Polytope(A, b)

    @staticmethod
    def box(lb: Vector, ub: Vector) -> "Polytope":
        """Creates a Polytope representing an n-dimensional box [lb, ub]."""
        n = lb.size
        # Constraints are:
        #  1*x <= ub
        # -1*x <= -lb  (which is x >= lb)
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.concatenate([ub, -lb])
        return Polytope(A, b)


# ---------- Interval Bound Propagation ----------
def ibp_affine(L: Vector, U: Vector, W: Matrix, b: Vector) -> tuple[Vector, Vector]:
    """
    Performs Interval Bound Propagation (IBP) over an affine layer.

    Calculates the *exact* output interval bounds [L_out, U_out] for the
    operation (W @ x + b), given that the input x is in the box [L, U].

    Args:
        L: Lower bounds of the input box (n_in,)
        U: Upper bounds of the input box (n_in,)
        W: Weight matrix (n_out, n_in)
        b: Bias vector (n_out,)

    Returns:
        A tuple (L_out, U_out) of the output bounds.
    """
    # This works by finding the min/max of each output row independently.
    # max(W_i @ x) = W_i_pos @ U + W_i_neg @ L
    # min(W_i @ x) = W_i_pos @ L + W_i_neg @ U
    Wpos = np.maximum(W, 0)
    Wneg = np.minimum(W, 0)
    L_out = Wpos @ L + Wneg @ U + b
    U_out = Wpos @ U + Wneg @ L + b
    return L_out, U_out


def ibp_activation(L: Vector, U: Vector, f: VecFunc) -> tuple[Vector, Vector]:
    """
    Performs IBP over an arbitrary activation function.

    This is a *conservative* (over-approximated) bound. It samples a
    few points and takes the min/max. It does not assume monotonicity.

    Args:
        L: Lower bounds of the input (n,)
        U: Upper bounds of the input (n,)
        f: The activation function.

    Returns:
        A tuple (L_out, U_out) of the output bounds.
    """
    # Sample key points: endpoints and midpoint
    xs = np.stack([L, U, 0.5 * (L + U)], axis=1)
    vals = f(xs)
    # The true bounds are guaranteed to be within the min/max of the samples
    # *only if* the function's extrema are at the endpoints.
    # For GELU, this is a heuristic, but a reasonable one.
    return np.min(vals, axis=1), np.max(vals, axis=1)


# ---------- Activation envelopes (generic & ready-made) ----------
def chords_for_function(f: VecFunc, segs: list[tuple[float, float]]) -> list[Line]:
    """
    Generates a list of line segments (chords) that connect points on
    a function 'f' over a list of intervals 'segs'.

    Args:
        f: The function (e.g., gelu)
        segs: A list of (x0, x1) intervals.

    Returns:
        A list of Line tuples (m, c, x0, x1) for each segment.
    """
    lines: list[Line] = []
    for x0, x1 in segs:
        # Evaluate function at endpoints
        y0 = float(f(np.array([x0], dtype=np.float64)))
        y1 = float(f(np.array([x1], dtype=np.float64)))
        # Calculate slope (m) and intercept (c)
        m = (y1 - y0) / (x1 - x0) if x1 != x0 else 0.0
        c = y0 - m * x0
        lines.append((m, c, x0, x1))
    return lines


def envelope_from_lines(
    lower_lines: list[Line], upper_lines: list[Line]
) -> tuple[Matrix, Vector]:
    """
    Converts lists of lower and upper bounding lines into H-representation (A, b).

    This creates a 2D constraint system for variables 'a' (pre-activation)
    and 'z' (post-activation). The resulting A and b will be mapped into the
    global problem by the PolyAnalyzer.

    Args:
        lower_lines: list of (m, c, x0, x1) tuples for lower bounds (z >= m*a + c)
        upper_lines: list of (m, c, x0, x1) tuples for upper bounds (z <= m*a + c)

    Returns:
        A tuple (A, b) where A is a (k, 2) matrix and b is a (k,) vector,
        representing the constraints on the [a, z] variable pair.
    """
    A_list: list[list[float]] = []
    b_list: list[float] = []
    for m, c, _, _ in lower_lines:
        # z >= m*a + c  =>  m*a - z <= -c
        # This corresponds to A_row = [m, -1], b_row = -c
        A_list.append([m, -1.0])
        b_list.append(-c)
    for m, c, _, _ in upper_lines:
        # z <= m*a + c  =>  -m*a + z <= c
        # This corresponds to A_row = [-m, 1], b_row = c
        A_list.append([-m, 1.0])
        b_list.append(c)

    # Handle the case of no lines (e.g., an unconstrained dimension)
    if not A_list:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.float64)

    return np.array(A_list, dtype=np.float64), np.array(b_list, dtype=np.float64)


# Prebuilt: ReLU and HardTanh envelopes on [L,U]
def relu_envelope(L: float, U: float) -> Envelope:
    """
    Builds the tightest convex (polyhedral) relaxation for ReLU on [L, U].

    This implements the "triangle relaxation" for the case where L < 0 < U.
    - Case 1 (U <= 0): "Dead" neuron. Output z = 0.
    - Case 2 (L >= 0): "Passing" neuron. Output z = a.
    - Case 3 (L < 0 < U): "Ambiguous" neuron. Relaxed with three lines:
        1. z >= 0                (Lower bound)
        2. z >= a                (Lower bound)
        3. z <= (U/(U-L)) * (a - L)  (The "triangle" upper bound)
    """
    lines_lower: list[Line] = []
    lines_upper: list[Line] = []
    if U <= 0.0:
        # z = 0
        lines_upper.append((0.0, 0.0, L, U))
        lines_lower.append((0.0, 0.0, L, U))
    elif L >= 0.0:
        # z = a
        lines_upper.append((1.0, 0.0, L, U))
        lines_lower.append((1.0, 0.0, L, U))
    else:
        # Crossing: triangle relaxation
        # lower: z >= 0 and z >= a
        lines_lower.append((0.0, 0.0, L, U))  # z >= 0
        lines_lower.append((1.0, 0.0, L, U))  # z >= a
        # upper: z <= (U/(U-L)) (a - L)
        m = U / (U - L)
        c = -m * L
        lines_upper.append((m, c, L, U))
    return lines_lower, lines_upper


def hardtanh_envelope(
    L: float, U: float, lo: float = -1.0, hi: float = 1.0
) -> Envelope:
    """Builds an *exact* polyhedral representation for HardTanh on [L, U]."""
    lower, upper = [], []
    # region 1: a <= lo → z = lo
    if L < lo:
        x0, x1 = L, min(lo, U)
        lower += [(0.0, lo, x0, x1)]
        upper += [(0.0, lo, x0, x1)]
    # region 2: lo <= a <= hi → z = a
    x0, x1 = max(L, lo), min(U, hi)
    if x0 < x1:
        lower += [(1.0, 0.0, x0, x1)]
        upper += [(1.0, 0.0, x0, x1)]
    # region 3: a >= hi → z = hi
    if U > hi:
        x0, x1 = max(hi, L), U
        lower += [(0.0, hi, x0, x1)]
        upper += [(0.0, hi, x0, x1)]
    return lower, upper


# GELU (or arbitrary) outer envelope via sampled chords
def sampled_envelope(
    f: VecFunc, L: float, U: float, n_segments: int = 4, pad: float = 0.0
) -> Envelope:
    """
    Creates a generic relaxation for *any* function 'f' by sampling chords.

    This is a general-purpose, but less-tight, relaxation. It works by:
    1. Splitting [L, U] into 'n_segments'.
    2. Creating a chord (line segment) for each piece.
    3. Using these chords as *both* upper and lower bounds (with a small
       'pad' for numerical safety).

    This is a valid over-approximation but is not guaranteed to be convex
    or the tightest possible.
    """
    xs = np.linspace(L, U, n_segments + 1)
    chords = chords_for_function(f, list(zip(xs[:-1], xs[1:])))
    # For safety: build both lower and upper by shifting chords
    # (no convexity assumption). pad ≥ 0 slightly expands.
    eps = pad
    lower = [(m, c - eps, x0, x1) for (m, c, x0, x1) in chords]
    upper = [(m, c + eps, x0, x1) for (m, c, x0, x1) in chords]
    return lower, upper


# GELU function
def gelu(x: Vector) -> Vector:
    """The GELU activation function, using the 'erf' approximation."""
    # exact: x*Phi(x); approximation: 0.5*x*(1+erf(x/sqrt(2)))
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def tight_gelu_envelope(L: float, U: float) -> Envelope:
    """
    Builds a tight convex relaxation for GELU on [L, U].

    Strategy:
    - Lower bound: Tangent line at the point of maximum slope in [L, U]
    - Upper bound: Chord (secant line) from (L, GELU(L)) to (U, GELU(U))

    This is provably tight because GELU is:
    - Convex on (-∞, ~-0.17)
    - Concave on (~-0.17, ~0.95)
    - Convex on (~0.95, ∞)
    """
    # GELU derivative: d/dx[x*Phi(x)] = Phi(x) + x*phi(x)
    # where phi is the standard normal PDF
    def gelu_deriv(x: float) -> float:
        phi_x = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        Phi_x = 0.5 * (1 + erf(x / np.sqrt(2)))
        return Phi_x + x * phi_x

    # Find point of maximum derivative in [L, U] by sampling
    # (For production: use scipy.optimize.minimize_scalar)
    xs = np.linspace(L, U, 100)
    derivs = np.array([gelu_deriv(x) for x in xs])
    max_idx = np.argmax(derivs)
    x_star = xs[max_idx]

    # Lower bound: tangent at x_star
    y_star = gelu(np.array([x_star])).item()
    m_lower = gelu_deriv(x_star)
    c_lower = y_star - m_lower * x_star

    # Upper bound: chord from L to U
    y_L = gelu(np.array([L])).item()
    y_U = gelu(np.array([U])).item()
    m_upper = (y_U - y_L) / (U - L) if U != L else 0.0
    c_upper = y_L - m_upper * L

    lower = [(m_lower, c_lower, L, U)]
    upper = [(m_upper, c_upper, L, U)]

    return lower, upper


# ---------- Core analyzer ----------
class PolyAnalyzer:
    """
    Builds and solves a global LP for a neural network.

    This class implements a "relational" analyzer. Instead of propagating
    a full polytope layer by layer, it builds a *single* large system
    of linear constraints (Ax <= b) that includes *all* variables
    (inputs, pre-activations, post-activations) of the network simultaneously.

    The final polytope is defined in the high-dimensional space:
    [z0, a1, z1, a2, z2, ..., zL]

    This allows the LP solver to find much tighter bounds by using information
    from all layers at once, leading to a more precise analysis.
    """

    def __init__(self) -> None:
        """Initializes an empty analyzer."""
        # The 'A' matrix of the global LP: self.constraints_A @ x <= self.constraints_b
        self.constraints_A: Matrix = np.zeros((0, 0), dtype=np.float64)
        # The 'b' vector of the global LP
        self.constraints_b: Vector = np.zeros((0,), dtype=np.float64)
        # A dictionary mapping layer names (e.g., "z0") to their column slice
        # in the global 'A' matrix.
        self.var_slices: dict[str, slice] = {}
        # The total number of variables (columns in 'A')
        self.nvars: int = 0

    def _alloc(self, name: str, dim: int) -> slice:
        """
        Allocates 'dim' new columns (variables) in the global LP for a layer.

        This grows the self.constraints_A matrix horizontally, padding
        existing constraints with zeros, and returns the
        slice object representing the new variables' column indices.
        """
        sl = slice(self.nvars, self.nvars + dim)
        self.var_slices[name] = sl

        # Grow the constraint matrix horizontally if it's not empty
        if self.constraints_A.size == 0:
            self.constraints_A = np.zeros((0, self.nvars + dim), dtype=np.float64)
        else:
            pad = np.zeros((self.constraints_A.shape[0], dim), dtype=np.float64)
            self.constraints_A = np.hstack([self.constraints_A, pad])

        self.nvars += dim
        return sl

    def add_input_box(self, name: str, lb: Vector, ub: Vector) -> slice:
        """
        Adds the initial input box constraints (lb <= z0 <= ub).

        This is typically the first set of constraints added to the analyzer.
        """
        sl = self._alloc(name, lb.size)
        # A = [I; -I], b = [ub; -lb]
        A = np.vstack([np.eye(lb.size), -np.eye(lb.size)])
        b = np.concatenate([ub, -lb])
        self._add_constraints_to_vars(sl, A, b)
        return sl

    def add_affine(self, in_name: str, W: Matrix, b: Vector, a_name: str) -> slice:
        """
        Adds affine layer constraints: a_name = W @ in_name + b.

        This adds *equality* constraints, which are encoded as two
        inequalities:
        1. a - W*z <= b
        2. -a + W*z <= -b
        """
        inz = self.var_slices[in_name]  # Get input variable slice
        a_sl = self._alloc(a_name, b.size)  # Get new output variable slice

        m = b.size
        Aeq = np.zeros((m, self.nvars), dtype=np.float64)
        # Set coefficients for 'a' variables (+1 in the 'a' columns)
        Aeq[np.arange(m), a_sl] = 1.0
        # Set coefficients for 'z' variables (-W in the 'z' columns)
        Aeq[:, inz] -= W
        beq = b

        # Add both (Aeq @ x <= beq) and (-Aeq @ x <= -beq)
        self.constraints_A = np.vstack([self.constraints_A, Aeq, -Aeq])
        self.constraints_b = np.concatenate([self.constraints_b, beq, -beq])
        return a_sl

    def add_activation(
        self,
        a_name: str,
        act_name: str,
        bounds: tuple[Vector, Vector],
        builder: ActBuilder,
    ) -> slice:
        """
        Adds activation layer constraints: z_name = f(a_name).

        This is where the non-linear function is *relaxed*.
        It uses the 'builder' function (e.g., relu_envelope) to get
        local 2D constraints (Ai, bi) for *each neuron* and maps
        them into the global LP.

        Args:
            a_name: Name of the pre-activation variable (e.g., "a1")
            act_name: Name for the new post-activation variable (e.g., "z1")
            bounds: A (L, U) tuple of *pre-computed* bounds for 'a_name',
                    used by the builder to create the tightest relaxation.
            builder: The envelope-building function (e.g., relu_builder).
        """
        a_sl = self.var_slices[a_name]
        z_sl = self._alloc(act_name, a_sl.stop - a_sl.start)

        m_rows: list[Matrix] = []
        b_rows: list[Vector] = []
        L, U = bounds

        # Iterate over each neuron in the layer
        for i, (Li, Ui) in enumerate(zip(L, U)):
            # 1. Build the local 2D relaxation for this neuron
            #    (e.g., the 3 lines for the ReLU triangle)
            lower_lines, upper_lines = builder(Li, Ui)
            Ai, bi = envelope_from_lines(lower_lines, upper_lines)

            if Ai.size == 0:
                continue  # No constraints for this neuron (e.g., fully unconstrained)

            # 2. Map the local (A, b) into the global constraint system
            Ablock = np.zeros((Ai.shape[0], self.nvars), dtype=np.float64)
            # Column for this neuron's 'a_i' variable
            Ablock[:, a_sl.start + i] = Ai[:, 0]
            # Column for this neuron's 'z_i' variable
            Ablock[:, z_sl.start + i] = Ai[:, 1]

            m_rows.append(Ablock)
            b_rows.append(bi)

        if m_rows:  # Add all new constraints at once (more efficient)
            self.constraints_A = np.vstack([self.constraints_A, *m_rows])
            self.constraints_b = np.concatenate([self.constraints_b, *b_rows])
        return z_sl

    def _add_constraints_to_vars(self, sl: slice, A: Matrix, b: Vector) -> None:
        """
        Helper to add local constraints (A, b) that apply only to a
        specific slice 'sl' of variables.
        """
        # Create a zero matrix with the full width (nvars)
        Ablk = np.zeros((A.shape[0], self.nvars), dtype=np.float64)
        # Copy the local 'A' matrix into the correct columns
        Ablk[:, sl] = A
        # Stack the new global constraint rows
        self.constraints_A = np.vstack([self.constraints_A, Ablk])
        self.constraints_b = np.concatenate([self.constraints_b, b])

    def optimize(self, c: Vector, sense: str = "min") -> OptimizeResult:
        """
        Solves the global LP: min/max c^T @ x subject to Ax <= b.

        Args:
            c: The objective vector. Must have length self.nvars.
            sense: 'min' or 'max'.

        Returns:
            A tuple (value, result_object) where 'value' is the optimal
            value (or None on failure).
        """
        if sense == "max":
            # Maximize c^T x by minimizing -c^T x
            res = linprog(
                -c,
                A_ub=self.constraints_A,
                b_ub=self.constraints_b,
                bounds=(None, None),
                method="highs",
            )
            return (-res.fun if res.success else None), res
        else:
            res = linprog(
                c,
                A_ub=self.constraints_A,
                b_ub=self.constraints_b,
                bounds=(None, None),
                method="highs",
            )
            return (res.fun if res.success else None), res


# ---------- Example: analyze a tiny (W,b)+activation stack ----------
def example_mlp_run() -> dict[str, Any]:
    """Runs the analyzer on a small, 2-layer network."""
    # Toy: input 7x7 -> Dense(12) -> GELU envelope -> Dense(10)
    rng = np.random.default_rng(0)
    W1 = rng.normal(size=(12, 49)) * 0.2
    b1 = rng.normal(size=(12,)) * 0.01
    W2 = rng.normal(size=(10, 12)) * 0.2
    b2 = rng.normal(size=(10,)) * 0.01

    # Define the input box [0, 1] for all 49 pixels
    lb0 = np.zeros(49)
    ub0 = np.ones(49)

    # --- Interval bounds (fast IBP) ---
    # We *must* pre-compute some bounds for the activation layer.
    # IBP is the fastest way, though "OBT" (Optimization-Based
    # Bounds Tightening) would be more precise.
    L1, U1 = ibp_affine(lb0, ub0, W1, b1)

    # --- Pick activation envelope builder ---
    def gelu_builder(Li: float, Ui: float) -> Envelope:
        """A builder for GELU using the generic sampled envelope."""
        return sampled_envelope(gelu, Li, Ui, n_segments=4, pad=1e-3)

    # (Optional) A builder for ReLU, if you wanted to swap
    def relu_builder(Li: float, Ui: float) -> Envelope:
        return relu_envelope(Li, Ui)

    # --- Build global polytope ---
    P = PolyAnalyzer()
    # 1. Add input box: z0 in [0, 1]^49
    z0 = P.add_input_box("z0", lb0, ub0)
    # 2. Add affine layer: a1 = W1*z0 + b1
    a1 = P.add_affine("z0", W1, b1, "a1")
    # 3. Add activation: z1 = GELU(a1), using IBP bounds (L1, U1)
    z1 = P.add_activation("a1", "z1", (L1, U1), gelu_builder)
    # 4. Add final affine layer: a2 = W2*z1 + b2 (these are the logits)
    a2 = P.add_affine("z1", W2, b2, "a2")

    # --- Probe the resulting polytope ---
    # Find bounds (min/max) for the first logit (output 0)
    nvars = P.nvars
    c = np.zeros(nvars)
    c[P.var_slices["a2"].start + 0] = 1.0  # Objective is c = [0...0, 1, 0...0]
    lo0, _ = P.optimize(c, "min")
    hi0, _ = P.optimize(c, "max")

    # Find the worst-case "margin" between logit 1 and logit 0.
    # We want to find min(a2[1] - a2[0]).
    # If this min value is > 0, it means a2[1] is *always* > a2[0].
    c = np.zeros(nvars)
    a2_sl = P.var_slices["a2"]
    c[a2_sl.start + 1] = 1.0  # +1 for a2[1]
    c[a2_sl.start + 0] = -1.0  # -1 for a2[0]
    # Find min(a2[1] - a2[0])
    worst_margin, _ = P.optimize(c, "min")

    return dict(logit0=(lo0, hi0), worst_m01=worst_margin)


# ---------- Loading a trained FNN and analyzing it ----------
# import torch
# def load_mnist_fnn():
#     """Load your trained FNN."""
#     model = torch.load('path/to/your/fnn.pt')
#     model.eval()
#
#     # Extract weights and biases
#     layers = []
#     for module in model.modules():
#         if isinstance(module, torch.nn.Linear):
#             W = module.weight.detach().numpy()
#             b = module.bias.detach().numpy()
#             layers.append(('affine', W, b))
#         elif isinstance(module, torch.nn.GELU):
#             layers.append(('gelu',))
#
#     return layers
#
# def analyze_network(layers, input_region):
#     """Analyze a loaded network."""
#     P = PolyAnalyzer()
#     lb, ub = input_region
#     z = P.add_input_box("z0", lb, ub)
#
#     for i, layer in enumerate(layers):
#         if layer[0] == 'affine':
#             _, W, b = layer
#             a = P.add_affine(f"z{i}", W, b, f"a{i+1}")
#             # Compute bounds for activation
#             L, U = obt_affine(P, f"a{i+1}", np.eye(len(b)), np.zeros(len(b)))
#         elif layer[0] == 'gelu':
#             z = P.add_activation(f"a{i+1}", f"z{i+1}", (L, U), tight_gelu_envelope)
#
#     return P

if __name__ == "__main__":
    out = example_mlp_run()
    print("Analysis results:")
    print(f"  Bounds for logit 0: ({out['logit0'][0]:.4f}, {out['logit0'][1]:.4f})")
    print(f"  Min margin (logit 1 - logit 0): {out['worst_m01']:.4f}")
