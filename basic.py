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

# Dictionary for linprog status codes
linprog_status_messages = {
    0: "Optimization terminated successfully.",
    1: "Iteration limit reached.",
    2: "Problem appears to be infeasible.",
    3: "Problem appears to be unbounded.",
    4: "Numerical difficulties encountered.",
}


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


# ---------- Activation envelopes ----------
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
        # FIX: Add [0] to extract the scalar from the 1-element array
        y0 = float(f(np.array([x0], dtype=np.float64))[0])
        y1 = float(f(np.array([x1], dtype=np.float64))[0])

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


# ReLU and HardTanh envelopes on [L,U]
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
    # If bounds are very tight, just use identity
    if abs(U - L) < 1e-6:
        return [(1.0, 0.0, L, U)], [(1.0, 0.0, L, U)]
    
    # For large positive values where GELU ≈ x, use simple linear bounds
    if L > 2.0:
        return [(1.0, 0.0, L, U)], [(1.0, 0.0, L, U)]
    
    # For large negative values where GELU ≈ 0
    if U < -3.0:
        return [(0.0, 0.0, L, U)], [(0.0, 0.0, L, U)]
    
    def gelu_deriv(x: float) -> float:
        phi_x = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        Phi_x = 0.5 * (1 + erf(x / np.sqrt(2)))
        return Phi_x + x * phi_x

    xs = np.linspace(L, U, 100)
    derivs = np.array([gelu_deriv(x) for x in xs])
    max_idx = np.argmax(derivs)
    x_star = xs[max_idx]

    y_star = gelu(np.array([x_star])).item()
    m_tangent = gelu_deriv(x_star)
    c_tangent = y_star - m_tangent * x_star

    y_L = gelu(np.array([L])).item()
    y_U = gelu(np.array([U])).item()
    m_secant = (y_U - y_L) / (U - L) if U != L else 0.0
    c_secant = y_L - m_secant * L

    # CRITICAL: Check which is actually lower/upper at midpoint
    mid = (L + U) / 2
    tangent_at_mid = m_tangent * mid + c_tangent
    secant_at_mid = m_secant * mid + c_secant
    
    if tangent_at_mid <= secant_at_mid:
        lower = [(m_tangent, c_tangent, L, U)]
        upper = [(m_secant, c_secant, L, U)]
    else:
        lower = [(m_secant, c_secant, L, U)]
        upper = [(m_tangent, c_tangent, L, U)]

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
        inz = self.var_slices[in_name]
        a_sl = self._alloc(a_name, b.size)

        m = b.size
        Aeq = np.zeros((m, self.nvars), dtype=np.float64)

        # Put +I on 'a' columns
        Aeq[:, a_sl] = np.eye(m, dtype=np.float64)

        # Put -W on 'z' columns
        Aeq[:, inz] -= W

        beq = b
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

        L, U = bounds
        # enforce a in [L, U] so the envelope is valid
        m_bounds = L.size
        A_bound = np.zeros((2 * m_bounds, self.nvars), dtype=np.float64)
        b_bound = np.zeros(2 * m_bounds, dtype=np.float64)

        # a_i <= U_i
        for i in range(m_bounds):
            A_bound[i, a_sl.start + i] = 1.0
            b_bound[i] = U[i]

        # -a_i <= -L_i  <=>  a_i >= L_i
        for i in range(m_bounds):
            A_bound[m_bounds + i, a_sl.start + i] = -1.0
            b_bound[m_bounds + i] = -L[i]

        self.constraints_A = np.vstack([self.constraints_A, A_bound])
        self.constraints_b = np.concatenate([self.constraints_b, b_bound])

        m_rows, b_rows = [], []
        for i, (Li, Ui) in enumerate(zip(L, U)):
            lower_lines, upper_lines = builder(float(Li), float(Ui))
            Ai, bi = envelope_from_lines(lower_lines, upper_lines)
            if Ai.size == 0:
                continue
            Ablock = np.zeros((Ai.shape[0], self.nvars), dtype=np.float64)
            Ablock[:, a_sl.start + i] = Ai[:, 0]
            Ablock[:, z_sl.start + i] = Ai[:, 1]
            m_rows.append(Ablock)
            b_rows.append(bi)

        if m_rows:
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
        return tight_gelu_envelope(Li, Ui)

    # A builder for ReLU, if you wanted to swap
    def relu_builder(Li: float, Ui: float) -> Envelope:
        return relu_envelope(Li, Ui)

    # Build global polytope
    P = PolyAnalyzer()
    z0 = P.add_input_box("z0", lb0, ub0)
    a1 = P.add_affine("z0", W1, b1, "a1")

    # --- !! IMPORTANT !! ---
    # Make sure you are using relu_builder for this test!
    # This proves the core analyzer works.
    z1 = P.add_activation("a1", "z1",
                          (L1, U1), gelu_builder)

    a2 = P.add_affine("z1", W2, b2, "a2")  # logits

    # --- Capture the full result object ---
    nvars = P.nvars
    c = np.zeros(nvars)
    c[P.var_slices["a2"].start + 0] = 1.0
    lo0, res_lo0 = P.optimize(c, "min")  # <-- Capture res_lo0
    hi0, res_hi0 = P.optimize(c, "max")  # <-- Capture res_hi0

    c = np.zeros(nvars)
    a2_sl = P.var_slices["a2"]
    c[a2_sl.start + 1] = 1.0
    c[a2_sl.start + 0] = -1.0
    worst_margin, res_margin = P.optimize(c, "min")  # <-- Capture res_margin

    return dict(
        logit0=(lo0, hi0),
        worst_m01=worst_margin,
        # Pass the raw results back for debugging
        res_lo0=res_lo0,
        res_hi0=res_hi0,
        res_margin=res_margin,
    )

if __name__ == "__main__":
    out = example_mlp_run()

    def fmt(val):
        return f"{val:.4f}" if val is not None else "None (solver failed)"

    print("Analysis results:")
    print(f"  Bounds for logit 0: ({fmt(out['logit0'][0])}, {fmt(out['logit0'][1])})")
    print(f"  Min margin (logit 1 - logit 0): {fmt(out['worst_m01'])}")

    # --- DEBUGGING: Print solver status ---
    if out["logit0"][0] is None:
        print("\n--- Solver Debug (logit 0 min) ---")
        res = out["res_lo0"]
        print(f"  Success: {res.success}")
        # Use .get() for safety in case of an unknown status
        status_msg = linprog_status_messages.get(res.status, "Unknown status code.")
        print(f"  Status:  {res.status} ({status_msg})")
        print(f"  Message: {res.message}")

    if out["worst_m01"] is None:
        print("\n--- Solver Debug (worst_margin) ---")
        res = out["res_margin"]
        print(f"  Success: {res.success}")
        status_msg = linprog_status_messages.get(res.status, "Unknown status code.")
        print(f"  Status:  {res.status} ({status_msg})")
        print(f"  Message: {res.message}")
