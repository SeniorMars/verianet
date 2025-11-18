import pyomo.environ as pyo
import numpy as np
from typing import Callable, Tuple, List

# here for the sake of testing
def relu_envelope(L: float, U: float):
    """
    Return (lower_lines, upper_lines) for ReLU on [L, U].
    Each line is (m, c, x_min, x_max), but we only use m, c.
    """
    # z = max(0, a)
    lower_lines = []
    upper_lines = []

    if U <= 0:
        # Always inactive: z = 0
        # lower:  z >= 0
        # upper:  z <= 0
        lower_lines.append((0.0, 0.0, L, U))
        upper_lines.append((0.0, 0.0, L, U))
    elif L >= 0:
        # Always active: z = a
        # lower:  z >= a
        # upper:  z <= a
        lower_lines.append((1.0, 0.0, L, U))
        upper_lines.append((1.0, 0.0, L, U))
    else:
        # Mixed case: convex hull
        # lower: z >= 0, z >= a
        lower_lines.append((0.0, 0.0, L, U))  # z >= 0
        lower_lines.append((1.0, 0.0, L, U))  # z >= a

        # upper: line through (L, 0) and (U, U)
        slope = U / (U - L)
        intercept = -slope * L
        upper_lines.append((slope, intercept, L, U))

    return lower_lines, upper_lines

def ibp_affine_keras(L: np.ndarray, U: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs IBP for weights in (in_dim, out_dim) format.
    Assumes math is: y = x @ W + b
    """
    Wpos = np.maximum(W, 0)
    Wneg = np.minimum(W, 0)
    
    # L_out_j = L @ Wpos_j + U @ Wneg_j + b_j
    # U_out_j = U @ Wpos_j + L @ Wneg_j + b_j
    L_out = L @ Wpos + U @ Wneg + b
    U_out = U @ Wpos + L @ Wneg + b
    return L_out, U_out

class PyomoPolyAnalyzer:
    """
    Builds a Pyomo model representing the polytope defined by a neural network.
    The polytope exists in the space of all layer activations.
    """
    
    def __init__(self):
        self.model = pyo.ConcreteModel()
        self.var_names = []   # Track variable names in order
        self.layer_dims = {}  # Map layer name to dimension
        
    def add_input_box(self, name: str, lb: np.ndarray, ub: np.ndarray):
        """Add input layer with box constraints."""
        n = len(lb)
        self.layer_dims[name] = n
        self.var_names.append(name)
        
        setattr(self.model, name, pyo.Var(range(n), domain=pyo.Reals))
        layer_vars = getattr(self.model, name)
        
        def input_lb_rule(model, i):
            return layer_vars[i] >= lb[i]
        
        def input_ub_rule(model, i):
            return layer_vars[i] <= ub[i]
        
        setattr(self.model, f"{name}_lb", pyo.Constraint(range(n), rule=input_lb_rule))
        setattr(self.model, f"{name}_ub", pyo.Constraint(range(n), rule=input_ub_rule))
        
        return name
    
    def add_affine(self, in_name: str, W: np.ndarray, b: np.ndarray, a_name: str):
        """
        Add affine transformation: a_name = in_name @ W + b
        Here W is assumed to have shape (n_in, m) = (input_dim, output_dim),
        as in Keras Dense layers.
        """
        n_in, m = W.shape          # W: (in_dim, out_dim)
        self.layer_dims[a_name] = m
        self.var_names.append(a_name)
        
        # Create variables for this layer (output of affine)
        setattr(self.model, a_name, pyo.Var(range(m), domain=pyo.Reals))
        a_vars = getattr(self.model, a_name)
        in_vars = getattr(self.model, in_name)
        
        # a[i] = sum_j in[j] * W[j, i] + b[i]
        def affine_rule(model, i):
            return a_vars[i] == sum(W[j, i] * in_vars[j] for j in range(n_in)) + b[i]
        
        setattr(self.model, f"{a_name}_affine", 
                pyo.Constraint(range(m), rule=affine_rule))
        
        return a_name
    
    def add_activation(self, a_name: str, z_name: str, 
                      bounds: Tuple[np.ndarray, np.ndarray],
                      builder: Callable):
        """
        Add activation layer with polyhedral relaxation.
        builder should return (lower_lines, upper_lines) for each neuron.
        """
        L, U = bounds
        n = len(L)
        self.layer_dims[z_name] = n
        self.var_names.append(z_name)
        
        setattr(self.model, z_name, pyo.Var(range(n), domain=pyo.Reals))
        z_vars = getattr(self.model, z_name)
        a_vars = getattr(self.model, a_name)
        
        constraint_idx = 0
        for i in range(n):
            Li, Ui = float(L[i]), float(U[i])
            lower_lines, upper_lines = builder(Li, Ui)
            
            # lower: z >= m*a + c
            for m_val, c_val, _, _ in lower_lines:
                m_val = float(m_val)
                c_val = float(c_val)
                # Use a factory function to properly capture the values
                def make_lower_rule(m_val, c_val, i):
                    def lower_rule(model):
                        return z_vars[i] >= m_val * a_vars[i] + c_val
                    return lower_rule
                cname = f"{z_name}_lower_{i}_{constraint_idx}"
                setattr(self.model, cname, pyo.Constraint(rule=make_lower_rule(m_val, c_val, i)))
                constraint_idx += 1
            
            # upper: z <= m*a + c
            for m_val, c_val, _, _ in upper_lines:
                m_val = float(m_val)
                c_val = float(c_val)
                # Use a factory function to properly capture the values
                def make_upper_rule(m_val, c_val, i):
                    def upper_rule(model):
                        return z_vars[i] <= m_val * a_vars[i] + c_val
                    return upper_rule
                cname = f"{z_name}_upper_{i}_{constraint_idx}"
                setattr(self.model, cname, pyo.Constraint(rule=make_upper_rule(m_val, c_val, i)))
                constraint_idx += 1
        
        return z_name
    
    def optimize(self, objective_expr, sense='min'):
        """
        Optimize over the polytope.
        objective_expr should be a Pyomo expression using the model variables.
        """
        # Delete previous objective if it exists to avoid warnings
        if hasattr(self.model, 'obj'):
            self.model.del_component('obj')
        
        if sense == 'min':
            self.model.obj = pyo.Objective(expr=objective_expr, sense=pyo.minimize)
        else:
            self.model.obj = pyo.Objective(expr=objective_expr, sense=pyo.maximize)
        
        solver = pyo.SolverFactory('glpk')  # change if you want gurobi, etc.
        results = solver.solve(self.model, tee=False)
        
        term = results.solver.termination_condition
        if term == pyo.TerminationCondition.optimal:
            obj_value = pyo.value(self.model.obj)
            return obj_value, results
        else:
            return None, results
    
    def get_polytope_description(self):
        """
        TODO: Extract all constraints and convert to A x <= b.
        """
        raise NotImplementedError("Not implemented yet")