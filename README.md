# Verianet

Polyhedral abstract interpretation experiments for a tiny MNIST classifier.

The model is a 7x7 MNIST MLP with architecture:

```text
49 input pixels -> 3 GELU neurons -> 3 GELU neurons -> 10 logits
```

Weights are stored at the repository root in `verysmallnn_weights.npz`; code
should access them through `verianet.paths.WEIGHTS_PATH`.

## Core Code

The canonical implementation lives in `verianet/`:

- `verianet.activations`: GELU, exact GELU interval bounds, sound linear GELU envelopes
- `verianet.bounds`: interval bound propagation for affine and activation layers
- `verianet.lp`: explicit `A @ x <= b` LP builder using SciPy/HiGHS
- `verianet.network`: weight loading and forward logits
- `verianet.objectives`: robustness, margin, and counterfactual-style LP objectives
- `verianet.paths`: stable project, weights, and results paths
- `verianet.refinement`: split-box LP refinement for tighter certified bounds
- `verianet.stats`: Hoeffding sample counts and clipped error intervals

Experiment scripts live in `scripts/experiments/`:

- `scripts.experiments.epsilon_robustness_test`: robustness sweep over digits and epsilon values
- `scripts.experiments.ideal_digits`: margin-optimized digit generation
- `scripts.experiments.finding`: counterfactual search with concrete-network validation
- `scripts.experiments.robustness_check`: small digit-0 robustness smoke check

Mechanistic interpretability scripts live in `scripts/mechanistic/`.
Additional exploratory scripts live in `scripts/exploratory/`, notebooks live in
`notebooks/`, generated outputs live in `results/`, and the static web demo lives
in `demo/`.

## Setup

Install the package with the tested core dependencies:

```bash
python3 -m pip install -e .
```

Install optional experiment stacks as needed:

```bash
python3 -m pip install -e ".[mnist]"
python3 -m pip install -e ".[legacy]"
```

`tensorflow` is only needed for scripts that load MNIST directly. `pyomo` and
`glpk` are only needed for older exploratory scripts that still use
`verianet.legacy.pyomo.PyomoPolyAnalyzer`. The core LP path in `verianet/`
uses SciPy and does not require Pyomo.

The pinned requirements files remain for non-editable installs and older
workflows:

```bash
python3 -m pip install -r requirements-core.txt
python3 -m pip install -r requirements.txt
```

## Quick Checks

Run unit tests:

```bash
python3 -m unittest
```

Run the lightweight SciPy LP demo:

```bash
python3 -m verianet.legacy.basic
```

Run an experiment script as a module from the repository root:

```bash
python3 -m scripts.experiments.ideal_digits
python3 -m scripts.experiments.finding
python3 -m scripts.experiments.epsilon_robustness_test
python3 -m scripts.mechanistic.simple_neuron_probe
```

The old root-level `basic.py` and `helper.py` compatibility wrappers have been
removed. Use `verianet.legacy.basic` and `verianet.legacy.pyomo` for legacy APIs.

## Notes

The old GELU tangent/secant helper was not sound on all intervals. It has been
replaced by a certified envelope that keeps curvature-aware tangent/secant lines
only after proving each line bounds exact GELU on the whole interval. See
`docs/gelu_envelopes.md` for the certificate. This restores the verification
interpretation: feasible LP regions over-approximate the true GELU network
instead of accidentally cutting through it.

For tighter bounds without switching to MILP, `verianet.refinement` can split
the input box and solve one refined LP per leaf. See `docs/refinement.md`.

Counterfactual generation has two statuses. `relaxed_feasible` means the LP
relaxation can reach the requested target margin, which is a certificate only
for the relaxed model. `concrete_valid` means the selected LP input also flips
or satisfies the target margin in the true GELU network. Because the relaxation
is an over-approximation, relaxed feasibility alone is not a proof that a true
network counterfactual exists.

The robustness sweep uses a two-sided Hoeffding bound for Bernoulli rates. For
95% confidence and +/-5% error this requires 738 samples per digit/epsilon
configuration; solver failures are counted as non-robust.
