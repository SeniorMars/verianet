# LP Refinement By Splitting

The base Verianet LP applies every activation envelope line globally over each
pre-activation interval `[L, U]`. This is sound, but it means a secant that is
valid only on a subinterval cannot be added to the same LP without a disjunction.

There are three ways to do better:

1. Add only globally certified lines.
   This is what `sound_gelu_envelope` does.

2. Split the input box into smaller boxes and solve one LP per leaf.
   This is what `verianet.refinement` does.

3. Add binary/disjunctive activation splitting.
   This becomes a MILP or a branch-and-bound verifier. Verianet does not do this
   in the core path yet.

## Why Input Splitting Helps

For each leaf input box:

```text
lb_leaf <= x <= ub_leaf
```

IBP computes narrower pre-activation intervals, so GELU envelopes can include
more certified tangent/secant cuts. The union of all leaf boxes exactly covers
the original input box.

For a maximization objective:

```text
max over root <=? root LP upper bound
max over root true network <= max_i leaf LP upper bound_i
```

Therefore robustness remains sound: if every leaf LP proves every competing
margin is `<= 0`, then the whole root box is certified robust.

## Nested Cuts

Each leaf LP inherits the activation cuts from the unsplit root LP and then adds
leaf-specific cuts. This makes each leaf relaxation a refinement of the root
relaxation, not merely a different relaxation. In practice this means the split
upper bound should be no looser than the root upper bound, up to solver
numerics.

## Cost

The tradeoff is direct: `max_leaves=8` solves up to eight LPs per objective.
For robustness with ten classes, this can mean up to `9 * max_leaves` LP solves
per input sample.
