# GELU Envelope Certification

The core LP relaxation uses affine bounds of the form

```text
lower: z >= m * a + c
upper: z <= m * a + c
```

for `z = GELU(a)` on a bounded pre-activation interval `[L, U]`.

## Exact Derivatives

For the exact GELU

```text
GELU(x) = x * Phi(x)
```

where `Phi` is the standard normal CDF:

```text
GELU'(x)  = Phi(x) + x * phi(x)
GELU''(x) = phi(x) * (2 - x^2)
```

Since `phi(x) > 0`, GELU is concave on `(-inf, -sqrt(2))`, convex on
`(-sqrt(2), sqrt(2))`, and concave on `(sqrt(2), inf)`.

## Line Certificate

To certify a lower line, define

```text
r(x) = GELU(x) - (m * x + c)
```

The line is sound iff `min_{x in [L, U]} r(x) >= 0`.

To certify an upper line, define

```text
r(x) = (m * x + c) - GELU(x)
```

The line is sound iff `min_{x in [L, U]} r(x) >= 0`.

For either residual, stationary points occur only when

```text
GELU'(x) = m
```

Because `GELU'` is monotone on each curvature segment, there is at most one root
inside each of:

```text
[L, U] intersect (-inf, -sqrt(2))
[L, U] intersect (-sqrt(2), sqrt(2))
[L, U] intersect (sqrt(2), inf)
```

Therefore the global residual minimum is found by checking the interval
endpoints and the finite set of roots of `GELU'(x) = m` on those segments.
`verianet.activations.certify_gelu_line` implements exactly this check.

## Tighter Relaxation

`verianet.activations.sound_gelu_envelope` starts with exact scalar GELU output
bounds, then proposes curvature-aware tangent candidates and secants between
adaptive sample points, stationary points, and curvature breakpoints. Every
candidate is kept only if `certify_gelu_line` proves it bounds GELU on the whole
interval `[L, U]`.

The current LP formulation applies every line globally over `[L, U]`; it does
not introduce binary variables or disjunctive activation regions. For that
reason, subinterval secants are not trusted merely because they are valid on a
subinterval. They are added only when the global certificate passes.
