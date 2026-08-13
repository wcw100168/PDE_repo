---
id: D003
type: definition
name: "Closed-Form Boundary-Compatible SBP Operator"
notation: "D_eta_new"
dependencies: ["A003", "A004"]
used_by: ["T2.1", "T3.2", "T3.3"]
source: "my_derivation"
date_created: 2026-08-13
---

# Definition D003 — Closed-Form Boundary-Compatible SBP Operator

## Statement

Given any symmetric boundary matrix $B$, the modified differentiation operator $D_\eta^{\text{new}} = D_\eta^{\text{ours}} + \Delta D_\eta$ on reference triangle $T$ is defined in closed form as:

$$
\Delta D_\eta = \frac{1}{2} W^{-1} (I + P)^T B (I - P)
$$

where $W$ is the diagonal volume quadrature weight matrix, and $P = V (V^T W V)^{-1} V^T W$ is the symmetric polynomial projection operator.

## Remarks

- Equivalent algebraically to Chen & Shu (2017) boundary operator formulation.
- Preserves exact polynomial accuracy up to degree $k$ while satisfying SBP boundary compatibility.

## Related

- Used in: T2.1, T3.2, T3.3
