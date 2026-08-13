# Open Questions

## High Priority

### Q001 — Metric-Compatible SBP Operator Construction
**Question:** How to design a curved metric-compatible SBP differential operator that eliminates both mass drift $\epsilon_{\text{mass}}$ in Two-Term Split (Split2) and energy source term $\epsilon_{\text{energy}}$ in Three-Term Split (Split3) without sacrificing high-order spatial accuracy?

**Context:** Non-polynomial metric terms ($\mathcal{J}, u^\xi, u^\eta$) cause geometric aliasing and break discrete SBP integral identities on spherical triangle elements.

**Related:** Split2, Split3, DGCL

**Status:** open

---

### Q002 — Interface Normal Misalignment ($R_{\text{proj}}$)
**Question:** How to enforce strict interface flux conservation and energy cancellation across adjacent curved triangular elements where face normal vectors possess microscopic orientation misalignment due to independent nonlinear radial projections?

**Context:** $R_{\text{proj}}$ causes microscopic interface energy leakage.

**Related:** Interface Numerical Flux, Penalty Flux

**Status:** open

---

## Medium Priority

### Q003 — Mathematical Proof of Radial Projection Stability
**Question:** Can we establish a rigorous differential geometric and spectral analysis proof explaining why the combination of "Radial Projection + 3D $\mathbb{R}^3$ Euclidean intrinsic computation" avoids numerical instability and pole singularities, whereas Equal-Area projection or 2D parameter plane solvers diverge?

**Context:** Currently empirical trial-and-error result ($2 \times 2$ experiment).

**Related:** Domain Mapping, Differential Geometry

**Status:** open

---

## Low Priority

### Q004 — Extension to Non-Linear System Equations
**Question:** How to generalize the scalar advection SBP-DG stability and conservation proofs to non-linear systems, specifically Spherical Shallow Water Equations and Euler Equations?

**Status:** open

