# Current Research State

**Last Updated:** 2026-08-13

## Current Objective

Design a metric-compatible SBP differential operator that achieves strict discrete mass conservation ($\frac{dM}{dt} = 0$) and unconditional $L^2$ energy stability ($\frac{dE}{dt} \le 0$) on spherical geometry without aliasing errors.

## Completed Results

- [x] **D001 — Radial Subdivided Octahedral Sphere Mapping**: Transformation $(\xi,\eta) \to \mathbf{x}_{\text{flat}} \to \mathbf{x}_s \in \mathbb{S}^2$ with intrinsic $\mathbb{R}^3$ metric computation.
- [x] **D002 — Preconditioned Cholesky Modal Orthogonalization**: $V = V_{\text{raw}}(L^T)^{-1}$ achieving $V^T W V = I$ to $O(10^{-16})$ residual.
- [x] **D003 — Closed-Form SBP Operator**: $\Delta D_\eta = \frac{1}{2} W^{-1}(I + P)^T B (I - P)$ under arbitrary boundary matrix $B$.
- [x] **T2.1 — Space Congruence Isomorphism Theorem**: Proved $D_\eta^{\text{new, A}} \equiv D_\eta^{\text{new, B}}$ under congruence coordinate transformation.
- [x] **T3.2 & T3.3 — Peirce Subspace Decomposition Theorem**: Fully decoupled uniqueness proof over $S_{PQ}, S_{QP}, S_{QQ}$ subspaces.
- [x] **E001 — Comparative Analysis of Formulations**: Conservative, Split2 (energy stable), and Split3 (mass conserving).

## Verification Status

| ID | Subject | LLM Audit | Symbolic | Numerical | Human | Formal |
|----|---------|-----------|----------|-----------|-------|--------|
| T2.1 | Space Congruence Isomorphism | Verified | Verified | Verified ($10^{-16}$) | Verified | — |
| T3.2 | Peirce Subspace Decomposition | Verified | Verified | Verified | Verified | — |
| T3.3 | Boundary Constraint Uniqueness | Verified | Verified | Verified | Verified | — |
| Split2 | Two-Term Split Energy Stability | Verified | Verified | Verified ($10^{-16}$) | Verified | — |
| Split3 | Three-Term Split Mass Drift | Verified | Verified | Verified ($10^{-16}$) | Verified | — |

## Known Issues

1. **Split2 Mass Drift ($\epsilon_{\text{mass}}$)**: Missing metric divergence correction leads to mesh-dependent mass drift.
2. **Split3 Energy Source Term ($\epsilon_{\text{energy}}$)**: Non-zero volume metric source term causes periodic energy oscillation ($10^{-11} \sim 10^{-5}$).
3. **Interface Normal Misalignment ($R_{\text{proj}}$)**: Microscopic normal vector misalignment across adjacent curved triangular elements.

## Next Actions

1. Construct metric-compatible SBP differential operators to eliminate $\epsilon_{\text{mass}}$ and $\epsilon_{\text{energy}}$ simultaneously.
2. Implement interface projection normal alignment algorithm to eliminate $R_{\text{proj}}$.
3. Extend SBP-DG solver to Spherical Shallow Water Equations.

