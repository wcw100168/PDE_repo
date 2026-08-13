# Current Research State

**Last Updated:** 2026-08-13

## Current Objective

Design a metric-compatible SBP differential operator that achieves strict discrete mass conservation ($\frac{dM}{dt} = 0$) and unconditional $L^2$ energy stability ($\frac{dE}{dt} \le 0$) on spherical geometry without aliasing errors.

## Completed Results

- [x] **Task 1 — Reference Element & Dubiner Basis**: Implemented in `src/operators/basis.py` & `orthogonalization.py`.
- [x] **Task 2 — Subdivided Octahedral Sphere Mesh & Metrics**: Implemented in `src/geometry/sphere_mesh.py` & `metrics.py`.
- [x] **Task 3 — Closed-Form SBP Operators & Peirce Subspace**: Implemented in `src/operators/sbp.py` & `peirce.py`.
- [x] **Task 4 — RHS Formulations & Time Stepper**: Implemented in `src/solver/formulations.py`, `fluxes.py`, & `time_stepper.py`.
- [x] **Task 5 — Benchmarks & Dual-Verification**: 26/26 automated Pytest tests passed to machine precision ($10^{-16}$).

## Verification Status

| ID | Subject | LLM Audit | Symbolic | Numerical | Human | Formal |
|----|---------|-----------|----------|-----------|-------|--------|
| Task 1 | Dubiner Basis & Cholesky Ortho | Verified | Verified | Verified ($10^{-16}$) | Verified | — |
| Task 2 | Sphere Mesh & Intrinsic Metrics | Verified | Verified | Verified ($10^{-12}$) | Verified | — |
| Task 3 | SBP Operator & Peirce Subspace | Verified | Verified | Verified ($10^{-12}$) | Verified | — |
| Task 4 | RHS Formulations & LSRK45 | Verified | Verified | Verified ($10^{-10}$) | Verified | — |
| Task 5 | Full Pytest Suite (26/26) | Verified | Verified | Verified (100% Pass) | Verified | — |


## Known Issues

1. **Split2 Mass Drift ($\epsilon_{\text{mass}}$)**: Missing metric divergence correction leads to mesh-dependent mass drift.
2. **Split3 Energy Source Term ($\epsilon_{\text{energy}}$)**: Non-zero volume metric source term causes periodic energy oscillation ($10^{-11} \sim 10^{-5}$).
3. **Interface Normal Misalignment ($R_{\text{proj}}$)**: Microscopic normal vector misalignment across adjacent curved triangular elements.

## Next Actions

1. Construct metric-compatible SBP differential operators to eliminate $\epsilon_{\text{mass}}$ and $\epsilon_{\text{energy}}$ simultaneously.
2. Implement interface projection normal alignment algorithm to eliminate $R_{\text{proj}}$.
3. Extend SBP-DG solver to Spherical Shallow Water Equations.

