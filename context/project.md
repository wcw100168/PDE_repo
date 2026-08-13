# Project Context

## Problem

We study high-order Summation-by-Parts Discontinuous Galerkin (SBP-DG) methods on three-dimensional embedded spherical surfaces ($\mathbb{S}^2 \subset \mathbb{R}^3$) for hyperbolic conservation laws, specifically scalar advection in global atmospheric modeling and climate prediction.

## Governing Equation

Scalar advection equation in curvilinear coordinates on the reference triangle $T$:

$$
\mathcal{J}\frac{\partial q}{\partial t} + \frac{\partial}{\partial \xi}(\mathcal{J} u^\xi q) + \frac{\partial}{\partial \eta}(\mathcal{J} u^\eta q) = 0, \quad (\xi, \eta) \in T
$$

where $T$ is the reference triangle, $\mathcal{J}$ is the Jacobian determinant of the metric mapping, and $u^\xi, u^\eta$ are contravariant velocity components projected onto the local tangent space.

## Domain & Mesh Geometry

- Physical domain: Embedded 2-sphere $\mathbb{S}^2 \subset \mathbb{R}^3$
- Reference domain: Unit reference triangle $T = \{(\xi, \eta) : \xi \ge 0, \eta \ge 0, \xi + \eta \le 1\}$
- Mesh type: Subdivided Octahedral Spherical Mesh (avoiding pole singularities and complex cubed-sphere corner seams)
- Mapping: Reference triangle $(\xi,\eta) \to$ Octahedral flat triangle $\mathbf{x}_{\text{flat}} \to$ Radial projection to unit sphere $\mathbf{x}_s = \mathbf{x}_{\text{flat}} / \|\mathbf{x}_{\text{flat}}\|$

## Numerical Method

- Spatial discretization: Nodal Discontinuous Galerkin (DG) with multidimensional Simplex Quadrature and SBP operators
- Basis / Orthogonalization: Preconditioned Cholesky orthogonalized Vandermonde matrix ($V^T W V = I$) via Jacobi/Dubiner orthogonal polynomials
- Formulations under study:
  1. Conservative form
  2. Two-Term Split form (Split2): Unconditional $L^2$ energy stable
  3. Three-Term Split form (Split3): Machine-precision discrete mass conserving
- Temporal discretization: Carpenter & Kennedy 5th-order 4-stage Low-Storage Runge-Kutta (LSRK45)

## Main Objectives

1. Develop a metric-compatible SBP differential operator that simultaneously achieves strict discrete mass conservation ($\frac{dM}{dt} = 0$) and unconditional $L^2$ energy stability ($\frac{dE}{dt} \le 0$).
2. Resolve geometric aliasing and discrete Geometric Conservation Law (DGCL) errors induced by non-polynomial metric terms.
3. Provide rigorous mathematical proofs for stability, modal Cholesky orthogonalization congruence, and Peirce subspace decomposition.

## Current Research Status

- [x] Preconditioned Cholesky modal orthogonalization ($O(10^{-16})$ residual)
- [x] Closed-form boundary-compatible SBP operator derivation & Peirce subspace decomposition proof
- [x] Space Congruence Isomorphism Theorem (Theorem 2.1)
- [x] Comparative numerical analysis of Conservative, Split2, and Split3 formulations
- [ ] Metric-compatible SBP operator for joint mass conservation & energy stability
- [ ] Interface projection normal alignment ($R_{\text{proj}}$ elimination)
- [ ] Extension to Spherical Shallow Water Equations & Euler Equations

## Current Focus

Constructing geometric-compatible SBP operators to eliminate mass drift $\epsilon_{\text{mass}}$ in Split2 while suppressing energy source terms $\epsilon_{\text{energy}}$ in Split3.

