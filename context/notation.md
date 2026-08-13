# Notation Contract

## Domain & Coordinates

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $\mathbb{S}^2$ | Unit 2-sphere embedded in $\mathbb{R}^3$ | $\mathbb{S}^2 \subset \mathbb{R}^3$ |
| $T$ | Reference triangle unit domain | $(\xi,\eta) \in T$ |
| $(\xi,\eta)$ | Reference coordinates | |
| $\mathbf{x}_s$ | Physical coordinates on sphere | $\mathbf{x}_s \in \mathbb{S}^2$ |
| $\mathbf{x}_{\text{flat}}$ | Octahedral flat plane coordinates | $\sum L_i \mathbf{v}_i$ |

## Geometry & Operators

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $\mathcal{J}$ | Jacobian determinant | $\mathcal{J} > 0$ always |
| $\mathbf{a}_1, \mathbf{a}_2$ | Covariant basis vectors | $\mathbf{a}_1 = \partial \mathbf{x}_s / \partial \xi, \mathbf{a}_2 = \partial \mathbf{x}_s / \partial \eta$ |
| $\mathbf{a}^1, \mathbf{a}^2$ | Contravariant basis vectors | $\mathbf{a}^1 = (\mathbf{a}_2 \times \mathbf{n}_{\text{surf}}) / \mathcal{J}$ |
| $u^\xi, u^\eta$ | Contravariant velocity components | Tangent space projections |
| $D_\xi, D_\eta$ | SBP 1st-derivative operator matrices | Discrete differential operators |
| $W, W_b$ | Volume & boundary quadrature weights | Diagonal matrices |
| $B_\xi, B_\eta$ | Boundary line-integral operators | Diagonal matrices |
| $V_{\text{raw}}, V$ | Raw & Cholesky-orthogonalized Vandermonde matrices | $V^T W V = I$ |
| $P, Q$ | Symmetric polynomial projection & orthogonal complement operators | $P = V(V^TWV)^{-1}V^TW$, $Q = I - P$ |
| $E$ | Extraction operator matrix | Extracts boundary nodal states |

## Solutions & Residuals

| Symbol | Meaning | Notes |
|--------|---------|-------|
| $q, \mathbf{q}$ | Scalar field / state vector | $q(\xi,\eta,t)$ |
| $\mathbf{q}_b$ | Boundary state node vector | |
| $\mathbf{F}_n^*$ | Interface numerical flux vector | Penalty / Upwind flux |
| $\epsilon_{\text{vol}}$ | Non-commutativity metric error | Metric aliasing |
| $\epsilon_{\text{energy}}$ | Metric-induced volume energy source term | Present in Split3 |
| $\epsilon_{\text{mass}}$ | Mass drift error | Present in Split2 |
| $R_{\text{SBP}}, R_{\text{geo}}, R_{\text{proj}}$ | Microscopic defect terms | SBP, geometry, and normal alignment defects |

---

## ⚠️ IMPORTANT: Forbidden Substitutions

以下替換在本專案中**嚴格禁止**：

```
Do NOT replace:
  J     with det(J)        — J is exclusively the Jacobian determinant scalar symbol J
  D     with ∂              — D refers to discrete SBP derivative operator matrix only
  q     with u              — q is state variable, u represents velocity field
  ξ, η  with x, y           — reference and physical coordinates must never be interchanged
  a_1   with a^1            — covariant and contravariant basis vectors are distinct
```

## Conventions

- Einstein summation convention: **not** used unless explicitly stated
- Index ranges: stated explicitly for each expression
- Discrete vs continuous: always distinguished; never substitute one for the other

