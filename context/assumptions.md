# Global Assumptions

## Active Assumptions

### A001 — Domain & Mesh Topology
The physical domain is the 2-sphere $\mathbb{S}^2 \subset \mathbb{R}^3$, discretized using a Subdivided Octahedral Spherical Mesh partitioned into non-overlapping conformal triangular elements $K_e$.

Used by: T2.1, T3.2, T3.3

---

### A002 — Radial Geometry Mapping
Mapping from reference triangle $(\xi,\eta) \in T$ to physical spherical surface $\mathbf{x}_s \in \mathbb{S}^2$ is constructed via flat octahedral transformation $\mathbf{x}_{\text{flat}} = \sum L_i \mathbf{v}_i$ followed by radial projection $\mathbf{x}_s = \mathbf{x}_{\text{flat}} / \|\mathbf{x}_{\text{flat}}\|$. The Jacobian determinant satisfies $\mathcal{J} > 0$ everywhere.

Used by: T2.1, Split2, Split3

---

### A003 — Multidimensional SBP Operator Property
The discrete derivative operators $D_\xi, D_\eta$ satisfy the SBP property on reference triangle $T$ with volume weight diagonal matrix $W > 0$ and boundary matrices $B_\xi, B_\eta$:

$$
W D_\xi + D_\xi^T W = B_\xi, \quad W D_\eta + D_\eta^T W = B_\eta
$$

Used by: T2.1, T3.2, T3.3

---

### A004 — Modal Cholesky Orthogonalization
The Vandermonde matrix $V$ is orthogonalized via Cholesky decomposition of the reference mass matrix $\hat{M} = V_{\text{raw}}^T W V_{\text{raw}} = L L^T$, resulting in $V = V_{\text{raw}}(L^T)^{-1}$ satisfying $V^T W V = I$ to machine precision ($\sim 10^{-16}$).

Used by: T2.1

---

### A005 — Peirce Subspace Decomposition
The operator matrix space $\mathbb{R}^{M \times M}$ is decomposed into orthogonal subspaces $S_{PP} \oplus S_{PQ} \oplus S_{QP} \oplus S_{QQ}$ via polynomial projection $P = V(V^T W V)^{-1} V^T W$ and its complement $Q = I - P$.

Used by: T3.2, T3.3

