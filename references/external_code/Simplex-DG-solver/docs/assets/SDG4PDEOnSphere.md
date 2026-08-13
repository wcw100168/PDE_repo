# Simplex quadrature nodal DG method for wave equations on sphere

**April 14, 2026**

---

## 1. $2n-1$ and $2n$ Quadrature points in 2-simplex

### Barycentric Coordinates
Consider the two-dimensional plane with coordinates $\mathbf{\xi} = (\xi, \eta)$. Let $T$ be the triangle with vertices
$$
\mathbf{\xi}_{v_1} =(0,1), \quad \mathbf{\xi}_{v_2} =(0,0), \quad \mathbf{\xi}_{v_3} =(1,0)
$$
Any point within the triangle can be expressed as
$$
(\xi,\eta) = \mathbf{\xi} = \sum_{i=0}^3 b_i \mathbf{\xi}_{v_i}, \quad \sum_{i=1}^3 b_i = 1, \quad b_i\ge 0
$$
where $b_i$ for $i=1,2,3$ are known as the barycentric coordinates. Numerical integration is often encounter in computations. To conduct numerical integration over $T$, numerical quadrature points and weights are introduced.

Below we quote the barycentric coordinates of the quadrature points and the associated weights for conducting integration of polynomials of degree at most degree $2n-1$ and $2n$, in Table 1 and Table 2.

**Table 1: Barycentric coordinates of quadrature integration points for integrating polynomials of degree at most $2n-1$ in a triangle.** $w^s_i$ are surface integration associated quadrature weights. $w^e_i$ are line integration quadrature weights associated with edge points.

| $n$ | Sym | $b_1$ | $b_2$ | $w^s_i$ | $w^e_i$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | $S_6$ | 0.2113248654051871 | 0.0000000000000000 | 0.16666666666666667 | 0.5000000000000000 |
| 2 | $S_6$ | 0.1127016653792583 | 0.0000000000000000 | 0.04166666666666666 | 0.2777777777777777 |
| 2 | $S_3$ | 0.5000000000000000 | 0.0000000000000000 | 0.09999999999999999 | 0.4444444444444444 |
| 2 | $S_1$ | 0.3333333333333333 | 0.3333333333333333 | 0.45000000000000000 | -- |
| 3 | $S_6$ | 0.06943184420297367| 0.0000000000000000 | 0.01509901487256561 | 0.1739274225687269 |
| 3 | $S_6$ | 0.3300094782075718 | 0.0000000000000000 | 0.04045654068298990 | 0.3260725774312731 |
| 3 | $S_6$ | 0.5841571139756568 | 0.1870738791912763 | 0.11111111111111111 | -- |
| 4 | $S_6$ | 0.04691007703066797| 0.0000000000000000 | 0.006601315081001592| 0.1184634425280944 |
| 4 | $S_6$ | 0.2307653449471584 | 0.0000000000000000 | 0.02053045968042892 | 0.2393143352496833 |
| 4 | $S_3$ | 0.5000000000000000 | 0.0000000000000000 | 0.01853708483394990 | 0.2844444444444446 |
| 4 | $S_3$ | 0.1394337314154536 | 0.1394337314154536 | 0.10542932962084440 | -- |
| 4 | $S_3$ | 0.4384239524408185 | 0.4384239524408185 | 0.12473673228977350 | -- |
| 4 | $S_1$ | 0.3333333333333333 | 0.3333333333333333 | 0.09109991119771331 | -- |

**Table 2: Barycentric coordinates of quadrature integration points for integrating polynomials of degree at most $2n$ in a triangle.** $w^s_i$ are surface integration associated quadrature weights.

| $n$ | Sym | $b_1$ | $b_2$ | $w^s_i$ |
| :--- | :--- | :--- | :--- | :--- |
| 1 | $S_3$ | 0.1666666666666666 | 0.1666666666666666 | 0.3333333333333333 |
| 2 | $S_3$ | 0.09157621350977067| 0.09157621350977067| 0.1099517436553218 |
| 2 | $S_3$ | 0.4459484909159648 | 0.4459484909159648 | 0.2233815896780115 |
| 3 | $S_3$ | 0.219429982549783  | 0.219429982549783  | 0.1713331241529809 |
| 3 | $S_3$ | 0.480137964112215  | 0.480137964112215  | 0.08073108959303095|
| 3 | $S_6$ | 0.1416190159239682 | 0.0193717243612408 | 0.04063455979366068|
| 4 | $S_6$ | 0.7284923929554044 | 0.2631128296346379 | 0.02723031417443505|
| 4 | $S_3$ | 0.4592925882927232 | 0.4592925882927232 | 0.09509163426728455|
| 4 | $S_3$ | 0.1705693077517602 | 0.1705693077517602 | 0.1032173705347182 |
| 4 | $S_3$ | 0.05054722831703096| 0.05054722831703096| 0.03245849762319804|
| 4 | $S_1$ | 0.3333333333333333 | 0.3333333333333333 | 0.1443156076777874 |

Based on these points we evaluate surface integration over a triangle, with $|T|$ being the area of the triangle, by a discrete summation as follows,
$$
\int_T f(\mathbf{\xi}) \, d\mathbf{\xi} = |T|\sum_{i=1}^N f(\mathbf{\xi}_i) w^s_i,
$$
provided that $f$ is a polynomial of degree at most $2n-1$ and $2n$. For evaluating line integrals on an edge of a triangle, we adopt the following quadrature formula:
$$
\int_{\partial T^\gamma} f(\mathbf{\xi}) \, d\mathbf{\xi} = |\partial T^{\gamma}| \sum_{i=1}^{n+1} f(\mathbf{\xi}^{\gamma}_i) w^e_i
$$
where $|\partial T^{\gamma}|$ denotes the length of the edge $\gamma$, $\mathbf{\xi}^\gamma_i$ are the quadrature points on the edge, and $w^e_i$ are the associated quadrature weights.

---

## 2. Orthogonal Polynomials on Simplex
We denote the triangle with vertices $(-1,1)$, $(-1,-1)$, and $(1,-1)$ by
$$
\mathsf{T}^2 = \{(r,s)\,|\, r,s \ge -1, \, r+s \le 1 \}.
$$
Let $n\in\{1,2,3,4\}$. On $\mathsf{T}^2$, let us introduce the orthogonal polynomials $\phi_{i,j}$ defined by
$$
\phi_m(r,s) = \Phi_{ij}(r,s) = \sqrt{2} P_i(a) P_j^{(2i+1,j)}(b) (1-b)^j, \quad 0 \le i,j \le n, \quad i+j\le n,
$$
$$
m = j+(n+1)i + 1- \frac{i}{2}(i-1), \quad a = 2 \cdot \frac{1+r}{1-s} - 1, \quad b = s
$$
where $P_i$ is the Legendre polynomial of degree $i$ and $P_j^{(2i+1,j)}$ is the Jabobi polynomials of degree $j$. Applying these basis functions we span the function space as
$$
\mathcal{V} = \text{span}\{ \Phi_{ij}(r,s)  \, | \, 0 \le i,j \le n, \quad i+j \le n \}
$$

Thus, for $n=1$, we have the basis functions:
- $\frac{\phi_1(r,s)}{\sqrt{2}} = \frac{\Phi_{00}}{\sqrt{2}} = P_0\left( \frac{2(1+r)}{1-s} \right) P^{(1,0)}_0(s) (1-s)^0 = 1$
- $\frac{\phi_2(r,s)}{\sqrt{2}} = \frac{\Phi_{01}}{\sqrt{2}} = P_0\left( \frac{2(1+r)}{1-s} \right) P^{(1,1)}_1(s) (1-s)^1 = 2-2(1-s)^2$
- $\frac{\phi_3(r,s)}{\sqrt{2}} = \frac{\Phi_{10}}{\sqrt{2}} = P_1\left( \frac{2(1+r)}{1-s} \right) P^{(3,0)}_0(s) (1-s)^0 = 2 \frac{1+r}{1-s}$

---

## 3. Approximation, Integration, and Differentiation

### Modal and nodal approximation
Based on these grid points, we may seek numerical approximations to a function $u$ defined on $T$.
Consider a function space $\mathcal{V}$ spanned by a set of linearly independent basis functions $\phi_k(\xi,\eta)$ for $i=1,2,\cdots,n$, i.e.,
$$
\mathcal{V} = \text{span} \{\phi_i(\xi,\eta)| i=1,2,...,n \}.
$$
Thus, a function $u(\xi,\eta) \in \mathcal{V}$ can be approximated as follows,
$$
u(\xi,\eta) = \sum_{i=1}^N a_i \phi_i(\xi,\eta),
$$
where $a_i$ are coefficients. Applying the quadrature points $(\xi_m,\eta_m)$ for $m=1,2,\cdots,M$ and denoting $u(\xi_m,\eta_m)=u_m$ for simplicity, we have
$$
\mathbf{u} = \mathbf{V} \mathbf{a},
$$
where
$$
\mathbf{u} = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_M \end{bmatrix}, \quad
\mathbf{V} = \begin{bmatrix} \phi_{1}(\xi_1,\eta_1) & \phi_{2}(\xi_1,\eta_1) & \cdots & \phi_{N}(\xi_1,\eta_1) \\ \phi_{1}(\xi_2,\eta_2) & \phi_{2}(\xi_2,\eta_2) & \cdots & \phi_{N}(\xi_2,\eta_2) \\ \vdots & \vdots & \ddots & \vdots \\ \phi_{1}(\xi_M,\eta_M) & \phi_{2}(\xi_M,\eta_M) & \cdots & \phi_{N}(\xi_M,\eta_M) \end{bmatrix}, \quad
\mathbf{a} = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_N \end{bmatrix}.
$$

Let $\mathbf{W} = \text{diag}(w^s_1,w^s_2,\cdots,w^s_M)$. Define the mass matrix over the triangle $T$ as
$$
\mathbf{M} = |T| \mathbf{V}^T \mathbf{W} \mathbf{V}.
$$
Since $\mathbf{W}$ is diagonal, we have $\mathbf{M}^T = \mathbf{M}$. Furthermore, for $\mathbf{a}$ being a non-zero vector and $\mathbf{W}$ being positive definite, we have
$$
\mathbf{a}^T \mathbf{M} \mathbf{a} = |T| (\mathbf{Va})^T \mathbf{W} (\mathbf{Va}) >0,
$$
indicating that $\mathbf{M}$ is not only symmetric but also positive definite. Thus, its inverse exists, denoted by $\mathbf{M}^{-1}$.
Then, multiplying $|T| \mathbf{V}^T \mathbf{W}$ to the nodal form, we have the vector $\mathbf{a}$ as follows:
$$
\mathbf{a} = |T| \mathbf{M}^{-1} \mathbf{V}^T \mathbf{W} \mathbf{u}
$$

### Numerical differentiation
To perform numerical differentiation on the vector $\mathbf{u}$, we construct the differentiation matrices for $\xi$ and $\eta$ directions.
Differentiating the modal form with respect to $\xi$:
$$
\partial_\xi u(\xi,\eta) = \sum_{i=1}^N a_i \partial_\xi \phi(\xi,\eta) \implies \mathbf{u}_\xi = \mathbf{V}_\xi \mathbf{a}
$$
Applying the relationship for $\mathbf{a}$:
$$
\mathbf{u}_\xi = \mathbf{V}_\xi \mathbf{a} = |T| \mathbf{V}_\xi \mathbf{M}^{-1} \mathbf{V}^T \mathbf{W} \mathbf{u} = \mathbf{D}_\xi \mathbf{u}
$$
Likewise, for the $\eta$ direction:
$$
\mathbf{D}_\eta = |T| \mathbf{V}_\eta \mathbf{M}^{-1} \mathbf{V}^T \mathbf{W}, \quad \mathbf{u}_\eta = \mathbf{D}_\eta \mathbf{u}.
$$

### Numerical Integration
From the divergence theorem:
$$
\oint_{\partial \mathcal{T}} \mathbf{n} \cdot (q \mathbf{F}) \, d \mathbf{\xi} = \int_{\mathcal{T}} q \nabla \cdot \mathbf{F} \, d\mathbf{\xi} + \int_{\mathcal{T}} \mathbf{F} \cdot \nabla q  \, d \mathbf{\xi}
$$
The discrete version is established as:
$$
(\mathbf{q}^e)^T \mathbf{W}^e \mathbf{f}_n^e = |T| \mathbf{1}^T \mathbf{W} \left( \mathbf{D}_\xi (\mathbf{q} \odot \mathbf{u}) + \mathbf{D}_\eta (\mathbf{q} \odot \mathbf{v}) \right)
$$

$$
= |T| \mathbf{q}^T \mathbf{W} (\mathbf{D}_\xi \mathbf{u} + \mathbf{D}_\eta \mathbf{v}) + |T| ( \mathbf{u}^T \mathbf{W} \mathbf{D}_\xi \mathbf{q} + \mathbf{v}^T \mathbf{W} \mathbf{D}_\eta \mathbf{q} )
$$
where $\mathbf{W}^e$ is the quadrature weight matrix for the three edges:
$$
\mathbf{W}^e = \begin{bmatrix} |\partial T^1| & 0 & 0 \\ 0 & |\partial T^2| & 0 \\ 0 & 0 & |\partial T^3| \end{bmatrix} \otimes \text{diag}(w_1^e,w_2^e,\cdots, w_{n+1}^e)
$$

---

## 4. Well-posed analysis for model wave equations
Consider the model wave equation:
$$
\partial_t q(\mathbf{\xi},t) + \nabla \cdot (\mathbf{V} q(\mathbf{\xi},t)) = 0, \quad \mathbf{\xi} \in \mathcal{T}
$$
Energy estimate leads to:
$$
\int_{\mathcal{T}} q^2(\mathbf{\xi},t) \, d \mathbf{\xi} \le e^{\alpha t} \left( \int_{\mathcal{T}} q_0^2(\mathbf{\xi}) \, d \mathbf{\xi} + \int_0^t \int_{\gamma} (-n \cdot \mathbf{V}) g^2(t') \, d\mathbf{\xi}  \, dt' \right)
$$

---

## 5. DG scheme for advection equation
We consider the following discretization:
$$
\partial_t \mathbf{q} = - \frac{1}{2} \left( \mathbf{D}_{\xi} (\mathbf{u} \odot \mathbf{q}) + \mathbf{D}_{\eta} (\mathbf{v} \odot \mathbf{q}) \right) - \frac{1}{2} (\mathbf{u} \odot \mathbf{D}_\xi \mathbf{q} + \mathbf{v} \odot \mathbf{D}_\eta \mathbf{q} ) - \frac{1}{2} (\mathbf{D}_\xi \mathbf{u} + \mathbf{D}_\eta \mathbf{v}) \odot  \mathbf{q} + (|T|)^{-1} \mathbf{W}^{-1} \mathbf{E}^T \mathbf{W}^e \mathbf{p}
$$
where $\mathbf{E} = [\mathbf{I} | \mathbf{0}]$ and $\mathbf{p}$ is the penalized boundary condition vector.

#### Upwind flux ($\tau=0$) and $q^*=g$
The boundary term $\rho$ satisfies:
$$
\mathbf{\rho} = \left\{ \begin{array}{ll} (-\mathbf{n} \cdot \mathbf{V}) q^2 \le 0 & \text{for out flowing } (\mathbf{n} \cdot \mathbf{V} > 0) \\ (\mathbf{n} \cdot \mathbf{V}) (q-g)^2 -\mathbf{n}\cdot \mathbf{V} g^2 \le (-\mathbf{n} \cdot \mathbf{V})g^2 & \text{for in flowing } (\mathbf{n} \cdot \mathbf{V} < 0) \end{array} \right.
$$
**Case 1: out flow: $\mathbf{n} \cdot \mathbf{V} \ge 0$.**
$$
q f^* = \frac{ \mathbf{n} \cdot \mathbf{V} q^2 + \mathbf{n} \cdot \mathbf{V} qg }{2} + \frac{\mathbf{n} \cdot \mathbf{V}}{2} (q^2-qg) = \mathbf{n} \cdot \mathbf{V} q^2, \quad 2 q(f-f^*) = 0
$$
**Case 2: inflow: $\mathbf{n} \cdot \mathbf{V} < 0$.**
$$
q f^* = \frac{ \mathbf{n} \cdot \mathbf{V} q^2 + \mathbf{n} \cdot \mathbf{V} qg }{2} - \frac{\mathbf{n} \cdot \mathbf{V}}{2} (q^2-qg) = \mathbf{n} \cdot \mathbf{V} qg, \quad 2\mathbf{q}(f-f^*) = 2( \mathbf{n} \cdot \mathbf{V} ) (q^2 - qg)
$$
We require that:
$$
2(\mathbf{q}^e)^T \mathbf{W}^e \mathbf{p} = \sum_{i=1}^{n+1} 2 |\partial T^1| q_i w^e_i p_i + \sum_{i=1}^{n+1} 2|\partial T^2| q_{n+1+i} w^e_i p_{n+1+i} + \sum_{i=1}^{n+1} 2|\partial T^2| q_{2(n+1)+i} w^e_i p_{2(n+1)+i}
$$
$$
= \oint_{\partial T} 2 q (f-f^*) \, d \mathbf{x}
$$

Thus,
$$
p_i = (f_i - f_i^*)
$$
where $f_i=f(\mathbf{\xi}_i)$ and $f_i^*(\mathbf{\xi}_i)$.

---

# Octahedron Sphere
