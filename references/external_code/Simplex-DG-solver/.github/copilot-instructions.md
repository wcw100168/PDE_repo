# GitHub Copilot Instructions

## Project Overview
This is the **Simplex-DG Solver**: a production-grade Python implementation of Spectral Element and Discontinuous Galerkin (DG) methods on simplicial meshes. Code quality and mathematical correctness are non-negotiable.

**Status:** Phase 2 Complete (Production-Ready)  
**Last Updated:** 2026-04-08  
**Version:** 0.2.0

---

## Code Quality Standards

### 1. Type Hints & Documentation
- **MANDATORY:** All functions must include complete type hints for parameters and return values
- **MANDATORY:** Every function must have a docstring including:
  - One-line summary
  - Extended description with mathematical context
  - Parameter descriptions with types
  - Return value description
  - Usage example (if applicable)
- **Recommended:** Include reference to published papers or formulas where applicable

**Example Pattern:**
```python
def compute_divergence(F: np.ndarray, rx_ry: tuple[np.ndarray, np.ndarray],
                       sx_sy: tuple[np.ndarray, np.ndarray],
                       D_matrix: np.ndarray) -> np.ndarray:
    """Compute divergence of vector field F on physical mesh.
    
    Computes ∇·F = ∂F_x/∂x + ∂F_y/∂y using the chain rule on affine-mapped
    elements. For a vector field F(r,s) in reference domain, divergence in
    physical coordinates is computed as:
    
    ∇·F = (1/|J|) * [Fᵣ * J_sy - Fₛ * J_ry] + [Fᵣ * J_ry + Fₛ * J_sy] / |J|
    
    Reference: Cockburn et al., "Runge-Kutta Discontinuous Galerkin Methods"
    
    Args:
        F: Vector field, shape (K, 2, N_nodes) where F[:,:,i] = [F_x, F_y] at node i
        rx_ry: Tuple of (∂x/∂r, ∂y/∂r) arrays, each shape (K, N_nodes)
        sx_sy: Tuple of (∂x/∂s, ∂y/∂s) arrays, each shape (K, N_nodes)
        D_matrix: Spectral differentiation matrix, shape (N_nodes, N_nodes)
        
    Returns:
        Divergence field, shape (K, N_nodes)
        
    Example:
        >>> K, N = 16, 10  # 16 elements, N² DOF per element
        >>> F = np.random.randn(K, 2, N)
        >>> rx_ry = (np.ones((K,N)), np.zeros((K,N)))
        >>> sx_sy = (np.zeros((K,N)), np.ones((K,N)))
        >>> D = vandermonde_deriv_matrix(n=3)
        >>> div_F = compute_divergence(F, rx_ry, sx_sy, D)
    """
```

### 2. NumPy Vectorization
- **MANDATORY for production code:** Use NumPy vectorized operations; loops only for setup
- **Why:** O(n) operations scale to 10k+ elements; loops cause science slowdown
- **Example:** `np.mgrid`, `np.meshgrid`, broadcasting preferred over `for k in range(K)`
- **Caching:** Use `@functools.lru_cache()` for repeated geometric computations

---

## CRITICAL: Mesh & Geometry Constraints

### 🔴 CCW Orientation Invariant
**MANDATORY:** All triangular elements in `EToV` must be stored in **Counter-Clockwise (CCW)** order.

- **Definition:** Vertices listed as counter-clockwise in physical (x,y) coordinates
- **Validation:** Signed area must be **strictly positive**: 
$$\text{SignedArea} = \frac{1}{2} \begin{vmatrix} x_1 & y_1 & 1 \\ x_2 & y_2 & 1 \\ x_3 & y_3 & 1 \end{vmatrix} > 0$$
- **Check Function:** `src/core/validation.py::check_element_orientation()`
- **Consequence of Violation:** DG fluxes compute with wrong sign; convergence fails

**Code Pattern for Validation:**
```python
# Always validate mesh orientation before processing
from src.core.validation import check_element_orientation

valid, areas = check_element_orientation(nodes, EToV)
if not valid:
    raise ValueError(f"Found {sum(areas <= 0)} elements with non-CCW orientation")
```

### 🔴 Jacobian Positivity Invariant
**MANDATORY:** All elements must have **strictly positive Jacobian determinant**.

- **Formula:** For affine map from reference $(r,s)$ to physical $(x,y)$:
$$|J| = \begin{vmatrix} \frac{\partial x}{\partial r} & \frac{\partial x}{\partial s} \\ \frac{\partial y}{\partial r} & \frac{\partial y}{\partial s} \end{vmatrix} > 0$$
- **Zero Tolerance:** Degenerate elements (|J| ≤ 0) are forbidden
- **Check Function:** `src/geometry/metrics.py::compute_geometric_factors()`
- **Consequence of Violation:** Quadrature integration breaks; results are meaningless

**Code Pattern:**
```python
from src.geometry.metrics import compute_geometric_factors

v1, v2, v3 = nodes[EToV[k]]  # k-th element vertices
factors = compute_geometric_factors(v1, v2, v3)
if factors['jacobian'] <= 0:
    raise ValueError(f"Element {k} has non-positive Jacobian: {factors['jacobian']}")
```

### 🔴 Reference Element Fixed
**MANDATORY:** The reference element is always the simplex triangle:
$$(r, s) \in \{(r, s) : r \geq 0, s \geq 0, r + s \leq 1\}$$

**Vertex Coordinates (Reference Domain):**
- Vertex 0 (r): $(1, 0)$
- Vertex 1 (s): $(0, 1)$
- Vertex 2 (top): $(0, 0)$

**Affine mappings:** Use `src/geometry/mappings.py::rs_to_xy()` for reference → physical

---

## Boundary Detection: Barycentric Coordinates

**MANDATORY Pattern:** Boundary nodes detected using barycentric coordinates $L_1, L_2, L_3$.

For a point in reference domain $(r, s)$, barycentric coordinates are:
$$L_1 = r, \quad L_2 = s, \quad L_3 = 1 - r - s$$

**Boundary Identification:**
- **Edge 0** (vertices 0→1, $r=0$): $L_3 = 0$
- **Edge 1** (vertices 1→2, $s=0$): $L_2 = 0$
- **Edge 2** (vertices 2→0, $r+s=1$): $L_1 = 0$

**Extraction Function:** `src/reconstruction/boundary.py::build_fmask_table1()`

**Code Pattern:**
```python
from src.reconstruction.boundary import build_fmask_table1

fmask, fmask_map = build_fmask_table1(r, s, tol=1e-10)
# fmask[j] = list of nodes on face j (j ∈ {0,1,2})
# fmask_map[k]=boundary_indices for element k
```

---

## Module Organization & Boundaries

### Stable Core Modules (Production-Ready)
- **`src/core/`**: Connectivity, data structures, generators
  - `connectivity.py::build_connectivity()` — EToE/EToF matrices
  - `generators.py::get_reference_data()` — Reference node/weight data
  - `validation.py` — Mesh quality checks

- **`src/geometry/`**: Transformations and metrics
  - `mappings.py::rs_to_xy()`, `xy_to_rs()` — Coordinate transforms
  - `metrics.py::compute_geometric_factors()` — Jacobian & shape factors

- **`src/reconstruction/`**: Operators and boundary extraction
  - `boundary.py::build_fmask_table1()` — Boundary node extraction
  - `operators.py::compute_divergence()`, gradient, Laplacian
  - `modal_reconstruct.py` — Modal expansion & reconstruction

### Experimental Code
- **`notebooks/experimental/`**: Research notebooks ONLY
  - Approved for prototyping; NOT for production use
  - Must be migrated to `src/` before use in pipelines

### Never mix:
- ❌ Duplicate function definitions (use `src/` modules)
- ❌ Unvalidated mesh objects (always run connectivity + orientation checks)
- ❌ Hardcoded reference element assumptions (use `src/bases/simplex_2d.py`)

---

## Development Workflow

### When Adding New Functions
1. **Extract to module first** (don't leave in notebook)
   - Target module per `API_FUNCTION_REGISTRY.md`
2. **Add comprehensive tests** in `src/tests/`
   - Unit tests for correctness
   - Convergence tests for operators (h-convergence with n_div ∈ {2,3,4})
3. **Update imports** in notebook to use `from src.module import func`
4. **Document in registry** — update `API_FUNCTION_REGISTRY.md`

### Validation Checklist
Before committing new code:
- [ ] **Mesh Validation**: `test_validation.py` passes all checks
- [ ] **Import Path Correct**: `from src.module import func` works locally
- [ ] **Type Hints**: All functions have full type annotations
- [ ] **Convergence Test**: h-convergence suite (n_div=2,3,4) produces expected rates
- [ ] **Documentation**: Mathematical formulas include paper references
- [ ] **Docstring Example**: Every function has runnable example code

---

## Performance Expectations

### Scaling Laws
- **Mesh operations:** O(n_elements) — linear in mesh size
- **Operator application:** O(n_elements × poly_order²) — spectral methods
- **Boundary detection:** O(n_elements × n_nodes) — vectorized, GPU-friendly

### Caching Strategy
```python
import functools

@functools.lru_cache(maxsize=128)
def compute_geometric_factors(v1_tuple, v2_tuple, v3_tuple):
    """Cached Jacobian computation for repeated element calls."""
    v1, v2, v3 = np.array(v1_tuple), np.array(v2_tuple), np.array(v3_tuple)
    # ... computation
```

### Batch Processing
Always offer both single-element and batch versions:
```python
# Single: compute_geometric_factors(v1, v2, v3) → {jacobian, rx, ry, ...}
# Batch:  compute_geometric_factors_batch(vertices_K3N2) → ({...}, {array_KN}, ...)
```

---

## Mathematical Notation Standards

When writing code comments and docstrings:
- Use $\LaTeX$ for inline math: $\partial u / \partial x$
- Use display math for complex formulas
- Always cite papers: "Reference: Author et al., 'Title' (Year)"
- Include both mathematical symbol and NumPy name:
  - "Jacobian determinant $|J|$ stored in `jacobian`"
  - "Local derivative $\partial/\partial r$ via `D_matrix`"

---

## Prohibited Patterns

❌ **Never do this:**
```python
# WRONG: Clockwise element ordering
EToV_wrong = np.array([[0, 2, 1], ...])  # CW → flux errors

# WRONG: Loop-based when vectorization exists
for k in range(K):
    div_F[k] = compute_divergence_single(F[k])  # Too slow, use batched

# WRONG: Unvalidated mesh
mesh = load_mesh(filename)
divergence = compute_divergence(F, mesh)  # No validation!

# WRONG: Duplicate function definitions
def generate_mesh():  # Already in src/bases/simplex_2d.py
    ...

# WRONG: Hardcoded reference element
nodes = np.array([[1,0], [0,1], [0,0]])  # Use get_reference_data()
```

**Correct alternatives:**
```python
# RIGHT: CCW validation first
valid, areas = check_element_orientation(nodes, EToV)
assert valid, "Non-CCW elements detected"

# RIGHT: Use batch operations
div_F = compute_divergence(F_batch, rx_ry_batch, sx_sy_batch, D_matrix)

# RIGHT: Import from src/
from src.bases.simplex_2d import get_reference_data
nodes, _ = get_reference_data(n=3)

# RIGHT: Validate mesh before use
from src.core.validation import validate_connectivity
EToE, EToF = validate_connectivity(build_connectivity(EToV))
```

---

## Code Review Checklist for AI Assistants

When suggesting code changes, ensure:
- [ ] All functions have type hints (`def func(x: np.ndarray) -> dict[str, np.ndarray]`)
- [ ] All parameters described in docstring with units/dimensions
- [ ] Mathematical formulas present (with paper references)
- [ ] Mesh validated before processing (CCW + positive Jacobian)
- [ ] Vectorization used (no performance-killing loops)
- [ ] Test cases demonstrate correctness
- [ ] Convergence rates verified (if applicable)

---

## API Function Reference

### Critical Functions (Must Know)

| ID | Function | Module | Signature | Priority |
|----|----|--------|-----------|----------|
| **F1** | `generate_subdivided_triangle()` | `src/bases/simplex_2d` | `(n_div: int) → (nodes, triangles)` | HIGH |
| **F2** | `compute_geometric_factors()` | `src/geometry/metrics` | `(v1, v2, v3) → Dict[str, float]` | **CRITICAL** |
| **F3** | `rs_to_xy()` / `xy_to_rs()` | `src/geometry/mappings` | `(r,s,v1,v2,v3) → (x,y)` | HIGH |
| **F4** | `compute_divergence()` | `src/reconstruction/operators` | `(F, rx_ry, sx_sy, D) → div_F` | HIGH |
| **F5** | `build_fmask_table1()` | `src/reconstruction/boundary` | `(r, s, tol) → fmask` | **CRITICAL** |
| **F6** | `global_face_extraction()` | `src/reconstruction/boundary` | Vectorized boundary extraction pattern | HIGH |
| **F7** | `exponent_pairs()` | `src/bases/simplex_2d` | `(total_degree) → [(i,j), ...]` | MEDIUM |
| **F8-9** | `F_vector_gaussian()` / `divF_exact_gaussian()` | `src/tests/test_data` | Test field generators | MEDIUM |

### Already-Extracted Functions (Use Directly)
- `get_reference_data(n)` — Reference quadrature nodes/weights
- `build_connectivity(EToV)` — Element connectivity matrices (EToE, EToF)
- `validate_connectivity()` — Connectivity verification
- `check_element_orientation()` — CCW validation
- `check_geometric_centroids()` — Centroid verification

---

## Connectivity Algorithm: EToE and EToF

### Purpose
Build Element-to-Element (EToE) and Element-to-Face (EToF) connectivity matrices for domain decomposition and inter-element communication.

**Function Location:** `src/core/connectivity::build_connectivity(EToV)`

### Local Face Convention (CCW Order)

| Face ID | Local Vertices | Global Edge | Boundary Condition |
|---------|----------------|-------------|-------------------|
| **0** | [0, 1] | $(v_0, v_1)$ | Edge 0→1 |
| **1** | [1, 2] | $(v_1, v_2)$ | Edge 1→2 |
| **2** | [2, 0] | $(v_2, v_0)$ | Edge 2→0 |

All face indices assume **Counter-Clockwise** element orientation.

### Algorithm: Vertex Pairing Hash Method

**Step 1: Initialize Boundary Defaults**
```python
EToE[k, :] = k           # Each element is its own neighbor (boundary)
EToF[k, :] = [0, 1, 2]   # Each face maps to itself (boundary)
```

**Step 2: Build Vertex Pairing Hash Table**
```python
For each element k and local face f in {0, 1, 2}:
    Extract local vertices: v1_local, v2_local (from face convention)
    Map to global: v1_global = EToV[k, v1_local], v2_global = EToV[k, v2_local]
    Create canonical edge key: edge_key = tuple(sorted([v1_global, v2_global]))
    Store: edge_dict[edge_key].append((k, f))
```

**Step 3: Process Interior Edges**
```python
For each edge_key in edge_dict:
    If len(edge_dict[edge_key]) == 2:  # Interior edge
        (k1, f1), (k2, f2) = element-face pairs
        Set: EToE[k1, f1] = k2,  EToE[k2, f2] = k1
        Set: EToF[k1, f1] = f2,  EToF[k2, f2] = f1
    Elif len(...) == 1:  # Boundary edge
        No action (boundary defaults already set)
```

### Complexity
- **Time:** $O(K)$ where $K =$ number of elements
- **Space:** $O(K)$ for hash table
- **Practical:** ~0.5ms for 10k elements

### Verification Pattern
```python
from src.core.connectivity import build_connectivity, validate_connectivity

EToE, EToF = build_connectivity(EToV)

# Optional: Verify correctness
validate_connectivity(EToV, EToE, EToF)
```

---

## Boundary Extraction: Fmask via Barycentric Coordinates

### Purpose
Extract node indices on each element face (boundary edges). Essential for flux computation in DG methods.

**Function Location:** `src/reconstruction/boundary::build_fmask_table1(r, s, tol=1e-10)`

### Barycentric Coordinate Basis

For reference element $(r, s) \in [0,1]^2$:
$$L_1 = r, \quad L_2 = s, \quad L_3 = 1 - r - s$$

### Face Node Detection
| Face | Detection Condition | Meaning |
|------|-------------------|---------|
| **0** | $L_3 < tol$ (i.e., $r + s \approx 1$) | Edge opposite vertex 2 |
| **1** | $L_2 < tol$ (i.e., $s \approx 0$) | Edge opposite vertex 1 |
| **2** | $L_1 < tol$ (i.e., $r \approx 0$) | Edge opposite vertex 0 |

### Output
Returns tuple: `(fmask, fmask_map)` where:
- `fmask[j]` = array of global node indices on face $j$ (length $N_{fp}$)
- `fmask_map` = mapping for per-element queries

### Robustness
- **Tolerance:** Default $10^{-10}$ accommodates floating-point errors
- **No Duplicates:** Barycentric detection ensures each node appears exactly once
- **Works with Non-Uniform Grids:** No explicit enumeration needed

### Example Usage
```python
from src.reconstruction.boundary import build_fmask_table1

r, s = nodes[:, 0], nodes[:, 1]
fmask, _ = build_fmask_table1(r, s, tol=1e-10)

boundary_nodes_face_0 = fmask[0]  # Nodes where r+s≈1
boundary_nodes_face_1 = fmask[1]  # Nodes where s≈0
boundary_nodes_face_2 = fmask[2]  # Nodes where r≈0
```

---

## Mesh Validation: CCW Orientation & Jacobian Positivity

### CCW Orientation Verification

All triangular elements must be stored in **Counter-Clockwise** order.

**Formula:** Signed area (computed via cross product)
$$\text{SignedArea} = \frac{1}{2} \begin{vmatrix} x_1 & y_1 & 1 \\ x_2 & y_2 & 1 \\ x_3 & y_3 & 1 \end{vmatrix} = \frac{1}{2}[(x_2-x_1)(y_3-y_1) - (x_3-x_1)(y_2-y_1)]$$

**Valid Range:** $\text{SignedArea} > 0$ (strictly positive)

**Validation Function:**
```python
from src.core.validation import check_element_orientation

valid, signed_areas = check_element_orientation(nodes, EToV)
if not valid:
    print(f"❌ Found {sum(signed_areas <= 0)} non-CCW elements")
```

### Jacobian Positivity Verification

Affine mapping from reference $(r, s)$ to physical $(x, y)$:
$$|J| = \frac{\partial x}{\partial r} \frac{\partial y}{\partial s} - \frac{\partial x}{\partial s} \frac{\partial y}{\partial r}$$

**Valid Range:** $|J| > 0$ (strictly positive, no degenerate elements)

**Validation Function:**
```python
from src.geometry.metrics import compute_geometric_factors

v1, v2, v3 = nodes[EToV[k]]
factors = compute_geometric_factors(v1, v2, v3)
jacobian = factors['jacobian']
assert jacobian > 0, f"Degenerate element {k}: J={jacobian}"
```

### Verified Mesh Properties
Example: Subdivided reference triangle with $n_{\text{div}} \in \{2,3,4\}$

| $n_{\text{div}}$ | **Elements** | **CCW** | **J > 0** | **Variance** |
|---|---|---|---|---|
| 2 | 4 | 4 ✓ | 4 ✓ | 0.0 |
| 3 | 9 | 9 ✓ | 9 ✓ | ~0.0 |
| 4 | 16 | 16 ✓ | 16 ✓ | 0.0 |

---

## Module Dependency Graph

```
src/
├── geometry/metrics.py          [compute_geometric_factors, MetricTensor]
│   ↑ required by: reconstruction.operators, reconstruction.boundary
│
├── geometry/mappings.py         [rs_to_xy, xy_to_rs, AffineMap]
│   ↑ required by: geometry.metrics, reconstruction.operators
│
├── reconstruction/operators.py  [compute_divergence, compute_gradient]
│   ↑ depends on: geometry.metrics, differentiation matrices
│
├── reconstruction/boundary.py   [build_fmask_table1, global_face_extraction]
│   ↑ depends on: numpy (no circular dependencies)
│
├── bases/simplex_2d.py          [generate_subdivided_triangle, exponent_pairs]
│   ↑ dependencies: numpy only
│
├── core/connectivity.py         [build_connectivity, validate_connectivity]
│   ↑ dependencies: numpy only
│
└── tests/test_data.py           [F_vector_gaussian, divF_exact_gaussian]
    ↑ provides: test fields for convergence validation
```

---

## Notebook Update Workflow: Extracting Code

### Pattern: Before → After Extraction

**Before (Duplicated in each notebook):**
```python
# f_mask.ipynb, Divergence.ipynb, Jacobian.ipynb
def compute_geometric_factors(v1, v2, v3):
    # ... 30+ lines ...
```

**After (Single import):**
```python
# Add at top of notebook
from src.geometry.metrics import compute_geometric_factors
```

### Function Extraction Steps

1. **Identify** duplicate in [API_FUNCTION_REGISTRY.md](API_FUNCTION_REGISTRY.md)
2. **Delete** function definition from notebook
3. **Add** import statement from proposed target module
4. **Verify** import works: `from src.module import func`
5. **Test** function call (names unchanged, behavior identical)

### Common Extraction Targets
- **Mesh generation:** `src/bases/simplex_2d::generate_subdivided_triangle()`
- **Jacobian:** `src/geometry/metrics::compute_geometric_factors()`
- **Transforms:** `src/geometry/mappings::rs_to_xy()`, `xy_to_rs()`
- **Operators:** `src/reconstruction/operators::compute_divergence()`
- **Boundaries:** `src/reconstruction/boundary::build_fmask_table1()`

---

## References & Documentation

- **Project Overview:** [README.md](../../README.md)
- **AI Assistant Guide:** [README_AI.md](../../README_AI.md)
- **Test Validation:** [test_validation.py](../../test_validation.py)

---

**Last Updated:** 2026-04-08  
**Maintained By:** Senior Developer & Project Auditor  
**Next Review:** As new functions are extracted
