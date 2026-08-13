"""
2D simplex orthonormal basis functions and derivatives.

Implements evaluation and differentiation of the Dubiner basis on the reference
triangle [-1, 1] × [0, 1] with collapsed coordinate mapping.
"""

import numpy as np
from ..numerics import jacobi_p, grad_jacobi_p


def evaluate_simplex_basis_2d(
    a: np.ndarray,
    b: np.ndarray,
    i: int,
    j: int
) -> np.ndarray:
    """
    Evaluate 2D simplex orthonormal basis of order (i, j) at collapsed coordinates (a, b).
    
    The 2D orthonormal basis is constructed from Jacobi polynomials as:
        psi_{i,j}(a, b) = sqrt(2) * P_i^{(0,0)}(a) * P_j^{(2i+1, 0)}(b) * (1-b)^i
    
    Parameters
    ----------
    a : np.ndarray
        First collapsed coordinate (array or scalar)
    b : np.ndarray
        Second collapsed coordinate (array or scalar)
    i : int
        First polynomial degree (i >= 0)
    j : int
        Second polynomial degree (j >= 0)
        
    Returns
    -------
    np.ndarray
        Basis function values, with shape matching input coordinates
    """
    # Evaluate Jacobi polynomials
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2 * i + 1, 0, j)

    # Combine into simplex basis
    return np.sqrt(2.0) * h1 * h2 * (1.0 - b) ** i


def dubiner_basis_derivative(
    a: np.ndarray,
    b: np.ndarray,
    i: int,
    j: int
) -> tuple:
    """
    Evaluate partial derivatives (dpsi_da, dpsi_db) of the 2D simplex basis in collapsed coordinates.
    
    Uses the chain rule and product rule with careful handling of the (1-b)^i singular term.
    
    Parameters
    ----------
    a : np.ndarray
        First collapsed coordinate
    b : np.ndarray
        Second collapsed coordinate
    i : int
        First polynomial degree
    j : int
        Second polynomial degree
        
    Returns
    -------
    tuple
        (dpsi_da, dpsi_db) partial derivatives in collapsed coordinate space
    """
    # Evaluate 1D polynomials and their derivatives
    p_i = jacobi_p(a, 0.0, 0.0, i)
    dp_i_da = grad_jacobi_p(a, 0.0, 0.0, i)
    
    p_j = jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    dp_j_db = grad_jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    
    sqrt2 = np.sqrt(2.0)
    
    # Initialize derivatives
    dpsi_dr = np.zeros_like(a)
    dpsi_ds = np.zeros_like(a)
    
    # Applying Chain Rule and canceling out (1-b) denominator analytically
    if i == 0:
        # If i=0, dp_i_da = 0, so the 'r' derivative is identically zero.
        dpsi_dr = np.zeros_like(a)
        dpsi_ds = sqrt2 * p_i * dp_j_db
    else:
        # Partial derivative w.r.t r
        dpsi_dr = sqrt2 * dp_i_da * p_j * 2.0 * ((1.0 - b) ** (i - 1))
        
        # Partial derivative w.r.t s
        # Splitting the product rule application for clarity
        term1 = dp_i_da * p_j * (1.0 + a) * ((1.0 - b) ** (i - 1))
        term2 = p_i * dp_j_db * ((1.0 - b) ** i)
        term3 = -i * p_i * p_j * ((1.0 - b) ** (i - 1))
        
        dpsi_ds = sqrt2 * (term1 + term2 + term3)
        
    return dpsi_dr, dpsi_ds


def dubiner_basis_index_to_order(n: int) -> tuple:
    """
    Convert flattened basis index to Dubiner order pair (p, q) in O(1) time.
    
    This function is a convenience wrapper for the canonical implementation
    in the numerics.filters module. See that module for details.
    
    Parameters
    ----------
    n : int
        Flattened basis index
        
    Returns
    -------
    tuple
        Order pair (p, q)
    """
    from ..numerics import dubiner_basis_index_to_order as index_to_order
    return index_to_order(n)


def generate_subdivided_triangle(n_div: int) -> tuple:
    r"""
    Subdivide reference triangle into n_div² smaller triangles on a uniform grid.

    Generates a regular triangulation of the reference triangle with vertices
    (0, 0), (1, 0), (0, 1) by creating a grid of nodes and connecting them
    in alternating orientation to tile the domain.

    The node coordinates follow a lexicographic ordering by (i, j) where:
    - i: horizontal index in [0, n_div]
    - j: vertical index in [0, n_div]
    - Constraint: i + j ≤ n_div

    This produces N_nodes = (n_div + 1)(n_div + 2)/2 nodes arranged in
    (n_div + 1) rows, and N_triangles = 2·n_div² small triangles.

    Parameters
    ----------
    n_div : int
        Number of edge subdivisions (n_div ≥ 1)

    Returns
    -------
    tuple
        (nodes, triangles) where:
        - nodes: np.ndarray, shape (N_nodes, 2), coordinates in [0, 1]²
        - triangles: list of lists, each entry [v1, v2, v3] is a CCW-oriented triangle

    Raises
    ------
    ValueError
        If n_div < 1

    Examples
    --------
    Generate a level-1 mesh with 2 subdivisions per edge:

    >>> nodes, triangles = generate_subdivided_triangle(2)
    >>> print(f"Nodes shape: {nodes.shape}")
    Nodes shape: (6, 2)
    >>> print(f"Number of triangles: {len(triangles)}")
    Number of triangles: 8

    Verify CCW orientation by checking signed areas:

    >>> for tri in triangles:
    ...     v_idx = tri
    ...     v1, v2, v3 = nodes[v_idx]
    ...     signed_area = 0.5 * ((v2[0]-v1[0])*(v3[1]-v1[1]) - (v3[0]-v1[0])*(v2[1]-v1[1]))
    ...     assert signed_area > 0, "Triangle not CCW oriented"

    Notes
    -----
    The subdivision pattern is:
    - For each (i, j) with i + j < n_div (interior corner):
        - Create "up" triangle: (i, j) → (i+1, j) → (i, j+1)
        - Create "down" triangle: (i+1, j) → (i+1, j+1) → (i, j+1)
    - All triangles are counter-clockwise (CCW) oriented for consistency with DG conventions
    """
    if not isinstance(n_div, (int, np.integer)) or n_div < 1:
        raise ValueError(f"n_div must be integer ≥ 1, got {n_div}")

    nodes = []
    node_idx = {}

    # Generate nodes in lexicographic order
    curr_idx = 0
    for j in range(n_div + 1):
        for i in range(n_div + 1 - j):
            x = i / n_div
            y = j / n_div
            nodes.append([x, y])
            node_idx[(i, j)] = curr_idx
            curr_idx += 1

    nodes = np.array(nodes, dtype=float)

    # Generate triangles with CCW orientation
    triangles = []
    for j in range(n_div):
        for i in range(n_div - j):
            # "Up" triangle: points (i,j), (i+1,j), (i,j+1)
            v1 = node_idx[(i, j)]
            v2 = node_idx[(i + 1, j)]
            v3 = node_idx[(i, j + 1)]
            triangles.append([v1, v2, v3])

            # "Down" triangle: points (i+1,j), (i+1,j+1), (i,j+1)
            # Only if this doesn't go outside the domain
            if i + j + 1 < n_div:
                v1 = node_idx[(i + 1, j)]
                v2 = node_idx[(i + 1, j + 1)]
                v3 = node_idx[(i, j + 1)]
                triangles.append([v1, v2, v3])

    return nodes, triangles


def exponent_pairs(total_degree: int) -> list:
    r"""
    Generate exponent pairs for 2D polynomial basis functions.

    For a given total polynomial degree p, generates all pairs (i, j) such that
    i + j = p, in canonical ordering for the 2D simplex polynomial space P^p_2.

    This utility is primarily used for enumerating monomials x^i y^j that form
    the basis of polynomial spaces in spectral methods.

    Parameters
    ----------
    total_degree : int
        Total polynomial degree p (i + j = p)

    Returns
    -------
    list of tuple
        Exponent pairs [(0, p), (1, p-1), ..., (p, 0)] in canonical order

    Examples
    --------
    Enumerate all monomials of degree 2:

    >>> pairs = exponent_pairs(2)
    >>> print(pairs)
    [(0, 2), (1, 1), (2, 0)]

    The number of basis functions in P^p_2:

    >>> total_basis_count = 0
    >>> for p in range(5):
    ...     pairs = exponent_pairs(p)
    ...     total_basis_count += len(pairs)
    >>> print(f"Total basis in P^4_2: {total_basis_count}")
    Total basis in P^4_2: 15

    Notes
    -----
    The total count of basis functions in P^p_2 is (p+1)(p+2)/2 = C(p+2, 2),
    which matches the binomial coefficient for choosing 2 items from p+2.

    This ordering is consistent with:
    - Jacobi polynomial orderings in collapsed coordinates
    - Standard spectral element method conventions
    - The Dubiner basis function ordering
    """
    if not isinstance(total_degree, (int, np.integer)) or total_degree < 0:
        raise ValueError(f"total_degree must be non-negative integer, got {total_degree}")

    return [(i, total_degree - i) for i in range(total_degree + 1)]
