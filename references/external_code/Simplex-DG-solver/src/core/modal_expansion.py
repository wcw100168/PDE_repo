"""
Modal expansion utilities (non-Dubiner path).

This module intentionally excludes Dubiner-Tikhonov implementation details.
Dubiner-specific functions live in core.dubiner_tikhonov.
"""

import numpy as np

from .data_structs import bary_to_cartesian_2d


# ==========================================
# Helper Math Functions
# ==========================================

def triangle_area(vertices_2d: np.ndarray) -> float:
    """
    Compute the area of a 2D triangle.

    Parameters:
    -----------
    vertices_2d : np.ndarray
        Triangle vertices (shape: [3, 2])

    Returns:
    --------
    float
        Area of the triangle
    """
    p0, p1, p2 = vertices_2d
    return 0.5 * abs(np.cross(p1 - p0, p2 - p0))


# ==========================================
# Standard Monomial Basis
# ==========================================

def monomial_powers(max_degree: int):
    """
    Generate all monomial powers up to max_degree.

    Returns powers (i, j) for monomials x^i * y^j with i+j <= max_degree,
    ordered by total degree then lexicographically.

    Parameters:
    -----------
    max_degree : int
        Maximum total degree

    Returns:
    --------
    list of tuple
        List of (i, j) power pairs
    """
    powers = []
    for total in range(max_degree + 1):
        for i in range(total + 1):
            j = total - i
            powers.append((i, j))
    return powers


def choose_basis_powers(n_points: int):
    """
    Choose monomial basis powers for given number of points.

    Selects enough distinct polynomial terms to form a square system.

    Parameters:
    -----------
    n_points : int
        Number of points to interpolate

    Returns:
    --------
    list of tuple
        List of (i, j) power pairs
    """
    degree = 0
    while (degree + 1) * (degree + 2) // 2 < n_points:
        degree += 1
    all_powers = monomial_powers(degree)
    return all_powers[:n_points]


def vandermonde_2d(x: np.ndarray, y: np.ndarray, powers):
    """
    Build 2D Vandermonde matrix using monomial basis.

    Parameters:
    -----------
    x, y : np.ndarray
        Coordinate arrays
    powers : list of tuple
        Monomial powers (i, j)

    Returns:
    --------
    np.ndarray
        Vandermonde matrix
    """
    cols = []
    for i, j in powers:
        cols.append((x ** i) * (y ** j))
    return np.column_stack(cols)


# ==========================================
# Modal Reconstruction Function
# ==========================================

def modal_reconstruct_at_bary(nodes, known_vals: np.ndarray, target_bary: np.ndarray,
                              vertices_2d: np.ndarray):
    """
    Reconstruct function values using standard monomial modal expansion.

    Builds a polynomial interpolant through known nodes and evaluates at target points.

    Parameters:
    -----------
    nodes : list of Node
        Known interior quadrature nodes
    known_vals : np.ndarray
        Function values at known nodes
    target_bary : np.ndarray
        Barycentric coordinates of target points (shape: [n_target, 3])
    vertices_2d : np.ndarray
        2D reference triangle vertices

    Returns:
    --------
    tuple of np.ndarray
        (target_xy, u_target_modal)
    """
    local_internal = np.array([n.local_coords for n in nodes], dtype=float)
    weights = np.array([n.weight for n in nodes], dtype=float)

    powers = choose_basis_powers(len(nodes))
    V = vandermonde_2d(local_internal[:, 0], local_internal[:, 1], powers)

    target_local = np.array([[b[2], b[0]] for b in target_bary], dtype=float)
    target_xy = np.array([bary_to_cartesian_2d(b, vertices_2d) for b in target_bary], dtype=float)
    V_target = vandermonde_2d(target_local[:, 0], target_local[:, 1], powers)

    W = np.diag(weights)
    M = V.T @ W @ V

    rhs = V.T @ W @ np.asarray(known_vals, dtype=float)
    a = np.linalg.solve(M, rhs)

    u_target_modal = V_target @ a
    return target_xy, u_target_modal
