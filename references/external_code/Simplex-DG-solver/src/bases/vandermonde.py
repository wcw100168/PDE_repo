"""
Vandermonde matrices for simplex basis functions.

Efficiently constructs Vandermonde matrices and their gradients for use in
collocation, interpolation, and modal expansion algorithms.
"""

import numpy as np
from .simplex_2d import evaluate_simplex_basis_2d, dubiner_basis_derivative
from ..geometry import collapsed_coords_transform


def vandermonde_2d_dubiner(r: np.ndarray, s: np.ndarray, N: int) -> np.ndarray:
    """
    Build Vandermonde matrix for 2D simplex orthonormal basis up to polynomial degree N.
    
    Evaluates all basis functions {psi_{i,j} : i + j <= N} at the given nodes.
    
    Parameters
    ----------
    r : np.ndarray
        First coordinate in reference element (shape: n_points)
    s : np.ndarray
        Second coordinate in reference element (shape: n_points)
    N : int
        Maximum polynomial degree (basis count: (N+1)(N+2)/2)
        
    Returns
    -------
    np.ndarray
        Vandermonde matrix of shape (n_points, num_basis)
    """
    # Transform to collapsed coordinates
    a, b = collapsed_coords_transform(r, s)
    n_points = len(r)

    # Total number of basis functions
    num_basis = (N + 1) * (N + 2) // 2
    V = np.zeros((n_points, num_basis), dtype=float)

    # Fill Vandermonde matrix column by column
    col_idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            V[:, col_idx] = evaluate_simplex_basis_2d(a, b, i, j)
            col_idx += 1

    return V


def grad_vandermonde_2d_dubiner(r: np.ndarray, s: np.ndarray, N: int) -> tuple:
    """
    Build gradient Vandermonde matrices (Vr, Vs) for 2D simplex basis up to polynomial degree N.
    
    Returns derivatives in the reference element coordinates (r, s) = (xi, eta).
    
    Parameters
    ----------
    r : np.ndarray
        First coordinate in reference element (shape: n_points)
    s : np.ndarray
        Second coordinate in reference element (shape: n_points)
    N : int
        Maximum polynomial degree
        
    Returns
    -------
    tuple
        (Vr, Vs) gradient Vandermonde matrices, each of shape (n_points, num_basis)
    """
    # Transform to collapsed coordinates
    a, b = collapsed_coords_transform(r, s)
    n_points = len(r)

    # Total number of basis functions
    num_basis = (N + 1) * (N + 2) // 2
    Vr = np.zeros((n_points, num_basis), dtype=float)
    Vs = np.zeros((n_points, num_basis), dtype=float)
    
    # Fill gradient matrices column by column
    col_idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            dpsi_dr, dpsi_ds = dubiner_basis_derivative(a, b, i, j)
            
            Vr[:, col_idx] = dpsi_dr
            Vs[:, col_idx] = dpsi_ds
            
            col_idx += 1

    return Vr, Vs
