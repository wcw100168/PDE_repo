"""
2D Simplex Dubiner Orthonormal Basis & Vandermonde Matrix Construction.

Implements Jacobi polynomials, collapsed coordinate transformations, and Dubiner
basis evaluations on reference triangular elements.
"""

from __future__ import annotations
import numpy as np
from scipy.special import eval_jacobi


def collapsed_coords_transform(r: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform reference triangle coordinates (r, s) in [-1, 1]^2 to collapsed coordinates (a, b) in [-1, 1]^2.
    
    Mapping:
        a = 2*(1+r)/(1-s) - 1  (if s != 1, else -1)
        b = s
    """
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    
    a = np.zeros_like(r)
    mask = (s != 1.0)
    a[mask] = 2.0 * (1.0 + r[mask]) / (1.0 - s[mask]) - 1.0
    a[~mask] = -1.0
    b = s
    return a, b


def jacobi_p(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """
    Evaluate 1D Jacobi polynomial P_n^{(alpha, beta)}(x).
    """
    return eval_jacobi(n, alpha, beta, x)


def evaluate_dubiner_basis_2d(a: np.ndarray, b: np.ndarray, i: int, j: int) -> np.ndarray:
    """
    Evaluate 2D Dubiner orthonormal basis psi_{i,j}(a, b) on collapsed coordinates:
        psi_{i,j}(a, b) = sqrt(2) * P_i^{(0,0)}(a) * P_j^{(2i+1, 0)}(b) * (1-b)^i
    """
    h1 = jacobi_p(a, 0.0, 0.0, i)
    h2 = jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    return np.sqrt(2.0) * h1 * h2 * ((1.0 - b) ** i)


def vandermonde_2d_dubiner(r: np.ndarray, s: np.ndarray, N: int) -> np.ndarray:
    """
    Construct raw Vandermonde matrix V_raw for 2D Dubiner basis up to polynomial degree N.
    
    Parameters
    ----------
    r, s : np.ndarray
        Coordinates on reference triangle.
    N : int
        Maximum polynomial degree. Number of basis functions: (N+1)(N+2)/2.
        
    Returns
    -------
    V_raw : np.ndarray of shape (n_points, num_basis)
    """
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    a, b = collapsed_coords_transform(r, s)
    n_points = len(r)
    num_basis = (N + 1) * (N + 2) // 2
    
    V_raw = np.zeros((n_points, num_basis), dtype=float)
    col_idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            V_raw[:, col_idx] = evaluate_dubiner_basis_2d(a, b, i, j)
            col_idx += 1
            
    return V_raw
