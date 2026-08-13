"""
Filter operations for modal coefficients: Tikhonov regularization and exponential filters.

These functions support spectral filtering and regularization of modal expansions,
critical for controlling high-frequency oscillations in DG methods.
"""

import numpy as np
from .orthogonal_polys import jacobi_p


def dubiner_basis_index_to_order(n: int) -> tuple:
    """
    Convert flattened basis index to Dubiner order pair (p, q) in O(1) time.
    
    Uses the analytical quadratic formula solution instead of iteration.
    The p-th polynomial degree d satisfies:
        d * (d + 1) / 2 <= n < (d + 1) * (d + 2) / 2
    
    Solving d² + d - 2n <= 0 using the quadratic formula:
        d = floor((-1 + sqrt(1 + 8n)) / 2)
    
    Parameters
    ----------
    n : int
        Flattened basis index (0-indexed)
        
    Returns
    -------
    tuple
        Order pair (p, q) where p, q >= 0 and p + q = d
    """
    # Closed-form solution: d = floor((-1 + sqrt(1 + 8n)) / 2)
    d = int((-1.0 + np.sqrt(1.0 + 8.0 * n)) / 2.0)
    
    # Compute position within this polynomial degree
    idx = n - d * (d + 1) // 2
    p = d - idx
    q = idx
    
    return p, q


def tikhonov_regularization_matrix(num_basis: int, lambda_reg: float = 1e-4) -> np.ndarray:
    """
    Create Tikhonov penalty matrix for regularizing modal expansion.
    
    Penalizes higher polynomial degrees with increasing regularization strength.
    The penalty weight is sqrt(lambda_reg) * (degree + 1) on the diagonal.
    
    Parameters
    ----------
    num_basis : int
        Number of basis functions (total)
    lambda_reg : float
        Regularization parameter (default: 1e-4)
        
    Returns
    -------
    np.ndarray
        Diagonal penalty matrix L of shape (num_basis, num_basis)
    """
    L = np.zeros((num_basis, num_basis), dtype=float)

    for n in range(num_basis):
        p, q = dubiner_basis_index_to_order(n)
        degree = p + q
        L[n, n] = np.sqrt(lambda_reg) * (degree + 1.0)

    return L


def apply_exponential_filter(
    coeffs: np.ndarray,
    num_basis: int,
    k: int,
    alpha: float = 36.0,
    filter_order: int = 8
) -> np.ndarray:
    """
    Apply spectral exponential filter to modal coefficients.
    
    High-frequency modes (large p+q) are damped according to:
        sigma(p+q) = exp(-alpha * (degree/k)^filter_order)
    
    This controls Gibbs oscillations near discontinuities while preserving
    low-frequency accuracy.
    
    Parameters
    ----------
    coeffs : np.ndarray
        Modal coefficients (shape: num_basis)
    num_basis : int
        Number of basis functions
    k : int
        Polynomial degree of reference element
    alpha : float
        Exponential decay parameter (default: 36)
    filter_order : int
        Polynomial order of decay (default: 8)
        
    Returns
    -------
    np.ndarray
        Filtered coefficients with same shape as input
    """
    filtered_coeffs = coeffs.copy()

    for n in range(num_basis):
        p, q = dubiner_basis_index_to_order(n)
        degree = p + q

        if degree == 0:
            continue

        # Normalize degree to [0, 1] range
        normalized_degree = degree / k if k > 0 else 0.0
        
        # Apply exponential decay on normalized degree
        sigma = np.exp(-alpha * (normalized_degree ** filter_order))
        filtered_coeffs[n] *= sigma

    return filtered_coeffs
