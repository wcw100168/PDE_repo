"""
Preconditioned Cholesky Modal Orthogonalization Module.

Eliminates Vandermonde matrix conditioning breakdown and ill-conditioning for high-order
polynomials on triangular elements via Cholesky decomposition of the reference mass matrix.
"""

from __future__ import annotations
import numpy as np


def cholesky_orthogonalize_vandermonde(V_raw: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Preconditioned Cholesky Modal Orthogonalization on raw Vandermonde matrix V_raw.
    
    Given:
        Reference Mass Matrix: M_hat = V_raw^T * W * V_raw
        Cholesky Factorization: M_hat = L * L^T  (L is lower triangular)
        Orthogonalized Vandermonde: V = V_raw * (L^T)^{-1}
        
    Satisfies:
        V^T * W * V = (L^{-1} * V_raw^T) * W * (V_raw * (L^T)^{-1})
                    = L^{-1} * M_hat * (L^T)^{-1}
                    = L^{-1} * (L * L^T) * (L^T)^{-1} = I
                    
    Parameters
    ----------
    V_raw : np.ndarray
        Raw Vandermonde matrix of shape (n_points, n_basis).
    W : np.ndarray
        Quadrature weight matrix (1D array of weights or 2D diagonal matrix).
        
    Returns
    -------
    V : np.ndarray
        Orthogonalized Vandermonde matrix of shape (n_points, n_basis).
    L : np.ndarray
        Lower triangular Cholesky factor of shape (n_basis, n_basis).
    """
    V_raw = np.asarray(V_raw, dtype=float)
    if W.ndim == 1:
        W_diag = np.diag(W)
    else:
        W_diag = W
        
    # Reference mass matrix calculation
    M_hat = V_raw.T @ W_diag @ V_raw
    
    # Cholesky factorization M_hat = L @ L.T
    L = np.linalg.cholesky(M_hat)
    
    # Compute V = V_raw @ (L.T)^{-1} via forward substitution
    # L.T * V.T = V_raw.T  => solve for V.T
    V_transposed = np.linalg.solve(L, V_raw.T)
    V = V_transposed.T
    
    return V, L


def compute_orthogonality_residual(V: np.ndarray, W: np.ndarray) -> float:
    """
    Compute maximum infinity norm residual || V^T * W * V - I ||_inf.
    """
    if W.ndim == 1:
        W_diag = np.diag(W)
    else:
        W_diag = W
        
    n_basis = V.shape[1]
    I_expected = np.eye(n_basis)
    M_ortho = V.T @ W_diag @ V
    return float(np.max(np.abs(M_ortho - I_expected)))
