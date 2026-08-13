"""
Derivative Vandermonde & Reference Differentiation Operator Matrices.

Computes partial derivatives Vr, Vs in reference coordinates (r, s)
and projects them to nodal differentiation matrices Dr, Ds.
"""

from __future__ import annotations
import numpy as np
from .basis import jacobi_p, collapsed_coords_transform
from scipy.special import eval_jacobi, gammaln


def jacobi_p_derivative(x: np.ndarray, alpha: float, beta: float, n: int) -> np.ndarray:
    """
    Derivative of orthonormal Jacobi polynomial d/dx P_n^{(alpha, beta)}(x).
    """
    if n == 0:
        return np.zeros_like(x)
    coeff = 0.5 * np.sqrt(n * (n + alpha + beta + 1.0))
    p_classical = eval_jacobi(n - 1, alpha + 1.0, beta + 1.0, x)
    
    # Normalization factor for P_n^{(alpha, beta)}
    log_num = (alpha + beta + 1.0) * np.log(2.0) + gammaln(n + alpha + 1.0) + gammaln(n + beta + 1.0)
    log_den = np.log(2.0 * n + alpha + beta + 1.0) + gammaln(n + 1.0) + gammaln(n + alpha + beta + 1.0)
    h_n = float(np.exp(log_num - log_den))
    
    return coeff * p_classical / np.sqrt(h_n)


def dubiner_basis_derivative(a: np.ndarray, b: np.ndarray, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate partial derivatives (dpsi_dr, dpsi_ds) in reference element coordinates (r, s).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    
    p_i = jacobi_p(a, 0.0, 0.0, i)
    dp_i_da = jacobi_p_derivative(a, 0.0, 0.0, i)
    
    p_j = jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    dp_j_db = jacobi_p_derivative(b, 2.0 * i + 1.0, 0.0, j)
    
    sqrt2 = np.sqrt(2.0)
    
    if i == 0:
        dpsi_dr = np.zeros_like(a)
        dpsi_ds = sqrt2 * p_i * dp_j_db
    else:
        dpsi_dr = sqrt2 * dp_i_da * p_j * 2.0 * ((1.0 - b) ** (i - 1))
        term1 = dp_i_da * p_j * (1.0 + a) * ((1.0 - b) ** (i - 1))
        term2 = p_i * dp_j_db * ((1.0 - b) ** i)
        term3 = -i * p_i * p_j * ((1.0 - b) ** (i - 1))
        dpsi_ds = sqrt2 * (term1 + term2 + term3)
        
    return dpsi_dr, dpsi_ds


def grad_vandermonde_2d_dubiner(r: np.ndarray, s: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build gradient Vandermonde matrices (Vr, Vs) for 2D Dubiner basis up to degree N.
    """
    a, b = collapsed_coords_transform(r, s)
    n_points = len(r)
    num_basis = (N + 1) * (N + 2) // 2
    
    Vr = np.zeros((n_points, num_basis), dtype=float)
    Vs = np.zeros((n_points, num_basis), dtype=float)
    
    col_idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            dpsi_dr, dpsi_ds = dubiner_basis_derivative(a, b, i, j)
            Vr[:, col_idx] = dpsi_dr
            Vs[:, col_idx] = dpsi_ds
            col_idx += 1
            
    return Vr, Vs


def differentiation_matrices_weighted(V: np.ndarray, Vr: np.ndarray, Vs: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute reference nodal differentiation matrices Dr, Ds:
        Projection: P_c = (V^T * W * V)^{-1} * V^T * W
        Dr = Vr * P_c
        Ds = Vs * P_c
    """
    W_diag = np.diag(W) if W.ndim == 1 else W
    M_hat = V.T @ W_diag @ V
    rhs = V.T @ W_diag
    P_c = np.linalg.solve(M_hat, rhs)
    
    Dr = Vr @ P_c
    Ds = Vs @ P_c
    return Dr, Ds
