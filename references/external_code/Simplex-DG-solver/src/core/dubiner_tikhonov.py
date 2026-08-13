"""
Dubiner basis and Tikhonov-regularized modal reconstruction utilities.

This module is intentionally separated from modal_expansion.py to avoid
mixing multiple reconstruction paradigms in one file.
"""

import numpy as np
from scipy.special import jacobi
from scipy.special import gamma

from .data_structs import bary_to_cartesian_2d


def collapsed_coords_transform(xi: np.ndarray, eta: np.ndarray):
    """
    Map (xi, eta) to collapsed coordinates (a, b).
    No singularity or tolerance handling.
    """
    a = 2.0 * (1.0 + xi) / (1.0 - eta) - 1.0
    b = eta
    return a, b


def dubiner_basis_index_to_order(n: int):
    """
    Convert flattened basis index to Dubiner order pair (p, q).
    """
    d = 0
    while (d + 1) * (d + 2) // 2 <= n:
        d += 1

    idx = n - d * (d + 1) // 2
    p = d - idx
    q = idx

    return p, q


def jacobi_p(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """
    Compute normalized Jacobi polynomial P_N^{(alpha, beta)} at x.
    Assumes alpha, beta > -1.
    """
    xp = np.atleast_1d(x)
    PL = np.zeros((N + 1, len(xp)))

    # P_0(x)
    gamma0 = (
        2**(alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )
    PL[0, :] = 1.0 / np.sqrt(gamma0)

    if N == 0:
        return PL[0, :].reshape(np.shape(x))

    # P_1(x)
    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    PL[1, :] = (
        (alpha + beta + 2) * xp / 2.0 + (alpha - beta) / 2.0
    ) / np.sqrt(gamma1)

    if N == 1:
        return PL[1, :].reshape(np.shape(x))

    # Recurrence initialization
    aold = (
        2.0 / (2.0 + alpha + beta)
        * np.sqrt((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0))
    )

    # Recurrence for higher orders
    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta

        anew = (
            2.0 / (h1 + 2.0)
            * np.sqrt(
                (i + 1.0)
                * (i + 1.0 + alpha + beta)
                * (i + 1.0 + alpha)
                * (i + 1.0 + beta)
                / (h1 + 1.0)
                / (h1 + 3.0)
            )
        )
        bnew = -(alpha**2 - beta**2) / (h1 * (h1 + 2.0))

        # Recurrence relation
        PL[i + 1, :] = (
            -aold * PL[i - 1, :] + (xp - bnew) * PL[i, :]
        ) / anew

        aold = anew

    return PL[N, :].reshape(np.shape(x))


def evaluate_simplex_basis_2d(
    a: np.ndarray, b: np.ndarray, i: int, j: int, jacobi_func
):
    """
    Evaluate 2D simplex orthonormal basis of order (i, j) at (a, b).
    jacobi_func must return normalized Jacobi polynomials.
    """
    h1 = jacobi_func(a, 0, 0, i)
    h2 = jacobi_func(b, 2 * i + 1, 0, j)

    return np.sqrt(2.0) * h1 * h2 * (1.0 - b) ** i


def vandermonde_2d_dubiner(
    r: np.ndarray, s: np.ndarray, N: int
):
    """
    Build Vandermonde matrix for 2D simplex basis up to order N.
    (i + j <= N)
    """
    a, b = collapsed_coords_transform(r, s)
    n_points = len(r)

    num_basis = (N + 1) * (N + 2) // 2
    V = np.zeros((n_points, num_basis), dtype=float)

    col_idx = 0
    for i in range(N + 1):
        for j in range(N - i + 1):
            V[:, col_idx] = evaluate_simplex_basis_2d(
                a, b, i, j, jacobi_p
            )
            col_idx += 1

    return V


def grad_jacobi_p(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """
    Compute the exact derivative of the normalized Jacobi polynomial.
    Based on Equation (A.2) from the PDF.
    """
    if N == 0:
        return np.zeros_like(x)
    
    # d/dx P_N^(alpha, beta) = sqrt(N * (N + alpha + beta + 1)) * P_{N-1}^(alpha+1, beta+1)
    coeff = np.sqrt(N * (N + alpha + beta + 1.0))
    
    return coeff * jacobi_p(x, alpha + 1.0, beta + 1.0, N - 1)


def dubiner_basis_derivative(a: np.ndarray, b: np.ndarray, i: int, j: int):
    """
    Evaluate the partial derivatives (dpsi_dr, dpsi_ds) of the 2D simplex 
    orthonormal basis of order (i, j) at collapsed coordinates (a, b).
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


def grad_vandermonde_2d_dubiner(r: np.ndarray, s: np.ndarray, N: int):
    """
    Build the gradient Vandermonde matrices (Vr, Vs) for 2D simplex basis up to order N.
    Matches the loop structure and signature of vandermonde_2d_dubiner.
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

def build_differentiation_matrices(V, V_xi, V_eta, w):
    """
    Construct numerical differentiation matrices with quadrature weights.

    Parameters:
      V : ndarray of shape (n_points, num_basis)
          Vandermonde matrix.

      V_xi : ndarray of shape (n_points, num_basis)
          Gradient Vandermonde matrix in the xi direction.

      V_eta : ndarray of shape (n_points, num_basis)
          Gradient Vandermonde matrix in the eta direction.

      w : ndarray of shape (n_points,)
          Quadrature weights (w_i^s).

    Returns:
      D_xi : ndarray
          Differentiation matrix in the xi direction.

      D_eta : ndarray
          Differentiation matrix in the eta direction.
    """

    W = np.diag(w)
    M_ref = V.T @ W @ V
    P = np.linalg.solve(M_ref, V.T @ W)
    D_xi = V_xi @ P
    D_eta = V_eta @ P

    return D_xi, D_eta


def tikhonov_regularization_matrix(num_basis: int, lambda_reg: float = 1e-4):
    """
    Create Tikhonov penalty matrix that penalizes higher modal degrees.
    """
    L = np.zeros((num_basis, num_basis), dtype=float)

    for n in range(num_basis):
        p, q = dubiner_basis_index_to_order(n)
        degree = p + q
        L[n, n] = np.sqrt(lambda_reg) * (degree + 1.0)

    return L


def apply_exponential_filter(coeffs: np.ndarray, num_basis: int, k: int,
                             alpha: float = 36.0, filter_order: int = 8):
    """
    Apply spectral exponential filter to modal coefficients.
    """
    filtered_coeffs = coeffs.copy()

    for n in range(num_basis):
        p, q = dubiner_basis_index_to_order(n)
        degree = p + q

        if degree == 0:
            continue

        normalized_degree = degree / k if k > 0 else 0.0
        sigma = np.exp(-alpha * (normalized_degree ** filter_order))
        filtered_coeffs[n] *= sigma

    return filtered_coeffs


def modal_reconstruct_at_bary_dubiner_tikhonov(nodes, known_vals: np.ndarray, target_bary: np.ndarray,
                                               vertices_2d: np.ndarray, k: int,
                                               lambda_reg: float = 1e-4, apply_filter: bool = False):
    """
    Modal reconstruction using Dubiner basis and Tikhonov regularization.
    """
    local_internal = np.array([n.local_coords for n in nodes], dtype=float)
    weights = np.array([n.weight for n in nodes], dtype=float)

    num_basis = (k + 1) * (k + 2) // 2

    xi_internal = 2.0 * local_internal[:, 0] - 1.0
    eta_internal = 2.0 * local_internal[:, 1] - 1.0
    V = vandermonde_2d_dubiner(xi_internal, eta_internal, num_basis)

    W = np.diag(weights)
    M = V.T @ W @ V

    L = tikhonov_regularization_matrix(num_basis, lambda_reg=lambda_reg)
    M_reg = M + lambda_reg ** 2 * (L.T @ L)

    rhs = V.T @ W @ np.asarray(known_vals, dtype=float)
    a = np.linalg.solve(M_reg, rhs)

    if apply_filter:
        a = apply_exponential_filter(a, num_basis, k)

    target_local = np.array([[b[2], b[0]] for b in target_bary], dtype=float)
    target_xy = np.array([bary_to_cartesian_2d(b, vertices_2d) for b in target_bary], dtype=float)

    xi_target = 2.0 * target_local[:, 0] - 1.0
    eta_target = 2.0 * target_local[:, 1] - 1.0
    V_target = vandermonde_2d_dubiner(xi_target, eta_target, num_basis)

    u_target = V_target @ a

    return target_xy, u_target


modal_reconstruct_at_bary_upgraded = modal_reconstruct_at_bary_dubiner_tikhonov
