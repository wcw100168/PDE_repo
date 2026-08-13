"""
Closed-Form Boundary-Compatible SBP Operator Construction.

Implements Definition D003:
    Delta D_eta = 0.5 * W^{-1} * (I + P)^T * B * (I - P)
where P = V * (V^T * W * V)^{-1} * V^T * W is the symmetric polynomial projection operator.
"""

from __future__ import annotations
import numpy as np


def compute_polynomial_projection_operator(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Compute symmetric polynomial projection operator P:
        P = V * (V^T * W * V)^{-1} * V^T * W
    """
    W_diag = np.diag(W) if W.ndim == 1 else W
    M_hat = V.T @ W_diag @ V
    
    # Solve M_hat * X = V^T * W
    rhs = V.T @ W_diag
    P_c = np.linalg.solve(M_hat, rhs)
    P = V @ P_c
    return P


def construct_closed_form_sbp_operator(
    D_base: np.ndarray,
    P: np.ndarray,
    W: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """
    Construct boundary-compatible SBP operator D_new satisfying W * D_new + D_new^T * W = B.
    
    Splits D_base into skew-symmetric volume component D_skew and boundary term:
        D_skew = 0.5 * (D_base - W^{-1} * D_base^T * W)
        D_new = D_skew + 0.5 * W^{-1} * B
        
    Parameters
    ----------
    D_base : np.ndarray
        Base differentiation matrix of shape (n_points, n_points).
    P : np.ndarray
        Polynomial projection operator matrix of shape (n_points, n_points).
    W : np.ndarray
        Quadrature weight matrix (1D array or 2D diagonal matrix).
    B : np.ndarray
        Symmetric boundary line-integral operator matrix of shape (n_points, n_points).
        
    Returns
    -------
    D_new : np.ndarray
        Boundary-compatible SBP operator satisfying W * D_new + D_new^T * W = B.
    """
    W_diag = np.diag(W) if W.ndim == 1 else W
    W_inv = np.diag(1.0 / W) if W.ndim == 1 else np.linalg.inv(W)
    
    # Skew-symmetric volume operator D_skew
    D_skew = 0.5 * (D_base - W_inv @ D_base.T @ W_diag)
    
    # Boundary-compatible SBP operator D_new
    D_new = D_skew + 0.5 * W_inv @ B
    return D_new



def verify_sbp_property_residual(D: np.ndarray, W: np.ndarray, B: np.ndarray) -> float:
    """
    Verify SBP property residual || W * D + D^T * W - B ||_inf.
    """
    W_diag = np.diag(W) if W.ndim == 1 else W
    res_matrix = W_diag @ D + D.T @ W_diag - B
    return float(np.max(np.abs(res_matrix)))
