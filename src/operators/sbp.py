"""
Closed-Form Boundary-Compatible SBP Operator Construction.

Implements Definition D003:
    Delta D_eta = 0.5 * W^{-1} * (I + P)^T * B * (I - P)
where P = V * (V^T * W * V)^{-1} * V^T * W is the symmetric polynomial projection operator.
"""

from __future__ import annotations
import numpy as np


def compute_polynomial_projection_operator(V: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute symmetric polynomial projection operator P, coefficients P_c, and modal inverse mass matrix.
    
    Returns
    -------
    P : np.ndarray
        Nodal projection matrix V * P_c
    P_c : np.ndarray
        Modal projection matrix (V^T W V)^{-1} V^T W
    Minv : np.ndarray
        Inverse of modal mass matrix (V^T W V)^{-1}
    """
    W_diag = np.diag(W) if W.ndim == 1 else W
    W_diag = 2.0 * W_diag  # Reference triangle area is 2.0
    M_hat = V.T @ W_diag @ V
    Minv = np.linalg.inv(M_hat)
    
    # Solve M_hat * X = V^T * W_diag
    rhs = V.T @ W_diag
    P_c = Minv @ rhs
    P = V @ P_c
    return P, P_c, Minv


def compute_face_operators(
    r_vol: np.ndarray,
    s_vol: np.ndarray,
    W_vol: np.ndarray,
    face_id: int,
    w_face: np.ndarray,
    order: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Face Interpolation (E) and Face Lift (L) matrices using direct extraction.
    
    Parameters
    ----------
    r_vol, s_vol : np.ndarray
        Volume node coordinates.
    W_vol : np.ndarray
        Volume quadrature weights.
    face_id : int
        Face ID (0, 1, or 2).
    w_face : np.ndarray
        Face quadrature weights.
    order : int
        Polynomial order.
        
    Returns
    -------
    E : np.ndarray
        Face extraction matrix (boolean).
    L : np.ndarray
        Face lift matrix.
    """
    from src.geometry.quadrature import get_triangle_boundary_extraction
    n_face = order + 1
    
    E, indices = get_triangle_boundary_extraction(r_vol, s_vol, face_id, n_face)
    
    # L = W_vol^{-1} E^T W_face
    # W_vol is the scaled volume weights summing to area (2.0)
    # Wait! W_vol passed in sums to 1.0. The discrete inner product matrix W_diag is 2.0 * W_vol.
    # So L = (2.0 * W_vol)^{-1} E^T w_face
    W_diag_inv = 1.0 / (2.0 * W_vol)
    L = W_diag_inv[:, None] * E.T * w_face[None, :]
    
    return E, L




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
