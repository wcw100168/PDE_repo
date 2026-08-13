"""
Peirce Subspace Decomposition Module.

Decomposes operator space R^{M x M} into four orthogonal subspaces:
    R^{M x M} = S_{PP} + S_{PQ} + S_{QP} + S_{QQ}
where P is the polynomial projection operator and Q = I - P.
"""

from __future__ import annotations
import numpy as np


def peirce_subspace_decomposition(A: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose matrix A into Peirce orthogonal components:
        A = A_PP + A_PQ + A_QP + A_QQ
    where:
        A_PP = P * A * P
        A_PQ = P * A * Q
        A_QP = Q * A * P
        A_QQ = Q * A * Q
    and Q = I - P.
    
    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (M, M).
    P : np.ndarray
        Polynomial projection matrix of shape (M, M).
        
    Returns
    -------
    A_PP, A_PQ, A_QP, A_QQ : tuple of np.ndarray
        Four orthogonal Peirce components, each of shape (M, M).
    """
    M = A.shape[0]
    I_mat = np.eye(M)
    Q = I_mat - P
    
    A_PP = P @ A @ P
    A_PQ = P @ A @ Q
    A_QP = Q @ A @ P
    A_QQ = Q @ A @ Q
    
    return A_PP, A_PQ, A_QP, A_QQ


def verify_peirce_orthogonality(P: np.ndarray) -> float:
    """
    Verify projection idempotency P^2 = P and complement Q^2 = Q, P * Q = 0.
    """
    M = P.shape[0]
    I_mat = np.eye(M)
    Q = I_mat - P
    
    p2_res = np.max(np.abs(P @ P - P))
    pq_res = np.max(np.abs(P @ Q))
    q2_res = np.max(np.abs(Q @ Q - Q))
    
    return float(max(p2_res, pq_res, q2_res))
