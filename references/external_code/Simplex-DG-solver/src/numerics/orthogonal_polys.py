"""
Orthogonal polynomial utilities: Jacobi polynomials and their derivatives.

This module provides normalized Jacobi polynomial evaluation and differentiation
using recurrence relations, suitable for spectral element methods.
"""

import numpy as np
from scipy.special import gamma


def jacobi_p(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """
    Compute normalized Jacobi polynomial P_N^{(alpha, beta)} at x.
    
    Uses three-term recurrence relation with normalization factors.
    Assumes alpha, beta > -1.
    
    Parameters
    ----------
    x : np.ndarray
        Evaluation points (can be scalar or array)
    alpha : float
        First Jacobi parameter (alpha > -1)
    beta : float
        Second Jacobi parameter (beta > -1)
    N : int
        Polynomial degree
        
    Returns
    -------
    np.ndarray
        Normalized Jacobi polynomial evaluated at x, with shape matching input
    """
    xp = np.atleast_1d(x)
    PL = np.zeros((N + 1, len(xp)))

    # P_0(x) normalization constant
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

    # P_1(x) normalization constant
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

    # Three-term recurrence for N >= 2
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

        # Recurrence: P_n = (x - b_n) * P_{n-1} / a_n - a_{n-1} * P_{n-2} / a_n
        PL[i + 1, :] = (
            -aold * PL[i - 1, :] + (xp - bnew) * PL[i, :]
        ) / anew

        aold = anew

    return PL[N, :].reshape(np.shape(x))


def grad_jacobi_p(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    """
    Compute the derivative of the normalized Jacobi polynomial.
    
    Uses the recurrence relation:
    d/dx P_N^(alpha, beta) = sqrt(N * (N + alpha + beta + 1)) * P_{N-1}^(alpha+1, beta+1)
    
    Parameters
    ----------
    x : np.ndarray
        Evaluation points (can be scalar or array)
    alpha : float
        First Jacobi parameter
    beta : float
        Second Jacobi parameter
    N : int
        Polynomial degree
        
    Returns
    -------
    np.ndarray
        Derivative of Jacobi polynomial at x, with shape matching input
    """
    if N == 0:
        return np.zeros_like(x)
    
    # d/dx P_N^(alpha, beta) = sqrt(N * (N + alpha + beta + 1)) * P_{N-1}^(alpha+1, beta+1)
    coeff = np.sqrt(N * (N + alpha + beta + 1.0))
    
    return coeff * jacobi_p(x, alpha + 1.0, beta + 1.0, N - 1)
