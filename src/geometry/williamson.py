"""
Williamson et al. (1992) Spherical Test Suite: Initial Conditions & Velocity Fields.

Implements:
1. Williamson Case 1: Solid body rotation of Gaussian bell (Zonal Flow alpha_0 = 0).
2. Williamson Case 2: Solid body rotation of Gaussian bell with inclination angle alpha_0 = pi/4.
"""

from __future__ import annotations
import numpy as np

# Earth radius and rotation period
EARTH_RADIUS = 6.37122e6  # meters
ROTATION_PERIOD = 12.0 * 86400.0  # 12 days in seconds


def gaussian_bell_initial_condition(X: np.ndarray, radius: float = 1.0, sigma: float = 0.5) -> np.ndarray:
    """
    Evaluate 3D Gaussian Bell initial condition centered at (1, 0, 0)^T.
    
    Parameters
    ----------
    X : np.ndarray
        Physical node coordinates on 2-sphere S^2 of shape (K, Np, 3) or (N_nodes, 3).
    radius : float
        Sphere radius.
    sigma : float
        Gaussian bell width parameter.
        
    Returns
    -------
    q0 : np.ndarray
        Initial state array of matching shape.
    """
    X = np.asarray(X, dtype=float)
    x0 = np.array([radius, 0.0, 0.0])
    dist_sq = np.sum((X - x0)**2, axis=-1)
    return np.exp(-dist_sq / (2.0 * sigma**2))


def rigid_body_rotation_velocity(
    X: np.ndarray,
    alpha0: float = np.pi / 4.0,
    period: float = ROTATION_PERIOD,
) -> np.ndarray:
    """
    Evaluate 3D velocity field u = omega x X for rigid body rotation with inclination angle alpha0.
    
    Parameters
    ----------
    X : np.ndarray
        Physical node coordinates on S^2 of shape (..., 3).
    alpha0 : float
        Inclination angle (alpha0 = 0 for zonal flow, alpha0 = pi/4 for inclined flow).
    period : float
        Rotation period in seconds.
        
    Returns
    -------
    u_3d : np.ndarray
        3D physical velocity vector of shape (..., 3).
    """
    omega_mag = 2.0 * np.pi / period
    omega_vec = omega_mag * np.array([np.sin(alpha0), 0.0, np.cos(alpha0)])
    
    # u = omega x X
    return np.cross(omega_vec, X)
