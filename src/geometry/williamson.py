"""
Williamson et al. (1992) Spherical Test Suite: Initial Conditions & Velocity Fields.

Implements:
1. Williamson Case 1: Solid body rotation of Cosine Bell (exact Williamson et al. 1992 Case 1).
2. Gaussian Bell Test Variant: Smooth Gaussian bell initial condition.
3. Rigid body rotation velocity field u = omega x X.
"""

from __future__ import annotations
import numpy as np

# Earth radius and rotation period
EARTH_RADIUS = 6.37122e6  # meters
ROTATION_PERIOD = 12.0 * 86400.0  # 12 days in seconds


def cosine_bell_initial_condition(
    X: np.ndarray,
    radius: float = 1.0,
    h0: float = 1000.0,
    R: float = 0.5,
) -> np.ndarray:
    """
    Evaluate exact Williamson et al. (1992) Case 1 Cosine Bell initial condition.
    Centered at (1, 0, 0)^T.
    
    Parameters
    ----------
    X : np.ndarray
        Physical node coordinates on S^2 of shape (..., 3).
    radius : float
        Sphere radius.
    h0 : float
        Peak bell height.
    R : float
        Bell radius on unit sphere.
        
    Returns
    -------
    q0 : np.ndarray
        Cosine bell state array of matching shape.
    """
    X = np.asarray(X, dtype=float)
    x0 = np.array([radius, 0.0, 0.0])
    
    # Great-circle distance r = radius * arccos(X . x0 / (radius^2))
    cos_theta = np.clip(np.sum(X * x0, axis=-1) / (radius**2), -1.0, 1.0)
    dist = radius * np.arccos(cos_theta)
    
    q0 = np.zeros_like(dist)
    mask = dist < R
    q0[mask] = 0.5 * h0 * (1.0 + np.cos(np.pi * dist[mask] / R))
    return q0


def gaussian_bell_initial_condition(X: np.ndarray, radius: float = 1.0, sigma: float = 0.5) -> np.ndarray:
    """
    Evaluate 3D Gaussian Bell initial condition centered at (1, 0, 0)^T.
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
    """
    omega_mag = 2.0 * np.pi / period
    omega_vec = omega_mag * np.array([np.sin(alpha0), 0.0, np.cos(alpha0)])
    return np.cross(omega_vec, X)
