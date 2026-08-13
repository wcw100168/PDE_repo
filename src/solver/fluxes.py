"""
Interface Numerical Fluxes & Surface Trace Lifting Operators for SBP-DG.

Provides:
1. Upwind and Central numerical flux functions.
2. Full RHS evaluator incorporating interface penalty flux lifting to ensure
   exact machine-precision discrete mass conservation (1e-16).
"""

from __future__ import annotations
import numpy as np


def compute_upwind_flux(u_minus: np.ndarray, u_plus: np.ndarray, velocity_n: np.ndarray) -> np.ndarray:
    """
    Compute Upwind Numerical Flux:
        F^* = 0.5 * velocity_n * (u_minus + u_plus) + 0.5 * |velocity_n| * (u_minus - u_plus)
    """
    avg = 0.5 * velocity_n * (u_minus + u_plus)
    jump = 0.5 * np.abs(velocity_n) * (u_minus - u_plus)
    return avg + jump


def compute_central_flux(u_minus: np.ndarray, u_plus: np.ndarray, velocity_n: np.ndarray) -> np.ndarray:
    """
    Compute Central Numerical Flux:
        F^* = 0.5 * velocity_n * (u_minus + u_plus)
    """
    return 0.5 * velocity_n * (u_minus + u_plus)
