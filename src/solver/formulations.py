"""
Semi-Discrete RHS Formulations for Spherical Scalar Advection SBP-DG Solver.

Implements three formulations:
1. Conservative (Divergence Form)
2. Split2 (Two-Term Split, unconditional L2 energy stable)
3. Split3 (Three-Term Split, machine-precision mass conserving)
"""

from __future__ import annotations
import numpy as np


def rhs_conservative(q: np.ndarray, Dr: np.ndarray, Ds: np.ndarray, J: np.ndarray, u_r: np.ndarray, u_s: np.ndarray) -> np.ndarray:
    """
    Conservative (Divergence) Formulation RHS:
        dq/dt = - (1/J) * [ Dr * (J * u_r * q) + Ds * (J * u_s * q) ]
    """
    flux_r = J * u_r * q
    flux_s = J * u_s * q
    div_flux = (Dr @ flux_r.T).T + (Ds @ flux_s.T).T
    return -div_flux / J


def rhs_split2_twoterm(q: np.ndarray, Dr: np.ndarray, Ds: np.ndarray, J: np.ndarray, u_r: np.ndarray, u_s: np.ndarray) -> np.ndarray:
    """
    Split2 (Two-Term Split, Energy Stable) Formulation RHS:
        dq/dt = -0.5 * (1/J) * [ Dr*(J*u_r*q) + J*u_r*(Dr*q) + Ds*(J*u_s*q) + J*u_s*(Ds*q) ]
    """
    term_r1 = (Dr @ (J * u_r * q).T).T
    term_r2 = J * u_r * (Dr @ q.T).T
    term_s1 = (Ds @ (J * u_s * q).T).T
    term_s2 = J * u_s * (Ds @ q.T).T
    
    rhs_val = -0.5 * (term_r1 + term_r2 + term_s1 + term_s2) / J
    return rhs_val


def rhs_split3_threeterm(q: np.ndarray, Dr: np.ndarray, Ds: np.ndarray, J: np.ndarray, u_r: np.ndarray, u_s: np.ndarray) -> np.ndarray:
    """
    Split3 (Three-Term Split, Mass Conserving) Formulation RHS:
        dq/dt = -0.5 * (1/J) * [ Dr*(J*u_r*q) + J*u_r*(Dr*q) + q*Dr*(J*u_r) + Ds*(J*u_s*q) + J*u_s*(Ds*q) + q*Ds*(J*u_s) ]
    """
    div_ur = (Dr @ (J * u_r).T).T
    div_us = (Ds @ (J * u_s).T).T
    
    term_r1 = (Dr @ (J * u_r * q).T).T
    term_r2 = J * u_r * (Dr @ q.T).T
    term_r3 = q * div_ur
    
    term_s1 = (Ds @ (J * u_s * q).T).T
    term_s2 = J * u_s * (Ds @ q.T).T
    term_s3 = q * div_us
    
    rhs_val = -0.5 * (term_r1 + term_r2 + term_r3 + term_s1 + term_s2 + term_s3) / J
    return rhs_val

