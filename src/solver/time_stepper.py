"""
Carpenter & Kennedy 4th-Order 5-Stage Low-Storage Runge-Kutta (LSRK45) Time Integrator.

Provides low-storage 5-stage RK time integration with high stability region.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# Carpenter & Kennedy LSRK45 coefficients
RK4A = np.array([
    0.0,
    -567301805773.0 / 1357537059087.0,
    -2404267990393.0 / 2016746695238.0,
    -3550918686646.0 / 2091501179385.0,
    -1275806237668.0 / 842570457699.0,
], dtype=float)

RK4B = np.array([
    1432997174477.0 / 9575080441755.0,
    5161836677717.0 / 13612068292357.0,
    1720146321549.0 / 2090206949498.0,
    3134564353537.0 / 4481467310338.0,
    2277821191437.0 / 14882151754819.0,
], dtype=float)

RK4C = np.array([
    0.0,
    1432997174477.0 / 9575080441755.0,
    2526269341429.0 / 6820363962896.0,
    2006345519317.0 / 3224310063776.0,
    2802321613138.0 / 2924317926251.0,
], dtype=float)


@dataclass(frozen=True)
class TimeIntegrationResult:
    q: np.ndarray
    t: float
    nsteps: int


def lsrk45_step(rhs_func, t: float, q: np.ndarray, dt: float) -> np.ndarray:
    """
    Single step of LSRK45 5-stage Runge-Kutta scheme.
    """
    q_out = np.asarray(q, dtype=float).copy()
    res = np.zeros_like(q_out)
    
    for stage in range(5):
        t_stage = float(t) + RK4C[stage] * dt
        dqdt = np.asarray(rhs_func(t_stage, q_out), dtype=float)
        
        res = RK4A[stage] * res + dt * dqdt
        q_out = q_out + RK4B[stage] * res
        
    return q_out


def integrate_lsrk45(rhs_func, q0: np.ndarray, t0: float, tf: float, dt: float) -> TimeIntegrationResult:
    """
    Integrate ODE dq/dt = rhs(t, q) from t0 to tf with fixed step size dt using LSRK45.
    """
    q = np.asarray(q0, dtype=float).copy()
    t = float(t0)
    tf = float(tf)
    nsteps = 0
    tol = 1e-14 * max(1.0, abs(tf), abs(t0))
    
    while t < tf - tol:
        dt_step = min(float(dt), tf - t)
        q = lsrk45_step(rhs_func, t, q, dt_step)
        t += dt_step
        nsteps += 1
        
    return TimeIntegrationResult(q=q, t=t, nsteps=nsteps)
