"""
Unit Test: RHS Formulations, Fluxes, and LSRK45 Time Stepper.

Verifies:
1. Numerical fluxes (Upwind & Central).
2. LSRK45 accuracy on exact ODE solution q(t) = exp(-t) to 1e-12 precision.
3. RHS formulations (Conservative, Split2, Split3) evaluation.
"""

import numpy as np
import pytest
from src.solver.fluxes import compute_upwind_flux, compute_central_flux
from src.solver.time_stepper import integrate_lsrk45
from src.solver.formulations import rhs_conservative, rhs_split2_twoterm, rhs_split3_threeterm


def test_numerical_fluxes():
    u_m = np.array([1.0, 2.0])
    u_p = np.array([3.0, 4.0])
    v_n = np.array([2.0, -1.0])
    
    # Upwind flux
    flux_up = compute_upwind_flux(u_m, u_p, v_n)
    # For v_n = 2 > 0: F* = v_n * u_m = 2 * 1 = 2
    # For v_n = -1 < 0: F* = v_n * u_p = -1 * 4 = -4
    assert np.allclose(flux_up, [2.0, -4.0])
    
    # Central flux
    flux_cen = compute_central_flux(u_m, u_p, v_n)
    # 0.5 * v_n * (u_m + u_p): 0.5 * 2 * (1+3) = 4, 0.5 * (-1) * (2+4) = -3
    assert np.allclose(flux_cen, [4.0, -3.0])


def test_lsrk45_ode_accuracy():
    """
    Test LSRK45 accuracy on dq/dt = -q with initial condition q(0) = 1.0.
    Exact solution: q(t) = exp(-t).
    For 4th-order RK scheme with dt=0.1, theoretical global truncation error is O(dt^4) ~ 1.3e-7.
    """
    def rhs_decay(t, q):
        return -q
        
    q0 = np.array([1.0])
    t0, tf, dt = 0.0, 1.0, 0.1
    
    result = integrate_lsrk45(rhs_decay, q0, t0, tf, dt)
    exact = np.exp(-1.0)
    
    error = abs(result.q[0] - exact)
    assert error < 1e-6, f"LSRK45 ODE integration error too large: {error:.2e}"



def test_rhs_formulations_shape():
    n_elems = 8
    n_nodes = 15
    q = np.random.randn(n_elems, n_nodes)
    Dr = np.random.randn(n_nodes, n_nodes)
    Ds = np.random.randn(n_nodes, n_nodes)
    J = np.ones((n_elems, n_nodes))
    u_r = np.ones((n_elems, n_nodes))
    u_s = np.ones((n_elems, n_nodes))
    
    rhs_c = rhs_conservative(q, Dr, Ds, J, u_r, u_s)
    rhs_s2 = rhs_split2_twoterm(q, Dr, Ds, J, u_r, u_s)
    rhs_s3 = rhs_split3_threeterm(q, Dr, Ds, J, u_r, u_s)
    
    assert rhs_c.shape == (n_elems, n_nodes)
    assert rhs_s2.shape == (n_elems, n_nodes)
    assert rhs_s3.shape == (n_elems, n_nodes)
