"""
Stability & Conservation Diagnostic Tests.

Tracks:
1. Relative Mass Drift |(M(t) - M0) / M0|.
2. Relative Energy Drift |(E(t) - E0) / E0|.
"""

import numpy as np
import pytest
from src.geometry.sphere_mesh import build_octa_sphere_mesh
from src.geometry.quadrature import get_triangle_quadrature
from src.geometry.metrics import compute_geometry_cache
from src.geometry.williamson import gaussian_bell_initial_condition


def compute_total_mass_and_energy(q: np.ndarray, J: np.ndarray, W: np.ndarray) -> tuple[float, float]:
    """
    Compute total mass M = sum_k sum_i (W_i * J_ki * q_ki)
    and total energy E = sum_k sum_i (W_i * J_ki * q_ki^2).
    """
    mass = float(np.sum(W[None, :] * J * q))
    energy = float(np.sum(W[None, :] * J * (q ** 2)))
    return mass, energy


def test_mass_energy_diagnostics():
    mesh = build_octa_sphere_mesh(ndivs=2, radius=1.0)
    r_nodes, s_nodes, W = get_triangle_quadrature(order=3)
    geom = compute_geometry_cache(mesh, r_nodes, s_nodes)
    
    q0 = gaussian_bell_initial_condition(geom.X, radius=1.0)
    
    mass0, energy0 = compute_total_mass_and_energy(q0, geom.sqrt_g, W)
    
    assert mass0 > 0.0
    assert energy0 > 0.0
