"""
Unit Test: Spherical Geometry Mesh & Intrinsic Metric Tensors.

Verifies:
1. Octahedral sphere mesh topology and element counts (K = 8 * ndivs^2).
2. Radius validation (all nodes lie on S^2).
3. Positive Jacobian J > 0 everywhere.
4. Unit normal vector ||n|| = 1.0.
5. Dual basis orthogonality: grad_r . a_1 = 1, grad_r . a_2 = 0, grad_s . a_1 = 0, grad_s . a_2 = 1.
"""

import numpy as np
import pytest
from src.geometry.quadrature import get_triangle_quadrature
from src.geometry.sphere_mesh import build_octa_sphere_mesh
from src.geometry.metrics import compute_geometry_cache


@pytest.mark.parametrize("ndivs", [1, 2, 4])
def test_octa_sphere_mesh_topology(ndivs: int):
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    expected_elements = 8 * (ndivs ** 2)
    assert mesh.elements.shape[0] == expected_elements
    assert mesh.elements.shape[1] == 3
    
    # Check all vertices lie on sphere radius 1.0
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-12)


def test_intrinsic_geometry_metrics():
    ndivs = 4
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    r_nodes, s_nodes, W = get_triangle_quadrature(order=4)
    
    geom = compute_geometry_cache(mesh, r_nodes, s_nodes)
    
    # 1. Check Jacobian positivity J > 0 everywhere
    assert np.all(geom.sqrt_g > 0.0)
    
    # 2. Check volume nodes lie on unit sphere
    node_radii = np.linalg.norm(geom.X, axis=2)
    assert np.allclose(node_radii, 1.0, atol=1e-12)
    
    # 3. Check unit normal vector ||n|| = 1.0
    normal_norms = np.linalg.norm(geom.normal, axis=2)
    assert np.allclose(normal_norms, 1.0, atol=1e-12)
    
    # 4. Check dual basis orthogonality
    rr = np.sum(geom.grad_r * geom.Xr, axis=2) # grad_r . a_1 == 1
    rs = np.sum(geom.grad_r * geom.Xs, axis=2) # grad_r . a_2 == 0
    sr = np.sum(geom.grad_s * geom.Xr, axis=2) # grad_s . a_1 == 0
    ss = np.sum(geom.grad_s * geom.Xs, axis=2) # grad_s . a_2 == 1
    
    assert np.max(np.abs(rr - 1.0)) < 1e-12, f"grad_r . a_1 residual: {np.max(np.abs(rr - 1.0)):.2e}"
    assert np.max(np.abs(rs)) < 1e-12, f"grad_r . a_2 residual: {np.max(np.abs(rs)):.2e}"
    assert np.max(np.abs(sr)) < 1e-12, f"grad_s . a_1 residual: {np.max(np.abs(sr)):.2e}"
    assert np.max(np.abs(ss - 1.0)) < 1e-12, f"grad_s . a_2 residual: {np.max(np.abs(ss - 1.0)):.2e}"
