"""
Intrinsic Metric Tensors & Surface Jacobian Computation on 2-Sphere S^2.

Computes physical coordinate mappings, covariant basis vectors (a_1, a_2),
contravariant basis vectors (a^1, a^2), Jacobian determinant J, and metric tensors g_ij.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .sphere_mesh import ManifoldMesh


@dataclass(frozen=True)
class GeometryCache:
    element_vertices: np.ndarray # Shape (K, 3, 3)
    X: np.ndarray                # Shape (K, Np, 3) Physical nodes on sphere
    Xr: np.ndarray               # Shape (K, Np, 3) Covariant basis a_1 = dX/dr
    Xs: np.ndarray               # Shape (K, Np, 3) Covariant basis a_2 = dX/ds
    normal: np.ndarray           # Shape (K, Np, 3) Unit normal vector n
    sqrt_g: np.ndarray           # Shape (K, Np) Jacobian determinant J = sqrt(det(g))
    g11: np.ndarray              # Shape (K, Np) Metric tensor g_11
    g12: np.ndarray              # Shape (K, Np) Metric tensor g_12
    g22: np.ndarray              # Shape (K, Np) Metric tensor g_22
    ginv11: np.ndarray           # Shape (K, Np) Inverse metric g^11
    ginv12: np.ndarray           # Shape (K, Np) Inverse metric g^12
    ginv22: np.ndarray           # Shape (K, Np) Inverse metric g^22
    grad_r: np.ndarray           # Shape (K, Np, 3) Contravariant basis a^1
    grad_s: np.ndarray           # Shape (K, Np, 3) Contravariant basis a^2


def _reference_shape_functions(r: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi1 = -(r + s) / 2.0
    phi2 = (1.0 + r) / 2.0
    phi3 = (1.0 + s) / 2.0
    return phi1, phi2, phi3


def _sphere_project_with_derivative(Y: np.ndarray, dY: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    normY = np.linalg.norm(Y, axis=-1, keepdims=True)
    if np.any(normY <= 0.0):
        raise ValueError("Projection to sphere received zero vector.")
    X = radius * Y / normY
    Y_dot_dY = np.sum(Y * dY, axis=-1, keepdims=True)
    dX = radius * (dY / normY - Y * Y_dot_dY / (normY**3))
    return X, dX


def map_reference_to_sphere_element(rs: np.ndarray, vertices: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map reference triangle nodes rs to physical nodes X on S^2 with analytic derivatives Xr, Xs.
    """
    rs = np.asarray(rs, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    r, s = rs[:, 0], rs[:, 1]
    phi1, phi2, phi3 = _reference_shape_functions(r, s)

    v1, v2, v3 = vertices[0], vertices[1], vertices[2]
    Y = phi1[:, None] * v1[None, :] + phi2[:, None] * v2[None, :] + phi3[:, None] * v3[None, :]

    dYdr = 0.5 * (v2 - v1)
    dYds = 0.5 * (v3 - v1)

    X, Xr = _sphere_project_with_derivative(Y=Y, dY=np.broadcast_to(dYdr, Y.shape), radius=radius)
    _, Xs = _sphere_project_with_derivative(Y=Y, dY=np.broadcast_to(dYds, Y.shape), radius=radius)
    return X, Xr, Xs


def compute_geometry_cache(mesh: ManifoldMesh, r_nodes: np.ndarray, s_nodes: np.ndarray) -> GeometryCache:
    """
    Compute full intrinsic geometry cache on 2-sphere S^2 for all mesh elements.
    
    Parameters
    ----------
    mesh : ManifoldMesh
    r_nodes, s_nodes : np.ndarray
        Quadrature node coordinates on reference element (shape Np).
        
    Returns
    -------
    GeometryCache
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    elements = np.asarray(mesh.elements, dtype=int)
    element_vertices = vertices[elements]

    K = elements.shape[0]
    Np = len(r_nodes)
    rs = np.column_stack([r_nodes, s_nodes])

    X = np.zeros((K, Np, 3), dtype=float)
    Xr = np.zeros((K, Np, 3), dtype=float)
    Xs = np.zeros((K, Np, 3), dtype=float)

    for k in range(K):
        X[k], Xr[k], Xs[k] = map_reference_to_sphere_element(rs, element_vertices[k], mesh.radius)

    cross = np.cross(Xr, Xs)
    sqrt_g = np.linalg.norm(cross, axis=2)

    if np.any(sqrt_g <= 0.0):
        raise ValueError("Degenerate surface Jacobian J encountered.")

    normal = cross / sqrt_g[:, :, None]

    g11 = np.sum(Xr * Xr, axis=2)
    g12 = np.sum(Xr * Xs, axis=2)
    g22 = np.sum(Xs * Xs, axis=2)
    gdet = sqrt_g * sqrt_g

    ginv11 = g22 / gdet
    ginv12 = -g12 / gdet
    ginv22 = g11 / gdet

    grad_r = ginv11[:, :, None] * Xr + ginv12[:, :, None] * Xs
    grad_s = ginv12[:, :, None] * Xr + ginv22[:, :, None] * Xs

    return GeometryCache(
        element_vertices=element_vertices,
        X=X,
        Xr=Xr,
        Xs=Xs,
        normal=normal,
        sqrt_g=sqrt_g,
        g11=g11,
        g12=g12,
        g22=g22,
        ginv11=ginv11,
        ginv12=ginv12,
        ginv22=ginv22,
        grad_r=grad_r,
        grad_s=grad_s,
    )
