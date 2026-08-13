"""
Boundary Trace Geometry Cache for Discontinuous Galerkin Solver.

Computes geometric quantities (positions, normals, Jacobians) exactly at
the boundary nodes of each element on the sphere.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.geometry.sphere_mesh import ManifoldMesh
from src.geometry.quadrature import get_edge_quadrature
from src.geometry.metrics import map_reference_to_sphere_element
from src.geometry.connectivity import ConnectivityCache


@dataclass(frozen=True)
class TraceGeometryCache:
    X_face: np.ndarray        # Shape (K, 3, Nf, 3): Physical coordinates on face
    face_jacobian: np.ndarray # Shape (K, 3, Nf): 1D line Jacobian determinant
    face_normal: np.ndarray   # Shape (K, 3, Nf, 3): Unit normal vector of the sphere surface
    face_conormal: np.ndarray # Shape (K, 3, Nf, 3): Outward unit normal tangent to the sphere surface


def _face_direction_rs(face_id: int) -> tuple[float, float]:
    """Return dr/dt, ds/dt for a given face mapped from parameter t in [0, 1]."""
    if face_id == 0:
        return -2.0, 2.0
    elif face_id == 1:
        return 0.0, -2.0
    elif face_id == 2:
        return 2.0, 0.0
    raise ValueError("face_id must be 0, 1, or 2.")


def build_trace_geometry_cache(mesh: ManifoldMesh, n_face_points: int) -> TraceGeometryCache:
    """
    Compute geometric cache on the boundaries of all elements.
    
    Parameters
    ----------
    mesh : ManifoldMesh
    n_face_points : int
        Number of quadrature points per face.
        
    Returns
    -------
    TraceGeometryCache
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    elements = np.asarray(mesh.elements, dtype=int)
    element_vertices = vertices[elements]
    
    K = elements.shape[0]
    Nf = n_face_points
    
    X_face = np.zeros((K, 3, Nf, 3), dtype=float)
    face_jacobian = np.zeros((K, 3, Nf), dtype=float)
    face_normal = np.zeros((K, 3, Nf, 3), dtype=float)
    face_conormal = np.zeros((K, 3, Nf, 3), dtype=float)
    
    for face_id in range(3):
        # Get reference nodes for this face
        r_f, s_f, _ = get_edge_quadrature(face_id, Nf)
        rs_face = np.column_stack([r_f, s_f])
        drdt, dsdt = _face_direction_rs(face_id)
        
        for k in range(K):
            # Evaluate mapping at face nodes
            Xf, Xr_f, Xs_f = map_reference_to_sphere_element(rs_face, element_vertices[k], mesh.radius)
            
            # Tangent vector to the edge on the manifold
            Xt = drdt * Xr_f + dsdt * Xs_f
            
            # Surface normal vector
            cross = np.cross(Xr_f, Xs_f)
            norm_cross = np.linalg.norm(cross, axis=1)
            nf = cross / norm_cross[:, None]
            
            # Tangent vector normalized
            tangent_norm = np.linalg.norm(Xt, axis=1)
            tf = Xt / tangent_norm[:, None]
            
            # Outward co-normal (tangent to surface, orthogonal to edge)
            cf = np.cross(tf, nf)
            cf_norm = np.linalg.norm(cf, axis=1)
            cf = cf / cf_norm[:, None]
            
            # Line Jacobian (since X is radial projection, exact line Jacobian needs |Y x Yt| / |Y|^2)
            # For exact machine precision, we should compute the analytic line Jacobian from the affine map.
            # But the derivative of the mapping Xr_f, Xs_f is already the exact derivative on the sphere.
            # So the tangent vector length |Xt| IS the exact line Jacobian dt -> ds on the sphere!
            jf = tangent_norm
            
            X_face[k, face_id] = Xf
            face_jacobian[k, face_id] = jf
            face_normal[k, face_id] = nf
            face_conormal[k, face_id] = cf
            
    return TraceGeometryCache(
        X_face=X_face,
        face_jacobian=face_jacobian,
        face_normal=face_normal,
        face_conormal=face_conormal
    )


def gather_neighbor_traces(
    trace_val: np.ndarray,
    conn: ConnectivityCache,
    boundary_value: float = np.nan
) -> np.ndarray:
    """
    Gather neighbor trace values across element interfaces.
    
    Parameters
    ----------
    trace_val : np.ndarray
        Shape (K, 3, Nf). Local trace values on the face.
    conn : ConnectivityCache
        Connectivity map containing EToE, EToF, face_flip.
    boundary_value : float
        Value to insert at boundaries where there is no neighbor.
        
    Returns
    -------
    neighbor_trace : np.ndarray
        Shape (K, 3, Nf). Values from the neighboring element.
    """
    K, n_faces, Nf = trace_val.shape
    neighbor_trace = np.full_like(trace_val, boundary_value)
    
    for k in range(K):
        for f in range(3):
            if conn.is_boundary[k, f]:
                continue
                
            nbr_k = conn.EToE[k, f]
            nbr_f = conn.EToF[k, f]
            
            nbr_val = trace_val[nbr_k, nbr_f, :]
            
            if conn.face_flip[k, f]:
                nbr_val = nbr_val[::-1]
                
            neighbor_trace[k, f, :] = nbr_val
            
    return neighbor_trace

