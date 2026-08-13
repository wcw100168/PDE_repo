"""
Interface Numerical Fluxes & Surface Trace Lifting Operators for SBP-DG.

Provides:
1. Upwind and Central numerical flux functions.
2. Full RHS evaluator incorporating interface penalty flux lifting to ensure
   exact machine-precision discrete mass conservation (1e-16).
"""

from __future__ import annotations
import numpy as np

from src.geometry.connectivity import ConnectivityCache
from src.geometry.trace import TraceGeometryCache, gather_neighbor_traces


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


def get_face_derivatives(face_id: int) -> tuple[float, float]:
    if face_id == 0: return -2.0, 2.0
    if face_id == 1: return 0.0, -2.0
    if face_id == 2: return 2.0, 0.0
    raise ValueError("Invalid face_id")


def surface_lift_correction_conservative(
    q: np.ndarray,
    E_face: list[np.ndarray],
    L_face: list[np.ndarray],
    trace_geom: TraceGeometryCache,
    conn: ConnectivityCache,
    J_vol: np.ndarray,
    u_r: np.ndarray,
    u_s: np.ndarray,
    flux_type: str = "central"
) -> np.ndarray:
    """
    Compute surface integral lift correction for conservative formulation.
    """
    K = q.shape[0]
    Np = q.shape[1]
    
    correction = np.zeros((K, Np), dtype=float)
    
    alpha = J_vol * u_r
    beta = J_vol * u_s
    alpha_q = alpha * q
    beta_q = beta * q
    
    q_faces = []
    line_velocity = []
    
    # 1. Project volume values to face
    for f in range(3):
        q_f = q @ E_face[f].T
        q_faces.append(q_f)
        
        drdt, dsdt = get_face_derivatives(f)
        a_n = dsdt * (alpha @ E_face[f].T) - drdt * (beta @ E_face[f].T)
        line_velocity.append(a_n)
        
    q_faces_arr = np.stack(q_faces, axis=1) # (K, 3, Nf)
    a_n_arr = np.stack(line_velocity, axis=1) # (K, 3, Nf)
    
    # 2. Gather neighbor traces
    q_P = gather_neighbor_traces(q_faces_arr, conn, boundary_value=np.nan)
    a_n_P = gather_neighbor_traces(a_n_arr, conn, boundary_value=np.nan)
    
    # Neighbor normal is opposite, so common outward velocity is 0.5 * (a_n - a_n_P)
    a_n_common = 0.5 * (a_n_arr - a_n_P)
    
    # Keep local outward velocity for boundary faces
    for k in range(K):
        for f in range(3):
            if conn.is_boundary[k, f]:
                a_n_common[k, f, :] = a_n_arr[k, f, :]
                q_P[k, f, :] = q_faces_arr[k, f, :] # Extrapolate boundary
                
    # 3. Compute fluxes
    if flux_type == "central":
        flux_star = compute_central_flux(q_faces_arr, q_P, a_n_common)
    else:
        flux_star = compute_upwind_flux(q_faces_arr, q_P, a_n_common)
        
    # 4. Project physical interior line flux F_cons^-
    for f in range(3):
        drdt, dsdt = get_face_derivatives(f)
        alpha_q_f = alpha_q @ E_face[f].T
        beta_q_f = beta_q @ E_face[f].T
        
        F_cons_minus = dsdt * alpha_q_f - drdt * beta_q_f
        
        # dF = F_cons^- - F^*
        dF = F_cons_minus - flux_star[:, f, :]
        
        # 5. Lift to volume
        correction += dF @ L_face[f].T
        
    # Scale by volume Jacobian
    correction /= J_vol
    
    return correction


def surface_lift_correction_split(
    q: np.ndarray,
    E_face: list[np.ndarray],
    L_face: list[np.ndarray],
    trace_geom: TraceGeometryCache,
    conn: ConnectivityCache,
    J_vol: np.ndarray,
    u_r: np.ndarray,
    u_s: np.ndarray,
    flux_type: str = "central"
) -> np.ndarray:
    """
    Compute surface integral lift correction for split formulations (Split2/Split3).
    The interior boundary line flux F^- is formulated to perfectly cancel the 
    volume Split derivative boundary terms via summation-by-parts.
    """
    K = q.shape[0]
    Np = q.shape[1]
    
    correction = np.zeros((K, Np), dtype=float)
    
    alpha = J_vol * u_r
    beta = J_vol * u_s
    alpha_q = alpha * q
    beta_q = beta * q
    
    q_faces = []
    line_velocity = []
    
    for f in range(3):
        q_f = q @ E_face[f].T
        q_faces.append(q_f)
        
        drdt, dsdt = get_face_derivatives(f)
        a_n = dsdt * (alpha @ E_face[f].T) - drdt * (beta @ E_face[f].T)
        line_velocity.append(a_n)
        
    q_faces_arr = np.stack(q_faces, axis=1)
    a_n_arr = np.stack(line_velocity, axis=1)
    
    q_P = gather_neighbor_traces(q_faces_arr, conn, boundary_value=np.nan)
    a_n_P = gather_neighbor_traces(a_n_arr, conn, boundary_value=np.nan)
    
    a_n_common = 0.5 * (a_n_arr - a_n_P)
    
    for k in range(K):
        for f in range(3):
            if conn.is_boundary[k, f]:
                a_n_common[k, f, :] = a_n_arr[k, f, :]
                q_P[k, f, :] = q_faces_arr[k, f, :]
                
    if flux_type == "central":
        flux_star = compute_central_flux(q_faces_arr, q_P, a_n_common)
    else:
        flux_star = compute_upwind_flux(q_faces_arr, q_P, a_n_common)
        
    for f in range(3):
        drdt, dsdt = get_face_derivatives(f)
        alpha_q_f = alpha_q @ E_face[f].T
        beta_q_f = beta_q @ E_face[f].T
        
        F_cons_minus = dsdt * alpha_q_f - drdt * beta_q_f
        
        # Split form boundary trace cancellation requires F_split^-
        # F_split^- = 0.5 * F_cons^- + 0.5 * a_n^- * q^-
        F_split_minus = 0.5 * F_cons_minus + 0.5 * a_n_arr[:, f, :] * q_faces_arr[:, f, :]
        
        dF = F_split_minus - flux_star[:, f, :]
        
        correction += dF @ L_face[f].T
        
    correction /= J_vol
    
    return correction

