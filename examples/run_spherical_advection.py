"""
Spherical Scalar Advection SBP-DG Solver Example.

Simulates Williamson et al. (1992) Case 1 (Cosine Bell Solid Body Rotation)
strictly using the refactored `src/` codebase across all THREE formulations simultaneously:
1. Conservative (Divergence Form) via `src.solver.formulations.rhs_conservative`
2. Split2 (Two-Term Split, Energy Stable) via `src.solver.formulations.rhs_split2_twoterm`
3. Split3 (Three-Term Split, Mass Conserving) via `src.solver.formulations.rhs_split3_threeterm`

Run command:
    python3 examples/run_spherical_advection.py
"""

import sys
import pathlib
import time
import numpy as np

# Ensure workspace root is in sys.path
workspace = pathlib.Path(__file__).resolve().parents[1]
if str(workspace) not in sys.path:
    sys.path.insert(0, str(workspace))

from src.geometry.sphere_mesh import build_octa_sphere_mesh
from src.geometry.quadrature import get_triangle_quadrature, get_edge_quadrature
from src.geometry.metrics import compute_geometry_cache
from src.geometry.connectivity import compute_connectivity
from src.geometry.trace import build_trace_geometry_cache
from src.geometry.williamson import (
    cosine_bell_initial_condition,
    rigid_body_rotation_velocity,
    ROTATION_PERIOD,
)
from src.operators.basis import vandermonde_2d_dubiner
from src.operators.derivatives import grad_vandermonde_2d_dubiner, differentiation_matrices_weighted
from src.operators.sbp import compute_polynomial_projection_operator, compute_face_operators
from src.solver.formulations import (
    rhs_conservative,
    rhs_split2_twoterm,
    rhs_split3_threeterm,
)
from src.solver.fluxes import surface_lift_correction_conservative, surface_lift_correction_split
from src.solver.time_stepper import lsrk45_step


def run_advection_demo(
    ndivs: int = 4,
    order: int = 3,
    n_steps: int = 50,
    dt: float = 0.001,
):
    print("=" * 85)
    print(f" Spherical SBP-DG Advection Simulation (ndivs={ndivs}, order={order})")
    print(" Strictly testing all THREE formulations simultaneously (src/ solver)")
    print("=" * 85)
    
    # 1. Build mesh & geometry cache
    print("[1/4] Generating Subdivided Octahedral Spherical Mesh...")
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    K = mesh.elements.shape[0]
    print(f"      Total spherical triangular elements K = {K}")
    
    r_nodes, s_nodes, W = get_triangle_quadrature(order=order)
    Np = len(r_nodes)
    Nf = order + 1
    print(f"      Quadrature nodes per element Np = {Np} (Order N={order})")
    print(f"      Trace nodes per face Nf = {Nf}")
    
    geom = compute_geometry_cache(mesh, r_nodes, s_nodes)
    conn = compute_connectivity(mesh.elements)
    trace = build_trace_geometry_cache(mesh, Nf)
    
    # 2. Build SBP differentiation matrices and face operators
    print("[2/4] Constructing Dubiner basis, SBP operators, and Face Lifts...")
    V = vandermonde_2d_dubiner(r_nodes, s_nodes, order)
    Vr, Vs = grad_vandermonde_2d_dubiner(r_nodes, s_nodes, order)
    
    # We now build face extraction matrices FIRST so they can be passed for Full-SBP correction
    E_face = []
    L_face = []
    w_face_list = []
    for face_id in range(3):
        r_f, s_f, w_f = get_edge_quadrature(face_id, Nf)
        E, L = compute_face_operators(r_nodes, s_nodes, W, face_id, w_f, order)
        E_face.append(E)
        L_face.append(L)
        w_face_list.append(w_f)
        
    Dr, Ds = differentiation_matrices_weighted(V, Vr, Vs, W, E_face, w_face_list)
    P, P_c, Minv = compute_polynomial_projection_operator(V, W)
    
    # 3. Explicitly compute velocity field and initial condition
    print("[3/4] Initializing Williamson Case 1 Cosine Bell & rigid-body velocity field...")
    u_velocity = rigid_body_rotation_velocity(geom.X, alpha0=np.pi / 4.0, period=ROTATION_PERIOD)
    max_speed = float(np.max(np.linalg.norm(u_velocity, axis=-1)))
    print(f"      Velocity field u initialized (max speed = {max_speed:.6e} m/s)")
    
    q0 = cosine_bell_initial_condition(geom.X, radius=1.0, h0=1000.0, R=0.5)
    
    u_r = np.sum(u_velocity * geom.grad_r, axis=2)
    u_s = np.sum(u_velocity * geom.grad_s, axis=2)
    J = geom.sqrt_g
    
    mass0 = float(np.sum(W[None, :] * J * q0))
    energy0 = float(np.sum(W[None, :] * J * (q0 ** 2)))
    print(f"      Initial Mass M0   = {mass0:.16e}")
    print(f"      Initial Energy E0 = {energy0:.16e}")
    
    # 4. Time Stepping across all 3 formulations simultaneously
    print(f"[4/4] Running {n_steps} LSRK54 time steps (dt={dt} s) for Conservative, Split2, Split3...")
    
    q_cons = q0.copy()
    q_split2 = q0.copy()
    q_split3 = q0.copy()
    
    def rhs_func_cons(t_val, q_val):
        surf = surface_lift_correction_conservative(
            q_val, E_face, L_face, trace, conn, J, u_r, u_s, flux_type="central"
        )
        return rhs_conservative(q_val, Dr, Ds, J, u_r, u_s, surface_correction=surf)
        
    def rhs_func_split2(t_val, q_val):
        surf = surface_lift_correction_split(
            q_val, E_face, L_face, trace, conn, J, u_r, u_s, flux_type="central"
        )
        return rhs_split2_twoterm(q_val, Dr, Ds, J, u_r, u_s, surface_correction=surf)
        
    def rhs_func_split3(t_val, q_val):
        surf = surface_lift_correction_split(
            q_val, E_face, L_face, trace, conn, J, u_r, u_s, flux_type="central"
        )
        return rhs_split3_threeterm(q_val, Dr, Ds, J, u_r, u_s, surface_correction=surf)
        
    t_start = time.time()
    t_curr = 0.0
    for step in range(n_steps):
        q_cons = lsrk45_step(rhs_func_cons, t_curr, q_cons, dt)
        q_split2 = lsrk45_step(rhs_func_split2, t_curr, q_split2, dt)
        q_split3 = lsrk45_step(rhs_func_split3, t_curr, q_split3, dt)
        t_curr += dt
    t_elapsed = time.time() - t_start
    
    # Final Mass and Energy
    mass_cons, energy_cons = float(np.sum(W[None, :] * J * q_cons)), float(np.sum(W[None, :] * J * (q_cons ** 2)))
    mass_s2, energy_s2 = float(np.sum(W[None, :] * J * q_split2)), float(np.sum(W[None, :] * J * (q_split2 ** 2)))
    mass_s3, energy_s3 = float(np.sum(W[None, :] * J * q_split3)), float(np.sum(W[None, :] * J * (q_split3 ** 2)))
    
    drift_mass_cons = abs(mass_cons - mass0) / mass0
    drift_energy_cons = abs(energy_cons - energy0) / energy0
    
    drift_mass_s2 = abs(mass_s2 - mass0) / mass0
    drift_energy_s2 = abs(energy_s2 - energy0) / energy0
    
    drift_mass_s3 = abs(mass_s3 - mass0) / mass0
    drift_energy_s3 = abs(energy_s3 - energy0) / energy0
    
    # Programmatic verification status checks (machine tolerance = 1e-12)
    status_cons = "Mass Conserving" if drift_mass_cons < 1e-12 else f"Drift={drift_mass_cons:.2e}"
    status_s2 = "Energy Stable" if drift_energy_s2 < 1e-12 else f"Drift={drift_energy_s2:.2e}"
    status_s3 = "Mass Conserving" if drift_mass_s3 < 1e-12 else f"Drift={drift_mass_s3:.2e}"
    
    print("\n" + "-" * 85)
    print(f" Execution Summary (Elapsed: {t_elapsed:.3f} s)")
    print("-" * 85)
    print(" Formulation           | Relative Mass Drift | Relative Energy Drift | Verification Status")
    print("-" * 85)
    print(f" 1. Conservative       | {drift_mass_cons:.16e} | {drift_energy_cons:.16e} | {status_cons}")
    print(f" 2. Split2 (Two-Term)  | {drift_mass_s2:.16e} | {drift_energy_s2:.16e} | {status_s2}")
    print(f" 3. Split3 (Three-Term)| {drift_mass_s3:.16e} | {drift_energy_s3:.16e} | {status_s3}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    run_advection_demo()

