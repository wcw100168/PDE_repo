"""
Spherical Scalar Advection SBP-DG Solver Example.

Simulates Williamson et al. (1992) Case 1 (Cosine Bell Solid Body Rotation)
strictly using the refactored `src/` codebase:
- Split2 (Two-Term Split, Energy Stable) formulation via `src.solver.formulations.rhs_split2_twoterm`
- Split3 (Three-Term Split, Mass Conserving) formulation via `src.solver.formulations.rhs_split3_threeterm`

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
from src.geometry.quadrature import get_triangle_quadrature
from src.geometry.metrics import compute_geometry_cache
from src.geometry.williamson import (
    cosine_bell_initial_condition,
    rigid_body_rotation_velocity,
    ROTATION_PERIOD,
)
from src.operators.basis import vandermonde_2d_dubiner
from src.operators.derivatives import grad_vandermonde_2d_dubiner, differentiation_matrices_weighted
from src.operators.sbp import compute_polynomial_projection_operator
from src.solver.formulations import rhs_split2_twoterm, rhs_split3_threeterm
from src.solver.time_stepper import lsrk45_step


def run_advection_demo(
    ndivs: int = 4,
    order: int = 3,
    n_steps: int = 50,
    dt: float = 0.001,
):
    print("=" * 80)
    print(f" Spherical SBP-DG Advection Simulation (ndivs={ndivs}, order={order})")
    print(" Strictly using refactored src/ formulations (Split2 vs Split3)")
    print("=" * 80)
    
    # 1. Build mesh & geometry cache
    print("[1/4] Generating Subdivided Octahedral Spherical Mesh...")
    mesh = build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    K = mesh.elements.shape[0]
    print(f"      Total spherical triangular elements K = {K}")
    
    r_nodes, s_nodes, W = get_triangle_quadrature(order=order)
    Np = len(r_nodes)
    print(f"      Quadrature nodes per element Np = {Np} (Order N={order})")
    
    geom = compute_geometry_cache(mesh, r_nodes, s_nodes)
    
    # 2. Build SBP differentiation matrices
    print("[2/4] Constructing Dubiner basis & SBP operators...")
    V = vandermonde_2d_dubiner(r_nodes, s_nodes, order)
    Vr, Vs = grad_vandermonde_2d_dubiner(r_nodes, s_nodes, order)
    Dr_base, Ds_base = differentiation_matrices_weighted(V, Vr, Vs, W)
    
    # Construct SBP operators with exact skew-symmetric volume component
    W_inv = np.diag(1.0 / W)
    W_diag = np.diag(W)
    Dr = 0.5 * (Dr_base - W_inv @ Dr_base.T @ W_diag)
    Ds = 0.5 * (Ds_base - W_inv @ Ds_base.T @ W_diag)
    
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
    
    # 4. Time Stepping strictly using src/ Split2 and Split3 RHS formulations
    print(f"[4/4] Running {n_steps} LSRK45 time steps (dt={dt} s)...")
    
    q_split2 = q0.copy()
    q_split3 = q0.copy()
    
    def rhs_func_split2(t_val, q_val):
        return rhs_split2_twoterm(q_val, Dr, Ds, J, u_r, u_s)
        
    def rhs_func_split3(t_val, q_val):
        return rhs_split3_threeterm(q_val, Dr, Ds, J, u_r, u_s)
        
    t_start = time.time()
    t_curr = 0.0
    for step in range(n_steps):
        q_split2 = lsrk45_step(rhs_func_split2, t_curr, q_split2, dt)
        q_split3 = lsrk45_step(rhs_func_split3, t_curr, q_split3, dt)
        t_curr += dt
    t_elapsed = time.time() - t_start
    
    # Final Mass and Energy
    mass_s2, energy_s2 = float(np.sum(W[None, :] * J * q_split2)), float(np.sum(W[None, :] * J * (q_split2 ** 2)))
    mass_s3, energy_s3 = float(np.sum(W[None, :] * J * q_split3)), float(np.sum(W[None, :] * J * (q_split3 ** 2)))
    
    drift_mass_s2 = abs(mass_s2 - mass0) / mass0
    drift_energy_s2 = abs(energy_s2 - energy0) / energy0
    
    drift_mass_s3 = abs(mass_s3 - mass0) / mass0
    drift_energy_s3 = abs(energy_s3 - energy0) / energy0
    
    # Programmatic verification status checks (machine tolerance = 1e-12)
    status_s2 = "Energy Stable" if drift_energy_s2 < 1e-12 else f"Drift={drift_energy_s2:.2e}"
    status_s3 = "Mass Conserving" if drift_mass_s3 < 1e-12 else f"Drift={drift_mass_s3:.2e}"
    
    print("\n" + "-" * 80)
    print(f" Execution Summary (Elapsed: {t_elapsed:.3f} s)")
    print("-" * 80)
    print(" Formulation | Relative Mass Drift | Relative Energy Drift | Verification Status")
    print("-" * 80)
    print(f" Split2 (Two-Term)  | {drift_mass_s2:.16e} | {drift_energy_s2:.16e} | {status_s2}")
    print(f" Split3 (Three-Term)| {drift_mass_s3:.16e} | {drift_energy_s3:.16e} | {status_s3}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_advection_demo()
