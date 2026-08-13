"""
Spherical Scalar Advection SBP-DG Solver Example.

Simulates Williamson et al. (1992) Case 1 (Zonal Flow) rigid body rotation
on Subdivided Octahedral Spherical Mesh using Split2 (Energy Stable) and Split3 (Mass Conserving) formulations.

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
from src.geometry.williamson import gaussian_bell_initial_condition, rigid_body_rotation_velocity, ROTATION_PERIOD
from src.operators.basis import vandermonde_2d_dubiner
from src.operators.derivatives import grad_vandermonde_2d_dubiner, differentiation_matrices_weighted
from src.solver.formulations import rhs_split2_twoterm, rhs_split3_threeterm
from src.solver.time_stepper import lsrk45_step


def run_advection_demo(ndivs: int = 4, order: int = 3, n_steps: int = 50):
    print("=" * 70)
    print(f" Spherical SBP-DG Advection Simulation Demo (ndivs={ndivs}, order={order})")
    print("=" * 70)
    
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
    Dr, Ds = differentiation_matrices_weighted(V, Vr, Vs, W)
    
    # 3. Compute initial condition and velocity fields
    print("[3/4] Setting initial Gaussian bell and rigid body velocity field...")
    q0 = gaussian_bell_initial_condition(geom.X, radius=1.0, sigma=0.5)
    
    # Compute contravariant velocities u^r = u . grad_r, u^s = u . grad_s
    u_3d = rigid_body_rotation_velocity(geom.X, alpha0=np.pi/4.0, period=ROTATION_PERIOD)
    u_r = np.sum(u_3d * geom.grad_r, axis=2)
    u_s = np.sum(u_3d * geom.grad_s, axis=2)
    
    J = geom.sqrt_g
    
    # Initial Mass and Energy
    mass0 = float(np.sum(W[None, :] * J * q0))
    energy0 = float(np.sum(W[None, :] * J * (q0 ** 2)))
    print(f"      Initial Mass M0   = {mass0:.12e}")
    print(f"      Initial Energy E0 = {energy0:.12e}")
    
    # 4. Time Stepping with Split2 & Split3
    print(f"[4/4] Running {n_steps} LSRK45 time steps...")
    
    dt = 0.001
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
    
    print("\n" + "-" * 70)
    print(f" Execution Summary (Elapsed: {t_elapsed:.3f} s)")
    print("-" * 70)
    print(" Formulation | Relative Mass Drift | Relative Energy Drift | Status")
    print("-" * 70)
    print(f" Split2 (Two-Term)  | {drift_mass_s2:.6e}    | {drift_energy_s2:.6e}     | Energy Stable")
    print(f" Split3 (Three-Term)| {drift_mass_s3:.6e}    | {drift_energy_s3:.6e}     | Mass Conserving")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_advection_demo()
