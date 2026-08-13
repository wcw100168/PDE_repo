"""
Spherical Scalar Advection SBP-DG Solver Example.

Simulates Williamson et al. (1992) Case 1 (Cosine Bell / Gaussian Bell Solid Body Rotation)
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
classmate_repo = workspace / "references" / "collaborators" / "Simplex-DG-solver"
if str(workspace) not in sys.path:
    sys.path.insert(0, str(workspace))
if str(classmate_repo / "src") not in sys.path:
    sys.path.insert(0, str(classmate_repo / "src"))

import simplex_dg.mesh.manifold as cm_mesh
import simplex_dg.mesh.connectivity as cm_conn
import simplex_dg.reference.operators as cm_ref
import simplex_dg.geometry.sphere as cm_geom
import simplex_dg.trace.cache as cm_trace
import simplex_dg.rhs.full as cm_full_rhs
import simplex_dg.time.lsrk54 as cm_time

from src.geometry.williamson import (
    cosine_bell_initial_condition,
    gaussian_bell_initial_condition,
    rigid_body_rotation_velocity,
    ROTATION_PERIOD,
)


def run_advection_demo(
    ndivs: int = 4,
    order: int = 3,
    n_steps: int = 50,
    dt: float = 0.001,
    ic_type: str = "cosine_bell",
):
    print("=" * 80)
    print(f" Spherical SBP-DG Advection Simulation Demo (ndivs={ndivs}, order={order}, ic={ic_type})")
    print("=" * 80)
    
    # 1. Build mesh & geometry cache
    print("[1/4] Generating Subdivided Octahedral Spherical Mesh...")
    mesh = cm_mesh.build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
    conn = cm_conn.build_connectivity_cache(mesh.elements)
    K = mesh.elements.shape[0]
    print(f"      Total spherical triangular elements K = {K}")
    
    ref = cm_ref.build_reference_cache(order=order, table="table1", sbp_variant="full-orth")
    Np = ref.rs.shape[0]
    print(f"      Quadrature nodes per element Np = {Np} (Order N={order})")
    
    geom = cm_geom.build_geometry_cache(mesh, ref)
    trace = cm_trace.build_trace_cache(ref, conn)
    
    # 2. Build SBP RHS caches
    print("[2/4] Constructing Full SBP Operators & Central Interface Flux Caches...")
    cache_split = cm_full_rhs.build_full_rhs_cache(ref, geom, trace, flux_type="central", volume_form="split")
    cache_cons = cm_full_rhs.build_full_rhs_cache(ref, geom, trace, flux_type="central", volume_form="conservative")
    
    # 3. Explicitly compute velocity field and initial condition
    print("[3/4] Explicitly initializing velocity field and initial tracer distribution...")
    u_velocity = rigid_body_rotation_velocity(geom.X, alpha0=np.pi / 4.0, period=ROTATION_PERIOD)
    max_speed = float(np.max(np.linalg.norm(u_velocity, axis=-1)))
    print(f"      Solid-body rotation velocity field u initialized (max speed = {max_speed:.6e} m/s)")
    
    if ic_type == "cosine_bell":
        q0 = cosine_bell_initial_condition(geom.X, radius=1.0, h0=1000.0, R=0.5)
        print("      Tracer IC: Williamson et al. (1992) Case 1 Cosine Bell")
    else:
        q0 = gaussian_bell_initial_condition(geom.X, radius=1.0, sigma=0.5)
        print("      Tracer IC: Smooth Gaussian Bell")
        
    J = geom.sqrt_g
    W = ref.weights
    
    mass0 = float(np.sum(W[None, :] * J * q0))
    energy0 = float(np.sum(W[None, :] * J * (q0 ** 2)))
    print(f"      Initial Mass M0   = {mass0:.16e}")
    print(f"      Initial Energy E0 = {energy0:.16e}")
    
    # 4. Time Stepping with Split2 & Split3 using LSRK54
    print(f"[4/4] Running {n_steps} LSRK54 time steps (dt={dt} s)...")
    
    q_split = q0.copy()
    q_cons = q0.copy()
    
    def rhs_split(t_val, q_val):
        return cm_full_rhs.full_rhs(q_val, cache_split)
        
    def rhs_cons(t_val, q_val):
        return cm_full_rhs.full_rhs(q_val, cache_cons)
        
    t_start = time.time()
    t_curr = 0.0
    for step in range(n_steps):
        q_split = cm_time.lsrk54_step(rhs_split, t_curr, q_split, dt)
        q_cons = cm_time.lsrk54_step(rhs_cons, t_curr, q_cons, dt)
        t_curr += dt
    t_elapsed = time.time() - t_start
    
    # Final Mass and Energy
    mass_split, energy_split = float(np.sum(W[None, :] * J * q_split)), float(np.sum(W[None, :] * J * (q_split ** 2)))
    mass_cons, energy_cons = float(np.sum(W[None, :] * J * q_cons)), float(np.sum(W[None, :] * J * (q_cons ** 2)))
    
    drift_mass_split = abs(mass_split - mass0) / mass0
    drift_energy_split = abs(energy_split - energy0) / energy0
    
    drift_mass_cons = abs(mass_cons - mass0) / mass0
    drift_energy_cons = abs(energy_cons - energy0) / energy0
    
    # Programmatic verification status checks (machine tolerance = 1e-12)
    status_split = "Energy Stable" if drift_energy_split < 1e-12 else f"Drift={drift_energy_split:.2e}"
    status_cons = "Mass Conserving" if drift_mass_cons < 1e-12 else f"Drift={drift_mass_cons:.2e}"
    
    print("\n" + "-" * 80)
    print(f" Execution Summary (Elapsed: {t_elapsed:.3f} s)")
    print("-" * 80)
    print(" Formulation | Relative Mass Drift | Relative Energy Drift | Verification Status")
    print("-" * 80)
    print(f" Split2 (Two-Term)  | {drift_mass_split:.16e} | {drift_energy_split:.16e} | {status_split}")
    print(f" Split3 (Three-Term)| {drift_mass_cons:.16e} | {drift_energy_cons:.16e} | {status_cons}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_advection_demo(ic_type="cosine_bell")
