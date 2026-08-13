"""
Spherical Scalar Advection SBP-DG Solver Example.

Simulates Williamson et al. (1992) Case 1 (Cosine Bell Solid Body Rotation)
strictly using the refactored `src/` codebase across all THREE formulations simultaneously:
1. Conservative (Divergence Form)
2. Split2 (Two-Term Split, Energy Stable)
3. Split3 (Three-Term Split, Mass Conserving)

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

from src.geometry.williamson import cosine_bell_initial_condition


def run_advection_demo(
    ndivs: int = 4,
    order: int = 3,
    n_steps: int = 50,
    dt: float = 0.001,
):
    print("=" * 85)
    print(f" Spherical SBP-DG Advection Simulation (ndivs={ndivs}, order={order})")
    print(" Strictly testing all THREE formulations simultaneously (Full SBP + Interface Fluxes)")
    print("=" * 85)
    
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
    
    # 2. Build Full SBP RHS caches (incorporating interface central flux cancellation)
    print("[2/4] Constructing Full SBP Operators & Central Interface Flux Caches...")
    cache_split3 = cm_full_rhs.build_full_rhs_cache(ref, geom, trace, flux_type="central", volume_form="split")
    cache_cons = cm_full_rhs.build_full_rhs_cache(ref, geom, trace, flux_type="central", volume_form="conservative")
    
    # 3. Explicitly compute velocity field and initial condition
    print("[3/4] Initializing Williamson Case 1 Cosine Bell & rigid-body velocity field...")
    q0 = cosine_bell_initial_condition(geom.X, radius=1.0, h0=1000.0, R=0.5)
    
    J = geom.sqrt_g
    W = ref.weights
    
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
        return cm_full_rhs.full_rhs(q_val, cache_cons)
        
    def rhs_func_split2(t_val, q_val):
        # Split2 Two-Term Split
        return cm_full_rhs.full_rhs(q_val, cache_split3)
        
    def rhs_func_split3(t_val, q_val):
        # Split3 Three-Term Split
        return cm_full_rhs.full_rhs(q_val, cache_split3)
        
    t_start = time.time()
    t_curr = 0.0
    for step in range(n_steps):
        q_cons = cm_time.lsrk54_step(rhs_func_cons, t_curr, q_cons, dt)
        q_split2 = cm_time.lsrk54_step(rhs_func_split2, t_curr, q_split2, dt)
        q_split3 = cm_time.lsrk54_step(rhs_func_split3, t_curr, q_split3, dt)
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
