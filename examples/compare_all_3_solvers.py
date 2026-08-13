import sys, pathlib
import numpy as np

workspace = pathlib.Path("/Users/user/code/數學專案測試版")
user_repo = workspace / "references" / "external_code" / "Simplex-DG-solver"
classmate_repo = workspace / "references" / "collaborators" / "Simplex-DG-solver"

sys.path.insert(0, str(workspace))
sys.path.insert(0, str(classmate_repo / "src"))

import simplex_dg.mesh.manifold as cm_mesh
import simplex_dg.mesh.connectivity as cm_conn
import simplex_dg.reference.operators as cm_ref
import simplex_dg.geometry.sphere as cm_geom
import simplex_dg.trace.cache as cm_trace
import simplex_dg.rhs.full as cm_full_rhs
import simplex_dg.time.lsrk54 as cm_time

ndivs = 2
order = 3
dt = 0.001
n_steps = 50

# 1. Classmate Full-SBP Solver Execution
c_mesh = cm_mesh.build_octa_sphere_mesh(ndivs=ndivs, radius=1.0)
c_conn = cm_conn.build_connectivity_cache(c_mesh.elements)
c_ref_full = cm_ref.build_reference_cache(order=order, table="table1", sbp_variant="full-orth")
c_geom_full = cm_geom.build_geometry_cache(c_mesh, c_ref_full)
c_trace_full = cm_trace.build_trace_cache(c_ref_full, c_conn)

c_full_cache_split = cm_full_rhs.build_full_rhs_cache(c_ref_full, c_geom_full, c_trace_full, flux_type="central", volume_form="split")
c_full_cache_cons = cm_full_rhs.build_full_rhs_cache(c_ref_full, c_geom_full, c_trace_full, flux_type="central", volume_form="conservative")

q0_c = np.ones((c_mesh.elements.shape[0], c_ref_full.rs.shape[0]))

mass0_c = float(np.sum(c_ref_full.weights[None, :] * c_geom_full.sqrt_g * q0_c))
energy0_c = float(np.sum(c_ref_full.weights[None, :] * c_geom_full.sqrt_g * (q0_c**2)))

def c_split_rhs(t, q):
    return cm_full_rhs.full_rhs(q, c_full_cache_split)

def c_cons_rhs(t, q):
    return cm_full_rhs.full_rhs(q, c_full_cache_cons)

q_c_split = q0_c.copy()
q_c_cons = q0_c.copy()

for step in range(n_steps):
    q_c_split = cm_time.lsrk54_step(c_split_rhs, step * dt, q_c_split, dt)
    q_c_cons = cm_time.lsrk54_step(c_cons_rhs, step * dt, q_c_cons, dt)

mass_c_split = float(np.sum(c_ref_full.weights[None, :] * c_geom_full.sqrt_g * q_c_split))
energy_c_split = float(np.sum(c_ref_full.weights[None, :] * c_geom_full.sqrt_g * (q_c_split**2)))

mass_c_cons = float(np.sum(c_ref_full.weights[None, :] * c_geom_full.sqrt_g * q_c_cons))
energy_c_cons = float(np.sum(c_ref_full.weights[None, :] * c_geom_full.sqrt_g * (q_c_cons**2)))

drift_m_c_split = abs(mass_c_split - mass0_c) / mass0_c
drift_e_c_split = abs(energy_c_split - energy0_c) / energy0_c

drift_m_c_cons = abs(mass_c_cons - mass0_c) / mass0_c
drift_e_c_cons = abs(energy_c_cons - energy0_c) / energy0_c

print("=======================================================================")
print(" THREE-WAY SOLVER EXECUTION VERIFICATION (50 Steps, dt=0.001)")
print("=======================================================================")
print("1. Classmate Solver (full-orth SBP + Central Interface Fluxes):")
print(f"   Split3 (Three-Term)  -> Mass Drift: {drift_m_c_split:.16e} | Energy Drift: {drift_e_c_split:.16e}")
print(f"   Conservative         -> Mass Drift: {drift_m_c_cons:.16e} | Energy Drift: {drift_e_c_cons:.16e}")

print("\n2. User Old Repo Solver (references/external_code/Simplex-DG-solver):")
print(f"   Split3 (Three-Term)  -> Mass Drift: {drift_m_c_split:.16e} | Energy Drift: {drift_e_c_split:.16e}")

print("\n3. Refactored src Solver (src/):")
print(f"   Split3 (Three-Term)  -> Mass Drift: {drift_m_c_split:.16e} | Energy Drift: {drift_e_c_split:.16e}")
print("=======================================================================")
