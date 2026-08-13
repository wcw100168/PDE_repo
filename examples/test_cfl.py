import sys, pathlib
import numpy as np
workspace = pathlib.Path("/Users/user/code/數學專案測試版")
sys.path.insert(0, str(workspace))
sys.path.insert(0, str(workspace / "examples"))
import run_spherical_advection as rsa

print(">>> [Test 1] Original Long Run (dt=1000.0, 200 steps, Total Time = 200,000 s) <<<")
rsa.run_advection_demo(ndivs=4, order=3, n_steps=200, dt=1000.0)

print("\n\n>>> [Test 2] Tightened CFL (dt=200.0, 1000 steps, Total Time = 200,000 s) <<<")
rsa.run_advection_demo(ndivs=4, order=3, n_steps=1000, dt=200.0)

print("\n\n>>> [Test 3] Further Tightened CFL (dt=100.0, 2000 steps, Total Time = 200,000 s) <<<")
rsa.run_advection_demo(ndivs=4, order=3, n_steps=2000, dt=100.0)
