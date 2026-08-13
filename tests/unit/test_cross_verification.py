"""
Cross-Verification Unit Test: Dual-Verification of User Code, Classmate Code, and Refactored Src.

Checks algebraic equivalence and numerical precision differences across all three implementations.
"""

import sys
import pathlib
import importlib.util
import numpy as np
import pytest

root_dir = pathlib.Path(__file__).resolve().parents[2]
user_src = root_dir / "references" / "external_code" / "Simplex-DG-solver" / "src"
classmate_src = root_dir / "references" / "collaborators" / "Simplex-DG-solver" / "src"

from src.geometry.quadrature import get_triangle_quadrature
from src.operators.basis import vandermonde_2d_dubiner as src_vandermonde
from src.operators.orthogonalization import cholesky_orthogonalize_vandermonde as src_cholesky


def load_module_from_path(name, path, is_pkg=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if is_pkg else None
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(classmate_src))
import simplex_dg.reference.basis as classmate_vdm
import simplex_dg.reference.indexing as classmate_idx

sys.path.insert(0, str(user_src))
load_module_from_path("user_dg", user_src / "__init__.py", is_pkg=True)
load_module_from_path("user_dg.numerics", user_src / "numerics" / "__init__.py", is_pkg=True)
load_module_from_path("user_dg.numerics.orthogonal_polys", user_src / "numerics" / "orthogonal_polys.py")
load_module_from_path("user_dg.geometry", user_src / "geometry" / "__init__.py", is_pkg=True)
load_module_from_path("user_dg.geometry.mappings", user_src / "geometry" / "mappings.py")
load_module_from_path("user_dg.bases", user_src / "bases" / "__init__.py", is_pkg=True)
load_module_from_path("user_dg.bases.simplex_2d", user_src / "bases" / "simplex_2d.py")
user_vdm = load_module_from_path("user_dg.bases.vandermonde", user_src / "bases" / "vandermonde.py")


def get_user_mode_indices(N: int) -> list[tuple[int, int]]:
    modes = []
    for i in range(N + 1):
        for j in range(N - i + 1):
            modes.append((i, j))
    return modes


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_vandermonde_raw_cross_verification(order: int):
    """
    Verify raw Vandermonde matrix algebraic equivalence across user, classmate, and refactored src.
    """
    r, s, W = get_triangle_quadrature(order=order)
    
    # 1. Refactored src
    V_src = src_vandermonde(r, s, order)
    
    # 2. User original code
    V_user = user_vdm.vandermonde_2d_dubiner(r, s, order)
    
    # 3. Classmate code
    V_classmate_raw = classmate_vdm.vandermonde2d(order, r, s)
    
    # Permute classmate's columns to match user/src mode ordering (i, j)
    classmate_modes = classmate_idx.mode_indices_2d(order)
    user_modes = get_user_mode_indices(order)
    
    classmate_perm = [classmate_modes.index(m) for m in user_modes]
    V_classmate = V_classmate_raw[:, classmate_perm]
    
    # Assert pairwise differences are below double-precision threshold (1e-14)
    diff_src_user = np.max(np.abs(V_src - V_user))
    diff_src_classmate = np.max(np.abs(V_src - V_classmate))
    diff_user_classmate = np.max(np.abs(V_user - V_classmate))
    
    assert diff_src_user < 1e-14, f"Src vs User mismatch for order {order}: {diff_src_user:.2e}"
    assert diff_src_classmate < 1e-14, f"Src vs Classmate mismatch for order {order}: {diff_src_classmate:.2e}"
    assert diff_user_classmate < 1e-14, f"User vs Classmate mismatch for order {order}: {diff_user_classmate:.2e}"


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_cholesky_orthogonalization_cross_verification(order: int):
    """
    Verify Cholesky orthogonalization yields identity mass matrix to 1e-14 for all three implementations.
    """
    r, s, W = get_triangle_quadrature(order=order)
    W_diag = np.diag(W)
    
    V_src = src_vandermonde(r, s, order)
    V_ortho_src, _ = src_cholesky(V_src, W)
    
    V_user = user_vdm.vandermonde_2d_dubiner(r, s, order)
    V_ortho_user, _ = src_cholesky(V_user, W)
    
    V_classmate = classmate_vdm.vandermonde2d(order, r, s)
    V_ortho_classmate, _ = src_cholesky(V_classmate, W)
    
    # Check orthogonality residual
    I_expected = np.eye(V_src.shape[1])
    res_src = np.max(np.abs(V_ortho_src.T @ W_diag @ V_ortho_src - I_expected))
    res_user = np.max(np.abs(V_ortho_user.T @ W_diag @ V_ortho_user - I_expected))
    res_classmate = np.max(np.abs(V_ortho_classmate.T @ W_diag @ V_ortho_classmate - I_expected))
    
    assert res_src < 1e-14, f"Src ortho residual: {res_src:.2e}"
    assert res_user < 1e-14, f"User ortho residual: {res_user:.2e}"
    assert res_classmate < 1e-14, f"Classmate ortho residual: {res_classmate:.2e}"
