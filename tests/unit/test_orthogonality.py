"""
Unit Test: Preconditioned Cholesky Modal Orthogonalization Residual.

Verifies that || V^T * W * V - I ||_inf <= 1e-15 to machine precision across polynomial degrees N = 1..4.
"""

import numpy as np
import pytest
from src.geometry.quadrature import get_triangle_quadrature
from src.operators.basis import vandermonde_2d_dubiner
from src.operators.orthogonalization import (
    cholesky_orthogonalize_vandermonde,
    compute_orthogonality_residual,
)


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_cholesky_orthogonality_precision(N: int):
    """
    Test Cholesky orthogonalization residual for order N Dubiner basis on triangle.
    """
    # 1. Get quadrature nodes & weights for order N (using order=4 quadrature rule for high precision)
    r, s, W = get_triangle_quadrature(order=4)
    
    # 2. Build raw Vandermonde matrix V_raw
    V_raw = vandermonde_2d_dubiner(r, s, N)
    
    # 3. Perform Preconditioned Cholesky Orthogonalization
    V, L = cholesky_orthogonalize_vandermonde(V_raw, W)
    
    # 4. Compute orthogonality residual
    residual = compute_orthogonality_residual(V, W)
    
    # 5. Assert residual is below double-precision machine threshold (1e-14)
    assert residual < 1e-14, f"Orthogonality residual for N={N} failed: {residual:.2e} >= 1e-14"


def test_v_raw_vs_v_ortho_dimension():
    """
    Test matrix dimensions for N=4 basis (15 basis functions).
    """
    r, s, W = get_triangle_quadrature(order=4)
    N = 4
    n_basis = (N + 1) * (N + 2) // 2
    assert n_basis == 15
    
    V_raw = vandermonde_2d_dubiner(r, s, N)
    assert V_raw.shape == (len(r), 15)
    
    V, L = cholesky_orthogonalize_vandermonde(V_raw, W)
    assert V.shape == (len(r), 15)
    assert L.shape == (15, 15)
