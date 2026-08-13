"""
Unit Test: Closed-Form SBP Operator Property & Peirce Subspace Decomposition.

Verifies:
1. SBP property residual || W * D_new + D_new^T * W - B ||_inf < 1e-12.
2. Projection idempotency P^2 = P and complement orthogonality P * Q = 0.
3. Peirce subspace decomposition completeness A_PP + A_PQ + A_QP + A_QQ == A.
"""

import numpy as np
import pytest
from src.geometry.quadrature import get_triangle_quadrature
from src.operators.basis import vandermonde_2d_dubiner
from src.operators.derivatives import grad_vandermonde_2d_dubiner, differentiation_matrices_weighted
from src.operators.sbp import (
    compute_polynomial_projection_operator,
    construct_closed_form_sbp_operator,
    verify_sbp_property_residual,
)
from src.operators.peirce import (
    peirce_subspace_decomposition,
    verify_peirce_orthogonality,
)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_closed_form_sbp_operator(order: int):
    r, s, W = get_triangle_quadrature(order=order)
    V = vandermonde_2d_dubiner(r, s, order)
    Vr, Vs = grad_vandermonde_2d_dubiner(r, s, order)
    
    Dr, Ds = differentiation_matrices_weighted(V, Vr, Vs, W)
    P = compute_polynomial_projection_operator(V, W)
    
    # Construct synthetic symmetric boundary matrix B
    n_points = len(r)
    B_sym = np.diag(np.linspace(0.1, 1.0, n_points))
    
    # Construct SBP operator
    Dr_sbp = construct_closed_form_sbp_operator(Dr, P, W, B_sym)
    
    # Verify SBP property residual || W * D + D^T * W - B ||_inf
    residual = verify_sbp_property_residual(Dr_sbp, W, B_sym)
    assert residual < 1e-12, f"SBP residual for order {order} failed: {residual:.2e} >= 1e-12"


def test_peirce_subspace_decomposition():
    order = 3
    r, s, W = get_triangle_quadrature(order=order)
    V = vandermonde_2d_dubiner(r, s, order)
    
    P = compute_polynomial_projection_operator(V, W)
    
    # 1. Verify projection idempotency & orthogonality
    ortho_res = verify_peirce_orthogonality(P)
    assert ortho_res < 1e-12, f"Peirce orthogonality residual failed: {ortho_res:.2e}"
    
    # 2. Verify matrix decomposition completeness
    n_points = len(r)
    A_rand = np.random.randn(n_points, n_points)
    
    A_PP, A_PQ, A_QP, A_QQ = peirce_subspace_decomposition(A_rand, P)
    A_reconstructed = A_PP + A_PQ + A_QP + A_QQ
    
    recon_res = np.max(np.abs(A_reconstructed - A_rand))
    assert recon_res < 1e-12, f"Peirce reconstruction failed: {recon_res:.2e}"
