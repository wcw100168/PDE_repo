"""
Operators package: Dubiner basis, Vandermonde matrices, Cholesky orthogonalization,
derivative matrices, closed-form SBP operators, and Peirce subspace decomposition.
"""

from .basis import (
    collapsed_coords_transform,
    jacobi_p,
    evaluate_dubiner_basis_2d,
    vandermonde_2d_dubiner,
)
from .orthogonalization import (
    cholesky_orthogonalize_vandermonde,
    compute_orthogonality_residual,
)
from .derivatives import (
    grad_vandermonde_2d_dubiner,
    differentiation_matrices_weighted,
)
from .sbp import (
    compute_polynomial_projection_operator,
    construct_closed_form_sbp_operator,
    verify_sbp_property_residual,
)
from .peirce import (
    peirce_subspace_decomposition,
    verify_peirce_orthogonality,
)

__all__ = [
    "collapsed_coords_transform",
    "jacobi_p",
    "evaluate_dubiner_basis_2d",
    "vandermonde_2d_dubiner",
    "cholesky_orthogonalize_vandermonde",
    "compute_orthogonality_residual",
    "grad_vandermonde_2d_dubiner",
    "differentiation_matrices_weighted",
    "compute_polynomial_projection_operator",
    "construct_closed_form_sbp_operator",
    "verify_sbp_property_residual",
    "peirce_subspace_decomposition",
    "verify_peirce_orthogonality",
]
