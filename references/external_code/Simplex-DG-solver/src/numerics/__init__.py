"""
Numerics module: Orthogonal polynomials and filter operations.
"""

from .orthogonal_polys import jacobi_p, grad_jacobi_p
from .filters import (
    tikhonov_regularization_matrix,
    apply_exponential_filter,
    dubiner_basis_index_to_order
)

__all__ = [
    "jacobi_p",
    "grad_jacobi_p",
    "tikhonov_regularization_matrix",
    "apply_exponential_filter",
    "dubiner_basis_index_to_order"
]
