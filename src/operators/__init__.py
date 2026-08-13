"""
Operators package: Dubiner basis, Vandermonde matrices, Cholesky orthogonalization, and SBP operators.
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

__all__ = [
    "collapsed_coords_transform",
    "jacobi_p",
    "evaluate_dubiner_basis_2d",
    "vandermonde_2d_dubiner",
    "cholesky_orthogonalize_vandermonde",
    "compute_orthogonality_residual",
]
