"""
Reconstruction module: Differentiation operators, boundary node extraction, and modal reconstruction.
"""

from .modal_reconstruct import (
    DubinerReconstructor,
    build_differentiation_matrices,
    modal_reconstruct_at_bary_dubiner_tikhonov,
    modal_reconstruct_at_bary_upgraded
)
from .boundary import (
    build_fmask_table1,
    validate_fmask,
    build_extraction_matrix_E,
    extract_boundary_nodes_global,
    display_fmask_dataframe
)
from .operators import (
    compute_divergence,
    compute_gradient,
    compute_laplacian,
    test_divergence_gaussian
)

__all__ = [
    # Reconstruction
    "DubinerReconstructor",
    "build_differentiation_matrices",
    "modal_reconstruct_at_bary_dubiner_tikhonov",
    "modal_reconstruct_at_bary_upgraded",
    # Boundary extraction
    "build_fmask_table1",
    "validate_fmask",
    "build_extraction_matrix_E",
    "extract_boundary_nodes_global",
    "display_fmask_dataframe",
    # Operators
    "compute_divergence",
    "compute_gradient",
    "compute_laplacian",
    "test_divergence_gaussian"
]
