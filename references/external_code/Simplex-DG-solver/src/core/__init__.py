"""
Linear Triangular Grid Rendering - Core Package

A modularized Python package for modal expansion and spectral methods
applied to linear triangular grid rendering.

Modules:
--------
- data_structs: Core data structures (Node) and coordinate utilities
- generators: Node generation from quadrature rules (Table 1/2)
- modal_expansion: Monomial / non-Dubiner modal expansion helpers
- dubiner_tikhonov: Dubiner basis and Tikhonov reconstruction pipeline
- render_utils: Visualization and rendering pipeline
"""

from .data_structs import (
    Node,
    get_orbit,
    bary_to_cartesian_2d,
    bary_to_cartesian_3d,
)

from .generators import (
    BaseNodeGenerator,
    Table1NodeGenerator,
    Table2NodeGenerator,
    build_nodes,
    get_extra_bary,
)

from .modal_expansion import (
    triangle_area,
    monomial_powers,
    choose_basis_powers,
    vandermonde_2d,
    modal_reconstruct_at_bary,
)

# Backward compatibility: re-export from refactored modules
from ..numerics import (
    jacobi_p,
    grad_jacobi_p,
    tikhonov_regularization_matrix,
    apply_exponential_filter,
    dubiner_basis_index_to_order,
)

from ..bases import (
    evaluate_simplex_basis_2d,
    dubiner_basis_derivative,
    vandermonde_2d_dubiner,
    grad_vandermonde_2d_dubiner,
)

from ..geometry import collapsed_coords_transform

from ..reconstruction import (
    DubinerReconstructor,
    build_differentiation_matrices,
    modal_reconstruct_at_bary_dubiner_tikhonov,
    modal_reconstruct_at_bary_upgraded,
)

from .render_utils import (
    VERTICES_2D,
    VERTICES_3D,
    exact_solution,
    render_linear_triangular_grid,
)

__all__ = [
    # data_structs
    "Node",
    "get_orbit",
    "bary_to_cartesian_2d",
    "bary_to_cartesian_3d",
    # generators
    "BaseNodeGenerator",
    "Table1NodeGenerator",
    "Table2NodeGenerator",
    "build_nodes",
    "get_extra_bary",
    # modal_expansion
    "triangle_area",
    "monomial_powers",
    "choose_basis_powers",
    "vandermonde_2d",
    "modal_reconstruct_at_bary",
    # dubiner_tikhonov
    "collapsed_coords_transform",
    "dubiner_basis_index_to_order",
    "dubiner_basis_function",
    "vandermonde_2d_dubiner",
    "dubiner_basis_derivative",
    "grad_vandermonde_2d_dubiner",
    "build_differentiation_matrices",
    "tikhonov_regularization_matrix",
    "apply_exponential_filter",
    "modal_reconstruct_at_bary_dubiner_tikhonov",
    "modal_reconstruct_at_bary_upgraded",
    # render_utils
    "VERTICES_2D",
    "VERTICES_3D",
    "exact_solution",
    "render_linear_triangular_grid",
]

__version__ = "0.1.0"
