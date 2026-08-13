"""
Geometry package: Quadrature rules, spherical mesh generation, and intrinsic metric computation.
"""

from .quadrature import get_triangle_quadrature
from .sphere_mesh import build_octa_sphere_mesh, ManifoldMesh
from .metrics import compute_geometry_cache, GeometryCache

__all__ = [
    "get_triangle_quadrature",
    "build_octa_sphere_mesh",
    "ManifoldMesh",
    "compute_geometry_cache",
    "GeometryCache",
]
