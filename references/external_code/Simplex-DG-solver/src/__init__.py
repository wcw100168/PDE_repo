"""
Simplex DG Solver - Spectral Element / DG Methods Implementation

Core modules:
- bases: Simplex basis functions, Vandermonde matrices, and mesh generation
- core: Data structures, generators, connectivity matrices, validation utilities
- geometry: Coordinate mappings, metric tensors, and affine transformations
- numerics: Orthogonal polynomials, filters, and polynomial operations
- reconstruction: Differentiation matrices, boundary extraction, operators
"""

from . import bases
from . import core
from . import geometry
from . import numerics
from . import reconstruction

__all__ = ["bases", "core", "geometry", "numerics", "reconstruction"]
__version__ = "0.2.0"  # After Phase 1 extraction
