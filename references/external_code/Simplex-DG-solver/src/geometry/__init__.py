"""
Geometry module: Coordinate systems, mappings, metrics, and data structures.
"""

from .data_structs import Node, get_orbit, bary_to_cartesian_2d, bary_to_cartesian_3d
from .mappings import (
    collapsed_coords_transform,
    rs_to_xy,
    xy_to_rs,
    AffineMap
)
from .metrics import (
    compute_geometric_factors,
    compute_geometric_factors_batch,
    MetricTensor
)

__all__ = [
    # Data structures
    "Node",
    "get_orbit",
    "bary_to_cartesian_2d",
    "bary_to_cartesian_3d",
    # Mappings
    "collapsed_coords_transform",
    "rs_to_xy",
    "xy_to_rs",
    "AffineMap",
    # Metrics
    "compute_geometric_factors",
    "compute_geometric_factors_batch",
    "MetricTensor"
]
