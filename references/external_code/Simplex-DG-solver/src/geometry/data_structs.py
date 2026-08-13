"""
Data structures and basic coordinate manipulation utilities.

Core dataclass for representing quadrature nodes and coordinate conversion
utilities for both 2D and 3D geometric operations.
"""

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Node:
    """
    Represents a quadrature node in the triangular domain.
    
    Attributes
    ----------
    node_id : int
        Unique identifier for the node
    barycentric : np.ndarray
        Barycentric coordinates [b1, b2, b3] in reference triangle
    weight : float
        Quadrature weight for integration
    local_coords : np.ndarray
        Local coordinates [b3, b1] used in modal basis
    global_coords : np.ndarray
        Global 3D coordinates on the octahedron face
    """
    node_id: int
    barycentric: np.ndarray
    weight: float
    local_coords: np.ndarray
    global_coords: np.ndarray


def get_orbit(b1: float, b2: float, atol: float = 1e-14):
    """
    Generate all distinct permutations of (b1, b2, b3=1-b1-b2).
    
    Filters out floating-point duplicates using tolerance.
    
    Parameters
    ----------
    b1, b2 : float
        Two barycentric coordinates
    atol : float
        Tolerance for floating-point comparison (default: 1e-14)
    
    Returns
    -------
    list of np.ndarray
        List of unique barycentric coordinate vectors
    """
    b3 = 1.0 - b1 - b2
    all_perms = list(itertools.permutations([b1, b2, b3]))
    unique_nodes = []
    for p in all_perms:
        if not any(np.allclose(p, q, atol=atol) for q in unique_nodes):
            unique_nodes.append(np.array(p, dtype=float))
    return unique_nodes


def bary_to_cartesian_2d(bary: np.ndarray, vertices_2d: np.ndarray) -> np.ndarray:
    """
    Convert barycentric coordinates to 2D Cartesian coordinates.
    
    Parameters
    ----------
    bary : np.ndarray
        Barycentric coordinates [b1, b2, b3]
    vertices_2d : np.ndarray
        2D reference triangle vertices (shape: [3, 2])
    
    Returns
    -------
    np.ndarray
        2D Cartesian coordinates
    """
    return bary @ vertices_2d


def bary_to_cartesian_3d(bary: np.ndarray, vertices_3d: np.ndarray) -> np.ndarray:
    """
    Convert barycentric coordinates to 3D Cartesian coordinates.
    
    Parameters
    ----------
    bary : np.ndarray
        Barycentric coordinates [b1, b2, b3]
    vertices_3d : np.ndarray
        3D reference triangle vertices (shape: [3, 3])
    
    Returns
    -------
    np.ndarray
        3D Cartesian coordinates
    """
    return bary @ vertices_3d
