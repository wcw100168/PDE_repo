"""
Mesh Connectivity Cache for Discontinuous Galerkin Solver.

Constructs Element-to-Element (EToE) and Element-to-Face (EToF) maps,
and determines if adjacent faces need orientation flipping.
"""

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


# In the reference triangle (v0, v1, v2):
# Face 0 connects v1 -> v2
# Face 1 connects v2 -> v0
# Face 2 connects v0 -> v1
LOCAL_FACE_VERTICES = np.array([
    [1, 2],
    [2, 0],
    [0, 1],
], dtype=int)


@dataclass(frozen=True)
class ConnectivityCache:
    EToE: np.ndarray          # Shape (K, 3): Element to Neighbor Element
    EToF: np.ndarray          # Shape (K, 3): Element to Neighbor Face ID
    is_boundary: np.ndarray   # Shape (K, 3): Boolean flag for boundary faces
    face_flip: np.ndarray     # Shape (K, 3): Boolean flag if neighbor face orientation is opposite


def compute_connectivity(elements: np.ndarray) -> ConnectivityCache:
    """
    Build EToE, EToF, and face orientation flip flags.
    
    Parameters
    ----------
    elements : np.ndarray
        Array of shape (K, 3) containing vertex indices for each element.
        
    Returns
    -------
    ConnectivityCache
    """
    elements = np.asarray(elements, dtype=int)
    K = elements.shape[0]
    
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("elements must have shape (K, 3)")
        
    EToE = -np.ones((K, 3), dtype=int)
    EToF = -np.ones((K, 3), dtype=int)
    is_boundary = np.ones((K, 3), dtype=bool)
    face_flip = np.zeros((K, 3), dtype=bool)
    
    # Map sorted tuple of vertex IDs to list of (element_id, face_id, v_start, v_end)
    edge_map = defaultdict(list)
    
    for k in range(K):
        elem_vids = elements[k]
        face_vids = elem_vids[LOCAL_FACE_VERTICES]
        for f in range(3):
            v1, v2 = face_vids[f]
            key = (min(v1, v2), max(v1, v2))
            edge_map[key].append((k, f, v1, v2))
            
    for key, entries in edge_map.items():
        if len(entries) == 1:
            # Boundary face
            continue
            
        if len(entries) != 2:
            raise ValueError(f"Non-manifold edge detected at vertices {key} shared by {len(entries)} elements.")
            
        (k1, f1, v1_start, v1_end), (k2, f2, v2_start, v2_end) = entries
        
        EToE[k1, f1] = k2
        EToF[k1, f1] = f2
        is_boundary[k1, f1] = False
        
        EToE[k2, f2] = k1
        EToF[k2, f2] = f1
        is_boundary[k2, f2] = False
        
        # Check relative orientation
        # If the start and end vertices are swapped between the two faces, they have compatible opposite orientations.
        # However, due to standard reference element traversal, an exact match in orientation requires the 
        # sequence of 1D trace nodes to be flipped during interpolation.
        if v1_start == v2_end and v1_end == v2_start:
            flip = True
        elif v1_start == v2_start and v1_end == v2_end:
            flip = False
        else:
            raise ValueError("Vertices matched but sequence is inconsistent.")
            
        face_flip[k1, f1] = flip
        face_flip[k2, f2] = flip
        
    return ConnectivityCache(
        EToE=EToE,
        EToF=EToF,
        is_boundary=is_boundary,
        face_flip=face_flip
    )
