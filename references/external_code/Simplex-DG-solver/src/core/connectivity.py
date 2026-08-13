"""
Global connectivity matrix construction for triangular meshes.

Implements the Vertex Pairing Hash method to build Element-to-Element (EToE)
and Element-to-Face (EToF) connectivity matrices from an Element-to-Vertex (EToV)
array. Designed for Counter-Clockwise (CCW) oriented triangular elements.

Key Structures:
    EToV (Element-to-Vertex): Input array of shape (N_elements, 3) containing
        global vertex indices for each triangular element.
    
    EToE (Element-to-Element): Output array of shape (N_elements, 3) where
        EToE[k, f] = ID of the element sharing face f of element k.
        Boundary faces satisfy EToE[k, f] = k.
    
    EToF (Element-to-Face): Output array of shape (N_elements, 3) where
        EToF[k, f] = local face index in the neighboring element.
        Boundary faces satisfy EToF[k, f] = f.
"""

import numpy as np


def build_connectivity(EToV):
    """
    Build global connectivity matrices EToE and EToF using Vertex Pairing Hash method.
    
    Constructs element-to-element and element-to-face connectivity matrices from
    an element-to-vertex array. Assumes all triangular elements are strictly
    Counter-Clockwise (CCW) oriented.
    
    Parameters
    ----------
    EToV : np.ndarray
        Element-to-Vertex connectivity array with shape (N_elements, 3).
        Each row contains the global vertex indices for a triangular element.
        Vertices must be 0-indexed.
        
    Returns
    -------
    EToE : np.ndarray
        Element-to-Element connectivity array, shape (N_elements, 3).
        EToE[k, f] contains the ID of the element sharing face f of element k.
        For boundary faces, EToE[k, f] = k (element maps to itself).
        
    EToF : np.ndarray
        Element-to-Face connectivity array, shape (N_elements, 3).
        EToF[k, f] contains the local face index in the neighboring element.
        For boundary faces, EToF[k, f] = f (face maps to itself).
        
    Notes
    -----
    Local Face Convention (CCW):
        - Face 0: connects local vertices [0, 1]
        - Face 1: connects local vertices [1, 2]
        - Face 2: connects local vertices [2, 0]
    
    Algorithm:
        1. Initialize EToE and EToF with boundary defaults (element→element, face→face).
        2. Build a hash dictionary mapping each global edge to its (element_id, face_id) pairs.
        3. For interior edges (appearing in 2 elements), update cross-element references.
        4. Boundary edges remain unchanged (map to self).
        
    Time Complexity: O(N_elements) for most practical meshes.
    Space Complexity: O(N_elements) for hash storage.
    
    Examples
    --------
    >>> EToV = np.array([[0, 1, 2], [2, 1, 3]])
    >>> EToE, EToF = build_connectivity(EToV)
    >>> print(EToE)
    [[0 1 0]
     [0 1 1]]
    >>> print(EToF)
    [[0 0 2]
     [1 1 2]]
    """
    
    N_elements = EToV.shape[0]
    
    # Step 0: Initialize connectivity matrices with boundary defaults
    EToE = np.zeros((N_elements, 3), dtype=int)
    EToF = np.zeros((N_elements, 3), dtype=int)
    
    # Default: each element maps to itself, each face maps to itself
    for k in range(N_elements):
        EToE[k] = k  # Element boundary default: maps to itself
        EToF[k] = np.arange(3)  # Face boundary default: maps to itself
    
    # Define local face-to-vertex mapping (CCW convention, 0-based indexing)
    # Face f connects vertices [local_faces[f, 0], local_faces[f, 1]]
    local_faces = np.array([
        [0, 1],  # Face 0: connects local vertices 0 and 1
        [1, 2],  # Face 1: connects local vertices 1 and 2
        [2, 0],  # Face 2: connects local vertices 2 and 0
    ])
    
    # Hash dictionary: edge_key -> list of (element_id, face_id) pairs
    # edge_key is a canonical tuple of sorted global vertex indices
    edge_hash = {}
    
    # Step 1: Build vertex pairing hash
    for k in range(N_elements):
        for f in range(3):
            # Get local vertex indices for this face
            v1_local, v2_local = local_faces[f]
            
            # Convert to global vertex IDs
            v1_global = EToV[k, v1_local]
            v2_global = EToV[k, v2_local]
            
            # Create canonical edge key (sorted to ensure uniqueness)
            edge_key = tuple(sorted([v1_global, v2_global]))
            
            # Store (element_id, face_id) pair in hash
            if edge_key not in edge_hash:
                edge_hash[edge_key] = []
            edge_hash[edge_key].append((k, f))
    
    # Step 2: Process edges and build bi-directional connectivity
    for edge_key, element_face_pairs in edge_hash.items():
        if len(element_face_pairs) == 2:
            # Interior edge: shared by exactly two elements
            (k1, f1), (k2, f2) = element_face_pairs
            
            # Cross-assign element connectivity (bi-directional)
            EToE[k1, f1] = k2
            EToE[k2, f2] = k1
            
            # Cross-assign face connectivity (bi-directional)
            EToF[k1, f1] = f2
            EToF[k2, f2] = f1
            
        elif len(element_face_pairs) == 1:
            # Boundary edge: only one element
            # Already initialized to boundary defaults, no action needed
            pass
        else:
            # Malformed mesh: edge shared by 3+ elements or 0 elements
            # This should not occur in a valid 2D triangulation
            raise ValueError(
                f"Invalid mesh topology: edge {edge_key} is shared by "
                f"{len(element_face_pairs)} elements."
            )
    
    return EToE, EToF


def validate_connectivity(EToV, EToE, EToF):
    """
    Validate connectivity matrices for consistency and correctness.
    
    Performs checks to ensure:
        - Bi-directional element relationships are consistent
        - Element and face indices are within valid ranges
        - Boundary elements map to themselves
        
    Parameters
    ----------
    EToV : np.ndarray
        Element-to-Vertex array, shape (N_elements, 3)
    EToE : np.ndarray
        Element-to-Element array, shape (N_elements, 3)
    EToF : np.ndarray
        Element-to-Face array, shape (N_elements, 3)
        
    Returns
    -------
    is_valid : bool
        True if all validation checks pass.
        
    Raises
    ------
    ValueError
        If any validation check fails.
    """
    
    N_elements = EToV.shape[0]
    
    # Check array shapes
    if EToE.shape != (N_elements, 3):
        raise ValueError(f"EToE shape {EToE.shape} does not match expected {(N_elements, 3)}")
    if EToF.shape != (N_elements, 3):
        raise ValueError(f"EToF shape {EToF.shape} does not match expected {(N_elements, 3)}")
    
    # Check element indices
    if np.any(EToE < -1) or np.any(EToE >= N_elements):
        raise ValueError(f"EToE contains invalid element indices")
    
    # Check face indices
    if np.any(EToF < -1) or np.any(EToF >= 3):
        raise ValueError(f"EToF contains invalid face indices")
    
    # Check bi-directional consistency (if not boundary)
    for k in range(N_elements):
        for f in range(3):
            k_neighbor = EToE[k, f]
            f_neighbor = EToF[k, f]
            
            # If not boundary (neighbor != self)
            if k_neighbor != k:
                # Verify reverse relationship exists
                if EToE[k_neighbor, f_neighbor] != k:
                    raise ValueError(
                        f"Inconsistent connectivity: EToE[{k}, {f}] = {k_neighbor}, "
                        f"but EToE[{k_neighbor}, {f_neighbor}] = {EToE[k_neighbor, f_neighbor]}"
                    )
                if EToF[k_neighbor, f_neighbor] != f:
                    raise ValueError(
                        f"Inconsistent face mapping: EToF[{k}, {f}] = {f_neighbor}, "
                        f"but EToF[{k_neighbor}, {f_neighbor}] = {EToF[k_neighbor, f_neighbor]}"
                    )
    
    return True


if __name__ == "__main__":
    # Test case: two triangles sharing edge [1, 2]
    EToV_test = np.array([
        [0, 1, 2],  # Element 0: vertices 0, 1, 2
        [2, 1, 3],  # Element 1: vertices 2, 1, 3
    ])
    
    print("=" * 80)
    print("TEST CASE: Build Connectivity for Two Triangles Sharing Edge [1, 2]")
    print("=" * 80)
    print()
    
    print("Input EToV (Element-to-Vertex):")
    print(EToV_test)
    print()
    
    EToE, EToF = build_connectivity(EToV_test)
    
    print("Output EToE (Element-to-Element):")
    print(EToE)
    print()
    
    print("Output EToF (Element-to-Face):")
    print(EToF)
    print()
    
    # Validate
    try:
        validate_connectivity(EToV_test, EToE, EToF)
        print("✓ Connectivity validation PASSED")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    print()
