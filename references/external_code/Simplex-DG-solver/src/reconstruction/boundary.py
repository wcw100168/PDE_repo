"""
Boundary node extraction and face connectivity utilities.

Implements Fmask construction for identifying boundary nodes and patterns
for efficient global boundary data extraction in discontinuous Galerkin methods.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd


def build_fmask_table1(
    bary_coords: np.ndarray,
    tol: float = 1e-10
) -> np.ndarray:
    r"""
    Extract boundary node indices using barycentric coordinates.

    Given volume node positions in barycentric coordinates (L1, L2, L3),
    identifies which nodes lie on each face by checking which barycentric
    coordinate is near zero. Returns a table mapping each face to its boundary nodes.

    The three faces of the reference triangle are defined by:
    - **Face 1:** L₃ = 0  (edge v₁v₂)
    - **Face 2:** L₁ = 0  (edge v₂v₃)
    - **Face 3:** L₂ = 0  (edge v₃v₁)

    Nodes on each face are sorted by the remaining barycentric coordinate
    to ensure consistent, canonical ordering (essential for flux computations).

    Parameters
    ----------
    bary_coords : np.ndarray
        Barycentric coordinates of all volume nodes, shape (Np, 3).
        Each row is [L1, L2, L3] with L1 + L2 + L3 = 1.
    tol : float, optional
        Tolerance for identifying boundary nodes. A coordinate < tol is
        considered zero (default: 1e-10).

    Returns
    -------
    np.ndarray
        Fmask matrix, shape (Nfp, 3) where Nfp is the number of nodes per face.
        - Column 0: Node indices on Face 1 (L3 = 0)
        - Column 1: Node indices on Face 2 (L1 = 0)
        - Column 2: Node indices on Face 3 (L2 = 0)
        Each column is sorted by the primary remaining barycentric coordinate.

    Examples
    --------
    Extract boundary nodes for a uniform quadrature on the reference triangle:

    >>> # 6 nodes: 3 vertices + 3 edge midpoints
    >>> bary_coords = np.array([
    ...     [1, 0, 0],  # v1
    ...     [0, 1, 0],  # v2
    ...     [0, 0, 1],  # v3
    ...     [0.5, 0.5, 0],  # Edge v1-v2
    ...     [0.5, 0, 0.5],  # Edge v2-v3
    ...     [0, 0.5, 0.5],  # Edge v3-v1
    ... ], dtype=float) / 1.0
    >>> fmask = build_fmask_table1(bary_coords)
    >>> print(fmask.shape)
    (3, 3)
    >>> print(fmask[:, 0])  # Face 1 nodes
    [0 3 1]

    Notes
    -----
    This function is robust to non-uniform node distributions since it relies on
    barycentric coordinates rather than spatial coordinates. It works with any
    nodal distribution (Gauss-Lobatto, Fekete, etc.) as long as barycentric
    coordinates are provided.
    """
    bary_coords = np.asarray(bary_coords, dtype=float)

    if bary_coords.shape[1] != 3:
        raise ValueError(f"Expected 3 barycentric coordinates, got {bary_coords.shape[1]}")

    fmask = []

    # Face 1: L3 = 0 (edge v1-v2)
    # Sort by L2 coordinate for consistent ordering
    f1 = np.where(bary_coords[:, 2] < tol)[0]
    if len(f1) > 0:
        f1_sorted = f1[np.argsort(bary_coords[f1, 1])]
        fmask.append(f1_sorted)
    else:
        fmask.append(np.array([], dtype=int))

    # Face 2: L1 = 0 (edge v2-v3)
    # Sort by L3 coordinate
    f2 = np.where(bary_coords[:, 0] < tol)[0]
    if len(f2) > 0:
        f2_sorted = f2[np.argsort(bary_coords[f2, 2])]
        fmask.append(f2_sorted)
    else:
        fmask.append(np.array([], dtype=int))

    # Face 3: L2 = 0 (edge v3-v1)
    # Sort by L1 coordinate
    f3 = np.where(bary_coords[:, 1] < tol)[0]
    if len(f3) > 0:
        f3_sorted = f3[np.argsort(bary_coords[f3, 0])]
        fmask.append(f3_sorted)
    else:
        fmask.append(np.array([], dtype=int))

    # Validate all faces have same number of nodes
    nfp_values = [len(f) for f in fmask]
    if len(set(nfp_values)) > 1:
        raise ValueError(
            f"Faces have unequal numbers of nodes: {nfp_values}. "
            f"Check barycentric coordinates or tolerance."
        )

    # Stack columns (Fortran order for later global extraction)
    return np.column_stack(fmask)


def validate_fmask(
    fmask: np.ndarray,
    num_volume_nodes: int,
    verbose: bool = False
) -> Tuple[bool, str]:
    """
    Validate Fmask table for consistency.

    Checks that:
    - All node indices are unique
    - All indices are within [0, num_volume_nodes)
    - All three faces have equal numbers of nodes
    - No duplicate entries across faces

    Parameters
    ----------
    fmask : np.ndarray
        Fmask table, shape (Nfp, 3)
    num_volume_nodes : int
        Total number of volume nodes (Np)
    verbose : bool, optional
        Print detailed validation output (default: False)

    Returns
    -------
    Tuple[bool, str]
        (is_valid, message)

    Examples
    --------
    >>> fmask = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    >>> is_valid, msg = validate_fmask(fmask, num_volume_nodes=9)
    >>> print(is_valid, msg)
    True All checks passed
    """
    nfp, nfaces = fmask.shape

    # Check all three faces
    if nfaces != 3:
        return False, f"Expected 3 faces, got {nfaces}"

    all_indices = fmask.flatten()

    # Check range
    if np.any(all_indices < 0) or np.any(all_indices >= num_volume_nodes):
        out_of_range = np.where((all_indices < 0) | (all_indices >= num_volume_nodes))[0]
        return False, f"Indices out of range [0, {num_volume_nodes}): {all_indices[out_of_range]}"

    # Check for duplicates
    if len(np.unique(all_indices)) != len(all_indices):
        return False, "Duplicate node indices found"

    # Check equal node count per face
    for face in range(3):
        if len(np.unique(fmask[:, face])) != nfp:
            return False, f"Face {face} has duplicate entries"

    if verbose:
        print(f"✓ Fmask validation passed:")
        print(f"  - Nodes per face: {nfp}")
        print(f"  - Total unique nodes: {len(all_indices)}")
        print(f"  - Index range: [{np.min(all_indices)}, {np.max(all_indices)}]")

    return True, "All checks passed"


def build_extraction_matrix_E(fmask: np.ndarray, num_volume_nodes: int) -> np.ndarray:
    """
    Build binary extraction matrix E^T for mapping volume to face data.

    Constructs a sparse representation of the extraction operator:
    Q_face = E^T @ Q_volume

    where E^T ∈ {0, 1}^(Np × 3·Nfp) and exactly one entry per row is 1.

    Parameters
    ----------
    fmask : np.ndarray
        Fmask table, shape (Nfp, 3)
    num_volume_nodes : int
        Total number of volume nodes (Np)

    Returns
    -------
    np.ndarray
        Extraction matrix E^T, shape (num_volume_nodes, 3*Nfp), dtype int ∈ {0, 1}

    Examples
    --------
    >>> fmask = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    >>> E_T = build_extraction_matrix_E(fmask, num_volume_nodes=9)
    >>> print(E_T.shape)
    (9, 9)
    >>> print(E_T[0, :])  # Node 0 on Face 1
    [1 0 0 0 0 0 0 0 0]
    """
    nfp = fmask.shape[0]
    num_face_nodes = 3 * nfp
    E_T = np.zeros((num_volume_nodes, num_face_nodes), dtype=int)

    for face in range(3):
        for local_i, global_idx in enumerate(fmask[:, face]):
            col_idx = face * nfp + local_i
            E_T[int(global_idx), col_idx] = 1

    return E_T


def extract_boundary_nodes_global(
    Q: np.ndarray,
    fmask: np.ndarray,
    order: str = "F"
) -> np.ndarray:
    r"""
    Vectorized extraction of boundary node data from all elements.

    Given a global state tensor Q (shape: n_var × Np × num_elements) and an Fmask,
    extracts boundary node values for all elements in a single vectorized operation.

    This implements the vectorized pattern from global tensor extraction:
    Q_face = Q[:, fmask_flat, :] where fmask is flattened in Fortran order.

    Mathematical basis:
    - Input Q contains all element nodal data
    - Output Q_face selects only boundary nodes across all elements
    - Face values are ordered as: [Face1 for all elements, Face2 for all elements, Face3...]

    Parameters
    ----------
    Q : np.ndarray
        Global state tensor, shape (n_var, Np, num_elements)
    fmask : np.ndarray
        Fmask table, shape (Nfp, 3)
    order : str, optional
        Flattening order for face grouping. Use 'F' (Fortran) to group by face
        (recommended for flux computation). Use 'C' for row-major grouping.
        (default: 'F')

    Returns
    -------
    np.ndarray
        Boundary state tensor, shape (n_var, 3*Nfp, num_elements)

    Raises
    ------
    ValueError
        If Q.shape[1] != Fmask.shape[0] (mismatch between Np in Q and Fmask)

    Examples
    --------
    Simulate extraction from a 2-element mesh with p=2 (6 nodes per element):

    >>> Q = np.random.randn(2, 6, 2)  # 2 variables, 6 nodes, 2 elements
    >>> fmask = np.array([[0, 3], [1, 4], [2, 5]])  # 2 nodes per face
    >>> Q_face = extract_boundary_nodes_global(Q, fmask)
    >>> print(Q_face.shape)
    (2, 6, 2)

    Notes
    -----
    This is a critical high-performance function for large-scale computations.
    The vectorized indexing avoids explicit Python loops and leverages NumPy's
    optimized C backend.
    """
    Q = np.asarray(Q, dtype=float)
    fmask = np.asarray(fmask, dtype=int)

    n_var, Np, num_elements = Q.shape
    nfp = fmask.shape[0]

    if Np != fmask.shape[0]:
        raise ValueError(
            f"Mismatch: Q has {Np} nodes per element, but Fmask expects {fmask.shape[0]}"
        )

    # Flatten Fmask in specified order to determine indexing pattern
    flat_fmask = fmask.flatten(order=order)

    # Vectorized extraction: Q[:, flat_fmask, :] selects boundary nodes
    Q_face = Q[:, flat_fmask, :]

    return Q_face


def display_fmask_dataframe(
    fmask: np.ndarray,
    highlight: bool = True
) -> pd.DataFrame:
    """
    Create a pandas DataFrame representation of Fmask for inspection.

    Useful for debugging and visualization in Jupyter notebooks.

    Parameters
    ----------
    fmask : np.ndarray
        Fmask table, shape (Nfp, 3)
    highlight : bool, optional
        If True, return styled DataFrame for better visibility (default: True)

    Returns
    -------
    pd.DataFrame
        DataFrame with Fmask data, columns=['Face1', 'Face2', 'Face3'],
        index = row number (local node index)

    Examples
    --------
    >>> fmask = np.array([[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    >>> df = display_fmask_dataframe(fmask)
    >>> print(df)
    """
    df = pd.DataFrame(
        fmask,
        columns=["Face1", "Face2", "Face3"],
        index=np.arange(fmask.shape[0])
    )
    df.index.name = "LocalNode"
    return df
