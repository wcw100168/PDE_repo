"""
Simplex Quadrature Rules for Triangular Elements (Table 1 & Table 2).

Provides quadrature node coordinates (r, s) and volume weights on reference triangle [-1, 1]^2.
"""

from __future__ import annotations
import numpy as np

# Table 1: High-order Simplex Quadrature rules for triangular elements
TABLE1_RAW = {
    1: [
        {"sym": "S6", "b1": 0.2113248654051871, "b2": 0.0, "ws": 0.16666666666666667, "we": 0.5000000000000000},
    ],
    2: [
        {"sym": "S6", "b1": 0.1127016653792583, "b2": 0.0, "ws": 0.04166666666666666, "we": 0.2777777777777777},
        {"sym": "S3", "b1": 0.5000000000000000, "b2": 0.0, "ws": 0.09999999999999999, "we": 0.4444444444444444},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.45000000000000000, "we": None},
    ],
    3: [
        {"sym": "S6", "b1": 0.06943184420297367, "b2": 0.0, "ws": 0.01509901487256561, "we": 0.1739274225687269},
        {"sym": "S6", "b1": 0.3300094782075718, "b2": 0.0, "ws": 0.04045654068298990, "we": 0.3260725774312731},
        {"sym": "S6", "b1": 0.5841571139756568, "b2": 0.1870738791912763, "ws": 0.11111111111111111, "we": None},
    ],
    4: [
        {"sym": "S6", "b1": 0.04691007703066797, "b2": 0.0, "ws": 0.006601315081001592, "we": 0.1184634425280944},
        {"sym": "S6", "b1": 0.2307653449471584, "b2": 0.0, "ws": 0.02053045968042892, "we": 0.2393143352496833},
        {"sym": "S3", "b1": 0.5000000000000000, "b2": 0.0, "ws": 0.01853708483394990, "we": 0.2844444444444446},
        {"sym": "S3", "b1": 0.1394337314154536, "b2": 0.1394337314154536, "ws": 0.10542932962084440, "we": None},
        {"sym": "S3", "b1": 0.4384239524408185, "b2": 0.4384239524408185, "ws": 0.12473673228977350, "we": None},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.09109991119771331, "we": None},
    ],
}


def _expand_barycentric(sym: str, b1: float, b2: float) -> np.ndarray:
    b3 = 1.0 - b1 - b2
    if sym == "S1":
        return np.array([[b1, b2, b3]])
    elif sym == "S3":
        return np.array([[b1, b2, b3], [b2, b3, b1], [b3, b1, b2]])
    elif sym == "S6":
        return np.array([
            [b1, b2, b3], [b1, b3, b2],
            [b2, b1, b3], [b2, b3, b1],
            [b3, b1, b2], [b3, b2, b1]
        ])
    else:
        raise ValueError(f"Unknown symmetry: {sym}")


def get_triangle_quadrature(order: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get quadrature nodes (r, s) and weights W for reference triangle [-1, 1]^2.
    
    Parameters
    ----------
    order : int
        Polynomial degree order (1 to 4).
        
    Returns
    -------
    r, s : np.ndarray
        Coordinates on reference triangle.
    W : np.ndarray
        Weights array of shape (n_nodes,).
    """
    if order not in TABLE1_RAW:
        raise ValueError(f"Order {order} not found in Table 1 quadrature rules.")
        
    bary_list = []
    w_list = []
    
    for row in TABLE1_RAW[order]:
        bary = _expand_barycentric(str(row["sym"]), float(row["b1"]), float(row["b2"]))
        ws = float(row["ws"])
        bary_list.append(bary)
        w_list.append(np.full(bary.shape[0], ws))
        
    bary_all = np.vstack(bary_list)
    W = np.concatenate(w_list)
    
    # Convert barycentric (L0, L1, L2) to reference triangle (r, s) in [-1, 1]^2
    # Vertices: V0 = (-1, -1), V1 = (1, -1), V2 = (-1, 1)
    # (r, s) = L0*V0 + L1*V1 + L2*V2
    L0, L1, L2 = bary_all[:, 0], bary_all[:, 1], bary_all[:, 2]
    r = L0 * (-1.0) + L1 * (1.0) + L2 * (-1.0)
    s = L0 * (-1.0) + L1 * (-1.0) + L2 * (1.0)
    
    return r, s, W


def get_edge_quadrature(face_id: int, n_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get 1D Gauss-Legendre quadrature for a reference triangle face.
    
    Parameters
    ----------
    face_id : int
        Face ID (0, 1, or 2).
        Face 0 corresponds to edge v1 -> v2 (r + s = 0)
        Face 1 corresponds to edge v2 -> v0 (r = -1)
        Face 2 corresponds to edge v0 -> v1 (s = -1)
    n_points : int
        Number of quadrature points on the face.
        
    Returns
    -------
    r, s : np.ndarray
        Coordinates on the reference triangle edge.
    w_edge : np.ndarray
        Weights mapped to parameter t in [0, 1].
    """
    from scipy.special import roots_legendre
    
    x, w = roots_legendre(n_points)
    t01 = 0.5 * (x + 1.0)
    w_edge = 0.5 * w
    
    if face_id == 0:
        r = 1.0 - 2.0 * t01
        s = -1.0 + 2.0 * t01
    elif face_id == 1:
        r = -np.ones_like(t01)
        s = 1.0 - 2.0 * t01
    elif face_id == 2:
        r = -1.0 + 2.0 * t01
        s = -np.ones_like(t01)
    else:
        raise ValueError("face_id must be 0, 1, or 2.")
        
    return r, s, w_edge


def get_triangle_boundary_extraction(
    r_vol: np.ndarray, 
    s_vol: np.ndarray, 
    face_id: int, 
    n_face: int, 
    atol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get a direct Boolean extraction matrix for a specific face.
    
    Parameters
    ----------
    r_vol, s_vol : np.ndarray
        Volume quadrature nodes.
    face_id : int
        Face ID (0 for r+s=0, 1 for r=-1, 2 for s=-1).
    n_face : int
        Expected number of nodes on the face.
    atol : float
        Tolerance for boundary matching.
        
    Returns
    -------
    E_ext : np.ndarray
        Boolean extraction matrix of shape (n_face, n_vol).
    indices : np.ndarray
        The indices of the volume nodes that lie on the face.
    """
    n_vol = len(r_vol)
    
    if face_id == 0:
        residual = r_vol + s_vol
        target = 0.0
        # For face 0 (v1 -> v2), parameter t in [0, 1] maps to r going from 1 to -1.
        # To match edge quadrature sorting (which maps x in [-1, 1] to t in [0, 1]),
        # we can sort by r descending.
        sort_key = -r_vol
    elif face_id == 1:
        residual = r_vol
        target = -1.0
        # For face 1 (v2 -> v0), s goes from 1 to -1.
        sort_key = -s_vol
    elif face_id == 2:
        residual = s_vol
        target = -1.0
        # For face 2 (v0 -> v1), r goes from -1 to 1.
        sort_key = r_vol
    else:
        raise ValueError("face_id must be 0, 1, or 2.")
        
    mask = np.isclose(residual, target, atol=atol, rtol=0.0)
    indices = np.flatnonzero(mask)
    
    if len(indices) != n_face:
        raise ValueError(f"Face {face_id}: Expected {n_face} nodes, found {len(indices)}")
        
    # Sort indices based on the 1D parameterization direction
    order = np.argsort(sort_key[indices])
    indices = indices[order]
    
    E_ext = np.zeros((n_face, n_vol), dtype=float)
    E_ext[np.arange(n_face), indices] = 1.0
    
    return E_ext, indices

