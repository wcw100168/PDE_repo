"""
Simplex Quadrature Rules for Triangular Elements (Table 1 & Table 2).

Provides quadrature node coordinates (r, s) and volume weights on reference triangle [-1, 1]^2.
"""

from __future__ import annotations
import numpy as np

# Table 1: High-order Simplex Quadrature rules for triangular elements
TABLE1_RAW = {
    1: [
        {"sym": "S6", "b1": 0.2113248654051871, "b2": 0.0, "ws": 0.16666666666666667},
    ],
    2: [
        {"sym": "S6", "b1": 0.1127016653792583, "b2": 0.0, "ws": 0.04166666666666666},
        {"sym": "S3", "b1": 0.5000000000000000, "b2": 0.0, "ws": 0.09999999999999999},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.45000000000000000},
    ],
    3: [
        {"sym": "S6", "b1": 0.06943184420297367, "b2": 0.0, "ws": 0.01509901487256561},
        {"sym": "S6", "b1": 0.3300094782075718, "b2": 0.0, "ws": 0.04045654068298990},
        {"sym": "S6", "b1": 0.5841571139756568, "b2": 0.1870738791912763, "ws": 0.11111111111111111},
    ],
    4: [
        {"sym": "S6", "b1": 0.04691007703066797, "b2": 0.0, "ws": 0.006601315081001592},
        {"sym": "S6", "b1": 0.2307653449471584, "b2": 0.0, "ws": 0.02053045968042892},
        {"sym": "S3", "b1": 0.5000000000000000, "b2": 0.0, "ws": 0.01853708483394990},
        {"sym": "S3", "b1": 0.1394337314154536, "b2": 0.1394337314154536, "ws": 0.10542932962084440},
        {"sym": "S3", "b1": 0.4384239524408185, "b2": 0.4384239524408185, "ws": 0.12473673228977350},
        {"sym": "S1", "b1": 0.3333333333333333, "b2": 0.3333333333333333, "ws": 0.09109991119771331},
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
