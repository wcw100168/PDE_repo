"""
Node generator classes for quadrature rules (Table 1 and Table 2).
"""

import numpy as np

from .data_structs import Node, get_orbit, bary_to_cartesian_2d, bary_to_cartesian_3d


class BaseNodeGenerator:
    """
    Abstract base class for node generation strategies.

    Parameters:
    -----------
    k : int
        Polynomial degree / basis parameter
    vertices_2d : np.ndarray
        2D reference triangle vertices (shape: [3, 2])
    vertices_3d : np.ndarray
        3D reference triangle vertices (shape: [3, 3])
    """

    def __init__(self, k: int, vertices_2d: np.ndarray, vertices_3d: np.ndarray):
        self.k = int(k)
        self.vertices_2d = np.asarray(vertices_2d, dtype=float)
        self.vertices_3d = np.asarray(vertices_3d, dtype=float)

    def generate(self):
        """Generate nodes. Must be implemented by subclasses."""
        raise NotImplementedError


class Table1NodeGenerator(BaseNodeGenerator):
    """
    Generate quadrature nodes based on Table 1 (exact integration rules).

    Supports polynomial degrees k=1,2,3,4.
    """

    QUADRATURE_RULES = {
        1: [
            (0.2113248654051871, 0.0000000000000000, 0.16666666666666667)
        ],
        2: [
            (0.1127016653792583, 0.0000000000000000, 0.04166666666666666),
            (0.5000000000000000, 0.0000000000000000, 0.09999999999999999),
            (0.3333333333333333, 0.3333333333333333, 0.45000000000000000)
        ],
        3: [
            (0.06943184420297367, 0.0000000000000000, 0.01509901487256561),
            (0.3300094782075718, 0.0000000000000000, 0.04045654068298990),
            (0.5841571139756568, 0.1870738791912763, 0.11111111111111111)
        ],
        4: [
            (0.04691007703066797, 0.0000000000000000, 0.006601315081001592),
            (0.2307653449471584, 0.0000000000000000, 0.02053045968042892),
            (0.5000000000000000, 0.0000000000000000, 0.01853708483394990),
            (0.1394337314154536, 0.1394337314154536, 0.10542932962084440),
            (0.4384239524408185, 0.4384239524408185, 0.12473673228977350),
            (0.3333333333333333, 0.3333333333333333, 0.09109991119771331)
        ]
    }

    def generate(self):
        """
        Generate nodes from Table 1 data.

        Returns:
        --------
        list of Node
            List of generated quadrature nodes
        """
        if self.k not in self.QUADRATURE_RULES:
            raise ValueError(f"Table 1 data for k={self.k} is not provided.")

        nodes = []
        node_id = 1

        for b1, b2, wt in self.QUADRATURE_RULES[self.k]:
            for bary in get_orbit(b1, b2):
                bary_mapped = np.array(bary, dtype=float)
                local = np.array([bary_mapped[2], bary_mapped[0]], dtype=float)
                g3 = bary_to_cartesian_3d(bary_mapped, self.vertices_3d)
                nodes.append(Node(node_id, bary_mapped, float(wt), local, g3))
                node_id += 1

        return nodes


class Table2NodeGenerator(BaseNodeGenerator):
    """
    Generate quadrature nodes based on Table 2 (interior nodal rules).

    Supports polynomial degrees k=1,2,3,4.
    """

    QUADRATURE_RULES = {
        1: [(0.1666666666666666, 0.1666666666666666, 0.3333333333333333)],
        2: [
            (0.09157621350977067, 0.09157621350977067, 0.1099517436553218),
            (0.4459484909159648, 0.4459484909159648, 0.2233815896780115),
        ],
        3: [
            (0.219429982549783, 0.219429982549783, 0.1713331241529809),
            (0.480137964112215, 0.480137964112215, 0.08073108959303095),
            (0.1416190159239682, 0.0193717243612408, 0.04063455979366068),
        ],
        4: [
            (0.7284923929554044, 0.2631128296346379, 0.02723031417443505),
            (0.4592925882927232, 0.4592925882927232, 0.09509163426728455),
            (0.1705693077517602, 0.1705693077517602, 0.1032173705347182),
            (0.05054722831703096, 0.05054722831703096, 0.03245849762319804),
            (0.3333333333333333, 0.3333333333333333, 0.1443156076777874),
        ],
    }

    @staticmethod
    def octa_face_map(u: float, v: float, w: float):
        """
        Map from normalized octahedron coordinates to [-1, 1]^3.

        Parameters:
        -----------
        u, v, w : float
            Normalized octahedron face coordinates

        Returns:
        --------
        np.ndarray
            Mapped coordinates x, y, z in [-1, 1]
        """
        den = (1.0 - u + v + w)
        eps = 1e-12
        if abs(den) < eps:
            den = eps if den >= 0 else -eps
        x = 2.0 * (1.0 + u) / den - 1.0
        y = 2.0 * (1.0 + v) / den - 1.0
        z = 2.0 * (1.0 + w) / den - 1.0
        return np.array([x, y, z], dtype=float)

    @staticmethod
    def mapped_to_bary(mapped_xyz: np.ndarray):
        """
        Convert mapped coordinates back to barycentric coordinates.

        Parameters:
        -----------
        mapped_xyz : np.ndarray
            Coordinates in [-1, 1]^3

        Returns:
        --------
        np.ndarray
            Normalized barycentric coordinates
        """
        xyz01 = (mapped_xyz + 1.0) / 2.0
        xyz01 = np.clip(xyz01, 1e-12, None)
        return xyz01 / np.sum(xyz01)

    def generate(self):
        """
        Generate nodes from Table 2 data.

        Returns:
        --------
        list of Node
            List of generated quadrature nodes
        """
        if self.k not in self.QUADRATURE_RULES:
            raise ValueError(f"Table 2 data for k={self.k} is not provided.")

        nodes = []
        node_id = 1

        for b1, b2, wt in self.QUADRATURE_RULES[self.k]:
            for bary in get_orbit(b1, b2):
                bary_mapped = np.array(bary, dtype=float)
                local = np.array([bary_mapped[2], bary_mapped[0]], dtype=float)
                g3 = bary_to_cartesian_3d(bary_mapped, self.vertices_3d)
                nodes.append(Node(node_id, bary_mapped, float(wt), local, g3))
                node_id += 1

        return nodes


def build_nodes(method: str, k: int, vertices_2d: np.ndarray, vertices_3d: np.ndarray):
    """
    Factory function to build nodes using the specified method.

    Parameters:
    -----------
    method : str
        Either "table1" or "table2"
    k : int
        Polynomial degree (1-4)
    vertices_2d : np.ndarray
        2D reference triangle vertices (shape: [3, 2])
    vertices_3d : np.ndarray
        3D reference triangle vertices (shape: [3, 3])

    Returns:
    --------
    list of Node
        List of generated nodes
    """
    method = method.lower().strip()
    if method == "table1":
        gen = Table1NodeGenerator(k, vertices_2d, vertices_3d)
    elif method == "table2":
        gen = Table2NodeGenerator(k, vertices_2d, vertices_3d)
    else:
        raise ValueError("method must be 'table1' or 'table2'")
    return gen.generate()


def get_extra_bary(method: str, k: int):
    """
    Generate additional barycentric points for boundary/symmetry correction.

    Injects symmetric corrective points based on the method and degree.

    Parameters:
    -----------
    method : str
        Either "table1" or "table2"
    k : int
        Polynomial degree

    Returns:
    --------
    np.ndarray
        Array of barycentric coordinates (shape: [n_extra, 3])
    """
    extra = []
    if method == 'table1':
        if k == 1:
            extra.append([1 / 3, 1 / 3, 1 / 3])
        elif k == 3:
            extra.append([1 / 3, 1 / 3, 1 / 3])
            extra.extend(get_orbit(0.7573626348863415, 0.1213186825568292))
            extra.extend(get_orbit(0.09353693959563815, 0.4532315302021809))
    elif method == 'table2':
        if k == 1:
            extra.extend(get_orbit(11 / 24, 11 / 24))

    return np.array(extra, dtype=float) if extra else np.empty((0, 3), dtype=float)


# Reference triangle vertices constants
# These standard definitions are imported from render_utils to avoid circular imports,
# but temporarily redefined here for compatibility with get_reference_data API
_VERTICES_2D = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3.0) / 2.0],
], dtype=float)
"""Reference equilateral triangle in 2D (barycentric → Cartesian)"""

_VERTICES_3D = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=float)
"""Octahedron first-quadrant face in 3D"""


def get_reference_data(method: str, k: int) -> dict:
    """
    Load reference element quadrature data for DG basis functions.

    Generates quadrature nodes on the reference simplex triangle using either Table 1
    (exact integration rules) or Table 2 (nodal rules) for polynomial degrees k ∈ {1,2,3,4}.
    Converts node data to barycentric and physical coordinates with associated quadrature weights.

    **Reference Coordinates:**
    - Barycentric: $\\lambda = (\\lambda_1, \\lambda_2, \\lambda_3)$ where $\\lambda_i \\in [0,1]$ and $\\sum \\lambda_i = 1$
    - Physical (r,s): $r = \\lambda_1$, $s = \\lambda_2$, so $(r, s) \\in [0,1]^2$

    Args:
        method: Quadrature rule method, either 'table1' (exact integration) or 'table2' (nodal points).
        k: Polynomial degree (1 ≤ k ≤ 4).

    Returns:
        Dictionary with keys:
            - 'nodes' (list): Reference nodes with barycentric coords, weights, local coords
            - 'bary_coords' (np.ndarray): Barycentric coordinates, shape (N_nodes, 3)
            - 'weights' (np.ndarray): Quadrature weights, shape (N_nodes,)
            - 'weights_1d' (np.ndarray): 1D boundary quadrature weights for edge integration, shape (nfp,)
            - 'xi' (np.ndarray): Transformed coordinate xi = 2*λ₃ - 1, shape (N_nodes,)
            - 'eta' (np.ndarray): Transformed coordinate eta = 2*λ₂ - 1, shape (N_nodes,)

    Example:
        >>> data = get_reference_data('table1', k=3)
        >>> xi, eta, w = data['xi'], data['eta'], data['weights']
        >>> w_1d = data['weights_1d']  # 1D boundary weights (4 points for k=3)
        >>> N_nodes = len(xi)
        >>> # Use for DG basis evaluation: Φ(xi,eta) on reference simplex

    Reference:
        Cockburn, Karniadakis, Shu (2000): \"Runge-Kutta Discontinuous Galerkin Methods\"
        Dubiner: \"Spectral Methods on Triangles and Other Domains\", J. Sci. Comput. (1991)
    """
    nodes = build_nodes(method, k, _VERTICES_2D, _VERTICES_3D)
    bary_coords = np.array([n.barycentric for n in nodes], dtype=float)
    weights = np.array([n.weight for n in nodes], dtype=float)
    xi = 2.0 * bary_coords[:, 2] - 1.0
    eta = 2.0 * bary_coords[:, 1] - 1.0

    # ========================================================================
    # Hardcoded 1D boundary quadrature weights (Matching Table 1 from PDF)
    # ========================================================================
    # These weights are for 1D edge integration in Surface term computation
    # The number of points per face (nfp) = k + 1, sum of weights = 1.0

    if method.lower().strip() in ['table1', 'table2']:
        if k == 1:
            # 2-point Gauss-Legendre (k=1 → 2 pts/face)
            w_1d = np.array([0.5000000000000000, 0.5000000000000000])
        elif k == 2:
            # 3-point Gauss-Legendre (k=2 → 3 pts/face)
            w_1d = np.array([0.2777777777777777, 0.4444444444444444, 0.2777777777777777])
        elif k == 3:
            # 4-point Gauss-Legendre (k=3 → 4 pts/face)
            w_1d = np.array([0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269])
        elif k == 4:
            # 5-point Gauss-Legendre (k=4 → 5 pts/face)
            w_1d = np.array([0.1184634425280944, 0.2393143352496833, 0.2844444444444446,
                             0.2393143352496833, 0.1184634425280944])
        else:
            raise ValueError(f"1D boundary weights not defined for k={k}")
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "nodes": nodes,
        "bary_coords": bary_coords,
        "weights": weights,
        "weights_1d": w_1d,
        "xi": xi,
        "eta": eta,
    }
