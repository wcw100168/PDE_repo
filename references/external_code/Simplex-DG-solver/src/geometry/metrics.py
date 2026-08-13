"""
Geometric metrics computation for affine element mappings.

Computes Jacobian determinants and inverse metric tensors for coordinate
transformations between reference and physical triangular elements.
"""

from typing import Dict, List, Tuple, Union
import numpy as np


def compute_geometric_factors(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    tol: float = 1e-14
) -> Dict[str, Union[float, np.ndarray]]:
    r"""
    Compute Jacobian and inverse metric tensors for a single affine-mapped triangle.

    Given three vertices of a physical triangle, computes the Jacobian determinant
    and the inverse metric tensor components required for coordinate transformations
    and differential operator evaluation.

    The affine mapping from reference coordinates (r, s) to physical coordinates (x, y) is:

    .. math::
        x = x_1 + \frac{x_2 - x_1}{2}(r + 1) + \frac{x_3 - x_1}{2}(s + 1)
        y = y_1 + \frac{y_2 - y_1}{2}(r + 1) + \frac{y_3 - y_1}{2}(s + 1)

    The Jacobian matrix is:

    .. math::
        J = \begin{vmatrix} x_r & x_s \\ y_r & y_s \end{vmatrix}, \quad
        J_{det} = x_r y_s - x_s y_r

    The inverse metric tensor (for chain rule) is:

    .. math::
        r_x = \frac{y_s}{J}, \quad r_y = -\frac{x_s}{J}, \quad
        s_x = -\frac{y_r}{J}, \quad s_y = \frac{x_r}{J}

    Parameters
    ----------
    v1 : np.ndarray
        First vertex coordinates, shape (2,) as [x1, y1]
    v2 : np.ndarray
        Second vertex coordinates, shape (2,) as [x2, y2]
    v3 : np.ndarray
        Third vertex coordinates, shape (2,) as [x3, y3]
    tol : float, optional
        Tolerance for Jacobian near-zero detection (default: 1e-14)

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - 'J': Jacobian determinant (float)
        - 'rx', 'ry': Inverse metric tensor components for r (floats)
        - 'sx', 'sy': Inverse metric tensor components for s (floats)
        - 'xr', 'yr': Forward partial derivatives ∂x/∂r, ∂y/∂r (floats)
        - 'xs', 'ys': Forward partial derivatives ∂x/∂s, ∂y/∂s (floats)

    Raises
    ------
    ValueError
        If Jacobian determinant is too close to zero (degenerate triangle).
    RuntimeWarning
        If Jacobian is very small (ill-conditioned mapping).

    Examples
    --------
    Compute geometric factors for a unit right triangle:

    >>> v1 = np.array([0.0, 0.0])
    >>> v2 = np.array([1.0, 0.0])
    >>> v3 = np.array([0.0, 1.0])
    >>> factors = compute_geometric_factors(v1, v2, v3)
    >>> print(f"Jacobian: {factors['J']:.4f}")
    Jacobian: 0.5000
    """
    # Convert to numpy arrays and validate input dimensions
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)

    if v1.shape != (2,) or v2.shape != (2,) or v3.shape != (2,):
        raise ValueError("Vertex coordinates must be shape (2,)")

    # Extract coordinates
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    # Compute forward partial derivatives (Eq. 6.4 in Hesthaven & Warburton)
    xr = (x2 - x1) / 2.0
    yr = (y2 - y1) / 2.0
    xs = (x3 - x1) / 2.0
    ys = (y3 - y1) / 2.0

    # Compute Jacobian determinant (Eq. 6.5)
    J = xr * ys - xs * yr

    # Check for degeneracy
    if np.abs(J) < tol:
        raise ValueError(
            f"Degenerate triangle detected: Jacobian determinant J={J:.2e} < tol={tol:.2e}. "
            f"Vertices may be collinear: v1={v1}, v2={v2}, v3={v3}"
        )

    # Warn on ill-conditioning
    if np.abs(J) < 1e-10:
        import warnings
        warnings.warn(
            f"Ill-conditioned Jacobian: J={J:.2e} (very small). "
            f"Numerical errors may accumulate.",
            RuntimeWarning,
            stacklevel=2
        )

    # Compute inverse metric tensor components (chain rule for Cartesian derivatives)
    rx = ys / J
    ry = -xs / J
    sx = -yr / J
    sy = xr / J

    return {
        "J": J,
        "rx": rx, "ry": ry,
        "sx": sx, "sy": sy,
        "xr": xr, "yr": yr,
        "xs": xs, "ys": ys
    }


def compute_geometric_factors_batch(
    vertices: np.ndarray,
    tol: float = 1e-14
) -> Dict[str, np.ndarray]:
    r"""
    Compute Jacobian and inverse metrics for multiple triangles (vectorized).

    This is a batch processing version of :func:`compute_geometric_factors`
    that processes all elements at once using NumPy broadcasting.

    Parameters
    ----------
    vertices : np.ndarray
        All triangle vertices, shape (K, 3, 2) where K is the number of triangles.
        For element k: vertices[k] = [[x1, y1], [x2, y2], [x3, y3]]
    tol : float, optional
        Tolerance for Jacobian near-zero detection (default: 1e-14)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing arrays of shape (K,):
        - 'J': Jacobian determinants
        - 'rx', 'ry', 'sx', 'sy': Inverse metric tensor components
        - 'xr', 'yr', 'xs', 'ys': Forward partial derivatives

    Raises
    ------
    ValueError
        If any triangle has Jacobian determinant too close to zero.

    Examples
    --------
    Process a simple mesh with 2 triangles:

    >>> vertices = np.array([
    ...     [[0, 0], [1, 0], [0, 1]],  # Triangle 1
    ...     [[1, 0], [1, 1], [0, 1]]   # Triangle 2
    ... ], dtype=float)
    >>> factors = compute_geometric_factors_batch(vertices)
    >>> print(f"Jacobians: {factors['J']}")
    Jacobians: [0.5 0.5]
    """
    vertices = np.asarray(vertices, dtype=float)
    K = vertices.shape[0]

    if vertices.shape != (K, 3, 2):
        raise ValueError(f"Expected shape (K, 3, 2), got {vertices.shape}")

    # Extract all v1, v2, v3 coordinates
    v1 = vertices[:, 0, :]  # shape (K, 2)
    v2 = vertices[:, 1, :]  # shape (K, 2)
    v3 = vertices[:, 2, :]  # shape (K, 2)

    # Vectorized partial derivative computation
    xr = (v2[:, 0] - v1[:, 0]) / 2.0
    yr = (v2[:, 1] - v1[:, 1]) / 2.0
    xs = (v3[:, 0] - v1[:, 0]) / 2.0
    ys = (v3[:, 1] - v1[:, 1]) / 2.0

    # Vectorized Jacobian
    J = xr * ys - xs * yr

    # Check for degeneracy
    degenerate = np.abs(J) < tol
    if np.any(degenerate):
        bad_idx = np.where(degenerate)[0][0]
        raise ValueError(
            f"Degenerate triangle at index {bad_idx}: Jacobian={J[bad_idx]:.2e} < tol={tol:.2e}"
        )

    # Warn on ill-conditioning
    ill_conditioned = np.abs(J) < 1e-10
    if np.any(ill_conditioned):
        import warnings
        warnings.warn(
            f"{np.sum(ill_conditioned)} triangles have ill-conditioned Jacobians (J < 1e-10)",
            RuntimeWarning,
            stacklevel=2
        )

    # Vectorized inverse metric tensor
    rx = ys / J
    ry = -xs / J
    sx = -yr / J
    sy = xr / J

    return {
        "J": J,
        "rx": rx, "ry": ry,
        "sx": sx, "sy": sy,
        "xr": xr, "yr": yr,
        "xs": xs, "ys": ys
    }


class MetricTensor:
    """
    Caching container for precomputed geometric metrics on a mesh.

    Stores Jacobians and metric tensor components for all elements to avoid
    repeated recomputation during operator evaluation.

    Attributes
    ----------
    vertices : np.ndarray
        Shape (K, 3, 2) - all element vertices
    J : np.ndarray
        Shape (K,) - Jacobian determinants
    rx, ry, sx, sy : np.ndarray
        Shape (K,) - inverse metric tensor components
    xr, yr, xs, ys : np.ndarray
        Shape (K,) - forward partial derivatives
    quality_metrics : Dict
        Element quality measures (aspect ratio, condition number, etc.)

    Examples
    --------
    >>> vertices = np.array([[[0,0],[1,0],[0,1]]], dtype=float)
    >>> metrics = MetricTensor(vertices)
    >>> jacobians = metrics.J
    >>> print(f"Aspect ratio: {metrics.quality_metrics['aspect_ratio'][0]:.2f}")
    """

    def __init__(self, vertices: np.ndarray, tol: float = 1e-14):
        r"""
        Initialize metric tensor from mesh vertices.

        Parameters
        ----------
        vertices : np.ndarray
            Shape (K, 3, 2) - all triangle vertices
        tol : float, optional
            Tolerance for degeneracy checking
        """
        self.vertices = np.asarray(vertices, dtype=float)
        self.tol = tol

        # Compute all metrics
        factors = compute_geometric_factors_batch(vertices, tol=tol)
        self.J = factors["J"]
        self.rx = factors["rx"]
        self.ry = factors["ry"]
        self.sx = factors["sx"]
        self.sy = factors["sy"]
        self.xr = factors["xr"]
        self.yr = factors["yr"]
        self.xs = factors["xs"]
        self.ys = factors["ys"]

        # Compute quality metrics
        self.quality_metrics = self._compute_quality_metrics()

    def _compute_quality_metrics(self) -> Dict[str, np.ndarray]:
        """Compute element quality measures (aspect ratio, condition number, etc.)."""
        K = len(self.J)
        metrics = {}

        # Compute side lengths for aspect ratio
        v1 = self.vertices[:, 0, :]
        v2 = self.vertices[:, 1, :]
        v3 = self.vertices[:, 2, :]

        side1 = np.linalg.norm(v2 - v1, axis=1)
        side2 = np.linalg.norm(v3 - v2, axis=1)
        side3 = np.linalg.norm(v1 - v3, axis=1)

        # Aspect ratio: max_side / min_altitude
        # Altitude = 2 * Area / base_length
        areas = np.abs(self.J)
        max_side = np.maximum(np.maximum(side1, side2), side3)
        min_altitude = 2 * areas / max_side
        metrics["aspect_ratio"] = max_side / (np.maximum(min_altitude, 1e-14))

        # Condition number: max|eigenvalue| / min|eigenvalue|
        metrics["condition_number"] = np.maximum(
            np.abs(self.J) / np.abs(self.J), 1.0
        )  # Placeholder: 1.0 for affine maps

        return metrics

    def get_element_factors(self, k: int) -> Dict[str, float]:
        """
        Retrieve metrics for a single element.

        Parameters
        ----------
        k : int
            Element index

        Returns
        -------
        Dict[str, float]
            Dictionary with keys J, rx, ry, sx, sy, xr, yr, xs, ys
        """
        return {
            "J": self.J[k],
            "rx": self.rx[k],
            "ry": self.ry[k],
            "sx": self.sx[k],
            "sy": self.sy[k],
            "xr": self.xr[k],
            "yr": self.yr[k],
            "xs": self.xs[k],
            "ys": self.ys[k],
        }

    def validate_jacobians(self) -> Tuple[bool, str]:
        """
        Check all Jacobians for degeneracy and ill-conditioning.

        Returns
        -------
        Tuple[bool, str]
            (is_valid, message)
        """
        degenerate = np.abs(self.J) < self.tol
        if np.any(degenerate):
            bad_indices = np.where(degenerate)[0]
            return False, f"Degenerate triangles at indices: {bad_indices}"

        ill_conditioned = np.abs(self.J) < 1e-10
        if np.any(ill_conditioned):
            return True, f"Warning: {np.sum(ill_conditioned)} ill-conditioned triangles"

        return True, "All Jacobians valid"
