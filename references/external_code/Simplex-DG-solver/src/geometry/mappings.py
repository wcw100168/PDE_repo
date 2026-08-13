"""
Coordinate transformations and geometric mappings.

Implements transformations between reference element coordinates, collapsed
coordinates used in spectral element methods, and physical coordinates.
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np


def collapsed_coords_transform(xi: np.ndarray, eta: np.ndarray) -> tuple:
    """
    Map reference triangle coordinates (xi, eta) to collapsed coordinates (a, b).

    The reference triangle is [-1, 1] × [0, 1] in (xi, eta) space.
    Collapsed coordinates (a, b) correspond to the Jacobi polynomial domain.

    Transformation:
        a = 2(1 + xi) / (1 - eta) - 1
        b = eta

    Parameters
    ----------
    xi : np.ndarray
        Reference element coordinate (array or scalar)
    eta : np.ndarray
        Reference element coordinate (array or scalar)

    Returns
    -------
    tuple
        (a, b) collapsed coordinates
    """
    a = np.empty_like(xi, dtype=float)
    mask = np.abs(1.0 - eta) > 1e-15
    a[mask] = 2.0 * (1.0 + xi[mask]) / (1.0 - eta[mask]) - 1.0
    a[~mask] = -1.0

    b = eta.copy()
    return a, b


def rs_to_xy(
    r: Union[float, np.ndarray],
    s: Union[float, np.ndarray],
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""
    Map reference coordinates (r, s) to physical coordinates (x, y) via affine transformation.

    Given three vertices of a physical triangle, maps reference element coordinates
    (r, s) ∈ [-1, 1]² to physical coordinates (x, y) using an affine (linear) map:

    .. math::
        x(r, s) = x_1 + \frac{x_2 - x_1}{2}(r + 1) + \frac{x_3 - x_1}{2}(s + 1)
        y(r, s) = y_1 + \frac{y_2 - y_1}{2}(r + 1) + \frac{y_3 - y_1}{2}(s + 1)

    This is the inverse of the reference-to-physical transformation for triangular
    spectral/DG methods (Hesthaven & Warburton 2008, Eq. 6.3).

    Parameters
    ----------
    r : Union[float, np.ndarray]
        Reference r-coordinate (scalar or array)
    s : Union[float, np.ndarray]
        Reference s-coordinate (scalar or array)
    v1 : np.ndarray
        Vertex 1 physical coordinates, shape (2,) as [x1, y1]
    v2 : np.ndarray
        Vertex 2 physical coordinates, shape (2,) as [x2, y2]
    v3 : np.ndarray
        Vertex 3 physical coordinates, shape (2,) as [x3, y3]

    Returns
    -------
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
        (x, y) physical coordinates with same shape as input r, s

    Examples
    --------
    Map reference coordinates to a unit right triangle:

    >>> v1 = np.array([0.0, 0.0])
    >>> v2 = np.array([1.0, 0.0])
    >>> v3 = np.array([0.0, 1.0])
    >>> r_ref = np.array([-1.0, 0.0, 1.0])
    >>> s_ref = np.array([-1.0, 0.0, 1.0])
    >>> x, y = rs_to_xy(r_ref, s_ref, v1, v2, v3)
    >>> print(x)  # Should be [0. , 0.5, 1.]
    """
    # Convert inputs to numpy arrays
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)

    # Validate vertex shapes
    if v1.shape != (2,) or v2.shape != (2,) or v3.shape != (2,):
        raise ValueError("Vertex coordinates must be shape (2,)")

    # Extract vertex coordinates
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    # Apply affine transformation (Eq. 6.3)
    x = x1 + (x2 - x1) * (r + 1.0) / 2.0 + (x3 - x1) * (s + 1.0) / 2.0
    y = y1 + (y2 - y1) * (r + 1.0) / 2.0 + (y3 - y1) * (s + 1.0) / 2.0

    return x, y


def xy_to_rs(
    x: Union[float, np.ndarray],
    y: Union[float, np.ndarray],
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    tol: float = 1e-14
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    r"""
    Map physical coordinates (x, y) to reference coordinates (r, s) via inverse affine transformation.

    Inverts the mapping from :func:`rs_to_xy`. Solves the linear system:

    .. math::
        \begin{pmatrix} x_2 - x_1 & x_3 - x_1 \\ y_2 - y_1 & y_3 - y_1 \end{pmatrix}
        \begin{pmatrix} (r+1)/2 \\ (s+1)/2 \end{pmatrix}
        = \begin{pmatrix} x - x_1 \\ y - y_1 \end{pmatrix}

    Parameters
    ----------
    x : Union[float, np.ndarray]
        Physical x-coordinate (scalar or array)
    y : Union[float, np.ndarray]
        Physical y-coordinate (scalar or array)
    v1 : np.ndarray
        Vertex 1 physical coordinates, shape (2,)
    v2 : np.ndarray
        Vertex 2 physical coordinates, shape (2,)
    v3 : np.ndarray
        Vertex 3 physical coordinates, shape (2,)
    tol : float, optional
        Tolerance for Jacobian degeneracy checking

    Returns
    -------
    Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
        (r, s) reference coordinates

    Raises
    ------
    ValueError
        If triangle is degenerate (determinant near zero)

    Examples
    --------
    >>> v1 = np.array([0.0, 0.0])
    >>> v2 = np.array([1.0, 0.0])
    >>> v3 = np.array([0.0, 1.0])
    >>> r, s = xy_to_rs(0.5, 0.5, v1, v2, v3)
    >>> print(f"r={r:.2f}, s={s:.2f}")
    r=0.00, s=1.00
    """
    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)

    # Validate vertex shapes
    if v1.shape != (2,) or v2.shape != (2,) or v3.shape != (2,):
        raise ValueError("Vertex coordinates must be shape (2,)")

    # Extract vertex coordinates
    x1, y1 = v1
    x2, y2 = v2
    x3, y3 = v3

    # Build the 2×2 Jacobian matrix (forward mapping derivatives)
    dx21 = x2 - x1
    dy21 = y2 - y1
    dx31 = x3 - x1
    dy31 = y3 - y1

    # Compute determinant
    J = dx21 * dy31 - dy21 * dx31

    if np.any(np.abs(J) < tol):
        raise ValueError(
            f"Degenerate triangle: Jacobian determinant={J:.2e} < tol={tol:.2e}. "
            f"Vertices may be collinear."
        )

    # Solve for (r+1)/2, (s+1)/2 using Cramer's rule
    dx1 = x - x1
    dy1 = y - y1

    alpha = (dx1 * dy31 - dy1 * dx31) / J  # (r + 1) / 2
    beta = (dx21 * dy1 - dy21 * dx1) / J   # (s + 1) / 2

    # Convert back to reference coordinates
    r = 2.0 * alpha - 1.0
    s = 2.0 * beta - 1.0

    return r, s


class AffineMap:
    """
    Wrapper class for affine coordinate transformations between reference and physical coordinates.

    Provides a convenient interface for evaluating forward and inverse mappings,
    and caches the triangle's geometric properties.

    Attributes
    ----------
    v1, v2, v3 : np.ndarray
        Vertex coordinates shape (2,)
    J : float
        Jacobian determinant
    xr, yr, xs, ys : float
        Forward partial derivatives

    Examples
    --------
    >>> v1 = np.array([0.0, 0.0])
    >>> v2 = np.array([1.0, 0.0])
    >>> v3 = np.array([0.0, 1.0])
    >>> mapping = AffineMap(v1, v2, v3)
    >>> x, y = mapping.forward(0.0, 0.0)  # Center of reference triangle
    >>> r, s = mapping.inverse(x, y)
    >>> print(f"Roundtrip: r={r:.6f}, s={s:.6f}")
    Roundtrip: r=0.000000, s=0.000000
    """

    def __init__(self, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, tol: float = 1e-14):
        """
        Initialize affine map with triangle vertices.

        Parameters
        ----------
        v1, v2, v3 : np.ndarray
            Vertex coordinates, shape (2,) each
        tol : float, optional
            Tolerance for degeneracy checking
        """
        from .metrics import compute_geometric_factors

        self.v1 = np.asarray(v1, dtype=float)
        self.v2 = np.asarray(v2, dtype=float)
        self.v3 = np.asarray(v3, dtype=float)
        self.tol = tol

        # Precompute geometric factors
        factors = compute_geometric_factors(self.v1, self.v2, self.v3, tol=tol)
        self.J = float(factors["J"])
        self.xr = float(factors["xr"])
        self.yr = float(factors["yr"])
        self.xs = float(factors["xs"])
        self.ys = float(factors["ys"])

    def forward(
        self, r: Union[float, np.ndarray], s: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Map reference coordinates to physical coordinates.

        Parameters
        ----------
        r, s : Union[float, np.ndarray]
            Reference coordinates

        Returns
        -------
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
            (x, y) physical coordinates
        """
        return rs_to_xy(r, s, self.v1, self.v2, self.v3)

    def inverse(
        self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Map physical coordinates to reference coordinates.

        Parameters
        ----------
        x, y : Union[float, np.ndarray]
            Physical coordinates

        Returns
        -------
        Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]
            (r, s) reference coordinates
        """
        return xy_to_rs(x, y, self.v1, self.v2, self.v3, tol=self.tol)

    def area(self) -> float:
        """Return physical element area (|J| is the area scaling factor)."""
        return float(abs(self.J))
