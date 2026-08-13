"""
Test data generators for DG operator verification and convergence studies.

Provides analytical solutions and test fields with known properties
for validating numerical implementations.
"""

from typing import Callable, Tuple
import numpy as np


def F_vector_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    xc: float = 1.0,
    yc: float = 1.0,
    alpha: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Gaussian bell vector field for testing divergence and gradient operators.

    Defines a smooth, compactly-supported-like vector field with equal components:

    .. math::
        F_x(x, y) = F_y(x, y) = \exp\left(-\alpha \| (x - x_c, y - y_c) \|^2 \right)

    Properties:
    - Smooth and infinitely differentiable everywhere
    - Centered at (xc, yc) with width controlled by alpha
    - Decays rapidly away from center (useful for localized testing)
    - Both components identical (simple divergence pattern)

    Parameters
    ----------
    x : np.ndarray
        x-coordinates (array or scalar)
    y : np.ndarray
        y-coordinates (array or scalar)
    xc : float, optional
        x-coordinate of Gaussian center (default: 1.0)
    yc : float, optional
        y-coordinate of Gaussian center (default: 1.0)
    alpha : float, optional
        Width parameter (larger α = narrower Gaussian, default: 5.0)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (Fx, Fy) vector field components

    Examples
    --------
    Evaluate Gaussian bell at a single point:

    >>> Fx, Fy = F_vector_gaussian(1.0, 1.0)
    >>> print(f"At center: Fx={Fx:.4f}, Fy={Fy:.4f}")
    At center: Fx=1.0000, Fy=1.0000

    Evaluate on a grid:

    >>> x = np.linspace(0, 2, 10)
    >>> y = np.linspace(0, 2, 10)
    >>> X, Y = np.meshgrid(x, y)
    >>> Fx, Fy = F_vector_gaussian(X, Y, xc=1.0, yc=1.0, alpha=10.0)
    >>> print(Fx.shape)
    (10, 10)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute Gaussian bell
    exponent = -alpha * ((x - xc) ** 2 + (y - yc) ** 2)
    val = np.exp(exponent)

    return val, val


def divF_exact_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    xc: float = 1.0,
    yc: float = 1.0,
    alpha: float = 5.0
) -> np.ndarray:
    r"""
    Analytical divergence of the Gaussian bell vector field.

    For the vector field F = (f, f) where f(x, y) = exp(-α||(x-xc, y-yc)||²),
    the divergence is:

    .. math::
        \nabla \cdot \mathbf{F} = \frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}

    where:

    .. math::
        \frac{\partial f}{\partial x} = -2\alpha(x - x_c) \exp\left(-\alpha \| (x - x_c, y - y_c) \|^2 \right)

    Similar for y-component.

    Parameters
    ----------
    x : np.ndarray
        x-coordinates (array or scalar)
    y : np.ndarray
        y-coordinates (array or scalar)
    xc : float, optional
        x-coordinate of Gaussian center (default: 1.0)
    yc : float, optional
        y-coordinate of Gaussian center (default: 1.0)
    alpha : float, optional
        Width parameter (default: 5.0)

    Returns
    -------
    np.ndarray
        Divergence ∇·F at the given points

    Examples
    --------
    Compute exact divergence at grid points:

    >>> x = np.linspace(0, 2, 5)
    >>> y = np.linspace(0, 2, 5)
    >>> X, Y = np.meshgrid(x, y)
    >>> div_F = divF_exact_gaussian(X, Y, xc=1.0, yc=1.0, alpha=5.0)
    >>> print(f"At center: div_F={div_F[2, 2]:.6f} (should be ~0)")
    At center: div_F=0.000000

    Notes
    -----
    In physical coordinates, the divergence has a zero at the center (xc, yc)
    and changes sign, making it a good test case for convergence studies.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute Gaussian bell
    dist_sq = (x - xc) ** 2 + (y - yc) ** 2
    val = np.exp(-alpha * dist_sq)

    # Compute partial derivatives
    dfdx = -2.0 * alpha * (x - xc) * val
    dfdy = -2.0 * alpha * (y - yc) * val

    # Divergence: both components are identical, so div = ∂Fx/∂x + ∂Fy/∂y
    div_F = dfdx + dfdy

    return div_F


def exponent_pairs(total_degree: int) -> list:
    r"""
    Generate exponent pairs for 2D polynomial basis functions.

    For a given total polynomial degree p, generates all pairs (i, j) such that
    i + j = p, following the ordering convention for 2D simplex polynomials.

    This is useful for enumerating monomials in the space P^p_2 (polynomials
    of total degree ≤ p on the 2D simplex).

    Parameters
    ----------
    total_degree : int
        Total polynomial degree p

    Returns
    -------
    list of tuple
        List of (i, j) pairs in order: [(0, p), (1, p-1), ..., (p, 0)]

    Examples
    --------
    Generate exponent pairs for cubic polynomials (p=3):

    >>> pairs = exponent_pairs(3)
    >>> print(pairs)
    [(0, 3), (1, 2), (2, 1), (3, 0)]

    Dimension of polynomial space:

    >>> p = 5
    >>> pairs = exponent_pairs(p)
    >>> n_basis = sum(len(exponent_pairs(k)) for k in range(p + 1))
    >>> print(f"Number of basis functions for P^{p}_2: {n_basis}")
    Number of basis functions for P^5_2: 21

    Notes
    -----
    The number of basis functions in P^p_2 is (p+1)(p+2)/2 = C(p+2, 2),
    which matches the dimension of the space of polynomials of total degree ≤ p.

    The ordering corresponds to:
    - Jacobi polynomial ordering in collapsed coordinates
    - Standard convention in spectral element methods (Hesthaven & Warburton)
    """
    if not isinstance(total_degree, (int, np.integer)) or total_degree < 0:
        raise ValueError(f"total_degree must be non-negative integer, got {total_degree}")

    return [(i, total_degree - i) for i in range(total_degree + 1)]


def convergence_test_field(
    x: np.ndarray,
    y: np.ndarray,
    field_type: str = "polynomial"
) -> np.ndarray:
    """
    Generate test fields with known analytical properties for convergence studies.

    Parameters
    ----------
    x, y : np.ndarray
        Physical coordinates
    field_type : str, optional
        Type of test field:
        - 'polynomial': Simple polynomial f(x,y) = x + y
        - 'harmonic': Harmonic field f(x,y) = sin(πx)cos(πy)
        - 'gaussian': Gaussian bell (smooth, localized)

    Returns
    -------
    np.ndarray
        Field values at (x, y)

    Notes
    -----
    Each field type allows testing of different aspects:
    - polynomial: Tests method order (should achieve full order)
    - harmonic: Tests oscillatory fields (Fourier content up to π frequency)
    - gaussian: Tests localized features and smooth decay
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if field_type == "polynomial":
        return x + y
    elif field_type == "harmonic":
        return np.sin(np.pi * x) * np.cos(np.pi * y)
    elif field_type == "gaussian":
        return np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)
    else:
        raise ValueError(f"Unknown field_type: {field_type}")


def generate_convergence_mesh_sequence(
    n_div_values: list,
) -> Tuple[list, list]:
    """
    Generate sequence of refined reference triangle meshes for h-convergence studies.

    Parameters
    ----------
    n_div_values : list of int
        Sequence of subdivision levels

    Returns
    -------
    Tuple[list, list]
        (meshes, mesh_sizes) where:
        - meshes: List of (nodes, triangles) tuples for each refinement
        - mesh_sizes: List of characteristic mesh sizes h

    Examples
    --------
    Generate a sequence of 3 refinement levels:

    >>> from src.bases.simplex_2d import generate_subdivided_triangle
    >>> n_divs = [1, 2, 3]
    >>> meshes = []
    >>> for n in n_divs:
    ...     nodes, triangles = generate_subdivided_triangle(n)
    ...     meshes.append((nodes, triangles))
    >>>
    >>> # Compute characteristic mesh size
    >>> h_values = [1.0 / n for n in n_divs]
    """
    # Import here to avoid circular dependency
    try:
        from ..bases.simplex_2d import generate_subdivided_triangle
    except ImportError:
        raise ImportError("Cannot import simplex_2d; ensure it has the generate_subdivided_triangle function")

    meshes = []
    mesh_sizes = []

    for n_div in n_div_values:
        nodes, triangles = generate_subdivided_triangle(n_div)
        meshes.append((nodes, triangles))

        # Characteristic mesh size: 1 / n_div (since reference triangle spans [0,1])
        h = 1.0 / max(n_div, 1)
        mesh_sizes.append(h)

    return meshes, mesh_sizes
