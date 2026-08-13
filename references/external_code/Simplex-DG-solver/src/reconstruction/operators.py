"""
Differential operators for spectral element methods.

Implements divergence, gradient, and curl operators on triangular elements
with support for coordinate transformations via metric tensors.
"""

from typing import Dict, Union, Tuple, Optional
import numpy as np


def compute_divergence(
    Fx: np.ndarray,
    Fy: np.ndarray,
    factors: Dict[str, Union[float, np.ndarray]],
    D_xi: Optional[np.ndarray] = None,
    D_eta: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""
    Compute divergence of a vector field via transformed reference coordinates.

    Given a vector field **F** = (Fx, Fy) defined at nodal points in a physical
    triangular element, computes its divergence ∇·**F** using differentiation
    matrices in reference coordinates combined with chain rule transformations.

    The operator is defined via the transformed divergence formula (Hesthaven &
    Warburton 2008, Eq. 6.6):

    .. math::
        \nabla \cdot \mathbf{F} = \frac{1}{J} \left[
            \frac{\partial}{\partial r}(y_s F_x - x_s F_y) +
            \frac{\partial}{\partial s}(-y_r F_x + x_r F_y)
        \right]

    where:
    - J is the Jacobian determinant
    - (xr, xs, yr, ys) are forward partial derivatives of the mapping
    - (D_xi, D_eta) are nodal differentiation matrices in reference space
    - The terms (y_s F_x - x_s F_y) and (-y_r F_x + x_r F_y) are the metric-scaled components

    Parameters
    ----------
    Fx : np.ndarray
        x-component of vector field, shape (Np,) where Np is the number of nodes
    Fy : np.ndarray
        y-component of vector field, shape (Np,)
    factors : Dict[str, float]
        Geometric factors obtained from :func:`~src.geometry.metrics.compute_geometric_factors`.
        Must contain keys: 'J', 'xr', 'yr', 'xs', 'ys'
        - 'J': Jacobian determinant (float)
        - 'xr', 'yr': ∂x/∂r, ∂y/∂r (floats)
        - 'xs', 'ys': ∂x/∂s, ∂y/∂s (floats)
    D_xi : np.ndarray, optional
        Differentiation matrix for r-direction, shape (Np, Np).
        If None, raises ValueError.
    D_eta : np.ndarray, optional
        Differentiation matrix for s-direction, shape (Np, Np).
        If None, raises ValueError.

    Returns
    -------
    np.ndarray
        Divergence field, shape (Np,). Each entry is ∇·**F** at that node.

    Raises
    ------
    ValueError
        - If D_xi or D_eta is None
        - If shape mismatch between field and differentiation matrices
        - If 'J' or metric factors missing from factors dict

    Examples
    --------
    Compute divergence of a uniform vector field on a unit right triangle.
    For constant F = (1, 1), exact divergence is ∇·F = ∂1/∂x + ∂1/∂y = 0:

    >>> import numpy as np
    >>> from src.geometry.metrics import compute_geometric_factors
    >>> from src.reconstruction import build_differentiation_matrices
    >>>
    >>> # Set up reference triangle
    >>> v1 = np.array([0.0, 0.0])
    >>> v2 = np.array([1.0, 0.0])
    >>> v3 = np.array([0.0, 1.0])
    >>> factors = compute_geometric_factors(v1, v2, v3)
    >>>
    >>> # Create mock differentiation matrices (typically from polynomial basis)
    >>> Np = 3  # Linear basis
    >>> D_xi = np.zeros((Np, Np))
    >>> D_eta = np.zeros((Np, Np))
    >>>
    >>> # Constant vector field
    >>> Fx = np.ones(Np)
    >>> Fy = np.ones(Np)
    >>>
    >>> # Compute divergence (should be ~0 for constant field)
    >>> div_F = compute_divergence(Fx, Fy, factors, D_xi, D_eta)
    >>> print(f"Divergence: {div_F}")

    Notes
    -----
    The formulation via transformed coordinates is essential for high-accuracy
    spectral methods since it avoids explicit computation of (∂Fx/∂x, ∂Fy/∂y) in
    physical coordinates, which would require inverting the Jacobian. Instead, it
    leverages the metric tensor and reference-space differentiation matrices.

    Complexity: O(Np²) due to matrix-vector products D_xi @ term_r and D_eta @ term_s.

    References
    ----------
    .. [HW] Hesthaven, J.S., & Warburton, T. (2008).
        "Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications."
        Springer, New York. Equation 6.6.
    """
    # Validate inputs
    if D_xi is None or D_eta is None:
        raise ValueError("D_xi and D_eta differentiation matrices are required")

    Fx = np.asarray(Fx, dtype=float)
    Fy = np.asarray(Fy, dtype=float)

    if Fx.shape != Fy.shape:
        raise ValueError(f"Fx and Fy shape mismatch: {Fx.shape} vs {Fy.shape}")

    Np = len(Fx)

    if D_xi.shape != (Np, Np) or D_eta.shape != (Np, Np):
        raise ValueError(
            f"Differentiation matrix shape mismatch. Expected ({Np}, {Np}), "
            f"got D_xi: {D_xi.shape}, D_eta: {D_eta.shape}"
        )

    # Extract required factors
    required_keys = ["J", "xr", "yr", "xs", "ys"]
    for key in required_keys:
        if key not in factors:
            raise ValueError(f"Missing required factor '{key}' in factors dict")

    J = factors["J"]
    xr = factors["xr"]
    yr = factors["yr"]
    xs = factors["xs"]
    ys = factors["ys"]

    # Compute metric-scaled field components (Eq. 6.6)
    term_r = ys * Fx - xs * Fy  # (y_s F_x - x_s F_y)
    term_s = -yr * Fx + xr * Fy  # (-y_r F_x + x_r F_y)

    # Apply differentiation matrices and scale by Jacobian inverse
    div_F = (D_xi @ term_r + D_eta @ term_s) / J

    return div_F


def compute_gradient(
    f: np.ndarray,
    factors: Dict[str, Union[float, np.ndarray]],
    D_xi: Optional[np.ndarray] = None,
    D_eta: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Compute gradient of a scalar field via transformed reference coordinates.

    Given a scalar function f defined at nodal points in a physical triangular element,
    computes its gradient ∇f = (∂f/∂x, ∂f/∂y) using differentiation matrices in
    reference coordinates combined with chain rule transformations.

    The operator is defined via:

    .. math::
        \frac{\partial f}{\partial x} = r_x \frac{\partial f}{\partial r} + s_x \frac{\partial f}{\partial s}
        \frac{\partial f}{\partial y} = r_y \frac{\partial f}{\partial r} + s_y \frac{\partial f}{\partial s}

    where (rx, ry, sx, sy) are inverse metric tensor components.

    Parameters
    ----------
    f : np.ndarray
        Scalar field, shape (Np,)
    factors : Dict[str, float]
        Geometric factors from :func:`~src.geometry.metrics.compute_geometric_factors`.
        Must contain: 'rx', 'ry', 'sx', 'sy'
    D_xi : np.ndarray, optional
        Differentiation matrix for r-direction, shape (Np, Np)
    D_eta : np.ndarray, optional
        Differentiation matrix for s-direction, shape (Np, Np)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (grad_x, grad_y) where each has shape (Np,)

    Raises
    ------
    ValueError
        If differentiation matrices are None or factors missing required keys

    Examples
    --------
    Compute gradient of f(x,y) = x + y on a unit triangle.
    Expected: ∇f = (1, 1) everywhere.

    >>> f = x + y  # Function values at nodepoints
    >>> grad_x, grad_y = compute_gradient(f, factors, D_xi, D_eta)
    """
    if D_xi is None or D_eta is None:
        raise ValueError("D_xi and D_eta differentiation matrices are required")

    f = np.asarray(f, dtype=float)
    Np = len(f)

    if D_xi.shape != (Np, Np) or D_eta.shape != (Np, Np):
        raise ValueError(
            f"Differentiation matrix shape mismatch. Expected ({Np}, {Np}), "
            f"got D_xi: {D_xi.shape}, D_eta: {D_eta.shape}"
        )

    # Extract inverse metric components
    required_keys = ["rx", "ry", "sx", "sy"]
    for key in required_keys:
        if key not in factors:
            raise ValueError(f"Missing required factor '{key}' in factors dict")

    rx = factors["rx"]
    ry = factors["ry"]
    sx = factors["sx"]
    sy = factors["sy"]

    # Compute reference-space derivatives
    df_dr = D_xi @ f
    df_ds = D_eta @ f

    # Apply chain rule (inverse transformation)
    grad_x = rx * df_dr + sx * df_ds
    grad_y = ry * df_dr + sy * df_ds

    return grad_x, grad_y


def compute_laplacian(
    f: np.ndarray,
    factors: Dict[str, Union[float, np.ndarray]],
    D_xi: Optional[np.ndarray] = None,
    D_eta: Optional[np.ndarray] = None,
) -> np.ndarray:
    r"""
    Compute Laplacian of a scalar field via transformed reference coordinates.

    Computes ∇²f = ∇·(∇f) by composing gradient and divergence operators.

    Parameters
    ----------
    f : np.ndarray
        Scalar field, shape (Np,)
    factors : Dict[str, float]
        Geometric factors
    D_xi : np.ndarray, optional
        Differentiation matrix for r-direction
    D_eta : np.ndarray, optional
        Differentiation matrix for s-direction

    Returns
    -------
    np.ndarray
        Laplacian ∇²f, shape (Np,)

    Notes
    -----
    This is implemented as the divergence of the gradient, which is the
    most numerically stable formulation for spectral methods.
    """
    # Compute gradient
    grad_x, grad_y = compute_gradient(f, factors, D_xi, D_eta)

    # Compute divergence of gradient
    laplacian = compute_divergence(grad_x, grad_y, factors, D_xi, D_eta)

    return laplacian


def test_divergence_gaussian(
    factors: Dict[str, Union[float, np.ndarray]],
    D_xi: np.ndarray,
    D_eta: np.ndarray,
    nodes_r: np.ndarray,
    nodes_s: np.ndarray,
    xc: float = 1.0,
    yc: float = 1.0,
    alpha: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Test divergence operator using Gaussian bell vector field.

    Provides a known analytical divergence for convergence testing and validation.

    Parameters
    ----------
    factors : Dict[str, float]
        Geometric factors
    D_xi, D_eta : np.ndarray
        Differentiation matrices
    nodes_r, nodes_s : np.ndarray
        Reference coordinates of nodes
    xc, yc : float, optional
        Center of Gaussian (default: (1, 1))
    alpha : float, optional
        Width parameter (default: 5.0)

    Returns
    -------
    Dict[str, np.ndarray]
        - 'div_F_numerical': Numerically computed divergence
        - 'div_F_exact': Analytically computed divergence
        - 'error': Point-wise error

    Notes
    -----
    The test field is: F(x, y) = exp(-α||x-xc||²) (both components equal)
    """
    # Map to physical coordinates
    from ..geometry.mappings import rs_to_xy

    vertices = np.array([
        [factors.get("v1", np.zeros(2))],
        [factors.get("v2", np.ones(2) * 0.5)],
        [factors.get("v3", np.array([0, 1]))],
    ])

    x, y = rs_to_xy(nodes_r, nodes_s,
                    factors.get("v1", np.zeros(2)),
                    factors.get("v2", np.ones(2) * 0.5),
                    factors.get("v3", np.array([0, 1])))

    # Evaluate Gaussian bell vector field
    val = np.exp(-alpha * ((x - xc) ** 2 + (y - yc) ** 2))
    Fx = val
    Fy = val

    # Numerical divergence
    div_F_numerical = compute_divergence(Fx, Fy, factors, D_xi, D_eta)

    # Analytical divergence
    grad_x = -2 * alpha * (x - xc) * val
    grad_y = -2 * alpha * (y - yc) * val
    div_F_exact = grad_x + grad_y

    error = np.abs(div_F_numerical - div_F_exact)

    return {
        "div_F_numerical": div_F_numerical,
        "div_F_exact": div_F_exact,
        "error": error,
        "l2_error": np.sqrt(np.mean(error ** 2)),
        "max_error": np.max(error),
    }
