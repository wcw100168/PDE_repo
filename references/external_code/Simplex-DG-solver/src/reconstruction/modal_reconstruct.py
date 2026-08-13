"""
Modal reconstruction using Dubiner basis with Tikhonov regularization.

Implements:
1. DubinerReconstructor class with matrix caching for high-performance CFD
2. Differentiation matrix construction for SEM/DG methods
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from ..bases import vandermonde_2d_dubiner, grad_vandermonde_2d_dubiner
from ..numerics import (
    tikhonov_regularization_matrix,
    apply_exponential_filter,
    dubiner_basis_index_to_order
)
from ..geometry import bary_to_cartesian_2d, collapsed_coords_transform


def build_differentiation_matrices(V, V_xi, V_eta, w):
    """
    Construct numerical differentiation matrices with quadrature weights.
    
    Solves for the polynomial coefficients using the mass matrix,
    then constructs operators for spatial derivatives.
    
    Parameters
    ----------
    V : np.ndarray
        Vandermonde matrix (n_points, num_basis)
    V_xi : np.ndarray
        Gradient Vandermonde matrix in xi direction (n_points, num_basis)
    V_eta : np.ndarray
        Gradient Vandermonde matrix in eta direction (n_points, num_basis)
    w : np.ndarray
        Quadrature weights (n_points,)
        
    Returns
    -------
    tuple
        (D_xi, D_eta) differentiation matrices
    """
    # Construct weighted mass matrix
    W = np.diag(w)
    M_ref = V.T @ W @ V
    
    # Compute projection operator P = M_ref^{-1} V^T W
    P = np.linalg.solve(M_ref, V.T @ W)
    
    # Differentiation matrices
    D_xi = V_xi @ P
    D_eta = V_eta @ P

    return D_xi, D_eta


class DubinerReconstructor:
    """
    High-performance modal reconstruction using Dubiner basis with Tikhonov regularization.
    
    This class implements matrix caching to avoid recomputing Vandermonde matrices,
    regularization matrices, and matrix decompositions on each reconstruction call.
    Critical for CFD simulations that require many local reconstructions.
    
    Architecture:
    - __init__: Precompute and cache V_ref, L, and LU/Cholesky decomposition of M_reg
    - reconstruct: Use cached decomposition for O(num_basis^2) reconstruction (vs O(num_basis^3) solve)
    
    Parameters
    ----------
    nodes : list of Node
        Quadrature nodes in reference element with barycentric/local coords
    k : int
        Polynomial degree of reference element
    vertices_2d : np.ndarray
        2D reference triangle vertices (shape: 3, 2)
    lambda_reg : float
        Tikhonov regularization parameter (default: 1e-4)
    
    Attributes
    ----------
    _V_ref : np.ndarray
        Cached Vandermonde matrix for reference nodes
    _M_ref : np.ndarray
        Cached mass matrix M = V_ref.T @ W @ V_ref
    _L : np.ndarray
        Cached Tikhonov penalty matrix
    _M_reg : np.ndarray
        Cached regularized mass matrix
    _M_reg_factor : tuple
        Cached Cholesky factorization of M_reg
    _num_basis : int
        Total number of basis functions
    _weights : np.ndarray
        Cached quadrature weights
    """
    
    def __init__(
        self,
        nodes,
        k: int,
        vertices_2d: np.ndarray,
        lambda_reg: float = 1e-4
    ):
        """
        Initialize reconstructor with matrix precomputation and caching.
        
        Parameters
        ----------
        nodes : list of Node
            Quadrature nodes with local_coords attribute
        k : int
            Polynomial degree
        vertices_2d : np.ndarray
            Reference triangle vertices
        lambda_reg : float
            Regularization strength
        """
        self.k = k
        self.vertices_2d = vertices_2d
        self.lambda_reg = lambda_reg
        
        # Extract quadrature data
        self._num_basis = (k + 1) * (k + 2) // 2
        local_internal = np.array([n.local_coords for n in nodes], dtype=float)
        self._weights = np.array([n.weight for n in nodes], dtype=float)
        
        # Convert local coords [b3, b1] to reference element [xi, eta]
        # local_coords = [b3, b1], and reference: xi = 2*b3 - 1, eta = 2*b1 - 1
        xi_internal = 2.0 * local_internal[:, 0] - 1.0
        eta_internal = 2.0 * local_internal[:, 1] - 1.0
        
        # Precompute Vandermonde matrix
        self._V_ref = vandermonde_2d_dubiner(xi_internal, eta_internal, k)
        
        # Precompute weighted mass matrix
        W = np.diag(self._weights)
        self._M_ref = self._V_ref.T @ W @ self._V_ref
        
        # Precompute Tikhonov regularization matrix
        self._L = tikhonov_regularization_matrix(self._num_basis, lambda_reg)
        
        # Precompute regularized mass matrix
        self._M_reg = self._M_ref + lambda_reg ** 2 * (self._L.T @ self._L)
        
        # Precompute Cholesky decomposition for fast forward/backward substitution
        # This is the key optimization: instead of np.linalg.solve on each call,
        # we use cho_solve with cached factorization (O(n^2) vs O(n^3))
        try:
            self._M_reg_factor = cho_factor(self._M_reg)
            self._use_cholesky = True
        except np.linalg.LinAlgError:
            # Fall back to LU if Cholesky fails (should not happen for well-conditioned problems)
            from scipy.linalg import lu_factor
            self._M_reg_factor = lu_factor(self._M_reg)
            self._use_cholesky = False
    
    def reconstruct(
        self,
        target_bary: np.ndarray,
        known_vals: np.ndarray,
        apply_filter: bool = False
    ) -> tuple:
        """
        Perform modal reconstruction at target locations using cached matrices.
        
        This method:
        1. Uses cached M_reg decomposition to solve for coefficients
        2. Constructs V_target Vandermonde matrix
        3. Evaluates basis at target locations
        
        Parameters
        ----------
        target_bary : np.ndarray
            Target barycentric coordinates (n_targets, 3)
        known_vals : np.ndarray
            Function values at quadrature nodes (n_nodes,)
        apply_filter : bool
            Whether to apply exponential spectral filter (default: False)
            
        Returns
        -------
        tuple
            (target_xy, u_target) where:
            - target_xy: 2D Cartesian coordinates (n_targets, 2)
            - u_target: Reconstructed function values (n_targets,)
        """
        # Solve regularized system using cached decomposition
        W = np.diag(self._weights)
        rhs = self._V_ref.T @ W @ np.asarray(known_vals, dtype=float)
        
        if self._use_cholesky:
            a = cho_solve(self._M_reg_factor, rhs)
        else:
            from scipy.linalg import lu_solve
            a = lu_solve(self._M_reg_factor, rhs)
        
        # Optional spectral filtering
        if apply_filter:
            a = apply_exponential_filter(a, self._num_basis, self.k)
        
        # Convert target barycentric to reference element coordinates
        # bary = [b1, b2, b3], local_coords = [b3, b1]
        target_local = np.array([[b[2], b[0]] for b in target_bary], dtype=float)
        
        # Convert to Cartesian coordinates
        target_xy = np.array(
            [bary_to_cartesian_2d(b, self.vertices_2d) for b in target_bary],
            dtype=float
        )
        
        # Reference element coordinates
        xi_target = 2.0 * target_local[:, 0] - 1.0
        eta_target = 2.0 * target_local[:, 1] - 1.0
        
        # Construct Vandermonde at target points
        V_target = vandermonde_2d_dubiner(xi_target, eta_target, self.k)
        
        # Evaluate reconstruction
        u_target = V_target @ a

        return target_xy, u_target
    
    def get_coefficients(
        self,
        known_vals: np.ndarray,
        apply_filter: bool = False
    ) -> np.ndarray:
        """
        Compute modal coefficients for given function values (without reconstruction).
        
        Useful for analysis, filtering, or transferring to other representations.
        
        Parameters
        ----------
        known_vals : np.ndarray
            Function values at quadrature nodes
        apply_filter : bool
            Whether to apply spectral filter
            
        Returns
        -------
        np.ndarray
            Modal coefficients (num_basis,)
        """
        W = np.diag(self._weights)
        rhs = self._V_ref.T @ W @ np.asarray(known_vals, dtype=float)
        
        if self._use_cholesky:
            a = cho_solve(self._M_reg_factor, rhs)
        else:
            from scipy.linalg import lu_solve
            a = lu_solve(self._M_reg_factor, rhs)
        
        if apply_filter:
            a = apply_exponential_filter(a, self._num_basis, self.k)
        
        return a


# Convenience function for backward compatibility
def modal_reconstruct_at_bary_dubiner_tikhonov(
    nodes,
    known_vals: np.ndarray,
    target_bary: np.ndarray,
    vertices_2d: np.ndarray,
    k: int,
    lambda_reg: float = 1e-4,
    apply_filter: bool = False
) -> tuple:
    """
    Functional interface to modal reconstruction (legacy compatibility).
    
    For production code, prefer using DubinerReconstructor class directly
    to avoid repeated matrix recomputation.
    
    Parameters
    ----------
    nodes : list of Node
        Quadrature nodes
    known_vals : np.ndarray
        Function values
    target_bary : np.ndarray
        Target barycentric coordinates
    vertices_2d : np.ndarray
        Reference triangle vertices
    k : int
        Polynomial degree
    lambda_reg : float
        Regularization parameter
    apply_filter : bool
        Apply spectral filter
        
    Returns
    -------
    tuple
        (target_xy, u_target)
    """
    reconstructor = DubinerReconstructor(nodes, k, vertices_2d, lambda_reg)
    return reconstructor.reconstruct(target_bary, known_vals, apply_filter)


# Backward compatibility alias
modal_reconstruct_at_bary_upgraded = modal_reconstruct_at_bary_dubiner_tikhonov
