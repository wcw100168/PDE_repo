"""
Rendering and visualization utilities for linear triangular grid.

Includes plotting functions, vertex definitions, and the main rendering engine.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import Delaunay

from .data_structs import bary_to_cartesian_2d
from .generators import build_nodes, get_extra_bary
from .dubiner_tikhonov import modal_reconstruct_at_bary_dubiner_tikhonov


# ==========================================
# Global Constants: Reference Triangle Vertices
# ==========================================

VERTICES_2D = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, np.sqrt(3.0) / 2.0],
], dtype=float)
"""Reference equilateral triangle in 2D (barycentric → Cartesian)"""

VERTICES_3D = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=float)
"""Octahedron first-quadrant face in 3D"""


# ==========================================
# Test Function
# ==========================================

def exact_solution(x, y, center: tuple[float, float] = (0.5, 0.3), beta: float = 15.0):
    """
    Exact test function: Gaussian bell centered at a configurable point.
    
    u(x, y) = exp(-beta * ((x - cx)^2 + (y - cy)^2))
    
    Parameters:
    -----------
    x, y : array-like
        Cartesian coordinates
    center : tuple[float, float]
        Gaussian center (cx, cy), default is (0.5, 0.3)
    beta : float
        Gaussian sharpness parameter, default is 15.0
    
    Returns:
    --------
    np.ndarray
        Function values
    """
    cx, cy = center
    return np.exp(-beta * ((x - cx) ** 2 + (y - cy) ** 2))


def _resolve_vertices(vertices_2d: np.ndarray | None,
                      vertices_3d: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Resolve optional custom vertices and validate basic geometry constraints."""
    resolved_2d = VERTICES_2D if vertices_2d is None else np.asarray(vertices_2d, dtype=float)
    resolved_3d = VERTICES_3D if vertices_3d is None else np.asarray(vertices_3d, dtype=float)

    if resolved_2d.shape != (3, 2):
        raise ValueError("vertices_2d must have shape (3, 2)")
    if resolved_3d.shape != (3, 3):
        raise ValueError("vertices_3d must have shape (3, 3)")

    # Reject degenerate 2D triangles because barycentric inversion requires non-zero area.
    det = np.linalg.det(np.array([
        [resolved_2d[0, 0], resolved_2d[1, 0], resolved_2d[2, 0]],
        [resolved_2d[0, 1], resolved_2d[1, 1], resolved_2d[2, 1]],
        [1.0, 1.0, 1.0],
    ], dtype=float))
    if abs(det) < 1e-12:
        raise ValueError("vertices_2d must define a non-degenerate triangle")

    return resolved_2d, resolved_3d


def _compute_plot_bounds(vertices_2d: np.ndarray) -> tuple[float, float, float, float]:
    """Compute plotting bounds from triangle vertices with a small margin."""
    x_min = float(np.min(vertices_2d[:, 0]))
    x_max = float(np.max(vertices_2d[:, 0]))
    y_min = float(np.min(vertices_2d[:, 1]))
    y_max = float(np.max(vertices_2d[:, 1]))

    dx = x_max - x_min
    dy = y_max - y_min

    pad_x = max(0.05 * dx, 1e-3)
    pad_y = max(0.05 * dy, 1e-3)

    return x_min - pad_x, x_max + pad_x, y_min - pad_y, y_max + pad_y


def _resolve_solution_center(vertices_2d_resolved: np.ndarray,
                             used_custom_vertices: bool,
                             solution_center: tuple[float, float] | None) -> tuple[float, float]:
    """Resolve Gaussian center: preserve legacy default unless custom vertices are provided."""
    if solution_center is not None:
        arr = np.asarray(solution_center, dtype=float).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError("solution_center must be a 2-element coordinate")
        return float(arr[0]), float(arr[1])

    if used_custom_vertices:
        centroid = np.mean(vertices_2d_resolved, axis=0)
        return float(centroid[0]), float(centroid[1])

    return 0.5, 0.3


# ==========================================
# Main Rendering Function
# ==========================================

def render_linear_triangular_grid(method: str = "table2", k: int = 3,
                                  lambda_reg: float = 1e-4, apply_filter: bool = False,
                                  vertices_2d: np.ndarray | None = None,
                                  vertices_3d: np.ndarray | None = None,
                                  solution_center: tuple[float, float] | None = None):
    """
    Render linear triangular grid with Dubiner-Tikhonov modal reconstruction.
    
    Demonstrates:
    1. Grid connectivity (known nodes + auxiliary boundary vertices)
    2. 2D exact vs. reconstructed solution comparison
    3. 3D surface plot comparison
    
    Parameters:
    -----------
    method : str
        Either "table1" or "table2" (default: "table2")
    k : int
        Polynomial degree / basis parameter, 1-4 (default: 3)
    lambda_reg : float
        Tikhonov regularization strength (default: 1e-4)
    apply_filter : bool
        Whether to apply exponential spectral filter (default: False)
    vertices_2d : np.ndarray | None
        Optional custom 2D triangle vertices with shape (3, 2).
        If None, uses built-in equilateral reference triangle.
    vertices_3d : np.ndarray | None
        Optional custom 3D triangle vertices with shape (3, 3).
        If None, uses built-in octahedron-face reference triangle.
    solution_center : tuple[float, float] | None
        Optional Gaussian center (cx, cy) for exact solution.
        If None and custom vertices are provided, uses triangle centroid.
        If None and default vertices are used, keeps legacy center (0.5, 0.3).
    """
    used_custom_vertices = vertices_2d is not None
    vertices_2d_resolved, vertices_3d_resolved = _resolve_vertices(vertices_2d, vertices_3d)
    resolved_center = _resolve_solution_center(vertices_2d_resolved, used_custom_vertices, solution_center)
    nodes = build_nodes(method, int(k), vertices_2d_resolved, vertices_3d_resolved)

    # Known points: evaluated at generated nodes
    known_xy = np.array([bary_to_cartesian_2d(n.barycentric, vertices_2d_resolved) for n in nodes])
    known_vals = exact_solution(known_xy[:, 0], known_xy[:, 1], center=resolved_center)

    # Target points: vertices + symmetric correction points (method/k dependent)
    vertex_bary = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    extra_bary = get_extra_bary(method, int(k))
    if len(extra_bary) > 0:
        target_bary = np.vstack([vertex_bary, extra_bary])
    else:
        target_bary = vertex_bary

    # Use Dubiner-Tikhonov modal reconstruction.
    aux_xy, aux_vals = modal_reconstruct_at_bary_dubiner_tikhonov(
        nodes, known_vals, target_bary, vertices_2d_resolved, int(k),
        lambda_reg=lambda_reg, apply_filter=apply_filter
    )

    # Merge point sets for triangulation
    all_xy = np.vstack([known_xy, aux_xy])
    all_vals = np.concatenate([known_vals, aux_vals])

    # Use Delaunay triangulation to establish topology
    delaunay = Delaunay(all_xy)
    triang = mtri.Triangulation(all_xy[:, 0], all_xy[:, 1], triangles=delaunay.simplices)

    # ---------- Plot 1: Actual Grid Connectivity ----------
    fig1, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.triplot(triang, 'k-', alpha=0.55, linewidth=1.2)

    # Plot known points
    ax.scatter(known_xy[:, 0], known_xy[:, 1], marker='o', color='red', s=55, zorder=5,
               label='Known Points (Generated Nodes)')

    # Annotate known points
    for i, (x, y) in enumerate(known_xy):
        ax.annotate(f'K{i}', (x, y), textcoords="offset points", xytext=(5, 5),
                    color='darkred', fontsize=11, fontweight='bold', zorder=10)

    # Plot auxiliary points
    ax.scatter(aux_xy[:, 0], aux_xy[:, 1], marker='o', facecolors='none', edgecolors='blue',
               linewidths=2.5, s=120, zorder=6,
               label='Auxiliary Vertices (A / E)')

    # Annotate auxiliary points: first 3 are A (vertices), rest are E (extras)
    for i, (x, y) in enumerate(aux_xy):
        label = f'A{i}' if i < 3 else f'E{i - 3}'
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 8),
                    color='darkblue', fontsize=11, fontweight='bold', zorder=10)

    ax.set_title(f'Actual Grid Connectivity | method={method}, k={k} (Dubiner-Tikhonov)', fontsize=14)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.14))
    plt.tight_layout()
    plt.show()

    # ---------- Plot 2: 2D Exact vs Linear Rendering ----------
    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))

    x0, x1, y0, y1 = _compute_plot_bounds(vertices_2d_resolved)
    gx, gy = np.mgrid[x0:x1:220j, y0:y1:220j]
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    # Create triangle interior mask using barycentric coordinates
    A = np.array([
        [vertices_2d_resolved[0, 0], vertices_2d_resolved[1, 0], vertices_2d_resolved[2, 0]],
        [vertices_2d_resolved[0, 1], vertices_2d_resolved[1, 1], vertices_2d_resolved[2, 1]],
        [1.0, 1.0, 1.0],
    ])
    A_inv = np.linalg.inv(A)
    bary_grid = (A_inv @ np.vstack([pts[:, 0], pts[:, 1], np.ones(len(pts))])).T
    inside = np.all(bary_grid >= -1e-10, axis=1).reshape(gx.shape)

    u_exact = exact_solution(gx, gy, center=resolved_center)
    u_exact_masked = np.ma.masked_where(~inside, u_exact)

    # Plot 2a: Exact solution
    im0 = axs2[0].imshow(
        u_exact_masked.T,
        extent=(x0, x1, y0, y1),
        origin='lower',
        cmap='viridis',
        vmin=0.0,
        vmax=1.0,
    )
    axs2[0].set_title('Exact Solution (2D)')
    axs2[0].set_aspect('equal')
    axs2[0].axis('off')

    # Plot 2b: Reconstructed solution
    im1 = axs2[1].tricontourf(all_xy[:, 0], all_xy[:, 1], all_vals, levels=30, cmap='viridis',
                              vmin=0.0, vmax=1.0)
    axs2[1].triplot(triang, 'k-', alpha=0.25, linewidth=0.8)
    axs2[1].set_title('Linear Triangular Grid Rendering (2D) - Dubiner-Tikhonov')
    axs2[1].set_aspect('equal')
    axs2[1].axis('off')

    # Shared colorbar
    cax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar2 = fig2.colorbar(im0, cax=cax, ax=axs2)
    cbar2.set_label('u(x, y)')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()

    # ---------- Plot 3: 3D Exact vs Linear Rendering ----------
    fig3 = plt.figure(figsize=(14, 6))

    ax3d_1 = fig3.add_subplot(121, projection='3d')
    surf_exact = ax3d_1.plot_surface(gx, gy, u_exact_masked, cmap='viridis', edgecolor='none',
                                     vmin=0.0, vmax=1.0, rstride=1, cstride=1)
    ax3d_1.set_title('Exact Surface (3D)')
    ax3d_1.set_xlabel('x')
    ax3d_1.set_ylabel('y')
    ax3d_1.set_zlabel('u')
    ax3d_1.view_init(elev=32, azim=-45)

    ax3d_2 = fig3.add_subplot(122, projection='3d')
    surf_tri = ax3d_2.plot_trisurf(all_xy[:, 0], all_xy[:, 1], all_vals, triangles=triang.triangles,
                                   cmap='viridis', vmin=0.0, vmax=1.0,
                                   edgecolor='black', linewidth=0.2, alpha=0.95)
    ax3d_2.set_title('Linear Triangular Mesh (3D) - Dubiner-Tikhonov')
    ax3d_2.set_xlabel('x')
    ax3d_2.set_ylabel('y')
    ax3d_2.set_zlabel('u')
    ax3d_2.set_zlim(0.0, 1.0)
    ax3d_2.view_init(elev=32, azim=-45)

    # Shared colorbar
    cax3 = fig3.add_axes([0.91, 0.18, 0.02, 0.65])
    cbar3 = fig3.colorbar(surf_exact, cax=cax3, ax=[ax3d_1, ax3d_2])
    cbar3.set_label('u(x, y)')

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.show()
