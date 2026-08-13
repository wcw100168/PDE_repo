"""
Geometric validation and visualization tools for mesh connectivity.

Provides functions to:
1. Verify geometric consistency of shared element faces
2. Visualize mesh topology with element adjacencies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches


def check_geometric_centroids(nodes, EToV, EToE, EToF, tol=1e-10):
    """
    Verify that shared faces between adjacent elements have matching physical midpoints.
    
    For each internal face defined by EToE/EToF, calculates the physical midpoint
    on both adjacent elements and verifies they match within floating-point tolerance.
    This serves as a geometric sanity check that the topological mapping is correct.
    
    Parameters
    ----------
    nodes : np.ndarray
        Physical coordinates of mesh vertices, shape (N_vertices, 2)
        Each row contains (x, y) coordinates
        
    EToV : np.ndarray
        Element-to-Vertex connectivity, shape (N_elements, 3)
        Each row contains global vertex indices for a triangular element
        
    EToE : np.ndarray
        Element-to-Element connectivity, shape (N_elements, 3)
        EToE[k, f] = ID of element sharing face f of element k
        Boundary faces satisfy EToE[k, f] = k
        
    EToF : np.ndarray
        Element-to-Face connectivity, shape (N_elements, 3)
        EToF[k, f] = local face index in the neighboring element
        
    tol : float
        Tolerance for floating-point comparison (default: 1e-10)
        
    Returns
    -------
    is_valid : bool
        True if all internal face midpoints match (assertion passes)
        
    Raises
    ------
    AssertionError
        If any internal face midpoints don't match within tolerance
        
    Notes
    -----
    Face Midpoint Calculation:
        For a face connecting global vertices v1 and v2:
        midpoint = (nodes[v1] + nodes[v2]) / 2
        
    Since elements should share complete faces (not just vertices), the midpoints
    of the same geometric face computed from either adjacent element must be identical.
    This check ensures no topological errors exist.
    
    Examples
    --------
    >>> nodes = np.array([[0,0], [1,0], [0,1], [1,1]])
    >>> EToV = np.array([[0,1,2], [1,3,2]])
    >>> EToE, EToF = build_connectivity(EToV)
    >>> check_geometric_centroids(nodes, EToV, EToE, EToF)
    ✓ Geometric validation PASSED
      Internal faces checked: 1
      All face midpoints match (within tolerance: 1e-10)
    True
    """
    
    # Local face-to-vertex mapping (CCW convention, 0-based indexing)
    local_faces = np.array([
        [0, 1],  # Face 0: connects local vertices 0 and 1
        [1, 2],  # Face 1: connects local vertices 1 and 2
        [2, 0],  # Face 2: connects local vertices 2 and 0
    ])
    
    N_elements = EToV.shape[0]
    internal_faces_checked = 0
    mismatches = []
    
    print("=" * 80)
    print("GEOMETRIC CENTROID VALIDATION")
    print("=" * 80)
    print()
    
    # Iterate through all elements and faces
    for k in range(N_elements):
        for f in range(3):
            # Skip boundary faces (EToE[k, f] == k means boundary)
            if EToE[k, f] == k:
                continue
            
            # Get neighbor element and face index
            k_neighbor = EToE[k, f]
            f_neighbor = EToF[k, f]
            
            # Get local vertex indices for this face on element k
            v1_local, v2_local = local_faces[f]
            # Get local vertex indices for same face on neighbor element
            v1_neighbor_local, v2_neighbor_local = local_faces[f_neighbor]
            
            # Convert to global vertex indices
            v1_global = EToV[k, v1_local]
            v2_global = EToV[k, v2_local]
            v1_neighbor_global = EToV[k_neighbor, v1_neighbor_local]
            v2_neighbor_global = EToV[k_neighbor, v2_neighbor_local]
            
            # Get physical coordinates from nodes array
            p1 = nodes[v1_global]
            p2 = nodes[v2_global]
            p1_neighbor = nodes[v1_neighbor_global]
            p2_neighbor = nodes[v2_neighbor_global]
            
            # Compute midpoints of the shared face
            midpoint_k = (p1 + p2) / 2.0
            midpoint_neighbor = (p1_neighbor + p2_neighbor) / 2.0
            
            # Check if midpoints match within tolerance
            if not np.allclose(midpoint_k, midpoint_neighbor, atol=tol):
                mismatch_dist = np.linalg.norm(midpoint_k - midpoint_neighbor)
                mismatches.append({
                    'elem_k': k,
                    'face_k': f,
                    'elem_neighbor': k_neighbor,
                    'face_neighbor': f_neighbor,
                    'midpoint_k': midpoint_k,
                    'midpoint_neighbor': midpoint_neighbor,
                    'distance': mismatch_dist,
                })
            
            internal_faces_checked += 1
    
    # Report results
    if mismatches:
        print(f"✗ Geometric validation FAILED")
        print(f"  Found {len(mismatches)} mismatched face midpoints:")
        print()
        for i, mismatch in enumerate(mismatches, 1):
            print(f"  {i}. Element {mismatch['elem_k']}, Face {mismatch['face_k']}")
            print(f"     ↔ Element {mismatch['elem_neighbor']}, Face {mismatch['face_neighbor']}")
            print(f"     Midpoint (Elem {mismatch['elem_k']}): {mismatch['midpoint_k']}")
            print(f"     Midpoint (Elem {mismatch['elem_neighbor']}): {mismatch['midpoint_neighbor']}")
            print(f"     Distance: {mismatch['distance']:.2e}")
            print()
        raise AssertionError(f"Geometric validation failed: {len(mismatches)} face midpoint mismatches")
    
    print(f"✓ Geometric validation PASSED")
    print(f"  Internal faces checked: {internal_faces_checked}")
    print(f"  All face midpoints match (within tolerance: {tol:.2e})")
    print()
    
    return True


def plot_connectivity_map(nodes, EToV, EToE, figsize=(12, 10), save_path=None):
    """
    Visualize the mesh with element IDs and directed element adjacency connections.
    
    Creates a comprehensive visualization showing:
    - Mesh edges colored by type (interior=black, boundary=orange)
    - Element centroids marked with their numeric IDs (red)
    - Directed arrows between adjacent elements (blue)
    - Vertex nodes (black dots)
    
    Parameters
    ----------
    nodes : np.ndarray
        Physical coordinates of mesh vertices, shape (N_vertices, 2)
        Each row contains (x, y) coordinates
        
    EToV : np.ndarray
        Element-to-Vertex connectivity, shape (N_elements, 3)
        Each row contains global vertex indices for a triangular element
        
    EToE : np.ndarray
        Element-to-Element connectivity, shape (N_elements, 3)
        Used to identify boundary vs. interior edges
        
    figsize : tuple
        Figure dimensions (width, height) in inches. Default: (12, 10)
        
    save_path : str, optional
        If provided, saves the figure to this path. Format inferred from extension.
        
    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes objects for further customization
        
    Notes
    -----
    Visual Legend:
        - Black lines (thin): Interior mesh edges
        - Orange lines (thick): Boundary mesh edges
        - Black dots: Mesh vertices
        - Red numbers: Element IDs at centroids
        - Blue curved arrows: Element adjacency connections
        
    The arrows use bezier curves (arc3,rad=0.1) to avoid overlap when
    multiple connections exist between regions.
    
    Examples
    --------
    >>> fig, ax = plot_connectivity_map(nodes, EToV, EToE)
    >>> plt.show()
    
    >>> fig, ax = plot_connectivity_map(nodes, EToV, EToE, 
    ...                                  save_path='mesh_connectivity.png')
    """
    
    print("Generating connectivity map visualization...")
    print()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    N_elements = EToV.shape[0]
    N_vertices = nodes.shape[0]
    
    # Local face-to-vertex mapping (CCW convention)
    local_faces = np.array([[0, 1], [1, 2], [2, 0]])
    
    # Track drawn edges and connections to avoid duplicates
    drawn_edges = set()
    drawn_connections = set()
    
    # ========================================================================
    # Step 1: Draw all mesh edges
    # ========================================================================
    print("Drawing mesh edges...")
    
    for k in range(N_elements):
        for f in range(3):
            v1_local, v2_local = local_faces[f]
            v1_global = EToV[k, v1_local]
            v2_global = EToV[k, v2_local]
            
            # Create canonical edge key (sorted) to avoid duplicates
            edge_key = tuple(sorted([v1_global, v2_global]))
            
            # Skip if already drawn
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            
            # Get physical coordinates
            p1 = nodes[v1_global]
            p2 = nodes[v2_global]
            
            # Determine if boundary or interior edge
            is_boundary = EToE[k, f] == k
            
            # Set visualization parameters
            if is_boundary:
                color = 'orange'
                linewidth = 2.5
                linestyle = '-'
                alpha = 0.8
            else:
                color = 'black'
                linewidth = 1.0
                linestyle = '-'
                alpha = 0.5
            
            # Draw edge
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, linewidth=linewidth, linestyle=linestyle, 
                   alpha=alpha, zorder=1)
    
    print(f"  {len(drawn_edges)} unique edges drawn")
    
    # ========================================================================
    # Step 2: Calculate and plot element centroids with IDs
    # ========================================================================
    print("Computing element centroids...")
    
    centroids = np.zeros((N_elements, 2))
    
    for k in range(N_elements):
        # Get vertices of element k
        v_indices = EToV[k]
        element_nodes = nodes[v_indices]
        
        # Compute centroid as mean of vertices
        centroid = element_nodes.mean(axis=0)
        centroids[k] = centroid
        
        # Plot centroid marker (red dot)
        ax.plot(centroid[0], centroid[1], 'ro', markersize=8, zorder=5)
        
        # Add element ID label
        ax.text(centroid[0], centroid[1], str(k), 
               fontsize=10, fontweight='bold', color='red',
               ha='center', va='center', zorder=6,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='red', alpha=0.7))
    
    print(f"  {N_elements} element centroids plotted")
    
    # ========================================================================
    # Step 3: Draw directed arrows between adjacent elements
    # ========================================================================
    print("Drawing element connectivity arrows...")
    
    for k in range(N_elements):
        for f in range(3):
            # Skip boundary edges
            if EToE[k, f] == k:
                continue
            
            k_neighbor = EToE[k, f]
            
            # Create canonical connection key to avoid duplicates
            conn_key = tuple(sorted([k, k_neighbor]))
            
            if conn_key in drawn_connections:
                continue
            drawn_connections.add(conn_key)
            
            # Draw arrow from element k to neighbor
            start = centroids[k]
            end = centroids[k_neighbor]
            
            # Use FancyArrowPatch for better visualization
            arrow = FancyArrowPatch(
                start, end,
                arrowstyle='<->', mutation_scale=15,
                color='blue', linewidth=1.5, alpha=0.6, zorder=3,
                connectionstyle="arc3,rad=0.1"
            )
            ax.add_patch(arrow)
    
    print(f"  {len(drawn_connections)} element connections drawn")
    
    # ========================================================================
    # Step 4: Plot vertex nodes
    # ========================================================================
    print("Plotting vertex nodes...")
    
    ax.scatter(nodes[:, 0], nodes[:, 1], c='black', s=30, zorder=4, 
              marker='o', alpha=0.6)
    
    print(f"  {N_vertices} vertices plotted")
    
    # ========================================================================
    # Step 5: Format plot
    # ========================================================================
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('x', fontsize=12, fontweight='bold')
    ax.set_ylabel('y', fontsize=12, fontweight='bold')
    ax.set_title('Mesh Connectivity Map with Element Adjacencies', 
                fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=1.0, label='Interior Edges'),
        Line2D([0], [0], color='orange', linewidth=2.5, label='Boundary Edges'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
              markersize=8, label='Element Centroids (ID in red)'),
        Line2D([0], [0], color='blue', linewidth=1.5, marker='>', 
              markersize=6, label='Element Connections'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
              markersize=5, label='Mesh Vertices'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
             framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    print()
    
    return fig, ax


if __name__ == "__main__":
    # Test with small subdivided mesh
    print("=" * 80)
    print("GEOMETRIC VALIDATION AND VISUALIZATION TEST")
    print("=" * 80)
    print()
    
    from src.core.connectivity import build_connectivity
    
    # Generate subdivided reference triangle (Level 2)
    def generate_subdivided_triangle(n_div):
        """Generate subdivided reference triangle mesh"""
        nodes = []
        for j in range(n_div + 1):
            for i in range(n_div + 1 - j):
                x = i / n_div
                y = j / n_div
                nodes.append((x, y))
        nodes = np.array(nodes)

        triangles = []
        node_idx = {}
        curr = 0
        for j in range(n_div + 1):
            for i in range(n_div + 1 - j):
                node_idx[(i, j)] = curr
                curr += 1
                
        for j in range(n_div):
            for i in range(n_div - j):
                triangles.append([node_idx[(i, j)], node_idx[(i+1, j)], node_idx[(i, j+1)]])
                if i + j + 1 < n_div:
                    triangles.append([node_idx[(i+1, j)], node_idx[(i+1, j+1)], node_idx[(i, j+1)]])
                    
        return nodes, np.array(triangles)
    
    # Generate mesh
    print("Generating subdivided mesh (n_div=2)...")
    nodes, triangles = generate_subdivided_triangle(2)
    EToV = triangles
    
    print(f"  Vertices: {nodes.shape[0]}")
    print(f"  Elements: {EToV.shape[0]}")
    print()
    
    # Build connectivity
    print("Building connectivity matrices...")
    EToE, EToF = build_connectivity(EToV)
    print()
    
    # Geometric validation
    print("Checking geometric consistency...")
    check_geometric_centroids(nodes, EToV, EToE, EToF)
    
    # Visualization
    print("Generating connectivity visualization...")
    fig, ax = plot_connectivity_map(nodes, EToV, EToE, 
                                    save_path='/tmp/connectivity_map.png')
    print("✓ Visualization complete")
    print()
    
    print("=" * 80)
    print("VALIDATION AND VISUALIZATION COMPLETE")
    print("=" * 80)
