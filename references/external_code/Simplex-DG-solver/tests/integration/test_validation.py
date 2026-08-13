#!/usr/bin/env python
"""
Test script for geometric validation and connectivity visualization.
"""

import numpy as np
import sys

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


if __name__ == '__main__':
    from src.core.connectivity import build_connectivity
    from src.core.validation import check_geometric_centroids, plot_connectivity_map
    
    print("=" * 80)
    print("GEOMETRIC VALIDATION AND VISUALIZATION TEST")
    print("=" * 80)
    print()
    
    # Generate mesh (Level 2 subdivision)
    print("Generating subdivided mesh (n_div=4)...")
    nodes, triangles = generate_subdivided_triangle(4)
    EToV = triangles
    
    print(f"  Mesh size: {EToV.shape[0]} elements, {nodes.shape[0]} vertices")
    print()
    
    # Build connectivity
    print("Building connectivity matrices...")
    EToE, EToF = build_connectivity(EToV)
    print()
    
    # Test 1: Geometric validation
    print("-" * 80)
    print("TEST 1: Geometric Centroid Validation")
    print("-" * 80)
    print()
    
    try:
        check_geometric_centroids(nodes, EToV, EToE, EToF)
        print("✓ TEST 1 PASSED: All geometric centroids match")
    except AssertionError as e:
        print(f"✗ TEST 1 FAILED: {e}")
        sys.exit(1)
    
    print()
    
    # Test 2: Visualization (without display)
    print("-" * 80)
    print("TEST 2: Connectivity Visualization")
    print("-" * 80)
    print()
    
    try:
        fig, ax = plot_connectivity_map(nodes, EToV, EToE, 
                                       save_path='/tmp/connectivity_map.png')
        print("✓ TEST 2 PASSED: Visualization generated and saved")
        print("  File: /tmp/connectivity_map.png")
    except Exception as e:
        print(f"✗ TEST 2 FAILED: {e}")
        sys.exit(1)
    
    print()
    print("=" * 80)
    print("ALL TESTS PASSED")
    print("=" * 80)
