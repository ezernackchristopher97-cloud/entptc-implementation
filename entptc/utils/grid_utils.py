"""
Grid Utility Functions
=======================

Utility functions for creating grid adjacency matrices.

"""

import numpy as np

def create_toroidal_grid(grid_size: int) -> np.ndarray:
 """
 Create toroidal grid adjacency matrix.
 
 Args:
 grid_size: size of grid (grid_size × grid_size)
 
 Returns:
 adjacency: (grid_size^2, grid_size^2) adjacency matrix
 """
 n_nodes = grid_size * grid_size
 adjacency = np.zeros((n_nodes, n_nodes))
 
 for i in range(grid_size):
 for j in range(grid_size):
 node = i * grid_size + j
 
 # Right neighbor (with wraparound)
 right = i * grid_size + ((j + 1) % grid_size)
 adjacency[node, right] = 1
 adjacency[right, node] = 1
 
 # Down neighbor (with wraparound)
 down = ((i + 1) % grid_size) * grid_size + j
 adjacency[node, down] = 1
 adjacency[down, node] = 1
 
 return adjacency
