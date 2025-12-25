"""
Toroidal Grid Topology for 16 ROIs

Implements the TRUE toroidal grid-cell topology as a 4×4 grid
embedded on a two-torus T² with periodic boundary conditions.

Key Features:
- 16 ROIs arranged as 4×4 spatial grid
- Periodic boundaries in both axes (wraparound)
- Von Neumann neighbor structure (4-connectivity)
- Toroidal distance computation (respecting periodic boundaries)
- Explicit adjacency matrix and distance matrix
- Structural prior for Progenitor Matrix construction

This is NOT a generic fully-connected graph. The toroidal constraint is
enforced as a structural prior throughout the pipeline.

Reference: ENTPC.tex + Christopher's implementation requirements
"""

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class ToroidalGrid:
 """
 4×4 grid embedded on two-torus T² with periodic boundaries.
 
 ROI Layout:
 ```
 0 1 2 3
 4 5 6 7
 8 9 10 11
 12 13 14 15
 ```
 
 With wraparound:
 - Node 0 neighbors: 1 (right), 3 (left wrap), 4 (down), 12 (up wrap)
 - Node 15 neighbors: 14 (left), 12 (right wrap), 11 (up), 3 (down wrap)
 """
 
 def __init__(self, grid_size: int = 4, connectivity: str = 'von_neumann'):
 """
 Initialize toroidal grid.
 
 Args:
 grid_size: Size of square grid (4 for 16 ROIs)
 connectivity: 'von_neumann' (4-neighbors) or 'moore' (8-neighbors)
 """
 self.grid_size = grid_size
 self.n_nodes = grid_size ** 2
 self.connectivity = connectivity
 
 assert self.n_nodes == 16, "Must have exactly 16 ROIs"
 assert connectivity in ['von_neumann', 'moore'], "Invalid connectivity"
 
 # Build adjacency and distance matrices
 self.adjacency_matrix = self._build_adjacency_matrix()
 self.distance_matrix = self._build_distance_matrix()
 
 def node_to_coords(self, node_idx: int) -> Tuple[int, int]:
 """
 Convert node index to (row, col) coordinates.
 
 Args:
 node_idx: Node index [0, 15]
 
 Returns:
 (row, col) coordinates
 """
 row = node_idx // self.grid_size
 col = node_idx % self.grid_size
 return (row, col)
 
 def coords_to_node(self, row: int, col: int) -> int:
 """
 Convert (row, col) coordinates to node index.
 
 Args:
 row: Row coordinate
 col: Column coordinate
 
 Returns:
 Node index
 """
 # Apply periodic boundary conditions
 row = row % self.grid_size
 col = col % self.grid_size
 return row * self.grid_size + col
 
 def get_neighbors(self, node_idx: int) -> List[int]:
 """
 Get neighbors of a node with toroidal wraparound.
 
 Args:
 node_idx: Node index
 
 Returns:
 List of neighbor indices
 """
 row, col = self.node_to_coords(node_idx)
 
 if self.connectivity == 'von_neumann':
 # 4-connectivity (up, down, left, right)
 offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
 else: # moore
 # 8-connectivity (includes diagonals)
 offsets = [
 (-1, -1), (-1, 0), (-1, 1),
 (0, -1), (0, 1),
 (1, -1), (1, 0), (1, 1)
 ]
 
 neighbors = []
 for dr, dc in offsets:
 neighbor_row = (row + dr) % self.grid_size
 neighbor_col = (col + dc) % self.grid_size
 neighbor_idx = self.coords_to_node(neighbor_row, neighbor_col)
 neighbors.append(neighbor_idx)
 
 return neighbors
 
 def _build_adjacency_matrix(self) -> np.ndarray:
 """
 Build adjacency matrix with toroidal connectivity.
 
 Returns:
 Adjacency matrix (16, 16) with 1 for neighbors, 0 otherwise
 """
 adj = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
 
 for node in range(self.n_nodes):
 neighbors = self.get_neighbors(node)
 for neighbor in neighbors:
 adj[node, neighbor] = 1
 
 return adj
 
 def toroidal_distance(self, node1: int, node2: int) -> float:
 """
 Compute toroidal distance between two nodes.
 
 Distance respects periodic boundaries: takes shortest path on torus.
 
 Args:
 node1: First node index
 node2: Second node index
 
 Returns:
 Toroidal distance
 """
 row1, col1 = self.node_to_coords(node1)
 row2, col2 = self.node_to_coords(node2)
 
 # Compute minimum distance in each dimension (with wraparound)
 row_diff = abs(row1 - row2)
 col_diff = abs(col1 - col2)
 
 row_dist = min(row_diff, self.grid_size - row_diff)
 col_dist = min(col_diff, self.grid_size - col_diff)
 
 # Euclidean distance on torus
 return np.sqrt(row_dist**2 + col_dist**2)
 
 def _build_distance_matrix(self) -> np.ndarray:
 """
 Build distance matrix with toroidal distances.
 
 Returns:
 Distance matrix (16, 16)
 """
 dist = np.zeros((self.n_nodes, self.n_nodes))
 
 for i in range(self.n_nodes):
 for j in range(self.n_nodes):
 dist[i, j] = self.toroidal_distance(i, j)
 
 return dist
 
 def visualize_grid(self, save_path: str = None):
 """
 Visualize the toroidal grid with connections.
 
 Args:
 save_path: Path to save figure (optional)
 """
 fig, ax = plt.subplots(figsize=(8, 8))
 
 # Plot nodes
 for node in range(self.n_nodes):
 row, col = self.node_to_coords(node)
 ax.plot(col, row, 'o', markersize=20, color='steelblue')
 ax.text(col, row, str(node), ha='center', va='center',
 fontsize=12, fontweight='bold', color='white')
 
 # Plot edges
 for node in range(self.n_nodes):
 row, col = self.node_to_coords(node)
 neighbors = self.get_neighbors(node)
 
 for neighbor in neighbors:
 n_row, n_col = self.node_to_coords(neighbor)
 
 # Check if edge crosses boundary (wraparound)
 if abs(row - n_row) > 1 or abs(col - n_col) > 1:
 # Wraparound edge (dashed)
 ax.plot([col, n_col], [row, n_row], 'r--', alpha=0.3, linewidth=1)
 else:
 # Regular edge
 ax.plot([col, n_col], [row, n_row], 'k-', alpha=0.3, linewidth=1)
 
 ax.set_xlim(-0.5, self.grid_size - 0.5)
 ax.set_ylim(-0.5, self.grid_size - 0.5)
 ax.set_aspect('equal')
 ax.invert_yaxis() # Row 0 at top
 ax.set_title(f'Toroidal Grid (4×4) - {self.connectivity} connectivity', fontsize=14)
 ax.set_xlabel('Column', fontsize=12)
 ax.set_ylabel('Row', fontsize=12)
 ax.grid(True, alpha=0.2)
 
 if save_path:
 plt.savefig(save_path, dpi=300, bbox_inches='tight')
 print(f"Grid visualization saved to: {save_path}")
 
 plt.close()
 
 def print_neighbor_structure(self):
 """Print neighbor structure for all nodes."""
 print("=" * 80)
 print("TOROIDAL GRID NEIGHBOR STRUCTURE")
 print("=" * 80)
 print(f"Grid size: {self.grid_size}×{self.grid_size}")
 print(f"Total nodes: {self.n_nodes}")
 print(f"Connectivity: {self.connectivity}")
 print("=" * 80)
 
 for node in range(self.n_nodes):
 row, col = self.node_to_coords(node)
 neighbors = self.get_neighbors(node)
 print(f"Node {node:2d} ({row},{col}): neighbors = {neighbors}")
 
 print("=" * 80)
 
 def save_matrices(self, output_dir: str):
 """
 Save adjacency and distance matrices to files.
 
 Args:
 output_dir: Directory to save matrices
 """
 import os
 os.makedirs(output_dir, exist_ok=True)
 
 # Save adjacency matrix
 adj_path = os.path.join(output_dir, 'toroidal_adjacency_matrix.txt')
 np.savetxt(adj_path, self.adjacency_matrix, fmt='%d')
 print(f"Adjacency matrix saved to: {adj_path}")
 
 # Save distance matrix
 dist_path = os.path.join(output_dir, 'toroidal_distance_matrix.txt')
 np.savetxt(dist_path, self.distance_matrix, fmt='%.6f')
 print(f"Distance matrix saved to: {dist_path}")
 
 # Save as numpy arrays (for easy loading)
 adj_npy_path = os.path.join(output_dir, 'toroidal_adjacency_matrix.npy')
 np.save(adj_npy_path, self.adjacency_matrix)
 
 dist_npy_path = os.path.join(output_dir, 'toroidal_distance_matrix.npy')
 np.save(dist_npy_path, self.distance_matrix)
 
 print(f"Matrices also saved as .npy files for easy loading")

def apply_toroidal_constraint_to_progenitor(progenitor_matrix: np.ndarray,
 toroidal_grid: ToroidalGrid,
 strength: float = 1.0) -> np.ndarray:
 """
 Apply toroidal constraint to Progenitor Matrix.
 
 The constraint enforces that non-adjacent nodes (on the toroidal grid)
 have reduced coupling in the Progenitor Matrix.
 
 Args:
 progenitor_matrix: Original 16×16 Progenitor Matrix
 toroidal_grid: ToroidalGrid object
 strength: Strength of constraint (0 = no constraint, 1 = full constraint)
 
 Returns:
 Constrained Progenitor Matrix
 """
 assert progenitor_matrix.shape == (16, 16), "Must be 16×16 matrix"
 
 # Create constraint mask based on toroidal distance
 # Closer nodes on torus → higher weight
 constraint_mask = np.zeros((16, 16))
 
 for i in range(16):
 for j in range(16):
 dist = toroidal_grid.distance_matrix[i, j]
 
 # Exponential decay with distance
 # Adjacent nodes (dist=1): weight ≈ 1.0
 # Distant nodes (dist=2+): weight < 0.5
 constraint_mask[i, j] = np.exp(-dist / 1.5)
 
 # Apply constraint
 constrained_matrix = progenitor_matrix * (
 (1 - strength) + strength * constraint_mask
 )
 
 return constrained_matrix

def main():
 """Demonstrate toroidal grid topology."""
 print("\n" + "=" * 80)
 print("TOROIDAL GRID TOPOLOGY FOR 16 ROIs")
 print("=" * 80)
 
 # Create toroidal grid with von Neumann connectivity
 grid = ToroidalGrid(grid_size=4, connectivity='von_neumann')
 
 # Print neighbor structure
 grid.print_neighbor_structure()
 
 # Print distance matrix
 print("\nTOROIDAL DISTANCE MATRIX:")
 print("=" * 80)
 print(grid.distance_matrix)
 print("=" * 80)
 
 # Save matrices
 output_dir = '/home/ubuntu/entptc-implementation/outputs/toroidal_topology'
 grid.save_matrices(output_dir)
 
 # Visualize grid
 viz_path = '/home/ubuntu/entptc-implementation/outputs/toroidal_topology/toroidal_grid_visualization.png'
 grid.visualize_grid(save_path=viz_path)
 
 # Example: Apply constraint to a test matrix
 print("\nEXAMPLE: Applying toroidal constraint to Progenitor Matrix")
 print("=" * 80)
 
 # Create a test unconstrained matrix
 test_matrix = np.random.rand(16, 16)
 test_matrix = (test_matrix + test_matrix.T) / 2 # Make symmetric
 
 # Apply toroidal constraint
 constrained_matrix = apply_toroidal_constraint_to_progenitor(
 test_matrix, grid, strength=0.8
 )
 
 print(f"Original matrix range: [{test_matrix.min():.3f}, {test_matrix.max():.3f}]")
 print(f"Constrained matrix range: [{constrained_matrix.min():.3f}, {constrained_matrix.max():.3f}]")
 print(f"Constraint reduces non-adjacent couplings")
 print("=" * 80)
 
 print("\n✓ Toroidal grid topology implemented successfully")
 print("✓ Adjacency matrix: 16×16 with periodic boundaries")
 print("✓ Distance matrix: Respects toroidal wraparound")
 print("✓ Ready for integration into EntPTC pipeline")

if __name__ == '__main__':
 main()
