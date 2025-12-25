"""
Delaunay Triangulation on Toroidal Manifold T³

Implements 3D periodic Delaunay triangulation for analyzing
the structure of point sets on the toroidal manifold T³. This provides
a natural definition of "neighborhood" and enables structural analysis
of the entropy field.

References:
- de Berg et al. (2008), Chapter 9: Delaunay Triangulations
- Handbook of Computational Geometry (Sack & Urrutia, 2000), Chapter 11
- Rycroft (2009): Voro++: A three-dimensional Voronoi cell library in C++

Per ENTPC.tex: This enables neighborhood analysis on T³ for studying
the geometric structure of the entropy field and identifying regions
of high/low organization.

"""

import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
import warnings

@dataclass
class ToroidalPoint:
 """
 Point on the toroidal manifold T³.
 
 Represented by three angles (θ₁, θ₂, θ₃) ∈ [0, 2π)³.
 """
 theta1: float
 theta2: float
 theta3: float
 index: int = -1 # Index in point set
 
 def __post_init__(self):
 """Wrap angles to [0, 2π)."""
 self.theta1 = np.mod(self.theta1, 2*np.pi)
 self.theta2 = np.mod(self.theta2, 2*np.pi)
 self.theta3 = np.mod(self.theta3, 2*np.pi)
 
 def to_array(self) -> np.ndarray:
 """Convert to numpy array."""
 return np.array([self.theta1, self.theta2, self.theta3])
 
 def distance_to(self, other: 'ToroidalPoint') -> float:
 """
 Compute geodesic distance to another point on T³.
 
 Uses angular distance with periodic boundaries.
 """
 def angular_distance(a, b):
 diff = abs(a - b)
 return min(diff, 2*np.pi - diff)
 
 d1 = angular_distance(self.theta1, other.theta1)
 d2 = angular_distance(self.theta2, other.theta2)
 d3 = angular_distance(self.theta3, other.theta3)
 
 return np.sqrt(d1**2 + d2**2 + d3**2)

@dataclass
class DelaunayTetrahedron:
 """
 Tetrahedron in the Delaunay triangulation.
 
 Defined by 4 vertex indices.
 """
 v0: int
 v1: int
 v2: int
 v3: int
 
 def vertices(self) -> List[int]:
 """Get list of vertex indices."""
 return [self.v0, self.v1, self.v2, self.v3]
 
 def contains_vertex(self, vertex_index: int) -> bool:
 """Check if tetrahedron contains a vertex."""
 return vertex_index in self.vertices()
 
 def volume(self, points: np.ndarray) -> float:
 """
 Compute volume of tetrahedron.
 
 Args:
 points: Array of point coordinates (n_points, 3)
 
 Returns:
 Volume of tetrahedron
 """
 # Get vertex coordinates
 p0 = points[self.v0]
 p1 = points[self.v1]
 p2 = points[self.v2]
 p3 = points[self.v3]
 
 # Volume = |det([p1-p0, p2-p0, p3-p0])| / 6
 mat = np.array([
 p1 - p0,
 p2 - p0,
 p3 - p0
 ])
 
 return abs(np.linalg.det(mat)) / 6.0

class PeriodicDelaunayT3:
 """
 Periodic Delaunay triangulation on T³.
 
 This implements Delaunay triangulation with periodic boundary conditions,
 which is essential for the toroidal topology.
 
 The algorithm works by:
 1. Replicating the point set in all 27 periodic images (3³ cells)
 2. Computing standard Delaunay triangulation
 3. Filtering tetrahedra to keep only those in the central cell
 4. Mapping vertex indices back to original points
 """
 
 def __init__(self, points: List[ToroidalPoint]):
 """
 Initialize periodic Delaunay triangulation.
 
 Args:
 points: List of points on T³
 """
 self.points = points
 self.n_points = len(points)
 
 # Assign indices
 for i, p in enumerate(points):
 p.index = i
 
 # Convert to array
 self.point_array = np.array([p.to_array() for p in points])
 
 # Compute triangulation
 self.tetrahedra = []
 self.neighbors = {} # vertex_index → set of neighbor indices
 self._compute_triangulation()
 
 def _compute_triangulation(self):
 """
 Compute periodic Delaunay triangulation.
 
 Uses the replication method: replicate points in 27 periodic images,
 compute standard Delaunay, then filter.
 """
 # Replicate points in 27 periodic images
 replicated_points = []
 point_to_original = [] # Maps replicated index to original index
 
 for dx in [-1, 0, 1]:
 for dy in [-1, 0, 1]:
 for dz in [-1, 0, 1]:
 offset = np.array([dx, dy, dz]) * 2*np.pi
 
 for i, p in enumerate(self.point_array):
 replicated_points.append(p + offset)
 point_to_original.append(i)
 
 replicated_points = np.array(replicated_points)
 
 # Compute standard Delaunay triangulation
 try:
 tri = Delaunay(replicated_points)
 except Exception as e:
 warnings.warn(f"Delaunay triangulation failed: {e}")
 return
 
 # Filter tetrahedra: keep only those with at least one vertex in central cell
 central_cell_indices = set(range(self.n_points, 2*self.n_points))
 
 for simplex in tri.simplices:
 # Map replicated indices to original indices
 original_indices = [point_to_original[idx] for idx in simplex]
 
 # Check if any vertex is in central cell (second replication)
 if any(idx in central_cell_indices for idx in simplex):
 # Create tetrahedron
 tet = DelaunayTetrahedron(
 v0=original_indices[0],
 v1=original_indices[1],
 v2=original_indices[2],
 v3=original_indices[3]
 )
 self.tetrahedra.append(tet)
 
 # Update neighbors
 for i in range(4):
 for j in range(i+1, 4):
 vi = original_indices[i]
 vj = original_indices[j]
 
 if vi not in self.neighbors:
 self.neighbors[vi] = set()
 if vj not in self.neighbors:
 self.neighbors[vj] = set()
 
 self.neighbors[vi].add(vj)
 self.neighbors[vj].add(vi)
 
 def get_neighbors(self, point_index: int) -> Set[int]:
 """
 Get indices of neighboring points in the Delaunay triangulation.
 
 Args:
 point_index: Index of the query point
 
 Returns:
 Set of neighbor indices
 """
 return self.neighbors.get(point_index, set())
 
 def get_neighbor_distances(self, point_index: int) -> Dict[int, float]:
 """
 Get distances to all neighbors.
 
 Args:
 point_index: Index of the query point
 
 Returns:
 Dictionary mapping neighbor index → distance
 """
 neighbors = self.get_neighbors(point_index)
 point = self.points[point_index]
 
 distances = {}
 for neighbor_idx in neighbors:
 neighbor = self.points[neighbor_idx]
 distances[neighbor_idx] = point.distance_to(neighbor)
 
 return distances
 
 def get_tetrahedra_containing_vertex(self, vertex_index: int) -> List[DelaunayTetrahedron]:
 """
 Get all tetrahedra containing a specific vertex.
 
 Args:
 vertex_index: Index of the vertex
 
 Returns:
 List of tetrahedra containing the vertex
 """
 return [tet for tet in self.tetrahedra if tet.contains_vertex(vertex_index)]
 
 def compute_local_density(self, point_index: int, radius: float = 0.5) -> float:
 """
 Compute local point density around a point.
 
 Density is defined as the number of neighbors within a given radius,
 divided by the volume of the ball.
 
 Args:
 point_index: Index of the query point
 radius: Radius of the ball
 
 Returns:
 Local density (points per unit volume)
 """
 point = self.points[point_index]
 neighbors = self.get_neighbors(point_index)
 
 # Count neighbors within radius
 count = 0
 for neighbor_idx in neighbors:
 neighbor = self.points[neighbor_idx]
 if point.distance_to(neighbor) <= radius:
 count += 1
 
 # Volume of ball in T³
 volume = (4.0/3.0) * np.pi * radius**3
 
 return count / volume
 
 def compute_edge_length_statistics(self) -> Dict[str, float]:
 """
 Compute statistics of edge lengths in the triangulation.
 
 Returns:
 Dictionary with 'mean', 'std', 'min', 'max' edge lengths
 """
 edge_lengths = []
 
 for tet in self.tetrahedra:
 vertices = tet.vertices()
 
 # Compute all 6 edge lengths
 for i in range(4):
 for j in range(i+1, 4):
 pi = self.points[vertices[i]]
 pj = self.points[vertices[j]]
 edge_lengths.append(pi.distance_to(pj))
 
 edge_lengths = np.array(edge_lengths)
 
 return {
 'mean': np.mean(edge_lengths),
 'std': np.std(edge_lengths),
 'min': np.min(edge_lengths),
 'max': np.max(edge_lengths)
 }
 
 def compute_volume_statistics(self) -> Dict[str, float]:
 """
 Compute statistics of tetrahedron volumes.
 
 Returns:
 Dictionary with 'mean', 'std', 'min', 'max' volumes
 """
 volumes = [tet.volume(self.point_array) for tet in self.tetrahedra]
 volumes = np.array(volumes)
 
 return {
 'mean': np.mean(volumes),
 'std': np.std(volumes),
 'min': np.min(volumes),
 'max': np.max(volumes),
 'total': np.sum(volumes)
 }

# Integration with EntPTC entropy field

class EntropyFieldStructureAnalyzer:
 """
 Analyzer for the geometric structure of the entropy field on T³.
 
 Uses Delaunay triangulation to identify regions of high/low organization
 and analyze the topology of the entropy landscape.
 """
 
 def __init__(self,
 entropy_field: callable,
 n_sample_points: int = 1000):
 """
 Initialize entropy field structure analyzer.
 
 Args:
 entropy_field: Function S(θ) → entropy value
 n_sample_points: Number of points to sample on T³
 """
 self.entropy_field = entropy_field
 self.n_sample_points = n_sample_points
 
 # Sample points on T³
 self.points = self._sample_points()
 
 # Compute entropy at each point
 self.entropy_values = np.array([
 entropy_field(p) for p in self.points
 ])
 
 # Compute Delaunay triangulation
 self.delaunay = PeriodicDelaunayT3(self.points)
 
 def _sample_points(self) -> List[ToroidalPoint]:
 """
 Sample points uniformly on T³.
 
 Returns:
 List of sampled points
 """
 # Use quasi-random sampling (Sobol sequence) for better coverage
 from scipy.stats import qmc
 
 sampler = qmc.Sobol(d=3, scramble=True)
 samples = sampler.random(n=self.n_sample_points)
 
 # Scale to [0, 2π)³
 samples *= 2*np.pi
 
 points = [
 ToroidalPoint(theta1=s[0], theta2=s[1], theta3=s[2])
 for s in samples
 ]
 
 return points
 
 def identify_local_minima(self, threshold: float = 0.1) -> List[int]:
 """
 Identify local minima of the entropy field.
 
 A point is a local minimum if its entropy is lower than all neighbors.
 
 Args:
 threshold: Minimum entropy difference to be considered a minimum
 
 Returns:
 List of point indices that are local minima
 """
 local_minima = []
 
 for i in range(self.n_sample_points):
 entropy_i = self.entropy_values[i]
 neighbors = self.delaunay.get_neighbors(i)
 
 # Check if lower than all neighbors
 is_minimum = True
 for j in neighbors:
 entropy_j = self.entropy_values[j]
 if entropy_j < entropy_i - threshold:
 is_minimum = False
 break
 
 if is_minimum and len(neighbors) > 0:
 local_minima.append(i)
 
 return local_minima
 
 def identify_local_maxima(self, threshold: float = 0.1) -> List[int]:
 """
 Identify local maxima of the entropy field.
 
 A point is a local maximum if its entropy is higher than all neighbors.
 
 Args:
 threshold: Minimum entropy difference to be considered a maximum
 
 Returns:
 List of point indices that are local maxima
 """
 local_maxima = []
 
 for i in range(self.n_sample_points):
 entropy_i = self.entropy_values[i]
 neighbors = self.delaunay.get_neighbors(i)
 
 # Check if higher than all neighbors
 is_maximum = True
 for j in neighbors:
 entropy_j = self.entropy_values[j]
 if entropy_j > entropy_i + threshold:
 is_maximum = False
 break
 
 if is_maximum and len(neighbors) > 0:
 local_maxima.append(i)
 
 return local_maxima
 
 def compute_entropy_gradient_field_statistics(self) -> Dict[str, float]:
 """
 Compute statistics of the entropy gradient field.
 
 Uses finite differences on the Delaunay neighbors.
 
 Returns:
 Dictionary with gradient statistics
 """
 gradient_norms = []
 
 for i in range(self.n_sample_points):
 neighbors = self.delaunay.get_neighbors(i)
 
 if len(neighbors) == 0:
 continue
 
 # Estimate gradient using neighbors
 entropy_i = self.entropy_values[i]
 point_i = self.points[i]
 
 gradient = np.zeros(3)
 weight_sum = 0.0
 
 for j in neighbors:
 entropy_j = self.entropy_values[j]
 point_j = self.points[j]
 
 # Direction from i to j
 direction = point_j.to_array() - point_i.to_array()
 
 # Wrap to [-π, π]
 direction = np.mod(direction + np.pi, 2*np.pi) - np.pi
 
 distance = np.linalg.norm(direction)
 
 if distance > 1e-10:
 # Gradient contribution
 gradient += (entropy_j - entropy_i) * direction / (distance**2)
 weight_sum += 1.0 / distance
 
 if weight_sum > 0:
 gradient /= weight_sum
 gradient_norms.append(np.linalg.norm(gradient))
 
 gradient_norms = np.array(gradient_norms)
 
 return {
 'mean': np.mean(gradient_norms),
 'std': np.std(gradient_norms),
 'min': np.min(gradient_norms),
 'max': np.max(gradient_norms)
 }
 
 def compute_structural_complexity(self) -> float:
 """
 Compute a measure of structural complexity of the entropy field.
 
 Complexity is measured by the variance of edge lengths in the
 Delaunay triangulation, normalized by the mean.
 
 Returns:
 Structural complexity measure
 """
 edge_stats = self.delaunay.compute_edge_length_statistics()
 
 # Coefficient of variation
 cv = edge_stats['std'] / edge_stats['mean']
 
 return cv

# Summary and integration notes

"""
Delaunay Triangulation for EntPTC:

1. **Natural Neighborhoods**: The Delaunay triangulation provides a principled
 definition of which points are "neighbors" on T³.

2. **Structural Analysis**: Edge length and volume statistics reveal information
 about the geometry of the entropy field.

3. **Critical Point Identification**: Local minima and maxima can be identified
 using the neighborhood structure.

4. **Gradient Estimation**: The Delaunay neighbors enable robust gradient
 estimation using finite differences.

Next steps for full integration:
- Connect with entropy.py to analyze the actual entropy field
- Use in progenitor.py for identifying stable states
- Visualize the Delaunay structure on T³
- Compare with direct gradient computations
"""
