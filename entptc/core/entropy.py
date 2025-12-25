"""
Entropy Field on T³ Toroidal Manifold

Reference: ENTPC.tex Definition 2.4, Section 2.2 (lines 258-262)

From ENTPC.tex:

Definition 2.4 (Entropy Field on T³): Let T³ = S¹ × S¹ × S¹ denote the 
3-dimensional torus. The entropy field S: T³ → ℝ assigns a real-valued entropy
to each point on the toroidal manifold. The gradient ∇S provides directional
information about entropy variation.

Section 2.2 (lines 258-262):
"The entropy field S on T³ encodes the informational structure of the system.
Points on T³ represent phase configurations, and S(θ₁, θ₂, θ₃) quantifies the
uncertainty or disorder at that configuration. The gradient ∇S = (∂S/∂θ₁, ∂S/∂θ₂, ∂S/∂θ₃)
guides the flow toward regions of lower entropy (higher organization)."

In EntPTC:
- T³ parameterized by angles (θ₁, θ₂, θ₃) ∈ [0, 2π)³
- Entropy computed from quaternionic state distributions
- Gradient ∇S used in Progenitor matrix: c_ij ∝ e^(-∇S_ij)
- Lower entropy → higher coherence → stronger connections
"""

import numpy as np
from typing import Tuple, Optional, Callable
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

class ToroidalManifold:
 """
 3-dimensional torus T³ = S¹ × S¹ × S¹.
 
 Per ENTPC.tex Definition 2.4:
 - Parameterized by angles (θ₁, θ₂, θ₃) ∈ [0, 2π)³
 - Each S¹ is a circle (1-sphere)
 - Periodic boundary conditions in all three directions
 """
 
 def __init__(self, resolution: int = 32):
 """
 Initialize toroidal manifold with discretization.
 
 Args:
 resolution: Number of grid points per angular dimension
 """
 self.resolution = resolution
 
 # Angular coordinates for each dimension
 self.theta = np.linspace(0, 2*np.pi, resolution, endpoint=False)
 
 # 3D grid of angles
 self.theta1, self.theta2, self.theta3 = np.meshgrid(
 self.theta, self.theta, self.theta, indexing='ij'
 )
 
 def to_cartesian(self, theta1: float, theta2: float, theta3: float,
 R: float = 2.0, r: float = 1.0) -> Tuple[float, float, float, float]:
 """
 Convert toroidal coordinates to 4D embedding space.
 
 T³ can be embedded in ℝ⁴. For visualization, using nested tori.
 
 Args:
 theta1, theta2, theta3: Angular coordinates in [0, 2π)
 R: Major radius of outer torus
 r: Minor radius
 
 Returns:
 (x, y, z, w) coordinates in ℝ⁴
 """
 # Nested torus embedding
 x = (R + r * np.cos(theta2)) * np.cos(theta1)
 y = (R + r * np.cos(theta2)) * np.sin(theta1)
 z = r * np.sin(theta2) * np.cos(theta3)
 w = r * np.sin(theta2) * np.sin(theta3)
 
 return x, y, z, w
 
 def distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
 """
 Compute geodesic distance on T³ between two points.
 
 For T³, geodesic distance is the sum of angular distances on each S¹.
 
 Args:
 point1: (θ₁, θ₂, θ₃)
 point2: (θ₁', θ₂', θ₃')
 
 Returns:
 Geodesic distance
 """
 # Angular distance on each circle (accounting for periodicity)
 def angular_distance(a, b):
 diff = abs(a - b)
 return min(diff, 2*np.pi - diff)
 
 d1 = angular_distance(point1[0], point2[0])
 d2 = angular_distance(point1[1], point2[1])
 d3 = angular_distance(point1[2], point2[2])
 
 # Euclidean distance in angle space
 return np.sqrt(d1**2 + d2**2 + d3**2)
 
 def normalize_angles(self, theta: np.ndarray) -> np.ndarray:
 """
 Normalize angles to [0, 2π) with periodic boundary conditions.
 
 Args:
 theta: Array of angles
 
 Returns:
 Normalized angles in [0, 2π)
 """
 return np.mod(theta, 2*np.pi)

class EntropyField:
 """
 Entropy Field S: T³ → ℝ
 
 Per ENTPC.tex Definition 2.4:
 - Maps points on T³ to real-valued entropy
 - Quantifies uncertainty/disorder at each phase configuration
 - Gradient ∇S guides flow toward organization
 
 Implementation:
 - Computed from quaternionic state distributions
 - Discretized on regular grid over T³
 - Interpolated for continuous queries
 """
 
 def __init__(self, manifold: ToroidalManifold):
 """
 Initialize entropy field on toroidal manifold.
 
 Args:
 manifold: ToroidalManifold instance defining T³
 """
 self.manifold = manifold
 self.resolution = manifold.resolution
 
 # Entropy values on grid (to be computed)
 self.entropy_grid = None
 
 # Interpolator for continuous queries
 self.interpolator = None
 
 def compute_from_quaternions(self, quaternion_field: np.ndarray,
 smoothing_sigma: float = 1.0):
 """
 Compute entropy field from quaternionic state distribution.
 
 Per ENTPC.tex: Entropy quantifies uncertainty in quaternionic states.
 
 Implementation:
 - Each point on T³ corresponds to a phase configuration
 - Quaternions at that configuration define a probability distribution
 - Entropy = -Σ p_i log(p_i) (Shannon entropy)
 
 Args:
 quaternion_field: Array of shape (res, res, res, 4) 
 Quaternion components at each grid point
 smoothing_sigma: Gaussian smoothing for regularization
 """
 assert quaternion_field.shape[:3] == (self.resolution, self.resolution, self.resolution), \
 f"Expected shape ({self.resolution}, {self.resolution}, {self.resolution}, 4), got {quaternion_field.shape}"
 
 # Compute entropy at each grid point
 entropy_grid = np.zeros((self.resolution, self.resolution, self.resolution))
 
 for i in range(self.resolution):
 for j in range(self.resolution):
 for k in range(self.resolution):
 # Get quaternion at this point
 q = quaternion_field[i, j, k]
 
 # Compute probability distribution from quaternion components
 # Use normalized squared components as probabilities
 p = q**2
 p_sum = np.sum(p)
 
 if p_sum > 1e-12:
 p = p / p_sum
 
 # Shannon entropy: -Σ p_i log(p_i)
 # Avoid log(0) by adding small epsilon
 entropy = -np.sum(p * np.log(p + 1e-12))
 else:
 # Zero quaternion → maximum entropy
 entropy = np.log(4) # log of number of components
 
 entropy_grid[i, j, k] = entropy
 
 # Apply Gaussian smoothing for regularization
 if smoothing_sigma > 0:
 entropy_grid = gaussian_filter(entropy_grid, sigma=smoothing_sigma, mode='wrap')
 
 self.entropy_grid = entropy_grid
 
 # Create interpolator for continuous queries
 self._build_interpolator()
 
 def compute_from_coherence(self, coherence_matrix: np.ndarray,
 smoothing_sigma: float = 1.0):
 """
 Compute entropy field from coherence matrix.
 
 Alternative construction: entropy inversely related to coherence.
 High coherence → low entropy (organized state)
 Low coherence → high entropy (disordered state)
 
 Args:
 coherence_matrix: 16×16 coherence matrix (PLV)
 smoothing_sigma: Gaussian smoothing for regularization
 """
 assert coherence_matrix.shape == (16, 16), \
 f"Expected 16×16 coherence matrix, got {coherence_matrix.shape}"
 
 # Map 16×16 matrix to T³ grid
 # Strategy: tile and interpolate to fill T³
 
 # Compute local entropy from coherence
 # Entropy ∝ -log(coherence)
 # Avoid log(0) by adding epsilon
 local_entropy = -np.log(coherence_matrix + 1e-6)
 
 # Normalize to [0, 1] range
 local_entropy = (local_entropy - local_entropy.min()) / (local_entropy.max() - local_entropy.min() + 1e-12)
 
 # Tile to fill T³ grid
 # 16×16 → 4×4 blocks → tile to resolution×resolution×resolution
 block_size = self.resolution // 4
 entropy_grid = np.zeros((self.resolution, self.resolution, self.resolution))
 
 for i in range(4):
 for j in range(4):
 for k in range(4):
 # Get entropy value from coherence matrix
 # Map (i,j,k) to coherence matrix indices
 coh_i = i
 coh_j = j * 4 + k
 
 if coh_j < 16:
 entropy_val = local_entropy[coh_i, coh_j]
 else:
 entropy_val = local_entropy.mean()
 
 # Fill corresponding block in grid
 i_start, i_end = i * block_size, (i + 1) * block_size
 j_start, j_end = j * block_size, (j + 1) * block_size
 k_start, k_end = k * block_size, (k + 1) * block_size
 
 entropy_grid[i_start:i_end, j_start:j_end, k_start:k_end] = entropy_val
 
 # Apply Gaussian smoothing
 if smoothing_sigma > 0:
 entropy_grid = gaussian_filter(entropy_grid, sigma=smoothing_sigma, mode='wrap')
 
 self.entropy_grid = entropy_grid
 
 # Create interpolator
 self._build_interpolator()
 
 def _build_interpolator(self):
 """Build interpolator for continuous entropy queries."""
 if self.entropy_grid is None:
 raise ValueError("Entropy grid not computed yet")
 
 # Create regular grid interpolator with periodic boundary conditions
 theta_coords = self.manifold.theta
 
 # Extend grid for periodic interpolation
 # Wrap first slice to end for periodicity
 extended_grid = np.concatenate([
 self.entropy_grid,
 self.entropy_grid[:1, :, :]
 ], axis=0)
 extended_grid = np.concatenate([
 extended_grid,
 extended_grid[:, :1, :]
 ], axis=1)
 extended_grid = np.concatenate([
 extended_grid,
 extended_grid[:, :, :1]
 ], axis=2)
 
 # Extended coordinates
 theta_extended = np.concatenate([theta_coords, [2*np.pi]])
 
 self.interpolator = RegularGridInterpolator(
 (theta_extended, theta_extended, theta_extended),
 extended_grid,
 method='linear',
 bounds_error=False,
 fill_value=None
 )
 
 def evaluate(self, theta1: float, theta2: float, theta3: float) -> float:
 """
 Evaluate entropy at point (θ₁, θ₂, θ₃) on T³.
 
 Args:
 theta1, theta2, theta3: Angular coordinates
 
 Returns:
 Entropy value S(θ₁, θ₂, θ₃)
 """
 if self.interpolator is None:
 raise ValueError("Entropy field not computed yet")
 
 # Normalize angles to [0, 2π)
 theta1 = np.mod(theta1, 2*np.pi)
 theta2 = np.mod(theta2, 2*np.pi)
 theta3 = np.mod(theta3, 2*np.pi)
 
 return float(self.interpolator([theta1, theta2, theta3]))
 
 def gradient(self, theta1: float, theta2: float, theta3: float,
 epsilon: float = 0.01) -> np.ndarray:
 """
 Compute gradient ∇S at point (θ₁, θ₂, θ₃).
 
 Per ENTPC.tex: ∇S = (∂S/∂θ₁, ∂S/∂θ₂, ∂S/∂θ₃)
 
 Used in Progenitor matrix: c_ij ∝ e^(-∇S_ij)
 
 Args:
 theta1, theta2, theta3: Angular coordinates
 epsilon: Finite difference step size
 
 Returns:
 Gradient vector [∂S/∂θ₁, ∂S/∂θ₂, ∂S/∂θ₃]
 """
 # Finite difference approximation
 dS_dtheta1 = (self.evaluate(theta1 + epsilon, theta2, theta3) - 
 self.evaluate(theta1 - epsilon, theta2, theta3)) / (2 * epsilon)
 
 dS_dtheta2 = (self.evaluate(theta1, theta2 + epsilon, theta3) - 
 self.evaluate(theta1, theta2 - epsilon, theta3)) / (2 * epsilon)
 
 dS_dtheta3 = (self.evaluate(theta1, theta2, theta3 + epsilon) - 
 self.evaluate(theta1, theta2, theta3 - epsilon)) / (2 * epsilon)
 
 return np.array([dS_dtheta1, dS_dtheta2, dS_dtheta3])
 
 def gradient_norm(self, theta1: float, theta2: float, theta3: float) -> float:
 """
 Compute norm of gradient |∇S|.
 
 Used in Progenitor matrix construction.
 
 Args:
 theta1, theta2, theta3: Angular coordinates
 
 Returns:
 Gradient norm |∇S|
 """
 grad = self.gradient(theta1, theta2, theta3)
 return np.linalg.norm(grad)
 
 def gradient_matrix(self, points: np.ndarray) -> np.ndarray:
 """
 Compute gradient norms for matrix of points.
 
 Per ENTPC.tex: Used in Progenitor matrix c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|
 
 Args:
 points: Array of shape (n, n, 3) where points[i,j] = (θ₁, θ₂, θ₃)
 
 Returns:
 Matrix of shape (n, n) with gradient norms
 """
 n = points.shape[0]
 assert points.shape == (n, n, 3), f"Expected shape (n, n, 3), got {points.shape}"
 
 grad_matrix = np.zeros((n, n))
 
 for i in range(n):
 for j in range(n):
 theta1, theta2, theta3 = points[i, j]
 grad_matrix[i, j] = self.gradient_norm(theta1, theta2, theta3)
 
 return grad_matrix
 
 def get_grid(self) -> np.ndarray:
 """
 Get full entropy grid.
 
 Returns:
 Array of shape (resolution, resolution, resolution)
 """
 if self.entropy_grid is None:
 raise ValueError("Entropy field not computed yet")
 
 return self.entropy_grid.copy()
 
 def get_statistics(self) -> dict:
 """
 Compute statistics of entropy field.
 
 Returns:
 Dictionary with min, max, mean, std of entropy
 """
 if self.entropy_grid is None:
 raise ValueError("Entropy field not computed yet")
 
 return {
 'min': float(np.min(self.entropy_grid)),
 'max': float(np.max(self.entropy_grid)),
 'mean': float(np.mean(self.entropy_grid)),
 'std': float(np.std(self.entropy_grid)),
 'median': float(np.median(self.entropy_grid))
 }

def create_entropy_field_from_progenitor(progenitor_matrix: np.ndarray,
 resolution: int = 32) -> EntropyField:
 """
 Create entropy field from Progenitor matrix.
 
 Convenience function for pipeline integration.
 
 Args:
 progenitor_matrix: 16×16 Progenitor matrix
 resolution: Grid resolution for T³
 
 Returns:
 EntropyField instance
 """
 # Extract coherence from Progenitor matrix
 # Progenitor = coherence * exp(-grad) * quaternion
 # Approximate coherence from matrix values
 coherence_matrix = np.abs(progenitor_matrix)
 
 # Normalize to [0, 1]
 coherence_matrix = coherence_matrix / (np.max(coherence_matrix) + 1e-12)
 
 # Create manifold and entropy field
 manifold = ToroidalManifold(resolution=resolution)
 entropy_field = EntropyField(manifold)
 
 # Compute from coherence
 entropy_field.compute_from_coherence(coherence_matrix)
 
 return entropy_field
