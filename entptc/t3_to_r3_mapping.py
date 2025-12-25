"""
T³→R³ Mapping for EntPTC Framework
===================================

Implements the 3-torus (T³) to R³ projection as specified in EntPTC model.

T³ is parameterized by three angular coordinates (θ₁, θ₂, θ₃) ∈ [0, 2π)³
R³ projection uses explicit embedding with documented normalization.

"""

import numpy as np
from typing import Tuple, Dict

# ============================================================================
# T³ COORDINATE SYSTEM
# ============================================================================

def compute_t3_coordinates(data: np.ndarray, fs: float, freq_ranges: Tuple[Tuple[float, float], ...] = None) -> np.ndarray:
 """
 Compute T³ coordinates (θ₁, θ₂, θ₃) from time series data.
 
 Each angular coordinate is derived from phase dynamics at different scales:
 - θ₁: Sub-delta phase (0.14-0.33 Hz) - slowest timescale
 - θ₂: Delta phase (0.5-4 Hz) - intermediate timescale 
 - θ₃: Theta phase (4-8 Hz) - fastest timescale
 
 Args:
 data: (n_rois, n_samples) array of time series
 fs: sampling rate (Hz)
 freq_ranges: tuple of (low, high) frequency ranges for (θ₁, θ₂, θ₃)
 Default: ((0.14, 0.33), (0.5, 4.0), (4.0, 8.0))
 
 Returns:
 t3_coords: (3, n_rois, n_samples) array of angular coordinates
 """
 from scipy.signal import butter, sosfiltfilt, hilbert
 
 if freq_ranges is None:
 freq_ranges = (
 (0.14, 0.33), # θ₁: Sub-delta (EntPTC control timescale)
 (0.5, 4.0), # θ₂: Delta
 (4.0, 8.0) # θ₃: Theta
 )
 
 n_rois, n_samples = data.shape
 t3_coords = np.zeros((3, n_rois, n_samples))
 
 for dim, (low, high) in enumerate(freq_ranges):
 # Bandpass filter
 sos = butter(4, [low, high], btype='band', fs=fs, output='sos')
 
 for roi in range(n_rois):
 # Filter
 filtered = sosfiltfilt(sos, data[roi])
 
 # Hilbert transform to get instantaneous phase
 analytic = hilbert(filtered)
 phase = np.angle(analytic) # Range: [-π, π]
 
 # Map to [0, 2π)
 phase = (phase + 2*np.pi) % (2*np.pi)
 
 t3_coords[dim, roi, :] = phase
 
 return t3_coords

def verify_t3_topology(t3_coords: np.ndarray) -> Dict[str, float]:
 """
 Verify that T³ coordinates have proper toroidal topology.
 
 Checks:
 - Angular coverage: all three dimensions should span [0, 2π)
 - Periodicity: phases should wrap around smoothly
 - Independence: three dimensions should be weakly correlated
 
 Args:
 t3_coords: (3, n_rois, n_samples) array of angular coordinates
 
 Returns:
 verification: dict of verification metrics
 """
 verification = {}
 
 for dim in range(3):
 phases = t3_coords[dim].flatten()
 
 # Angular coverage
 coverage = (phases.max() - phases.min()) / (2*np.pi) * 100
 verification[f'theta{dim+1}_coverage_pct'] = float(coverage)
 
 # Circular variance (1 = uniform, 0 = concentrated)
 mean_dir = np.mean(np.exp(1j * phases))
 circ_var = 1 - np.abs(mean_dir)
 verification[f'theta{dim+1}_circular_variance'] = float(circ_var)
 
 # Cross-dimensional correlation
 for i in range(3):
 for j in range(i+1, 3):
 # Circular correlation
 phase_i = t3_coords[i].flatten()
 phase_j = t3_coords[j].flatten()
 
 corr = np.abs(np.mean(np.exp(1j * (phase_i - phase_j))))
 verification[f'theta{i+1}_theta{j+1}_coupling'] = float(corr)
 
 return verification

# ============================================================================
# T³→R³ PROJECTION
# ============================================================================

def project_t3_to_r3(t3_coords: np.ndarray, projection_type: str = 'stereographic') -> np.ndarray:
 """
 Project T³ coordinates to R³.
 
 Multiple projection types available:
 - 'stereographic': Stereographic projection (conformal)
 - 'cylindrical': Cylindrical projection (preserves one angle)
 - 'embedding': Direct 3D embedding in R⁶ then PCA to R³
 
 Args:
 t3_coords: (3, n_rois, n_samples) array of angular coordinates
 projection_type: type of projection
 
 Returns:
 r3_coords: (3, n_rois, n_samples) array of R³ coordinates
 """
 if projection_type == 'stereographic':
 return _stereographic_projection(t3_coords)
 elif projection_type == 'cylindrical':
 return _cylindrical_projection(t3_coords)
 elif projection_type == 'embedding':
 return _embedding_projection(t3_coords)
 else:
 raise ValueError(f"Unknown projection type: {projection_type}")

def _stereographic_projection(t3_coords: np.ndarray) -> np.ndarray:
 """
 Stereographic projection from T³ to R³.
 
 Maps (θ₁, θ₂, θ₃) → (x, y, z) using:
 x = sin(θ₁) * cos(θ₂)
 y = sin(θ₁) * sin(θ₂)
 z = cos(θ₁) * sin(θ₃)
 
 This preserves angular relationships and is conformal.
 """
 theta1, theta2, theta3 = t3_coords[0], t3_coords[1], t3_coords[2]
 
 x = np.sin(theta1) * np.cos(theta2)
 y = np.sin(theta1) * np.sin(theta2)
 z = np.cos(theta1) * np.sin(theta3)
 
 r3_coords = np.stack([x, y, z], axis=0)
 
 return r3_coords

def _cylindrical_projection(t3_coords: np.ndarray) -> np.ndarray:
 """
 Cylindrical projection from T³ to R³.
 
 Maps (θ₁, θ₂, θ₃) → (x, y, z) using:
 x = θ₁ / (2π) (normalized angle)
 y = cos(θ₂)
 z = sin(θ₃)
 """
 theta1, theta2, theta3 = t3_coords[0], t3_coords[1], t3_coords[2]
 
 x = theta1 / (2*np.pi) # Normalize to [0, 1]
 y = np.cos(theta2)
 z = np.sin(theta3)
 
 r3_coords = np.stack([x, y, z], axis=0)
 
 return r3_coords

def _embedding_projection(t3_coords: np.ndarray) -> np.ndarray:
 """
 Embedding projection from T³ to R³ via R⁶.
 
 First embed T³ in R⁶ using standard torus embedding:
 (cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), cos(θ₃), sin(θ₃))
 
 Then project to R³ using PCA (principal components).
 """
 from sklearn.decomposition import PCA
 
 theta1, theta2, theta3 = t3_coords[0], t3_coords[1], t3_coords[2]
 
 # Embed in R⁶
 r6_coords = np.stack([
 np.cos(theta1),
 np.sin(theta1),
 np.cos(theta2),
 np.sin(theta2),
 np.cos(theta3),
 np.sin(theta3)
 ], axis=0) # Shape: (6, n_rois, n_samples)
 
 # Reshape for PCA: (n_samples * n_rois, 6)
 n_rois, n_samples = theta1.shape
 r6_flat = r6_coords.reshape(6, -1).T # Shape: (n_samples * n_rois, 6)
 
 # PCA to R³
 pca = PCA(n_components=3)
 r3_flat = pca.fit_transform(r6_flat) # Shape: (n_samples * n_rois, 3)
 
 # Reshape back: (3, n_rois, n_samples)
 r3_coords = r3_flat.T.reshape(3, n_rois, n_samples)
 
 return r3_coords

def normalize_r3_coordinates(r3_coords: np.ndarray, method: str = 'unit_sphere') -> np.ndarray:
 """
 Normalize R³ coordinates.
 
 Methods:
 - 'unit_sphere': Project onto unit sphere (preserves angles)
 - 'unit_variance': Standardize to zero mean, unit variance per dimension
 - 'unit_cube': Scale to [0, 1]³
 
 Args:
 r3_coords: (3, n_rois, n_samples) array
 method: normalization method
 
 Returns:
 normalized: (3, n_rois, n_samples) normalized array
 """
 if method == 'unit_sphere':
 # Project onto unit sphere
 norms = np.linalg.norm(r3_coords, axis=0, keepdims=True)
 norms = np.where(norms == 0, 1, norms) # Avoid division by zero
 normalized = r3_coords / norms
 
 elif method == 'unit_variance':
 # Standardize per dimension
 normalized = np.zeros_like(r3_coords)
 for dim in range(3):
 data = r3_coords[dim]
 mean = data.mean()
 std = data.std()
 if std > 0:
 normalized[dim] = (data - mean) / std
 else:
 normalized[dim] = data - mean
 
 elif method == 'unit_cube':
 # Scale to [0, 1]³
 normalized = np.zeros_like(r3_coords)
 for dim in range(3):
 data = r3_coords[dim]
 min_val = data.min()
 max_val = data.max()
 if max_val > min_val:
 normalized[dim] = (data - min_val) / (max_val - min_val)
 else:
 normalized[dim] = 0.5 # Constant data → center of cube
 
 else:
 raise ValueError(f"Unknown normalization method: {method}")
 
 return normalized

# ============================================================================
# INVARIANT COMPUTATION ON T³ AND R³
# ============================================================================

def compute_t3_invariants(t3_coords: np.ndarray, adjacency: np.ndarray) -> Dict[str, float]:
 """
 Compute geometric invariants on T³.
 
 Invariants:
 - Phase velocity: |dθ/dt| for each dimension
 - Phase winding: circulation around adjacent ROIs
 - Trajectory curvature: |d²θ/dt²|
 - Entropy flow: rate of change of phase entropy
 
 Args:
 t3_coords: (3, n_rois, n_samples) array
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 invariants: dict of invariant values
 """
 invariants = {}
 
 n_dims, n_rois, n_samples = t3_coords.shape
 
 # Phase velocity per dimension
 for dim in range(n_dims):
 phases = t3_coords[dim]
 
 # Unwrap phases for velocity computation
 phases_unwrapped = np.unwrap(phases, axis=1)
 velocities = np.diff(phases_unwrapped, axis=1)
 
 mean_velocity = np.abs(velocities).mean()
 invariants[f'theta{dim+1}_phase_velocity'] = float(mean_velocity)
 
 # Phase winding (circulation around adjacent ROIs)
 for dim in range(n_dims):
 phases = t3_coords[dim]
 
 windings = []
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 winding = np.abs(np.mean(np.exp(1j * phase_diff)))
 windings.append(winding)
 
 if len(windings) > 0:
 invariants[f'theta{dim+1}_phase_winding'] = float(np.mean(windings))
 else:
 invariants[f'theta{dim+1}_phase_winding'] = 0.0
 
 # Trajectory curvature
 for dim in range(n_dims):
 phases = t3_coords[dim]
 phases_unwrapped = np.unwrap(phases, axis=1)
 
 velocities = np.diff(phases_unwrapped, axis=1)
 accelerations = np.diff(velocities, axis=1)
 
 curvature = np.abs(accelerations).mean()
 invariants[f'theta{dim+1}_trajectory_curvature'] = float(curvature)
 
 return invariants

def compute_r3_invariants(r3_coords: np.ndarray, adjacency: np.ndarray) -> Dict[str, float]:
 """
 Compute geometric invariants on R³.
 
 Invariants:
 - Trajectory length: integrated path length
 - Spatial spread: variance in each dimension
 - Neighbor distance: mean distance to adjacent ROIs
 - Trajectory alignment: correlation of velocity vectors
 
 Args:
 r3_coords: (3, n_rois, n_samples) array
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 invariants: dict of invariant values
 """
 invariants = {}
 
 n_dims, n_rois, n_samples = r3_coords.shape
 
 # Trajectory length
 velocities = np.diff(r3_coords, axis=2) # Shape: (3, n_rois, n_samples-1)
 speeds = np.linalg.norm(velocities, axis=0) # Shape: (n_rois, n_samples-1)
 trajectory_length = speeds.sum(axis=1).mean()
 invariants['trajectory_length'] = float(trajectory_length)
 
 # Spatial spread per dimension
 for dim in range(n_dims):
 variance = np.var(r3_coords[dim])
 invariants[f'r{dim+1}_spatial_variance'] = float(variance)
 
 # Neighbor distance
 neighbor_distances = []
 for t in range(n_samples):
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 dist = np.linalg.norm(r3_coords[:, i, t] - r3_coords[:, j, t])
 neighbor_distances.append(dist)
 
 if len(neighbor_distances) > 0:
 invariants['mean_neighbor_distance'] = float(np.mean(neighbor_distances))
 else:
 invariants['mean_neighbor_distance'] = 0.0
 
 # Trajectory alignment (velocity correlation between adjacent ROIs)
 alignments = []
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 vel_i = velocities[:, i, :] # Shape: (3, n_samples-1)
 vel_j = velocities[:, j, :]
 
 # Flatten and correlate
 vel_i_flat = vel_i.flatten()
 vel_j_flat = vel_j.flatten()
 
 if len(vel_i_flat) > 0 and len(vel_j_flat) > 0:
 corr = np.corrcoef(vel_i_flat, vel_j_flat)[0, 1]
 if not np.isnan(corr):
 alignments.append(abs(corr))
 
 if len(alignments) > 0:
 invariants['trajectory_alignment'] = float(np.mean(alignments))
 else:
 invariants['trajectory_alignment'] = 0.0
 
 return invariants

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def entptc_t3_to_r3_pipeline(data: np.ndarray, fs: float, adjacency: np.ndarray, 
 projection_type: str = 'stereographic',
 normalization: str = 'unit_variance') -> Dict:
 """
 Complete T³→R³ pipeline for EntPTC framework.
 
 Args:
 data: (n_rois, n_samples) time series
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 projection_type: 'stereographic', 'cylindrical', or 'embedding'
 normalization: 'unit_sphere', 'unit_variance', or 'unit_cube'
 
 Returns:
 results: dict containing:
 - t3_coords: (3, n_rois, n_samples) T³ coordinates
 - r3_coords: (3, n_rois, n_samples) R³ coordinates
 - t3_verification: topology verification metrics
 - t3_invariants: geometric invariants on T³
 - r3_invariants: geometric invariants on R³
 """
 # Compute T³ coordinates
 t3_coords = compute_t3_coordinates(data, fs)
 
 # Verify T³ topology
 t3_verification = verify_t3_topology(t3_coords)
 
 # Project to R³
 r3_coords = project_t3_to_r3(t3_coords, projection_type=projection_type)
 
 # Normalize R³ coordinates
 r3_coords = normalize_r3_coordinates(r3_coords, method=normalization)
 
 # Compute invariants
 t3_invariants = compute_t3_invariants(t3_coords, adjacency)
 r3_invariants = compute_r3_invariants(r3_coords, adjacency)
 
 results = {
 't3_coords': t3_coords,
 'r3_coords': r3_coords,
 't3_verification': t3_verification,
 't3_invariants': t3_invariants,
 'r3_invariants': r3_invariants,
 'projection_type': projection_type,
 'normalization': normalization
 }
 
 return results

if __name__ == '__main__':
 print("T³→R³ Mapping for EntPTC Framework")
 print("Use: from entptc.t3_to_r3_mapping import entptc_t3_to_r3_pipeline")
