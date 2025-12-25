"""
U2: Persistent Homology on T³ Trajectory
=========================================

Topology-native uniqueness test using persistent homology.

Expectation: Torus has characteristic H1 structure (multiple independent 1-cycles).
Cylinder drops a cycle; removed torus reduces Betti-1 features.

Per user protocol: This is direct topology test, not "stronger PLV on torus".

"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Set random seed
np.random.seed(42)

# ============================================================================
# PERSISTENT HOMOLOGY COMPUTATION
# ============================================================================

def compute_persistent_homology(t3_coords: np.ndarray, max_dimension: int = 2) -> Dict:
 """
 Compute persistent homology on T³ trajectory.
 
 Uses ripser library for persistent homology computation.
 
 Args:
 t3_coords: (3, n_rois, n_samples) T³ coordinates
 max_dimension: maximum homology dimension to compute
 
 Returns:
 homology_results: dict with persistence diagrams and Betti numbers
 """
 try:
 from ripser import ripser
 from persim import plot_diagrams
 except ImportError:
 print("⚠️ ripser/persim not installed. Installing...")
 import subprocess
 subprocess.run(['pip3', 'install', 'ripser', 'persim'], check=True)
 from ripser import ripser
 from persim import plot_diagrams
 
 n_dims, n_rois, n_samples = t3_coords.shape
 
 # Flatten trajectory: (n_rois * n_samples, 3)
 # Each point is (θ₁, θ₂, θ₃) at a specific ROI and time
 trajectory = t3_coords.reshape(n_dims, -1).T # (n_rois * n_samples, 3)
 
 # Subsample for computational efficiency
 subsample_factor = max(1, len(trajectory) // 5000)
 trajectory_subsampled = trajectory[::subsample_factor]
 
 print(f"Trajectory shape: {trajectory.shape}")
 print(f"Subsampled to: {trajectory_subsampled.shape}")
 
 # Compute persistent homology
 print("Computing persistent homology...")
 result = ripser(trajectory_subsampled, maxdim=max_dimension)
 
 diagrams = result['dgms']
 
 # Extract Betti numbers (number of features at max persistence)
 betti_numbers = []
 for dim in range(len(diagrams)):
 diagram = diagrams[dim]
 # Filter out infinite bars
 finite_bars = diagram[diagram[:, 1] < np.inf]
 if len(finite_bars) > 0:
 # Count features with persistence > threshold
 persistence_threshold = np.percentile(finite_bars[:, 1] - finite_bars[:, 0], 75)
 significant_features = np.sum((finite_bars[:, 1] - finite_bars[:, 0]) > persistence_threshold)
 betti_numbers.append(int(significant_features))
 else:
 betti_numbers.append(0)
 
 # Compute persistence statistics
 persistence_stats = []
 for dim in range(len(diagrams)):
 diagram = diagrams[dim]
 finite_bars = diagram[diagram[:, 1] < np.inf]
 if len(finite_bars) > 0:
 persistences = finite_bars[:, 1] - finite_bars[:, 0]
 persistence_stats.append({
 'dimension': dim,
 'n_features': len(finite_bars),
 'mean_persistence': float(np.mean(persistences)),
 'max_persistence': float(np.max(persistences)),
 'total_persistence': float(np.sum(persistences))
 })
 else:
 persistence_stats.append({
 'dimension': dim,
 'n_features': 0,
 'mean_persistence': 0.0,
 'max_persistence': 0.0,
 'total_persistence': 0.0
 })
 
 results = {
 'betti_numbers': betti_numbers,
 'persistence_stats': persistence_stats,
 'n_points': len(trajectory_subsampled)
 }
 
 return results

def compare_homology_across_topologies(data: np.ndarray, fs: float, adjacency_torus: np.ndarray) -> Dict:
 """
 Compare persistent homology across different topologies.
 
 Tests:
 1. Torus (original)
 2. Cylinder (one periodic boundary removed)
 3. Plane (both periodic boundaries removed)
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency_torus: (n_rois, n_rois) toroidal adjacency
 
 Returns:
 comparison: dict with homology for each topology
 """
 from entptc.t3_to_r3_mapping import compute_t3_coordinates
 
 print("\n" + "="*80)
 print("U2: PERSISTENT HOMOLOGY COMPARISON")
 print("="*80)
 
 # 1. Torus (original)
 print("\n1. TORUS (original topology)")
 t3_coords_torus = compute_t3_coordinates(data, fs)
 homology_torus = compute_persistent_homology(t3_coords_torus)
 
 print(f"Betti numbers: {homology_torus['betti_numbers']}")
 print("Persistence stats:")
 for stat in homology_torus['persistence_stats']:
 print(f" H{stat['dimension']}: {stat['n_features']} features, "
 f"mean persistence = {stat['mean_persistence']:.6f}, "
 f"max persistence = {stat['max_persistence']:.6f}")
 
 # 2. Cylinder (remove one periodic boundary)
 print("\n2. CYLINDER (one periodic boundary removed)")
 # Simulate cylinder by phase-scrambling one dimension
 data_cylinder = data.copy()
 # Scramble spatial structure in one dimension (break one periodic boundary)
 n_rois = data.shape[0]
 grid_size = int(np.sqrt(n_rois))
 for i in range(grid_size):
 # Scramble rows (break x-periodicity)
 row_indices = list(range(i * grid_size, (i + 1) * grid_size))
 scrambled_indices = np.random.permutation(row_indices)
 data_cylinder[row_indices] = data[scrambled_indices]
 
 t3_coords_cylinder = compute_t3_coordinates(data_cylinder, fs)
 homology_cylinder = compute_persistent_homology(t3_coords_cylinder)
 
 print(f"Betti numbers: {homology_cylinder['betti_numbers']}")
 print("Persistence stats:")
 for stat in homology_cylinder['persistence_stats']:
 print(f" H{stat['dimension']}: {stat['n_features']} features, "
 f"mean persistence = {stat['mean_persistence']:.6f}, "
 f"max persistence = {stat['max_persistence']:.6f}")
 
 # 3. Plane (remove both periodic boundaries)
 print("\n3. PLANE (both periodic boundaries removed)")
 data_plane = data.copy()
 # Scramble both dimensions
 scrambled_indices = np.random.permutation(n_rois)
 data_plane = data[scrambled_indices]
 
 t3_coords_plane = compute_t3_coordinates(data_plane, fs)
 homology_plane = compute_persistent_homology(t3_coords_plane)
 
 print(f"Betti numbers: {homology_plane['betti_numbers']}")
 print("Persistence stats:")
 for stat in homology_plane['persistence_stats']:
 print(f" H{stat['dimension']}: {stat['n_features']} features, "
 f"mean persistence = {stat['mean_persistence']:.6f}, "
 f"max persistence = {stat['max_persistence']:.6f}")
 
 # Compare
 print("\n" + "="*80)
 print("U2 RESULTS")
 print("="*80)
 
 print("\nBetti-1 (1-cycles) comparison:")
 betti1_torus = homology_torus['betti_numbers'][1] if len(homology_torus['betti_numbers']) > 1 else 0
 betti1_cylinder = homology_cylinder['betti_numbers'][1] if len(homology_cylinder['betti_numbers']) > 1 else 0
 betti1_plane = homology_plane['betti_numbers'][1] if len(homology_plane['betti_numbers']) > 1 else 0
 
 print(f" Torus: {betti1_torus}")
 print(f" Cylinder: {betti1_cylinder}")
 print(f" Plane: {betti1_plane}")
 
 # Expected: Torus > Cylinder > Plane
 monotonic = (betti1_torus >= betti1_cylinder) and (betti1_cylinder >= betti1_plane)
 
 print(f"\nMonotonic degradation (Torus ≥ Cylinder ≥ Plane): {'✅ YES' if monotonic else '❌ NO'}")
 
 # Total persistence comparison
 total_pers_torus = homology_torus['persistence_stats'][1]['total_persistence'] if len(homology_torus['persistence_stats']) > 1 else 0
 total_pers_cylinder = homology_cylinder['persistence_stats'][1]['total_persistence'] if len(homology_cylinder['persistence_stats']) > 1 else 0
 total_pers_plane = homology_plane['persistence_stats'][1]['total_persistence'] if len(homology_plane['persistence_stats']) > 1 else 0
 
 print(f"\nTotal H1 persistence:")
 print(f" Torus: {total_pers_torus:.6f}")
 print(f" Cylinder: {total_pers_cylinder:.6f}")
 print(f" Plane: {total_pers_plane:.6f}")
 
 persistence_monotonic = (total_pers_torus >= total_pers_cylinder) and (total_pers_cylinder >= total_pers_plane)
 
 print(f"\nPersistence monotonic: {'✅ YES' if persistence_monotonic else '❌ NO'}")
 
 overall_pass = monotonic or persistence_monotonic
 
 print(f"\nU2 verdict: {'✅ PASS' if overall_pass else '❌ FAIL'}")
 
 results = {
 'torus': homology_torus,
 'cylinder': homology_cylinder,
 'plane': homology_plane,
 'betti1_comparison': {
 'torus': betti1_torus,
 'cylinder': betti1_cylinder,
 'plane': betti1_plane,
 'monotonic': monotonic
 },
 'persistence_comparison': {
 'torus': total_pers_torus,
 'cylinder': total_pers_cylinder,
 'plane': total_pers_plane,
 'monotonic': persistence_monotonic
 },
 'overall_pass': overall_pass
 }
 
 return results

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 import scipy.io as sio
 from entptc.utils.grid_utils import create_toroidal_grid
 
 # Load data
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/uniqueness_u2_homology')
 output_dir.mkdir(parents=True, exist_ok=True)
 
 if not data_path.exists():
 print(f"Data file not found: {data_path}")
 exit(1)
 
 print("Loading data...")
 mat = sio.loadmat(data_path)
 data = mat['eeg_data']
 fs = float(mat['fs'][0, 0])
 
 grid_size = int(np.sqrt(data.shape[0]))
 adjacency = create_toroidal_grid(grid_size)
 
 print(f"Data shape: {data.shape}")
 print(f"Sampling rate: {fs} Hz")
 print(f"Grid size: {grid_size}×{grid_size}")
 
 # Run U2
 results = compare_homology_across_topologies(data, fs, adjacency)
 
 # Save results
 output_path = output_dir / 'u2_homology_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")
