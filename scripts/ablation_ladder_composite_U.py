"""
Ablation Ladder on Composite Signature U
=========================================

Per locked protocol: Apply ablations to full signature U (7 components), assess signature separation.

Ablations:
1. Boundary removal (torus → cylinder → plane)
2. Adjacency scramble (degree-preserved randomization)
3. Phase destruction (matched-spectrum surrogates)
4. Channel randomization (negative control)

Uniqueness assessed by signature separation under ablations, NOT single-metric monotonicity.

"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import json
from typing import Dict, List
from composite_invariant_signature import compute_composite_signature_U

# Set random seed
np.random.seed(42)

# ============================================================================
# ABLATION 1: BOUNDARY REMOVAL
# ============================================================================

def ablation_boundary_removal(data: np.ndarray, grid_size: int, mode: str = 'cylinder') -> np.ndarray:
 """
 Remove periodic boundaries.
 
 Args:
 data: (n_rois, n_samples) array
 grid_size: size of grid
 mode: 'cylinder' (remove one boundary) or 'plane' (remove both)
 
 Returns:
 data_ablated: (n_rois, n_samples) with boundaries removed
 """
 n_rois, n_samples = data.shape
 data_ablated = data.copy()
 
 if mode == 'cylinder':
 # Remove x-periodicity: scramble rows
 for i in range(grid_size):
 row_indices = list(range(i * grid_size, (i + 1) * grid_size))
 scrambled_indices = np.random.permutation(row_indices)
 data_ablated[row_indices] = data[scrambled_indices]
 
 elif mode == 'plane':
 # Remove both periodicities: scramble rows and columns
 for i in range(grid_size):
 row_indices = list(range(i * grid_size, (i + 1) * grid_size))
 scrambled_indices = np.random.permutation(row_indices)
 data_ablated[row_indices] = data[scrambled_indices]
 
 for j in range(grid_size):
 col_indices = list(range(j, n_rois, grid_size))
 scrambled_indices = np.random.permutation(col_indices)
 data_ablated[col_indices] = data_ablated[scrambled_indices]
 
 return data_ablated

# ============================================================================
# ABLATION 2: ADJACENCY SCRAMBLE
# ============================================================================

def ablation_adjacency_scramble(adjacency: np.ndarray, n_swaps: int = None) -> np.ndarray:
 """
 Randomize adjacency matrix with degree preservation.
 
 Args:
 adjacency: (n_rois, n_rois) adjacency matrix
 n_swaps: number of edge swaps
 
 Returns:
 adjacency_scrambled: (n_rois, n_rois) randomized adjacency
 """
 from uniqueness_U3_causal_ablation import randomize_adjacency_degree_preserved
 return randomize_adjacency_degree_preserved(adjacency, n_swaps)

# ============================================================================
# ABLATION 3: PHASE DESTRUCTION
# ============================================================================

def ablation_phase_destruction(data: np.ndarray) -> np.ndarray:
 """
 Generate matched-spectrum surrogate (phase-scrambled).
 
 Args:
 data: (n_rois, n_samples) array
 
 Returns:
 data_scrambled: (n_rois, n_samples) phase-scrambled
 """
 from uniqueness_U1_FIX import generate_matched_spectrum_surrogate
 return generate_matched_spectrum_surrogate(data)

# ============================================================================
# ABLATION 4: CHANNEL RANDOMIZATION (NEGATIVE CONTROL)
# ============================================================================

def ablation_channel_randomization(data: np.ndarray) -> np.ndarray:
 """
 Randomly permute channels (negative control).
 
 Args:
 data: (n_rois, n_samples) array
 
 Returns:
 data_permuted: (n_rois, n_samples) with channels permuted
 """
 n_rois = data.shape[0]
 permutation = np.random.permutation(n_rois)
 return data[permutation]

# ============================================================================
# SIGNATURE DISTANCE METRIC
# ============================================================================

def compute_signature_distance(U1: Dict, U2: Dict) -> float:
 """
 Compute normalized Euclidean distance between two signatures.
 Per-component normalization to prevent scale dominance.
 
 Args:
 U1, U2: composite signatures
 
 Returns:
 distance: normalized distance
 """
 # Flatten signatures to vectors with per-component normalization
 def flatten_signature_normalized(U1, U2):
 vec1 = []
 vec2 = []
 for component_key in sorted(U1.keys()):
 component1 = U1[component_key]
 component2 = U2[component_key]
 for metric_key in sorted(component1.keys()):
 val1 = component1[metric_key]
 val2 = component2[metric_key]
 
 # Normalize each metric by its baseline magnitude
 baseline = abs(val1) + 1e-10
 vec1.append(val1 / baseline)
 vec2.append(val2 / baseline)
 
 return np.array(vec1), np.array(vec2)
 
 vec1_norm, vec2_norm = flatten_signature_normalized(U1, U2)
 
 # Euclidean distance in normalized space
 distance = np.linalg.norm(vec1_norm - vec2_norm) / np.sqrt(len(vec1_norm))
 
 return distance

# ============================================================================
# ABLATION LADDER RUNNER
# ============================================================================

def run_ablation_ladder(data: np.ndarray, fs: float, adjacency: np.ndarray, n_iterations: int = 20) -> Dict:
 """
 Run full ablation ladder on composite signature U.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 n_iterations: number of iterations per ablation
 
 Returns:
 results: dict with ablation results
 """
 print("\n" + "="*80)
 print("ABLATION LADDER ON COMPOSITE SIGNATURE U")
 print("="*80)
 
 n_rois = data.shape[0]
 grid_size = int(np.sqrt(n_rois))
 
 # Baseline signature
 print("\nComputing baseline signature...")
 U_baseline = compute_composite_signature_U(data, fs, adjacency)
 
 # ========================================================================
 # ABLATION 1: BOUNDARY REMOVAL
 # ========================================================================
 
 print("\n" + "-"*80)
 print("ABLATION 1: BOUNDARY REMOVAL")
 print("-"*80)
 
 distances_cylinder = []
 distances_plane = []
 
 for i in range(n_iterations):
 if i % 5 == 0:
 print(f" Iteration {i}/{n_iterations}")
 
 # Cylinder (one boundary removed)
 data_cylinder = ablation_boundary_removal(data, grid_size, mode='cylinder')
 U_cylinder = compute_composite_signature_U(data_cylinder, fs, adjacency)
 distance = compute_signature_distance(U_baseline, U_cylinder)
 distances_cylinder.append(distance)
 
 # Plane (both boundaries removed)
 data_plane = ablation_boundary_removal(data, grid_size, mode='plane')
 U_plane = compute_composite_signature_U(data_plane, fs, adjacency)
 distance = compute_signature_distance(U_baseline, U_plane)
 distances_plane.append(distance)
 
 mean_distance_cylinder = np.mean(distances_cylinder)
 mean_distance_plane = np.mean(distances_plane)
 
 print(f"\nMean distance (cylinder): {mean_distance_cylinder:.4f}")
 print(f"Mean distance (plane): {mean_distance_plane:.4f}")
 
 # ========================================================================
 # ABLATION 2: ADJACENCY SCRAMBLE
 # ========================================================================
 
 print("\n" + "-"*80)
 print("ABLATION 2: ADJACENCY SCRAMBLE")
 print("-"*80)
 
 distances_adjacency = []
 
 for i in range(n_iterations):
 if i % 5 == 0:
 print(f" Iteration {i}/{n_iterations}")
 
 adjacency_scrambled = ablation_adjacency_scramble(adjacency)
 U_adjacency = compute_composite_signature_U(data, fs, adjacency_scrambled)
 distance = compute_signature_distance(U_baseline, U_adjacency)
 distances_adjacency.append(distance)
 
 mean_distance_adjacency = np.mean(distances_adjacency)
 
 print(f"\nMean distance (adjacency scramble): {mean_distance_adjacency:.4f}")
 
 # ========================================================================
 # ABLATION 3: PHASE DESTRUCTION
 # ========================================================================
 
 print("\n" + "-"*80)
 print("ABLATION 3: PHASE DESTRUCTION")
 print("-"*80)
 
 distances_phase = []
 
 for i in range(n_iterations):
 if i % 5 == 0:
 print(f" Iteration {i}/{n_iterations}")
 
 data_scrambled = ablation_phase_destruction(data)
 U_phase = compute_composite_signature_U(data_scrambled, fs, adjacency)
 distance = compute_signature_distance(U_baseline, U_phase)
 distances_phase.append(distance)
 
 mean_distance_phase = np.mean(distances_phase)
 
 print(f"\nMean distance (phase destruction): {mean_distance_phase:.4f}")
 
 # ========================================================================
 # ABLATION 4: CHANNEL RANDOMIZATION (NEGATIVE CONTROL)
 # ========================================================================
 
 print("\n" + "-"*80)
 print("ABLATION 4: CHANNEL RANDOMIZATION (NEGATIVE CONTROL)")
 print("-"*80)
 
 distances_channel = []
 
 for i in range(n_iterations):
 if i % 5 == 0:
 print(f" Iteration {i}/{n_iterations}")
 
 data_permuted = ablation_channel_randomization(data)
 U_channel = compute_composite_signature_U(data_permuted, fs, adjacency)
 distance = compute_signature_distance(U_baseline, U_channel)
 distances_channel.append(distance)
 
 mean_distance_channel = np.mean(distances_channel)
 
 print(f"\nMean distance (channel randomization): {mean_distance_channel:.4f}")
 
 # ========================================================================
 # SIGNATURE SEPARATION ANALYSIS
 # ========================================================================
 
 print("\n" + "="*80)
 print("SIGNATURE SEPARATION ANALYSIS")
 print("="*80)
 
 # Geometry-targeted ablations should show larger distances than negative control
 geometry_targeted_distances = [mean_distance_cylinder, mean_distance_plane, mean_distance_adjacency]
 max_geometry_distance = max(geometry_targeted_distances)
 
 # Pass if geometry-targeted ablations show > 2x distance of negative control
 pass_criterion = max_geometry_distance > 2 * mean_distance_channel
 
 print(f"\nMax geometry-targeted distance: {max_geometry_distance:.4f}")
 print(f"Negative control distance: {mean_distance_channel:.4f}")
 print(f"Ratio: {max_geometry_distance / (mean_distance_channel + 1e-10):.2f}x")
 print(f"\nPass criterion (>2x): {'✅ PASS' if pass_criterion else '❌ FAIL'}")
 
 results = {
 'baseline_signature': U_baseline,
 'ablation_1_boundary_removal': {
 'cylinder_distance': float(mean_distance_cylinder),
 'plane_distance': float(mean_distance_plane)
 },
 'ablation_2_adjacency_scramble': {
 'distance': float(mean_distance_adjacency)
 },
 'ablation_3_phase_destruction': {
 'distance': float(mean_distance_phase)
 },
 'ablation_4_channel_randomization': {
 'distance': float(mean_distance_channel)
 },
 'signature_separation': {
 'max_geometry_distance': float(max_geometry_distance),
 'negative_control_distance': float(mean_distance_channel),
 'ratio': float(max_geometry_distance / (mean_distance_channel + 1e-10)),
 'pass': bool(pass_criterion)
 }
 }
 
 return results

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 from entptc.utils.grid_utils import create_toroidal_grid
 
 # Load data
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/ablation_ladder')
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
 
 # Run ablation ladder
 results = run_ablation_ladder(data, fs, adjacency, n_iterations=10) # Reduced for speed
 
 # Save results
 output_path = output_dir / 'ablation_ladder_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")
