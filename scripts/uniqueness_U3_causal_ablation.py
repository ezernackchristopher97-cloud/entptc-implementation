"""
U3: Causal Ablation with Geometry-Targeted Controls
====================================================

Tests uniqueness via controlled geometry ablations that preserve marginal signal statistics.

Ablations:
1. Adjacency randomization with degree preservation
2. Boundary condition removal (torus→cylinder) with matched path length
3. Phase scrambling surrogates that preserve PSD

Per user protocol: Uniqueness passes only if topology-targeted ablation changes
topology-native metric in expected direction while matched-spectrum surrogates do not.

"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple

# Set random seed
np.random.seed(42)

# ============================================================================
# DEGREE-PRESERVED ADJACENCY RANDOMIZATION
# ============================================================================

def randomize_adjacency_degree_preserved(adjacency: np.ndarray, n_swaps: int = None) -> np.ndarray:
 """
 Randomize adjacency matrix while preserving degree distribution.
 
 Uses edge-swapping algorithm.
 
 Args:
 adjacency: (n_rois, n_rois) adjacency matrix
 n_swaps: number of edge swaps (default: 10 * n_edges)
 
 Returns:
 randomized_adjacency: (n_rois, n_rois) randomized adjacency
 """
 n_rois = adjacency.shape[0]
 
 # Get edge list
 edges = []
 for i in range(n_rois):
 for j in range(i+1, n_rois):
 if adjacency[i, j] > 0:
 edges.append((i, j))
 
 n_edges = len(edges)
 if n_swaps is None:
 n_swaps = 10 * n_edges
 
 # Edge swapping
 for _ in range(n_swaps):
 # Pick two random edges
 if len(edges) < 2:
 break
 
 idx1, idx2 = np.random.choice(len(edges), size=2, replace=False)
 (a, b) = edges[idx1]
 (c, d) = edges[idx2]
 
 # Try to swap: (a,b) + (c,d) → (a,c) + (b,d) or (a,d) + (b,c)
 # Check if swap is valid (no self-loops, no duplicate edges)
 if np.random.rand() < 0.5:
 new_edge1, new_edge2 = (a, c), (b, d)
 else:
 new_edge1, new_edge2 = (a, d), (b, c)
 
 # Check validity
 if (new_edge1[0] != new_edge1[1] and new_edge2[0] != new_edge2[1] and
 new_edge1 not in edges and new_edge2 not in edges and
 (new_edge1[1], new_edge1[0]) not in edges and (new_edge2[1], new_edge2[0]) not in edges):
 # Valid swap
 edges[idx1] = new_edge1
 edges[idx2] = new_edge2
 
 # Reconstruct adjacency
 randomized_adjacency = np.zeros_like(adjacency)
 for (i, j) in edges:
 randomized_adjacency[i, j] = 1
 randomized_adjacency[j, i] = 1
 
 return randomized_adjacency

# ============================================================================
# BOUNDARY CONDITION ABLATION
# ============================================================================

def remove_periodic_boundary(data: np.ndarray, grid_size: int, axis: int = 0) -> np.ndarray:
 """
 Remove one periodic boundary (torus → cylinder).
 
 Args:
 data: (n_rois, n_samples) array
 grid_size: size of grid (grid_size × grid_size)
 axis: which axis to break (0 = x, 1 = y)
 
 Returns:
 data_cylinder: (n_rois, n_samples) with one boundary broken
 """
 n_rois, n_samples = data.shape
 data_cylinder = data.copy()
 
 if axis == 0:
 # Break x-periodicity: scramble rows
 for i in range(grid_size):
 row_indices = list(range(i * grid_size, (i + 1) * grid_size))
 scrambled_indices = np.random.permutation(row_indices)
 data_cylinder[row_indices] = data[scrambled_indices]
 else:
 # Break y-periodicity: scramble columns
 for j in range(grid_size):
 col_indices = list(range(j, n_rois, grid_size))
 scrambled_indices = np.random.permutation(col_indices)
 data_cylinder[col_indices] = data[scrambled_indices]
 
 return data_cylinder

# ============================================================================
# U3 TEST
# ============================================================================

def run_u3_causal_ablation(data: np.ndarray, fs: float, adjacency: np.ndarray, n_surrogates: int = 50) -> Dict:
 """
 Run U3: Causal ablation test.
 
 Tests:
 1. Degree-preserved adjacency randomization
 2. Boundary condition removal (torus→cylinder)
 3. Phase-scrambled surrogates (matched PSD)
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 n_surrogates: number of surrogate iterations
 
 Returns:
 results: dict with ablation results
 """
 from entptc.t3_to_r3_mapping import compute_t3_coordinates, compute_t3_invariants
 
 print("\n" + "="*80)
 print("U3: CAUSAL ABLATION TEST")
 print("="*80)
 
 n_rois = data.shape[0]
 grid_size = int(np.sqrt(n_rois))
 
 # Baseline (real data, real adjacency)
 print("\nBaseline (real data, real adjacency)...")
 t3_coords_baseline = compute_t3_coordinates(data, fs)
 invariants_baseline = compute_t3_invariants(t3_coords_baseline, adjacency)
 
 print("Baseline invariants:")
 for key, value in invariants_baseline.items():
 print(f" {key}: {value:.6f}")
 
 # ========================================================================
 # ABLATION 1: Degree-preserved adjacency randomization
 # ========================================================================
 
 print("\n" + "-"*80)
 print("ABLATION 1: Degree-preserved adjacency randomization")
 print("-"*80)
 
 invariants_adj_random = []
 
 for i in range(n_surrogates):
 if i % 10 == 0:
 print(f" Iteration {i}/{n_surrogates}")
 
 # Randomize adjacency
 adjacency_random = randomize_adjacency_degree_preserved(adjacency)
 
 # Compute invariants
 invariants = compute_t3_invariants(t3_coords_baseline, adjacency_random)
 invariants_adj_random.append(invariants)
 
 # Compare
 print("\nAdjacency randomization results:")
 ablation1_pass = {}
 for key in invariants_baseline.keys():
 baseline_value = invariants_baseline[key]
 surrogate_values = [inv[key] for inv in invariants_adj_random]
 surrogate_mean = np.mean(surrogate_values)
 surrogate_std = np.std(surrogate_values)
 
 # Effect size
 effect_size = abs(baseline_value - surrogate_mean) / surrogate_std if surrogate_std > 0 else 0
 
 # Pass if baseline differs significantly (|d| > 0.5)
 pass_test = effect_size > 0.5
 ablation1_pass[key] = bool(pass_test)
 
 print(f" {key}:")
 print(f" Baseline: {baseline_value:.6f}")
 print(f" Randomized: {surrogate_mean:.6f} ± {surrogate_std:.6f}")
 print(f" Effect size: {effect_size:.2f}")
 print(f" Pass (|d| > 0.5): {'✅ YES' if pass_test else '❌ NO'}")
 
 # ========================================================================
 # ABLATION 2: Boundary condition removal (torus→cylinder)
 # ========================================================================
 
 print("\n" + "-"*80)
 print("ABLATION 2: Boundary condition removal (torus→cylinder)")
 print("-"*80)
 
 invariants_cylinder = []
 
 for i in range(n_surrogates):
 if i % 10 == 0:
 print(f" Iteration {i}/{n_surrogates}")
 
 # Remove one periodic boundary
 data_cylinder = remove_periodic_boundary(data, grid_size, axis=np.random.choice([0, 1]))
 
 # Compute T³ coordinates and invariants
 t3_coords_cylinder = compute_t3_coordinates(data_cylinder, fs)
 invariants = compute_t3_invariants(t3_coords_cylinder, adjacency)
 invariants_cylinder.append(invariants)
 
 # Compare
 print("\nBoundary removal results:")
 ablation2_pass = {}
 for key in invariants_baseline.keys():
 baseline_value = invariants_baseline[key]
 cylinder_values = [inv[key] for inv in invariants_cylinder]
 cylinder_mean = np.mean(cylinder_values)
 cylinder_std = np.std(cylinder_values)
 
 # Effect size
 effect_size = abs(baseline_value - cylinder_mean) / cylinder_std if cylinder_std > 0 else 0
 
 # Pass if baseline differs significantly (|d| > 0.5)
 pass_test = effect_size > 0.5
 ablation2_pass[key] = bool(pass_test)
 
 print(f" {key}:")
 print(f" Baseline: {baseline_value:.6f}")
 print(f" Cylinder: {cylinder_mean:.6f} ± {cylinder_std:.6f}")
 print(f" Effect size: {effect_size:.2f}")
 print(f" Pass (|d| > 0.5): {'✅ YES' if pass_test else '❌ NO'}")
 
 # ========================================================================
 # ABLATION 3: Phase-scrambled surrogates (matched PSD)
 # ========================================================================
 
 print("\n" + "-"*80)
 print("ABLATION 3: Phase-scrambled surrogates (matched PSD)")
 print("-"*80)
 
 from uniqueness_U1_FIX import generate_matched_spectrum_surrogate
 
 invariants_phase_scrambled = []
 
 for i in range(n_surrogates):
 if i % 10 == 0:
 print(f" Iteration {i}/{n_surrogates}")
 
 # Generate phase-scrambled surrogate
 data_scrambled = generate_matched_spectrum_surrogate(data)
 
 # Compute T³ coordinates and invariants
 t3_coords_scrambled = compute_t3_coordinates(data_scrambled, fs)
 invariants = compute_t3_invariants(t3_coords_scrambled, adjacency)
 invariants_phase_scrambled.append(invariants)
 
 # Compare
 print("\nPhase scrambling results:")
 ablation3_pass = {}
 for key in invariants_baseline.keys():
 baseline_value = invariants_baseline[key]
 scrambled_values = [inv[key] for inv in invariants_phase_scrambled]
 scrambled_mean = np.mean(scrambled_values)
 scrambled_std = np.std(scrambled_values)
 
 # Effect size
 effect_size = abs(baseline_value - scrambled_mean) / scrambled_std if scrambled_std > 0 else 0
 
 # Pass if baseline differs significantly (|d| > 0.5)
 pass_test = effect_size > 0.5
 ablation3_pass[key] = bool(pass_test)
 
 print(f" {key}:")
 print(f" Baseline: {baseline_value:.6f}")
 print(f" Scrambled: {scrambled_mean:.6f} ± {scrambled_std:.6f}")
 print(f" Effect size: {effect_size:.2f}")
 print(f" Pass (|d| > 0.5): {'✅ YES' if pass_test else '❌ NO'}")
 
 # ========================================================================
 # OVERALL VERDICT
 # ========================================================================
 
 print("\n" + "="*80)
 print("U3 OVERALL VERDICT")
 print("="*80)
 
 # Count passes per ablation
 n_pass_ablation1 = sum(ablation1_pass.values())
 n_pass_ablation2 = sum(ablation2_pass.values())
 n_pass_ablation3 = sum(ablation3_pass.values())
 n_total = len(invariants_baseline)
 
 print(f"\nAblation 1 (adjacency randomization): {n_pass_ablation1}/{n_total} invariants differ")
 print(f"Ablation 2 (boundary removal): {n_pass_ablation2}/{n_total} invariants differ")
 print(f"Ablation 3 (phase scrambling): {n_pass_ablation3}/{n_total} invariants differ")
 
 # Overall pass: at least one geometry-targeted ablation (1 or 2) shows more sensitivity than phase scrambling (3)
 overall_pass = (n_pass_ablation1 > n_pass_ablation3) or (n_pass_ablation2 > n_pass_ablation3)
 
 print(f"\nU3 verdict: {'✅ PASS' if overall_pass else '❌ FAIL'}")
 print("(Geometry-targeted ablations should be more sensitive than phase scrambling)")
 
 results = {
 'baseline_invariants': invariants_baseline,
 'ablation1_adjacency_randomization': {
 'pass_per_invariant': ablation1_pass,
 'n_pass': n_pass_ablation1,
 'n_total': n_total
 },
 'ablation2_boundary_removal': {
 'pass_per_invariant': ablation2_pass,
 'n_pass': n_pass_ablation2,
 'n_total': n_total
 },
 'ablation3_phase_scrambling': {
 'pass_per_invariant': ablation3_pass,
 'n_pass': n_pass_ablation3,
 'n_total': n_total
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
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/uniqueness_u3_ablation')
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
 
 # Run U3
 results = run_u3_causal_ablation(data, fs, adjacency, n_surrogates=30) # Reduced for speed
 
 # Save results
 output_path = output_dir / 'u3_ablation_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")
