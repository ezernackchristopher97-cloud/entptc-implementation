"""
Stage C Uniqueness Tests U1-U3

MANDATORY uniqueness tests for Stage C projection validation:
U1: Topology-specific ablation ladder (torus→cylinder, randomize, phase scramble)
U2: Non-toroidal manifold controls (S², random walk, matched spectrum)
U3: Resolution consistency (4×4, 6×6, 8×8 grids)

"""

import numpy as np
import scipy.signal as signal
from scipy.spatial.distance import pdist, squareform
import json
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# U1: TOPOLOGY-SPECIFIC ABLATION LADDER
# ============================================================================

def create_toroidal_grid(grid_size=4):
 """
 Create toroidal grid adjacency matrix with periodic boundaries.
 
 Args:
 grid_size: size of grid (grid_size x grid_size)
 
 Returns:
 adjacency: (n_rois, n_rois) adjacency matrix
 """
 n_rois = grid_size * grid_size
 adjacency = np.zeros((n_rois, n_rois))
 
 for i in range(grid_size):
 for j in range(grid_size):
 idx = i * grid_size + j
 
 # 4-neighbors with periodic boundaries
 neighbors = [
 ((i-1) % grid_size) * grid_size + j, # up
 ((i+1) % grid_size) * grid_size + j, # down
 i * grid_size + ((j-1) % grid_size), # left
 i * grid_size + ((j+1) % grid_size) # right
 ]
 
 for neighbor in neighbors:
 adjacency[idx, neighbor] = 1
 
 return adjacency

def create_cylindrical_grid(grid_size=4):
 """
 Create cylindrical grid (periodic in one dimension only).
 
 Args:
 grid_size: size of grid (grid_size x grid_size)
 
 Returns:
 adjacency: (n_rois, n_rois) adjacency matrix
 """
 n_rois = grid_size * grid_size
 adjacency = np.zeros((n_rois, n_rois))
 
 for i in range(grid_size):
 for j in range(grid_size):
 idx = i * grid_size + j
 
 # 4-neighbors with periodic boundary in j only
 neighbors = []
 
 # up (no wrap)
 if i > 0:
 neighbors.append((i-1) * grid_size + j)
 
 # down (no wrap)
 if i < grid_size - 1:
 neighbors.append((i+1) * grid_size + j)
 
 # left (wrap)
 neighbors.append(i * grid_size + ((j-1) % grid_size))
 
 # right (wrap)
 neighbors.append(i * grid_size + ((j+1) % grid_size))
 
 for neighbor in neighbors:
 adjacency[idx, neighbor] = 1
 
 return adjacency

def randomize_adjacency(adjacency):
 """
 Randomize adjacency while preserving degree distribution.
 
 Args:
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 adjacency_random: (n_rois, n_rois) randomized adjacency matrix
 """
 n_rois = adjacency.shape[0]
 degrees = np.sum(adjacency, axis=1).astype(int)
 
 adjacency_random = np.zeros_like(adjacency)
 
 for i in range(n_rois):
 # Select random neighbors (excluding self)
 possible_neighbors = list(range(n_rois))
 possible_neighbors.remove(i)
 neighbors = np.random.choice(possible_neighbors, size=degrees[i], replace=False)
 
 for neighbor in neighbors:
 adjacency_random[i, neighbor] = 1
 adjacency_random[neighbor, i] = 1 # Symmetric
 
 return adjacency_random

def phase_scramble_data(data):
 """
 Phase scramble data to destroy coherence while preserving amplitude spectrum.
 
 Args:
 data: (n_rois, n_samples) array
 
 Returns:
 data_scrambled: (n_rois, n_samples) phase-scrambled array
 """
 n_rois, n_samples = data.shape
 data_scrambled = np.zeros_like(data)
 
 for i in range(n_rois):
 # FFT
 fft = np.fft.fft(data[i])
 
 # Randomize phase
 amplitude = np.abs(fft)
 phase = np.random.uniform(-np.pi, np.pi, size=len(fft))
 
 # Reconstruct
 fft_scrambled = amplitude * np.exp(1j * phase)
 data_scrambled[i] = np.real(np.fft.ifft(fft_scrambled))
 
 return data_scrambled

def run_u1_ablation_ladder(data, fs, metric_func):
 """
 Run U1: Topology-specific ablation ladder.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 metric_func: function that computes organization metric from data
 
 Returns:
 results: dict of results for each ablation
 """
 print("\n" + "="*80)
 print("U1: TOPOLOGY-SPECIFIC ABLATION LADDER")
 print("="*80)
 
 grid_size = int(np.sqrt(data.shape[0]))
 
 # Baseline: Intact torus
 adjacency_torus = create_toroidal_grid(grid_size)
 metric_torus = metric_func(data, adjacency_torus)
 print(f"Intact torus: metric = {metric_torus:.6f}")
 
 # Ablation 1: Break periodic boundary (torus → cylinder)
 adjacency_cylinder = create_cylindrical_grid(grid_size)
 metric_cylinder = metric_func(data, adjacency_cylinder)
 collapse_cylinder = (metric_torus - metric_cylinder) / metric_torus * 100
 print(f"Cylinder (one periodic boundary): metric = {metric_cylinder:.6f}, collapse = {collapse_cylinder:.1f}%")
 
 # Ablation 2: Randomize neighbor adjacency
 adjacency_random = randomize_adjacency(adjacency_torus)
 metric_random = metric_func(data, adjacency_random)
 collapse_random = (metric_torus - metric_random) / metric_torus * 100
 print(f"Randomized adjacency: metric = {metric_random:.6f}, collapse = {collapse_random:.1f}%")
 
 # Ablation 3: Phase scramble (destroy coherence)
 data_scrambled = phase_scramble_data(data)
 metric_scrambled = metric_func(data_scrambled, adjacency_torus)
 collapse_scrambled = (metric_torus - metric_scrambled) / metric_torus * 100
 print(f"Phase scrambled: metric = {metric_scrambled:.6f}, collapse = {collapse_scrambled:.1f}%")
 
 # Check monotonic degradation
 metrics = [metric_torus, metric_cylinder, metric_random, metric_scrambled]
 is_monotonic = all(metrics[i] >= metrics[i+1] for i in range(len(metrics)-1))
 print(f"\nMonotonic degradation: {'✅ YES' if is_monotonic else '❌ NO'}")
 
 results = {
 'intact_torus': float(metric_torus),
 'cylinder': {'metric': float(metric_cylinder), 'collapse_pct': float(collapse_cylinder)},
 'randomized': {'metric': float(metric_random), 'collapse_pct': float(collapse_random)},
 'phase_scrambled': {'metric': float(metric_scrambled), 'collapse_pct': float(collapse_scrambled)},
 'monotonic_degradation': is_monotonic
 }
 
 return results

# ============================================================================
# U2: NON-TOROIDAL MANIFOLD CONTROLS
# ============================================================================

def generate_s2_trajectory(n_samples, n_rois=16):
 """
 Generate trajectory on 2-sphere S².
 
 Args:
 n_samples: number of time samples
 n_rois: number of ROIs
 
 Returns:
 data: (n_rois, n_samples) synthetic data from S² trajectory
 """
 # Generate trajectory on S² (spherical coordinates)
 t = np.linspace(0, 4*np.pi, n_samples)
 theta = np.pi/4 * np.sin(0.5 * t) # Latitude oscillation
 phi = t # Longitude rotation
 
 # Convert to Cartesian
 x = np.cos(theta) * np.cos(phi)
 y = np.cos(theta) * np.sin(phi)
 z = np.sin(theta)
 
 # Project to ROIs (spatial sampling)
 data = np.zeros((n_rois, n_samples))
 for i in range(n_rois):
 # Each ROI samples a different combination of coordinates
 weight_x = np.sin(2*np.pi*i/n_rois)
 weight_y = np.cos(2*np.pi*i/n_rois)
 weight_z = np.sin(4*np.pi*i/n_rois)
 
 data[i] = weight_x * x + weight_y * y + weight_z * z
 
 # Add noise
 data[i] += 0.1 * np.random.randn(n_samples)
 
 return data

def generate_random_walk(n_samples, n_rois=16):
 """
 Generate random walk control.
 
 Args:
 n_samples: number of time samples
 n_rois: number of ROIs
 
 Returns:
 data: (n_rois, n_samples) random walk data
 """
 data = np.cumsum(np.random.randn(n_rois, n_samples), axis=1)
 
 # Normalize
 data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
 
 return data

def generate_matched_spectrum_surrogate(data):
 """
 Generate surrogate data with matched power spectrum.
 
 Args:
 data: (n_rois, n_samples) original data
 
 Returns:
 surrogate: (n_rois, n_samples) surrogate data with matched spectrum
 """
 n_rois, n_samples = data.shape
 surrogate = np.zeros_like(data)
 
 for i in range(n_rois):
 # FFT
 fft = np.fft.fft(data[i])
 
 # Keep amplitude, randomize phase
 amplitude = np.abs(fft)
 phase = np.random.uniform(-np.pi, np.pi, size=len(fft))
 
 # Reconstruct
 fft_surrogate = amplitude * np.exp(1j * phase)
 surrogate[i] = np.real(np.fft.ifft(fft_surrogate))
 
 return surrogate

def run_u2_manifold_controls(data, fs, metric_func):
 """
 Run U2: Non-toroidal manifold controls.
 
 Args:
 data: (n_rois, n_samples) original toroidal data
 fs: sampling rate (Hz)
 metric_func: function that computes organization metric from data
 
 Returns:
 results: dict of results for each control
 """
 print("\n" + "="*80)
 print("U2: NON-TOROIDAL MANIFOLD CONTROLS")
 print("="*80)
 
 n_rois, n_samples = data.shape
 grid_size = int(np.sqrt(n_rois))
 adjacency_torus = create_toroidal_grid(grid_size)
 
 # Baseline: Original toroidal data
 metric_torus = metric_func(data, adjacency_torus)
 print(f"Original toroidal data: metric = {metric_torus:.6f}")
 
 # Control 1: S² trajectory
 data_s2 = generate_s2_trajectory(n_samples, n_rois)
 metric_s2 = metric_func(data_s2, adjacency_torus)
 diff_s2 = (metric_torus - metric_s2) / metric_torus * 100
 print(f"S² trajectory control: metric = {metric_s2:.6f}, difference = {diff_s2:.1f}%")
 
 # Control 2: Random walk
 data_rw = generate_random_walk(n_samples, n_rois)
 metric_rw = metric_func(data_rw, adjacency_torus)
 diff_rw = (metric_torus - metric_rw) / metric_torus * 100
 print(f"Random walk control: metric = {metric_rw:.6f}, difference = {diff_rw:.1f}%")
 
 # Control 3: Matched spectrum surrogate
 data_surrogate = generate_matched_spectrum_surrogate(data)
 metric_surrogate = metric_func(data_surrogate, adjacency_torus)
 diff_surrogate = (metric_torus - metric_surrogate) / metric_torus * 100
 print(f"Matched spectrum surrogate: metric = {metric_surrogate:.6f}, difference = {diff_surrogate:.1f}%")
 
 # Check if toroidal signature is unique
 is_unique = all([abs(diff_s2) > 30, abs(diff_rw) > 30, abs(diff_surrogate) > 30])
 print(f"\nToroidal signature unique (>30% difference from all controls): {'✅ YES' if is_unique else '❌ NO'}")
 
 results = {
 'original_torus': float(metric_torus),
 's2_trajectory': {'metric': float(metric_s2), 'diff_pct': float(diff_s2)},
 'random_walk': {'metric': float(metric_rw), 'diff_pct': float(diff_rw)},
 'matched_spectrum': {'metric': float(metric_surrogate), 'diff_pct': float(diff_surrogate)},
 'unique_signature': is_unique
 }
 
 return results

# ============================================================================
# U3: RESOLUTION CONSISTENCY
# ============================================================================

def resample_to_grid(data, original_grid_size, target_grid_size):
 """
 Resample data from original grid to target grid resolution.
 
 Args:
 data: (n_rois_original, n_samples) array
 original_grid_size: original grid size
 target_grid_size: target grid size
 
 Returns:
 data_resampled: (n_rois_target, n_samples) resampled array
 """
 n_samples = data.shape[1]
 n_rois_target = target_grid_size * target_grid_size
 
 # Reshape to 2D grid
 data_grid = data.reshape(original_grid_size, original_grid_size, n_samples)
 
 # Interpolate to target resolution
 from scipy.ndimage import zoom
 zoom_factor = target_grid_size / original_grid_size
 data_grid_resampled = zoom(data_grid, (zoom_factor, zoom_factor, 1), order=1)
 
 # Reshape back to (n_rois, n_samples)
 data_resampled = data_grid_resampled.reshape(n_rois_target, n_samples)
 
 return data_resampled

def run_u3_resolution_consistency(data, fs, metric_func, original_grid_size=4):
 """
 Run U3: Resolution consistency test.
 
 Args:
 data: (n_rois, n_samples) original data at base resolution
 fs: sampling rate (Hz)
 metric_func: function that computes organization metric from data
 original_grid_size: original grid size (default 4x4)
 
 Returns:
 results: dict of results for each resolution
 """
 print("\n" + "="*80)
 print("U3: RESOLUTION CONSISTENCY")
 print("="*80)
 
 resolutions = [4, 6, 8]
 metrics = []
 
 for grid_size in resolutions:
 # Resample data to target resolution
 if grid_size == original_grid_size:
 data_resampled = data
 else:
 data_resampled = resample_to_grid(data, original_grid_size, grid_size)
 
 # Create adjacency for this resolution
 adjacency = create_toroidal_grid(grid_size)
 
 # Compute metric
 metric = metric_func(data_resampled, adjacency)
 metrics.append(metric)
 
 print(f"Grid {grid_size}×{grid_size} ({grid_size**2} ROIs): metric = {metric:.6f}")
 
 # Compute coefficient of variation
 cv = np.std(metrics) / np.mean(metrics) * 100
 print(f"\nCoefficient of variation across resolutions: {cv:.1f}%")
 
 # Check consistency (CV < 30%)
 is_consistent = cv < 30
 print(f"Resolution consistency (CV < 30%): {'✅ YES' if is_consistent else '❌ NO'}")
 
 results = {
 'resolutions': [f"{g}x{g}" for g in resolutions],
 'metrics': [float(m) for m in metrics],
 'cv_pct': float(cv),
 'consistent': is_consistent
 }
 
 return results

# ============================================================================
# EXAMPLE METRIC FUNCTION (PLACEHOLDER)
# ============================================================================

def example_organization_metric(data, adjacency):
 """
 Example organization metric (phase winding or trajectory alignment).
 
 This is a placeholder. Replace with actual metric from stage_c_sanity_patch.py
 
 Args:
 data: (n_rois, n_samples) array
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 metric: scalar organization metric
 """
 # Compute phase
 from scipy.signal import hilbert
 phases = np.angle(hilbert(data, axis=1))
 
 # Compute phase winding (simplified)
 phase_diffs = []
 n_rois = data.shape[0]
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 phase_diffs.append(phase_diff)
 
 if len(phase_diffs) == 0:
 return 0.0
 
 phase_diffs = np.array(phase_diffs)
 
 # Phase winding metric
 winding = np.abs(np.mean(np.exp(1j * phase_diffs)))
 
 return winding

# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_uniqueness_tests(data, fs, metric_func, output_dir):
 """
 Run all uniqueness tests U1-U3.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 metric_func: function that computes organization metric from data
 output_dir: directory to save results
 
 Returns:
 all_results: dict of all uniqueness test results
 """
 all_results = {}
 
 # U1: Ablation ladder
 u1_results = run_u1_ablation_ladder(data, fs, metric_func)
 all_results['u1_ablation_ladder'] = u1_results
 
 # U2: Manifold controls
 u2_results = run_u2_manifold_controls(data, fs, metric_func)
 all_results['u2_manifold_controls'] = u2_results
 
 # U3: Resolution consistency
 u3_results = run_u3_resolution_consistency(data, fs, metric_func)
 all_results['u3_resolution_consistency'] = u3_results
 
 # Overall verdict
 u1_pass = u1_results['monotonic_degradation']
 u2_pass = u2_results['unique_signature']
 u3_pass = u3_results['consistent']
 
 all_pass = u1_pass and u2_pass and u3_pass
 
 print("\n" + "="*80)
 print("UNIQUENESS TESTS SUMMARY")
 print("="*80)
 print(f"U1 (Ablation Ladder): {'✅ PASS' if u1_pass else '❌ FAIL'}")
 print(f"U2 (Manifold Controls): {'✅ PASS' if u2_pass else '❌ FAIL'}")
 print(f"U3 (Resolution Consistency): {'✅ PASS' if u3_pass else '❌ FAIL'}")
 print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
 
 all_results['overall_verdict'] = {
 'u1_pass': u1_pass,
 'u2_pass': u2_pass,
 'u3_pass': u3_pass,
 'all_pass': all_pass
 }
 
 # Save results
 output_path = Path(output_dir) / 'uniqueness_tests_results.json'
 with open(output_path, 'w') as f:
 json.dump(all_results, f, indent=2)
 print(f"\n✅ Uniqueness test results saved to {output_path}")
 
 return all_results

if __name__ == '__main__':
 print("Stage C Uniqueness Tests U1-U3")
 print("Provides functions for uniqueness testing.")
 print("Import and use in stage_c_dataset2_analysis_fixed.py")
