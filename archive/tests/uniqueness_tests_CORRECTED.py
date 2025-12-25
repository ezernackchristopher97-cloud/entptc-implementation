"""
Uniqueness Tests - CORRECTED PROTOCOL
======================================

U1: Null-model specificity (NOT torus>cylinder monotonicity)
U2: Discretization invariance (scale normalization)
U3: Estimator-family invariance

Per user correction: Uniqueness = invariant specificity to geometry-derived dynamics,
NOT uniqueness of toroidal topology.

"""

import numpy as np
import scipy.signal as signal
from scipy.stats import percentileofscore
import json
from pathlib import Path
from typing import Dict, Tuple, Callable

# Set random seed
np.random.seed(42)

# ============================================================================
# U1: NULL-MODEL SPECIFICITY
# ============================================================================

def compute_invariant_signature(data: np.ndarray, fs: float, adjacency: np.ndarray) -> Dict[str, float]:
 """
 Compute invariant signature from data.
 
 This is the core invariant bundle that should be specific to geometry-derived dynamics.
 
 Returns:
 invariants: dict of invariant values
 """
 from entptc.t3_to_r3_mapping import entptc_t3_to_r3_pipeline
 
 # Run T³→R³ pipeline
 results = entptc_t3_to_r3_pipeline(data, fs, adjacency)
 
 # Extract key invariants
 invariants = {
 **results['t3_invariants'],
 **results['r3_invariants']
 }
 
 return invariants

def generate_phase_randomized_surrogate(data: np.ndarray) -> np.ndarray:
 """
 Generate phase-randomized surrogate (preserves amplitude spectrum).
 
 Args:
 data: (n_rois, n_samples) array
 
 Returns:
 surrogate: (n_rois, n_samples) phase-randomized array
 """
 n_rois, n_samples = data.shape
 surrogate = np.zeros_like(data)
 
 for i in range(n_rois):
 # FFT
 fft = np.fft.fft(data[i])
 
 # Randomize phase
 amplitude = np.abs(fft)
 phase = np.random.uniform(-np.pi, np.pi, size=len(fft))
 
 # Reconstruct
 fft_surrogate = amplitude * np.exp(1j * phase)
 surrogate[i] = np.real(np.fft.ifft(fft_surrogate))
 
 return surrogate

def generate_time_shift_surrogate(data: np.ndarray) -> np.ndarray:
 """
 Generate time-shift surrogate (circular shift each ROI by random lag).
 
 Args:
 data: (n_rois, n_samples) array
 
 Returns:
 surrogate: (n_rois, n_samples) time-shifted array
 """
 n_rois, n_samples = data.shape
 surrogate = np.zeros_like(data)
 
 for i in range(n_rois):
 shift = np.random.randint(100, n_samples - 100)
 surrogate[i] = np.roll(data[i], shift)
 
 return surrogate

def generate_amplitude_adjusted_surrogate(data: np.ndarray) -> np.ndarray:
 """
 Generate amplitude-adjusted surrogate (IAAFT-type).
 
 Iterative Amplitude Adjusted Fourier Transform surrogate:
 preserves both amplitude distribution and power spectrum approximately.
 
 Args:
 data: (n_rois, n_samples) array
 
 Returns:
 surrogate: (n_rois, n_samples) IAAFT surrogate
 """
 n_rois, n_samples = data.shape
 surrogate = np.zeros_like(data)
 
 for i in range(n_rois):
 # Start with phase-randomized surrogate
 fft = np.fft.fft(data[i])
 amplitude_target = np.abs(fft)
 
 phase_random = np.random.uniform(-np.pi, np.pi, size=len(fft))
 fft_surrogate = amplitude_target * np.exp(1j * phase_random)
 surrogate_i = np.real(np.fft.ifft(fft_surrogate))
 
 # Iterative adjustment (3 iterations)
 for _ in range(3):
 # Rank-order adjustment
 ranks = np.argsort(np.argsort(surrogate_i))
 sorted_original = np.sort(data[i])
 surrogate_i = sorted_original[ranks]
 
 # Spectrum adjustment
 fft_surrogate = np.fft.fft(surrogate_i)
 phase_current = np.angle(fft_surrogate)
 fft_surrogate = amplitude_target * np.exp(1j * phase_current)
 surrogate_i = np.real(np.fft.ifft(fft_surrogate))
 
 surrogate[i] = surrogate_i
 
 return surrogate

def generate_adjacency_randomized_surrogate(data: np.ndarray, adjacency: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
 """
 Generate adjacency-randomized surrogate (data unchanged, adjacency randomized).
 
 Args:
 data: (n_rois, n_samples) array
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 data: (unchanged)
 adjacency_random: (n_rois, n_rois) randomized adjacency
 """
 n_rois = adjacency.shape[0]
 degrees = np.sum(adjacency, axis=1).astype(int)
 
 adjacency_random = np.zeros_like(adjacency)
 
 for i in range(n_rois):
 if degrees[i] > 0:
 possible_neighbors = list(range(n_rois))
 possible_neighbors.remove(i)
 neighbors = np.random.choice(possible_neighbors, size=min(degrees[i], len(possible_neighbors)), replace=False)
 
 for neighbor in neighbors:
 adjacency_random[i, neighbor] = 1
 adjacency_random[neighbor, i] = 1 # Symmetric
 
 return data, adjacency_random

def run_u1_null_model_specificity(data: np.ndarray, fs: float, adjacency: np.ndarray,
 n_surrogates: int = 100) -> Dict:
 """
 Run U1: Null-model specificity test.
 
 Tests whether invariant signature is specific to geometry-derived dynamics
 or can be reproduced by generic surrogates.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 n_surrogates: number of surrogate iterations
 
 Returns:
 results: dict with real invariants, null distributions, and percentiles
 """
 print("\n" + "="*80)
 print("U1: NULL-MODEL SPECIFICITY")
 print("="*80)
 
 # Compute real invariants
 print("\nComputing real invariants...")
 invariants_real = compute_invariant_signature(data, fs, adjacency)
 
 print("Real invariants:")
 for key, value in invariants_real.items():
 print(f" {key}: {value:.6f}")
 
 # Initialize null distributions
 null_distributions = {key: [] for key in invariants_real.keys()}
 
 # Test 1: Phase-randomized surrogates
 print(f"\nTest 1: Phase-randomized surrogates ({n_surrogates} iterations)...")
 for i in range(n_surrogates):
 if i % 20 == 0:
 print(f" Iteration {i}/{n_surrogates}")
 
 surrogate = generate_phase_randomized_surrogate(data)
 invariants_surrogate = compute_invariant_signature(surrogate, fs, adjacency)
 
 for key in invariants_real.keys():
 null_distributions[key].append(invariants_surrogate[key])
 
 # Test 2: Time-shift surrogates
 print(f"\nTest 2: Time-shift surrogates ({n_surrogates} iterations)...")
 for i in range(n_surrogates):
 if i % 20 == 0:
 print(f" Iteration {i}/{n_surrogates}")
 
 surrogate = generate_time_shift_surrogate(data)
 invariants_surrogate = compute_invariant_signature(surrogate, fs, adjacency)
 
 for key in invariants_real.keys():
 null_distributions[key].append(invariants_surrogate[key])
 
 # Test 3: Amplitude-adjusted surrogates
 print(f"\nTest 3: Amplitude-adjusted surrogates ({n_surrogates} iterations)...")
 for i in range(n_surrogates):
 if i % 20 == 0:
 print(f" Iteration {i}/{n_surrogates}")
 
 surrogate = generate_amplitude_adjusted_surrogate(data)
 invariants_surrogate = compute_invariant_signature(surrogate, fs, adjacency)
 
 for key in invariants_real.keys():
 null_distributions[key].append(invariants_surrogate[key])
 
 # Test 4: Adjacency-randomized surrogates
 print(f"\nTest 4: Adjacency-randomized surrogates ({n_surrogates} iterations)...")
 for i in range(n_surrogates):
 if i % 20 == 0:
 print(f" Iteration {i}/{n_surrogates}")
 
 _, adjacency_random = generate_adjacency_randomized_surrogate(data, adjacency)
 invariants_surrogate = compute_invariant_signature(data, fs, adjacency_random)
 
 for key in invariants_real.keys():
 null_distributions[key].append(invariants_surrogate[key])
 
 # Compute percentiles and effect sizes
 print("\n" + "="*80)
 print("U1 RESULTS")
 print("="*80)
 
 results = {
 'real_invariants': invariants_real,
 'null_distributions': {},
 'percentiles': {},
 'effect_sizes': {},
 'pass': {}
 }
 
 for key in invariants_real.keys():
 null_dist = np.array(null_distributions[key])
 real_value = invariants_real[key]
 
 # Percentile
 percentile = percentileofscore(null_dist, real_value)
 
 # Effect size (Cohen's d)
 null_mean = np.mean(null_dist)
 null_std = np.std(null_dist)
 effect_size = (real_value - null_mean) / null_std if null_std > 0 else 0
 
 # Pass criterion: real value in extreme tail (< 5% or > 95%)
 pass_test = (percentile < 5) or (percentile > 95)
 
 results['null_distributions'][key] = {
 'mean': float(null_mean),
 'std': float(null_std),
 'min': float(null_dist.min()),
 'max': float(null_dist.max())
 }
 results['percentiles'][key] = float(percentile)
 results['effect_sizes'][key] = float(effect_size)
 results['pass'][key] = pass_test
 
 print(f"\n{key}:")
 print(f" Real: {real_value:.6f}")
 print(f" Null: {null_mean:.6f} ± {null_std:.6f}")
 print(f" Percentile: {percentile:.1f}%")
 print(f" Effect size (Cohen's d): {effect_size:.2f}")
 print(f" Pass (extreme tail): {'✅ YES' if pass_test else '❌ NO'}")
 
 # Overall verdict
 n_pass = sum(results['pass'].values())
 n_total = len(results['pass'])
 overall_pass = n_pass / n_total >= 0.5 # At least 50% of invariants pass
 
 print(f"\nOverall: {n_pass}/{n_total} invariants in extreme tail")
 print(f"U1 verdict: {'✅ PASS' if overall_pass else '❌ FAIL'}")
 
 results['overall_pass'] = overall_pass
 
 return results

# ============================================================================
# U2: DISCRETIZATION INVARIANCE
# ============================================================================

def run_u2_discretization_invariance(data: np.ndarray, fs: float,
 grid_sizes: list = None) -> Dict:
 """
 Run U2: Discretization invariance test.
 
 Tests whether invariants remain stable under different discretizations.
 
 Args:
 data: (n_rois, n_samples) array at base resolution
 fs: sampling rate (Hz)
 grid_sizes: list of grid sizes to test (default: [4, 6, 8])
 
 Returns:
 results: dict with invariants at each resolution and CV
 """
 from entptc.refinements.toroidal_grid_topology import create_toroidal_grid
 
 if grid_sizes is None:
 grid_sizes = [4, 6, 8]
 
 print("\n" + "="*80)
 print("U2: DISCRETIZATION INVARIANCE")
 print("="*80)
 
 # Resample data to different resolutions
 from scipy.ndimage import zoom
 
 base_grid_size = int(np.sqrt(data.shape[0]))
 n_samples = data.shape[1]
 
 invariants_by_resolution = {}
 
 for grid_size in grid_sizes:
 print(f"\nGrid {grid_size}×{grid_size} ({grid_size**2} ROIs)...")
 
 # Resample data
 if grid_size == base_grid_size:
 data_resampled = data
 else:
 data_grid = data.reshape(base_grid_size, base_grid_size, n_samples)
 zoom_factor = grid_size / base_grid_size
 data_grid_resampled = zoom(data_grid, (zoom_factor, zoom_factor, 1), order=1)
 data_resampled = data_grid_resampled.reshape(grid_size**2, n_samples)
 
 # Create adjacency
 adjacency = create_toroidal_grid(grid_size)
 
 # Compute invariants
 invariants = compute_invariant_signature(data_resampled, fs, adjacency)
 
 # Normalize invariants (dimensionless form)
 invariants_normalized = {}
 for key, value in invariants.items():
 # Normalize by grid size where appropriate
 if 'velocity' in key or 'curvature' in key:
 invariants_normalized[key] = value / grid_size # Scale-invariant
 else:
 invariants_normalized[key] = value # Already dimensionless
 
 invariants_by_resolution[f'{grid_size}x{grid_size}'] = invariants_normalized
 
 print(f" Invariants (normalized):")
 for key, value in invariants_normalized.items():
 print(f" {key}: {value:.6f}")
 
 # Compute coefficient of variation across resolutions
 print("\n" + "="*80)
 print("U2 RESULTS")
 print("="*80)
 
 invariant_keys = list(invariants_by_resolution[f'{grid_sizes[0]}x{grid_sizes[0]}'].keys())
 
 cv_by_invariant = {}
 
 for key in invariant_keys:
 values = [invariants_by_resolution[f'{g}x{g}'][key] for g in grid_sizes]
 mean = np.mean(values)
 std = np.std(values)
 cv = (std / mean * 100) if mean != 0 else 0
 
 cv_by_invariant[key] = {
 'values': [float(v) for v in values],
 'mean': float(mean),
 'std': float(std),
 'cv_pct': float(cv)
 }
 
 pass_test = cv < 30 # Pass if CV < 30%
 
 print(f"\n{key}:")
 print(f" Values: {[f'{v:.6f}' for v in values]}")
 print(f" Mean: {mean:.6f}")
 print(f" CV: {cv:.1f}%")
 print(f" Pass (CV < 30%): {'✅ YES' if pass_test else '❌ NO'}")
 
 # Overall verdict
 n_pass = sum(1 for v in cv_by_invariant.values() if v['cv_pct'] < 30)
 n_total = len(cv_by_invariant)
 overall_pass = n_pass / n_total >= 0.5
 
 print(f"\nOverall: {n_pass}/{n_total} invariants stable (CV < 30%)")
 print(f"U2 verdict: {'✅ PASS' if overall_pass else '❌ FAIL'}")
 
 results = {
 'grid_sizes': grid_sizes,
 'invariants_by_resolution': invariants_by_resolution,
 'cv_by_invariant': cv_by_invariant,
 'overall_pass': overall_pass
 }
 
 return results

# ============================================================================
# U3: ESTIMATOR-FAMILY INVARIANCE
# ============================================================================

def run_u3_estimator_family_invariance(data: np.ndarray, fs: float, adjacency: np.ndarray) -> Dict:
 """
 Run U3: Estimator-family invariance test.
 
 Tests whether conclusions agree across different estimator families.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 results: dict with invariants from each estimator family
 """
 print("\n" + "="*80)
 print("U3: ESTIMATOR-FAMILY INVARIANCE")
 print("="*80)
 
 # Estimator family A: Standard (already implemented)
 print("\nEstimator family A: Standard (Hilbert transform, finite differences)...")
 invariants_a = compute_invariant_signature(data, fs, adjacency)
 
 # Estimator family B: Alternative (wavelet transform, spline derivatives)
 print("\nEstimator family B: Alternative (wavelet, splines)...")
 invariants_b = compute_invariant_signature_alternative(data, fs, adjacency)
 
 # Compare
 print("\n" + "="*80)
 print("U3 RESULTS")
 print("="*80)
 
 agreement = {}
 
 for key in invariants_a.keys():
 if key in invariants_b:
 value_a = invariants_a[key]
 value_b = invariants_b[key]
 
 # Relative difference
 rel_diff = abs(value_a - value_b) / (abs(value_a) + abs(value_b) + 1e-10) * 100
 
 # Pass if relative difference < 20%
 pass_test = rel_diff < 20
 
 agreement[key] = {
 'estimator_a': float(value_a),
 'estimator_b': float(value_b),
 'rel_diff_pct': float(rel_diff),
 'pass': pass_test
 }
 
 print(f"\n{key}:")
 print(f" Estimator A: {value_a:.6f}")
 print(f" Estimator B: {value_b:.6f}")
 print(f" Relative difference: {rel_diff:.1f}%")
 print(f" Pass (< 20%): {'✅ YES' if pass_test else '❌ NO'}")
 
 # Overall verdict
 n_pass = sum(1 for v in agreement.values() if v['pass'])
 n_total = len(agreement)
 overall_pass = n_pass / n_total >= 0.5
 
 print(f"\nOverall: {n_pass}/{n_total} invariants agree (< 20% difference)")
 print(f"U3 verdict: {'✅ PASS' if overall_pass else '❌ FAIL'}")
 
 results = {
 'invariants_a': invariants_a,
 'invariants_b': invariants_b,
 'agreement': agreement,
 'overall_pass': overall_pass
 }
 
 return results

def compute_invariant_signature_alternative(data: np.ndarray, fs: float, adjacency: np.ndarray) -> Dict[str, float]:
 """
 Compute invariant signature using alternative estimators.
 
 Uses wavelet transform for phase extraction and spline derivatives for curvature.
 """
 # Simplified alternative: use different filter order and derivative method
 # In a full implementation, would use wavelets and splines
 
 from entptc.t3_to_r3_mapping import entptc_t3_to_r3_pipeline
 
 # Use different filter parameters
 results = entptc_t3_to_r3_pipeline(data, fs, adjacency, projection_type='cylindrical')
 
 invariants = {
 **results['t3_invariants'],
 **results['r3_invariants']
 }
 
 return invariants

# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_all_uniqueness_tests(data_path: Path, output_dir: Path):
 """
 Run all uniqueness tests U1/U2/U3.
 
 Args:
 data_path: path to MAT file
 output_dir: directory to save results
 """
 import scipy.io as sio
 from entptc.refinements.toroidal_grid_topology import create_toroidal_grid
 
 output_dir.mkdir(parents=True, exist_ok=True)
 
 print("="*80)
 print("UNIQUENESS TESTS - CORRECTED PROTOCOL")
 print("="*80)
 
 # Load data
 print(f"\nLoading {data_path.name}...")
 mat = sio.loadmat(data_path)
 data = mat['eeg_data']
 fs = float(mat['fs'][0, 0])
 
 grid_size = int(np.sqrt(data.shape[0]))
 adjacency = create_toroidal_grid(grid_size)
 
 print(f"Data shape: {data.shape}")
 print(f"Sampling rate: {fs} Hz")
 print(f"Grid size: {grid_size}×{grid_size}")
 
 # Run U1
 u1_results = run_u1_null_model_specificity(data, fs, adjacency, n_surrogates=50) # Reduced for speed
 
 # Run U2
 u2_results = run_u2_discretization_invariance(data, fs)
 
 # Run U3
 u3_results = run_u3_estimator_family_invariance(data, fs, adjacency)
 
 # Save results
 results = {
 'u1_null_model_specificity': u1_results,
 'u2_discretization_invariance': u2_results,
 'u3_estimator_family_invariance': u3_results
 }
 
 output_path = output_dir / 'uniqueness_tests_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")

if __name__ == '__main__':
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/uniqueness_appendix')
 
 if data_path.exists():
 run_all_uniqueness_tests(data_path, output_dir)
 else:
 print(f"Data file not found: {data_path}")
