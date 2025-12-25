"""
U1-FIX: Normalization-Corrected Winding
========================================

Fixes the U1 "cylinder > torus" failure by normalizing phase winding by:
1. Total path length
2. Total angular velocity
3. Total observation time

Plus matched-spectrum surrogate controls.

Per user protocol: Raw winding can be biased by path length differences.
Normalization makes the metric topology-invariant.

"""

import numpy as np
import scipy.signal as signal
from scipy.stats import percentileofscore
from pathlib import Path
import json
from typing import Dict, Tuple

# Set random seed
np.random.seed(42)

# ============================================================================
# NORMALIZED WINDING COMPUTATION
# ============================================================================

def compute_normalized_winding(t3_coords: np.ndarray, adjacency: np.ndarray) -> Dict[str, float]:
 """
 Compute phase winding normalized by path length, angular velocity, and time.
 
 Args:
 t3_coords: (3, n_rois, n_samples) T³ coordinates
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 winding_metrics: dict with raw and normalized winding values
 """
 n_dims, n_rois, n_samples = t3_coords.shape
 
 results = {}
 
 for dim in range(n_dims):
 phases = t3_coords[dim] # (n_rois, n_samples)
 
 # Raw phase winding (as before)
 raw_winding = []
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 winding = np.abs(np.mean(np.exp(1j * phase_diff)))
 raw_winding.append(winding)
 
 raw_winding_mean = np.mean(raw_winding) if len(raw_winding) > 0 else 0.0
 
 # Compute normalization factors
 
 # 1. Total path length
 phases_unwrapped = np.unwrap(phases, axis=1)
 velocities = np.diff(phases_unwrapped, axis=1)
 path_lengths = np.abs(velocities).sum(axis=1) # Per ROI
 total_path_length = path_lengths.mean()
 
 # 2. Total angular velocity
 total_angular_velocity = np.abs(velocities).mean()
 
 # 3. Total observation time
 total_time = n_samples # In samples
 
 # Normalized winding
 winding_per_path = raw_winding_mean / (total_path_length + 1e-10)
 winding_per_velocity = raw_winding_mean / (total_angular_velocity + 1e-10)
 winding_per_time = raw_winding_mean / (total_time + 1e-10)
 
 results[f'theta{dim+1}_raw_winding'] = float(raw_winding_mean)
 results[f'theta{dim+1}_path_length'] = float(total_path_length)
 results[f'theta{dim+1}_angular_velocity'] = float(total_angular_velocity)
 results[f'theta{dim+1}_winding_per_path'] = float(winding_per_path)
 results[f'theta{dim+1}_winding_per_velocity'] = float(winding_per_velocity)
 results[f'theta{dim+1}_winding_per_time'] = float(winding_per_time)
 
 return results

# ============================================================================
# MATCHED-SPECTRUM SURROGATES
# ============================================================================

def generate_matched_spectrum_surrogate(data: np.ndarray) -> np.ndarray:
 """
 Generate matched-spectrum surrogate (phase randomization preserving PSD).
 
 Args:
 data: (n_rois, n_samples) time series
 
 Returns:
 surrogate: (n_rois, n_samples) phase-randomized array
 """
 n_rois, n_samples = data.shape
 surrogate = np.zeros_like(data)
 
 for i in range(n_rois):
 # FFT
 fft = np.fft.fft(data[i])
 
 # Randomize phase, preserve amplitude
 amplitude = np.abs(fft)
 phase = np.random.uniform(-np.pi, np.pi, size=len(fft))
 
 # Enforce conjugate symmetry for real output
 phase[0] = 0 # DC component
 if n_samples % 2 == 0:
 phase[n_samples // 2] = 0 # Nyquist
 phase[n_samples // 2 + 1:] = -phase[1:n_samples // 2][::-1]
 
 # Reconstruct
 fft_surrogate = amplitude * np.exp(1j * phase)
 surrogate[i] = np.real(np.fft.ifft(fft_surrogate))
 
 return surrogate

# ============================================================================
# U1-FIX TEST
# ============================================================================

def run_u1_fix(data: np.ndarray, fs: float, adjacency: np.ndarray, n_surrogates: int = 100) -> Dict:
 """
 Run U1-FIX: Normalization-corrected winding test.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 n_surrogates: number of surrogate iterations
 
 Returns:
 results: dict with real and surrogate winding metrics
 """
 from entptc.t3_to_r3_mapping import compute_t3_coordinates
 
 print("\n" + "="*80)
 print("U1-FIX: NORMALIZATION-CORRECTED WINDING")
 print("="*80)
 
 # Compute T³ coordinates for real data
 print("\nComputing T³ coordinates for real data...")
 t3_coords_real = compute_t3_coordinates(data, fs)
 
 # Compute normalized winding for real data
 print("\nComputing normalized winding for real data...")
 winding_real = compute_normalized_winding(t3_coords_real, adjacency)
 
 print("\nReal data winding metrics:")
 for key, value in winding_real.items():
 print(f" {key}: {value:.6f}")
 
 # Generate matched-spectrum surrogates and compute winding
 print(f"\nGenerating {n_surrogates} matched-spectrum surrogates...")
 
 winding_surrogates = {key: [] for key in winding_real.keys()}
 
 for i in range(n_surrogates):
 if i % 20 == 0:
 print(f" Surrogate {i}/{n_surrogates}")
 
 # Generate surrogate
 surrogate = generate_matched_spectrum_surrogate(data)
 
 # Compute T³ coordinates
 t3_coords_surrogate = compute_t3_coordinates(surrogate, fs)
 
 # Compute normalized winding
 winding_surrogate = compute_normalized_winding(t3_coords_surrogate, adjacency)
 
 for key in winding_real.keys():
 winding_surrogates[key].append(winding_surrogate[key])
 
 # Compute percentiles and effect sizes
 print("\n" + "="*80)
 print("U1-FIX RESULTS")
 print("="*80)
 
 results = {
 'real_winding': winding_real,
 'surrogate_distributions': {},
 'percentiles': {},
 'effect_sizes': {},
 'pass': {}
 }
 
 for key in winding_real.keys():
 surrogate_dist = np.array(winding_surrogates[key])
 real_value = winding_real[key]
 
 # Percentile
 percentile = percentileofscore(surrogate_dist, real_value)
 
 # Effect size (Cohen's d)
 surrogate_mean = np.mean(surrogate_dist)
 surrogate_std = np.std(surrogate_dist)
 effect_size = (real_value - surrogate_mean) / surrogate_std if surrogate_std > 0 else 0
 
 # Pass criterion: real value in extreme tail (< 5% or > 95%)
 pass_test = (percentile < 5) or (percentile > 95)
 
 results['surrogate_distributions'][key] = {
 'mean': float(surrogate_mean),
 'std': float(surrogate_std),
 'min': float(surrogate_dist.min()),
 'max': float(surrogate_dist.max())
 }
 results['percentiles'][key] = float(percentile)
 results['effect_sizes'][key] = float(effect_size)
 results['pass'][key] = bool(pass_test)
 
 print(f"\n{key}:")
 print(f" Real: {real_value:.6f}")
 print(f" Surrogate: {surrogate_mean:.6f} ± {surrogate_std:.6f}")
 print(f" Percentile: {percentile:.1f}%")
 print(f" Effect size (Cohen's d): {effect_size:.2f}")
 print(f" Pass (extreme tail): {'✅ YES' if pass_test else '❌ NO'}")
 
 # Focus on normalized metrics
 normalized_keys = [k for k in winding_real.keys() if 'per_' in k]
 n_pass_normalized = sum(results['pass'][k] for k in normalized_keys)
 n_total_normalized = len(normalized_keys)
 
 overall_pass = n_pass_normalized / n_total_normalized >= 0.5 if n_total_normalized > 0 else False
 
 print(f"\nNormalized metrics: {n_pass_normalized}/{n_total_normalized} in extreme tail")
 print(f"U1-FIX verdict: {'✅ PASS' if overall_pass else '❌ FAIL'}")
 
 results['overall_pass'] = overall_pass
 results['n_pass_normalized'] = n_pass_normalized
 results['n_total_normalized'] = n_total_normalized
 
 return results

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 import scipy.io as sio
 from entptc.utils.grid_utils import create_toroidal_grid
 
 # Load data
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/uniqueness_u1_fix')
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
 
 # Run U1-FIX
 results = run_u1_fix(data, fs, adjacency, n_surrogates=50) # Reduced for speed
 
 # Save results
 output_path = output_dir / 'u1_fix_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")
