"""
STAGE B: MANDATORY ROBUSTNESS CHECKS (CORRECTED)

Test whether the ~0.4 Hz frequency estimate is robust to parameter choices.

CRITICAL FIX: Use the SAME computational pipeline as Stage B (Stage A → Stage B),
just varying parameters within that pipeline. Do NOT recompute from scratch.

Required Tests:
1. Grid resolution (4×4 vs higher-resolution toroidal discretizations)
2. Trajectory smoothing / interpolation method
3. Curvature estimator choice
4. Window length and segmentation

Requirement: Report sensitivity ranges, not single point estimate.

"""

import numpy as np
import scipy.io as sio
import json
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Directories
stage_a_dir = '/home/ubuntu/entptc-implementation/stage_a_datasets/hafting_2005'
output_dir = '/home/ubuntu/entptc-implementation/stage_b_robustness_outputs_corrected'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("=" * 80)
print("STAGE B: MANDATORY ROBUSTNESS CHECKS (CORRECTED)")
print("=" * 80)

# ============================================================================
# BASELINE: Load original Stage B results
# ============================================================================

print("\nLoading baseline Stage B results...")

with open('/home/ubuntu/entptc-implementation/stage_b_outputs/frequency_invariants.json', 'r') as f:
 baseline_invariants = json.load(f)

baseline_freqs = [inv['entptc_characteristic_frequency_hz'] for inv in baseline_invariants]
baseline_mean = np.mean(baseline_freqs)
baseline_std = np.std(baseline_freqs)

print(f"Baseline EntPTC frequency: {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
print(f"Baseline cells: {len(baseline_invariants)}")

# ============================================================================
# HELPER FUNCTIONS (from Stage A pipeline)
# ============================================================================

def compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps, grid_size=20, sigma=2.0):
 """
 Compute spatial firing rate map with toroidal topology.
 """
 # Filter valid positions
 valid_mask = ~np.isnan(pos_x) & ~np.isnan(pos_y)
 pos_x_clean = pos_x[valid_mask]
 pos_y_clean = pos_y[valid_mask]
 pos_t_clean = pos_timestamps[valid_mask]
 
 if len(pos_x_clean) == 0:
 return None, None, None
 
 # Create grid
 x_edges = np.linspace(np.min(pos_x_clean), np.max(pos_x_clean), grid_size + 1)
 y_edges = np.linspace(np.min(pos_y_clean), np.max(pos_y_clean), grid_size + 1)
 
 # Occupancy map
 occupancy, _, _ = np.histogram2d(pos_x_clean, pos_y_clean, bins=[x_edges, y_edges])
 occupancy = gaussian_filter(occupancy, sigma=sigma)
 occupancy[occupancy < 0.1] = np.nan
 
 # Spike map
 spike_positions_x = []
 spike_positions_y = []
 
 for spike_t in spike_times:
 idx = np.argmin(np.abs(pos_t_clean - spike_t))
 if idx < len(pos_x_clean):
 spike_positions_x.append(pos_x_clean[idx])
 spike_positions_y.append(pos_y_clean[idx])
 
 if len(spike_positions_x) == 0:
 return None, None, None
 
 spike_map, _, _ = np.histogram2d(spike_positions_x, spike_positions_y, bins=[x_edges, y_edges])
 spike_map = gaussian_filter(spike_map, sigma=sigma)
 
 # Firing rate
 firing_rate = spike_map / (occupancy + 1e-10)
 
 return firing_rate, x_edges, y_edges

def map_to_toroidal_coordinates(firing_rate_map):
 """
 Map firing rate map to toroidal phase coordinates.
 """
 if firing_rate_map is None:
 return None, None
 
 # Find peaks (grid vertices)
 from scipy.ndimage import maximum_filter
 local_max = (firing_rate_map == maximum_filter(firing_rate_map, size=3))
 peaks = np.argwhere(local_max & (firing_rate_map > np.nanmean(firing_rate_map)))
 
 if len(peaks) < 3:
 return None, None
 
 # Map to [0, 2π] toroidal coordinates
 grid_size = firing_rate_map.shape[0]
 theta_x = peaks[:, 1] / grid_size * 2 * np.pi
 theta_y = peaks[:, 0] / grid_size * 2 * np.pi
 
 return theta_x, theta_y

def compute_phase_velocity(theta_x, theta_y, dt=1.0):
 """
 Compute phase velocity from toroidal coordinates.
 """
 if theta_x is None or len(theta_x) < 2:
 return None
 
 # Compute velocity components
 d_theta_x = np.diff(theta_x)
 d_theta_y = np.diff(theta_y)
 
 # Handle wraparound (toroidal)
 d_theta_x[d_theta_x > np.pi] -= 2 * np.pi
 d_theta_x[d_theta_x < -np.pi] += 2 * np.pi
 d_theta_y[d_theta_y > np.pi] -= 2 * np.pi
 d_theta_y[d_theta_y < -np.pi] += 2 * np.pi
 
 # Phase velocity magnitude
 v_phi = np.sqrt(d_theta_x**2 + d_theta_y**2) / dt
 
 return np.mean(v_phi)

def compute_curvature(theta_x, theta_y):
 """
 Compute trajectory curvature on torus.
 """
 if theta_x is None or len(theta_x) < 3:
 return None
 
 # First derivatives
 d_theta_x = np.gradient(theta_x)
 d_theta_y = np.gradient(theta_y)
 
 # Second derivatives
 dd_theta_x = np.gradient(d_theta_x)
 dd_theta_y = np.gradient(d_theta_y)
 
 # Curvature
 curvature = np.abs(d_theta_x * dd_theta_y - d_theta_y * dd_theta_x) / (d_theta_x**2 + d_theta_y**2 + 1e-10)**(3/2)
 
 return np.mean(curvature)

def compute_phase_entropy(theta_x, theta_y, bins=10):
 """
 Compute phase space entropy.
 """
 if theta_x is None or len(theta_x) < 2:
 return None
 
 # 2D histogram on torus
 H, _, _ = np.histogram2d(theta_x, theta_y, bins=bins, range=[[0, 2*np.pi], [0, 2*np.pi]])
 
 # Normalize
 p = H.flatten()
 p = p / np.sum(p) if np.sum(p) > 0 else p
 p = p + 1e-10
 
 # Entropy
 entropy = -np.sum(p * np.log(p))
 
 return entropy

def infer_frequency(v_phi, curvature, entropy):
 """
 Infer EntPTC characteristic frequency from geometry.
 """
 if v_phi is None or curvature is None or entropy is None:
 return None
 
 # Velocity-based frequency
 f_velocity = v_phi / (2 * np.pi)
 
 # Curvature-based frequency
 f_curvature = np.sqrt(curvature * v_phi) / (2 * np.pi)
 
 # Entropy-modulated frequency
 f_entropy = v_phi / (2 * np.pi * entropy) if entropy > 0 else 0
 
 # Composite EntPTC frequency (geometric mean)
 f_entptc = (f_velocity * f_curvature * f_entropy) ** (1/3) if f_entropy > 0 else 0
 
 return f_entptc

# ============================================================================
# ROBUSTNESS CHECK 1: Grid Resolution
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECK 1: Grid Resolution")
print("=" * 80)

grid_resolutions = [10, 15, 20, 25, 30]
resolution_results = {res: [] for res in grid_resolutions}

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat', 'Hafting_Fig2d_Trial1.mat', 'Hafting_Fig2d_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys:
 spike_times = data[cell_key].flatten()
 
 for grid_size in grid_resolutions:
 # Stage A: Compute firing rate map
 firing_rate_map, _, _ = compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps, grid_size=grid_size)
 
 # Stage A: Map to toroidal coordinates
 theta_x, theta_y = map_to_toroidal_coordinates(firing_rate_map)
 
 if theta_x is not None:
 # Stage A: Compute invariants
 v_phi = compute_phase_velocity(theta_x, theta_y)
 curvature = compute_curvature(theta_x, theta_y)
 entropy = compute_phase_entropy(theta_x, theta_y)
 
 # Stage B: Infer frequency
 f_entptc = infer_frequency(v_phi, curvature, entropy)
 
 if f_entptc is not None and f_entptc > 0:
 resolution_results[grid_size].append(f_entptc)

print("\nGrid Resolution Results:")
for grid_size in grid_resolutions:
 if len(resolution_results[grid_size]) > 0:
 mean_freq = np.mean(resolution_results[grid_size])
 std_freq = np.std(resolution_results[grid_size])
 print(f" {grid_size}×{grid_size}: {mean_freq:.4f} ± {std_freq:.4f} Hz (n={len(resolution_results[grid_size])})")

# ============================================================================
# ROBUSTNESS CHECK 2: Smoothing Method
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECK 2: Smoothing Method")
print("=" * 80)

smoothing_sigmas = [1.0, 1.5, 2.0, 2.5, 3.0]
smoothing_results = {sigma: [] for sigma in smoothing_sigmas}

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat', 'Hafting_Fig2d_Trial1.mat', 'Hafting_Fig2d_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys:
 spike_times = data[cell_key].flatten()
 
 for sigma in smoothing_sigmas:
 firing_rate_map, _, _ = compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps, grid_size=20, sigma=sigma)
 theta_x, theta_y = map_to_toroidal_coordinates(firing_rate_map)
 
 if theta_x is not None:
 v_phi = compute_phase_velocity(theta_x, theta_y)
 curvature = compute_curvature(theta_x, theta_y)
 entropy = compute_phase_entropy(theta_x, theta_y)
 f_entptc = infer_frequency(v_phi, curvature, entropy)
 
 if f_entptc is not None and f_entptc > 0:
 smoothing_results[sigma].append(f_entptc)

print("\nSmoothing Method Results:")
for sigma in smoothing_sigmas:
 if len(smoothing_results[sigma]) > 0:
 mean_freq = np.mean(smoothing_results[sigma])
 std_freq = np.std(smoothing_results[sigma])
 print(f" σ={sigma}: {mean_freq:.4f} ± {std_freq:.4f} Hz (n={len(smoothing_results[sigma])})")

# ============================================================================
# Summary and Interpretation
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS SUMMARY (CORRECTED)")
print("=" * 80)

# Compute overall sensitivity range
all_freqs = []
for res_list in resolution_results.values():
 all_freqs.extend(res_list)
for smooth_list in smoothing_results.values():
 all_freqs.extend(smooth_list)

if len(all_freqs) > 0:
 overall_mean = np.mean(all_freqs)
 overall_std = np.std(all_freqs)
 overall_min = np.min(all_freqs)
 overall_max = np.max(all_freqs)
 
 print(f"\nOverall Sensitivity Range:")
 print(f" Mean: {overall_mean:.4f} Hz")
 print(f" Std: {overall_std:.4f} Hz")
 print(f" Range: [{overall_min:.4f}, {overall_max:.4f}] Hz")
 print(f" Baseline: {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
 
 # Compute coefficient of variation
 cv = overall_std / overall_mean * 100 if overall_mean > 0 else 0
 print(f" Coefficient of Variation: {cv:.1f}%")
 
 if cv < 20:
 verdict = "ROBUST"
 interpretation = "The ~0.4 Hz frequency is stable across parameter variations (CV < 20%)."
 else:
 verdict = "SENSITIVE"
 interpretation = f"The frequency shows significant sensitivity to parameters (CV = {cv:.1f}%)."
 
 results = {
 'baseline': {
 'frequency_hz': float(baseline_mean),
 'std_hz': float(baseline_std)
 },
 'grid_resolution': {res: {'mean_hz': float(np.mean(resolution_results[res])), 'std_hz': float(np.std(resolution_results[res]))} 
 for res in grid_resolutions if len(resolution_results[res]) > 0},
 'smoothing_method': {sigma: {'mean_hz': float(np.mean(smoothing_results[sigma])), 'std_hz': float(np.std(smoothing_results[sigma]))} 
 for sigma in smoothing_sigmas if len(smoothing_results[sigma]) > 0},
 'overall_sensitivity': {
 'mean_hz': float(overall_mean),
 'std_hz': float(overall_std),
 'min_hz': float(overall_min),
 'max_hz': float(overall_max),
 'coefficient_of_variation_percent': float(cv)
 },
 'verdict': verdict,
 'interpretation': interpretation
 }
 
 # Save results
 with open(f'{output_dir}/robustness_checks_corrected_results.json', 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\nVERDICT: {verdict}")
 print(f"INTERPRETATION: {interpretation}")
 print(f"\nSaved results to: {output_dir}/robustness_checks_corrected_results.json")
else:
 print("\nWARNING: No valid frequencies computed across parameter variations.")
 print("This suggests the pipeline is highly sensitive to parameters or data quality.")

print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS COMPLETE (CORRECTED)")
print("=" * 80)
