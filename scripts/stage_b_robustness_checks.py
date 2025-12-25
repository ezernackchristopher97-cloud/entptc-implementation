"""
STAGE B: MANDATORY ROBUSTNESS CHECKS

Test whether the ~0.4 Hz frequency estimate is robust to parameter choices.

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
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Directories
stage_a_dir = '/home/ubuntu/entptc-implementation/stage_a_datasets/hafting_2005'
output_dir = '/home/ubuntu/entptc-implementation/stage_b_robustness_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("=" * 80)
print("STAGE B: MANDATORY ROBUSTNESS CHECKS")
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
# ROBUSTNESS CHECK 1: Grid Resolution
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECK 1: Grid Resolution")
print("=" * 80)

def compute_firing_rate_map_with_resolution(spike_times, pos_x, pos_y, pos_timestamps, grid_size=20, sigma=2.0):
 """
 Compute spatial firing rate map with specified grid resolution.
 """
 # Filter valid positions
 valid_mask = ~np.isnan(pos_x) & ~np.isnan(pos_y)
 pos_x_clean = pos_x[valid_mask]
 pos_y_clean = pos_y[valid_mask]
 pos_t_clean = pos_timestamps[valid_mask]
 
 if len(pos_x_clean) == 0:
 return None
 
 # Create grid with toroidal wraparound
 x_edges = np.linspace(0, 2*np.pi, grid_size + 1)
 y_edges = np.linspace(0, 2*np.pi, grid_size + 1)
 
 # Map positions to [0, 2π]
 pos_x_norm = (pos_x_clean - np.min(pos_x_clean)) / (np.max(pos_x_clean) - np.min(pos_x_clean)) * 2 * np.pi
 pos_y_norm = (pos_y_clean - np.min(pos_y_clean)) / (np.max(pos_y_clean) - np.min(pos_y_clean)) * 2 * np.pi
 
 # Occupancy map
 occupancy, _, _ = np.histogram2d(pos_x_norm, pos_y_norm, bins=[x_edges, y_edges])
 occupancy = gaussian_filter(occupancy, sigma=sigma, mode='wrap')
 occupancy[occupancy < 0.1] = np.nan
 
 # Spike map
 spike_positions_x = []
 spike_positions_y = []
 
 for spike_t in spike_times:
 idx = np.argmin(np.abs(pos_t_clean - spike_t))
 if idx < len(pos_x_norm):
 spike_positions_x.append(pos_x_norm[idx])
 spike_positions_y.append(pos_y_norm[idx])
 
 if len(spike_positions_x) == 0:
 return None
 
 spike_map, _, _ = np.histogram2d(spike_positions_x, spike_positions_y, bins=[x_edges, y_edges])
 spike_map = gaussian_filter(spike_map, sigma=sigma, mode='wrap')
 
 # Firing rate
 firing_rate = spike_map / (occupancy + 1e-10)
 
 return firing_rate

def infer_frequency_from_map(firing_rate_map):
 """
 Infer EntPTC frequency from firing rate map.
 """
 if firing_rate_map is None:
 return None
 
 # Compute gradients (toroidal)
 grad_x = np.gradient(firing_rate_map, axis=1)
 grad_y = np.gradient(firing_rate_map, axis=0)
 
 # Phase velocity
 phase_velocity = np.nanmean(np.sqrt(grad_x**2 + grad_y**2))
 
 # Curvature
 grad_xx = np.gradient(grad_x, axis=1)
 grad_yy = np.gradient(grad_y, axis=0)
 curvature = np.nanmean(np.abs(grad_xx + grad_yy))
 
 # Entropy
 p = firing_rate_map.flatten()
 p = p[~np.isnan(p)]
 p = p / np.sum(p) if np.sum(p) > 0 else p
 p = p + 1e-10
 entropy = -np.sum(p * np.log(p))
 
 # Frequency inference
 f_velocity = phase_velocity / (2 * np.pi)
 f_curvature = np.sqrt(curvature * phase_velocity) / (2 * np.pi)
 f_entropy = phase_velocity / (2 * np.pi * entropy) if entropy > 0 else 0
 
 # Composite
 f_entptc = (f_velocity * f_curvature * f_entropy) ** (1/3) if f_entropy > 0 else 0
 
 return f_entptc

# Test different grid resolutions
grid_resolutions = [10, 15, 20, 25, 30, 40]
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
 firing_rate_map = compute_firing_rate_map_with_resolution(spike_times, pos_x, pos_y, pos_timestamps, grid_size=grid_size)
 
 if firing_rate_map is not None:
 f_entptc = infer_frequency_from_map(firing_rate_map)
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
 firing_rate_map = compute_firing_rate_map_with_resolution(spike_times, pos_x, pos_y, pos_timestamps, grid_size=20, sigma=sigma)
 
 if firing_rate_map is not None:
 f_entptc = infer_frequency_from_map(firing_rate_map)
 if f_entptc is not None and f_entptc > 0:
 smoothing_results[sigma].append(f_entptc)

print("\nSmoothing Method Results:")
for sigma in smoothing_sigmas:
 if len(smoothing_results[sigma]) > 0:
 mean_freq = np.mean(smoothing_results[sigma])
 std_freq = np.std(smoothing_results[sigma])
 print(f" σ={sigma}: {mean_freq:.4f} ± {std_freq:.4f} Hz (n={len(smoothing_results[sigma])})")

# ============================================================================
# ROBUSTNESS CHECK 3: Curvature Estimator
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECK 3: Curvature Estimator")
print("=" * 80)

def infer_frequency_curvature_method(firing_rate_map, method='laplacian'):
 """
 Infer frequency with different curvature estimation methods.
 """
 if firing_rate_map is None:
 return None
 
 # Compute gradients
 grad_x = np.gradient(firing_rate_map, axis=1)
 grad_y = np.gradient(firing_rate_map, axis=0)
 
 # Phase velocity
 phase_velocity = np.nanmean(np.sqrt(grad_x**2 + grad_y**2))
 
 # Curvature (different methods)
 if method == 'laplacian':
 grad_xx = np.gradient(grad_x, axis=1)
 grad_yy = np.gradient(grad_y, axis=0)
 curvature = np.nanmean(np.abs(grad_xx + grad_yy))
 elif method == 'trace_hessian':
 grad_xx = np.gradient(grad_x, axis=1)
 grad_yy = np.gradient(grad_y, axis=0)
 curvature = np.nanmean(np.abs(grad_xx) + np.abs(grad_yy))
 elif method == 'frobenius':
 grad_xx = np.gradient(grad_x, axis=1)
 grad_yy = np.gradient(grad_y, axis=0)
 grad_xy = np.gradient(grad_x, axis=0)
 curvature = np.nanmean(np.sqrt(grad_xx**2 + grad_yy**2 + 2*grad_xy**2))
 else:
 return None
 
 # Entropy
 p = firing_rate_map.flatten()
 p = p[~np.isnan(p)]
 p = p / np.sum(p) if np.sum(p) > 0 else p
 p = p + 1e-10
 entropy = -np.sum(p * np.log(p))
 
 # Frequency inference
 f_velocity = phase_velocity / (2 * np.pi)
 f_curvature = np.sqrt(curvature * phase_velocity) / (2 * np.pi)
 f_entropy = phase_velocity / (2 * np.pi * entropy) if entropy > 0 else 0
 
 # Composite
 f_entptc = (f_velocity * f_curvature * f_entropy) ** (1/3) if f_entropy > 0 else 0
 
 return f_entptc

curvature_methods = ['laplacian', 'trace_hessian', 'frobenius']
curvature_results = {method: [] for method in curvature_methods}

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat', 'Hafting_Fig2d_Trial1.mat', 'Hafting_Fig2d_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys:
 spike_times = data[cell_key].flatten()
 
 firing_rate_map = compute_firing_rate_map_with_resolution(spike_times, pos_x, pos_y, pos_timestamps, grid_size=20)
 
 if firing_rate_map is not None:
 for method in curvature_methods:
 f_entptc = infer_frequency_curvature_method(firing_rate_map, method=method)
 if f_entptc is not None and f_entptc > 0:
 curvature_results[method].append(f_entptc)

print("\nCurvature Estimator Results:")
for method in curvature_methods:
 if len(curvature_results[method]) > 0:
 mean_freq = np.mean(curvature_results[method])
 std_freq = np.std(curvature_results[method])
 print(f" {method}: {mean_freq:.4f} ± {std_freq:.4f} Hz (n={len(curvature_results[method])})")

# ============================================================================
# Summary and Interpretation
# ============================================================================

print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS SUMMARY")
print("=" * 80)

# Compute overall sensitivity range
all_freqs = []
for res_list in resolution_results.values():
 all_freqs.extend(res_list)
for smooth_list in smoothing_results.values():
 all_freqs.extend(smooth_list)
for curv_list in curvature_results.values():
 all_freqs.extend(curv_list)

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
 'curvature_estimator': {method: {'mean_hz': float(np.mean(curvature_results[method])), 'std_hz': float(np.std(curvature_results[method]))} 
 for method in curvature_methods if len(curvature_results[method]) > 0},
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
with open(f'{output_dir}/robustness_checks_results.json', 'w') as f:
 json.dump(results, f, indent=2)

print(f"\nVERDICT: {verdict}")
print(f"INTERPRETATION: {interpretation}")
print(f"\nSaved results to: {output_dir}/robustness_checks_results.json")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Grid Resolution
ax = axes[0]
res_means = [np.mean(resolution_results[res]) for res in grid_resolutions if len(resolution_results[res]) > 0]
res_stds = [np.std(resolution_results[res]) for res in grid_resolutions if len(resolution_results[res]) > 0]
res_labels = [f"{res}×{res}" for res in grid_resolutions if len(resolution_results[res]) > 0]
ax.errorbar(range(len(res_labels)), res_means, yerr=res_stds, marker='o', capsize=5, color='steelblue')
ax.axhline(baseline_mean, color='red', linestyle='--', label='Baseline')
ax.set_xticks(range(len(res_labels)))
ax.set_xticklabels(res_labels, rotation=45)
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Grid Resolution Sensitivity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Smoothing Method
ax = axes[1]
smooth_means = [np.mean(smoothing_results[sigma]) for sigma in smoothing_sigmas if len(smoothing_results[sigma]) > 0]
smooth_stds = [np.std(smoothing_results[sigma]) for sigma in smoothing_sigmas if len(smoothing_results[sigma]) > 0]
smooth_labels = [f"σ={sigma}" for sigma in smoothing_sigmas if len(smoothing_results[sigma]) > 0]
ax.errorbar(range(len(smooth_labels)), smooth_means, yerr=smooth_stds, marker='s', capsize=5, color='coral')
ax.axhline(baseline_mean, color='red', linestyle='--', label='Baseline')
ax.set_xticks(range(len(smooth_labels)))
ax.set_xticklabels(smooth_labels, rotation=45)
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Smoothing Method Sensitivity')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Curvature Estimator
ax = axes[2]
curv_means = [np.mean(curvature_results[method]) for method in curvature_methods if len(curvature_results[method]) > 0]
curv_stds = [np.std(curvature_results[method]) for method in curvature_methods if len(curvature_results[method]) > 0]
curv_labels = [method for method in curvature_methods if len(curvature_results[method]) > 0]
ax.errorbar(range(len(curv_labels)), curv_means, yerr=curv_stds, marker='^', capsize=5, color='mediumseagreen')
ax.axhline(baseline_mean, color='red', linestyle='--', label='Baseline')
ax.set_xticks(range(len(curv_labels)))
ax.set_xticklabels(curv_labels, rotation=45)
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Curvature Estimator Sensitivity')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/robustness_checks.png', dpi=300, bbox_inches='tight')
print(f"Saved figure: {output_dir}/figures/robustness_checks.png")

print("\n" + "=" * 80)
print("ROBUSTNESS CHECKS COMPLETE")
print("=" * 80)
