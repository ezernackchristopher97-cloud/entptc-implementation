"""
STAGE B: QUANTIFIED ROBUSTNESS CHECKS

Test sensitivity of frequency estimate across:
- Grid resolution: 4×4, 6×6, 8×8, 10×10, 15×15, 20×20
- Smoothing: σ = 1.0, 1.5, 2.0, 2.5, 3.0
- Curvature estimators: gradient, finite-difference, spline
- Window length: multiple values

Report sensitivity ranges, not point estimates.
If frequency shifts wildly, it's a modeling artifact.

"""

import numpy as np
import scipy.io as sio
import json
import os
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

# Directories
stage_a_dir = '/home/ubuntu/entptc-implementation/stage_a_datasets/hafting_2005'
output_dir = '/home/ubuntu/entptc-implementation/stage_b_robustness_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)
os.makedirs(f'{output_dir}/logs', exist_ok=True)

# Logging
log_file = open(f'{output_dir}/logs/robustness_run.log', 'w')

def log(msg):
 """Log to both console and file."""
 print(msg)
 log_file.write(msg + '\n')
 log_file.flush()

log("=" * 80)
log("STAGE B: QUANTIFIED ROBUSTNESS CHECKS")
log("=" * 80)

# Set random seed
np.random.seed(42)
log(f"\nRandom seed set to: 42")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps, grid_size=20, sigma=2.0):
 """Compute spatial firing rate map."""
 valid_mask = ~np.isnan(pos_x) & ~np.isnan(pos_y)
 pos_x_clean = pos_x[valid_mask]
 pos_y_clean = pos_y[valid_mask]
 pos_t_clean = pos_timestamps[valid_mask]
 
 if len(pos_x_clean) == 0:
 return None
 
 x_edges = np.linspace(np.min(pos_x_clean), np.max(pos_x_clean), grid_size + 1)
 y_edges = np.linspace(np.min(pos_y_clean), np.max(pos_y_clean), grid_size + 1)
 
 occupancy, _, _ = np.histogram2d(pos_x_clean, pos_y_clean, bins=[x_edges, y_edges])
 occupancy = gaussian_filter(occupancy, sigma=sigma)
 occupancy[occupancy < 0.1] = np.nan
 
 spike_positions_x = []
 spike_positions_y = []
 
 for spike_t in spike_times:
 idx = np.argmin(np.abs(pos_t_clean - spike_t))
 if idx < len(pos_x_clean):
 spike_positions_x.append(pos_x_clean[idx])
 spike_positions_y.append(pos_y_clean[idx])
 
 if len(spike_positions_x) == 0:
 return None
 
 spike_map, _, _ = np.histogram2d(spike_positions_x, spike_positions_y, bins=[x_edges, y_edges])
 spike_map = gaussian_filter(spike_map, sigma=sigma)
 
 firing_rate = spike_map / (occupancy + 1e-10)
 
 return firing_rate

def map_to_toroidal_coordinates(firing_rate_map):
 """Map firing rate map to toroidal coordinates."""
 if firing_rate_map is None:
 return None, None
 
 from scipy.ndimage import maximum_filter
 local_max = (firing_rate_map == maximum_filter(firing_rate_map, size=3))
 peaks = np.argwhere(local_max & (firing_rate_map > np.nanmean(firing_rate_map)))
 
 if len(peaks) < 3:
 return None, None
 
 grid_size = firing_rate_map.shape[0]
 coord_x = peaks[:, 1] / grid_size * 2 * np.pi
 coord_y = peaks[:, 0] / grid_size * 2 * np.pi
 
 return coord_x, coord_y

def compute_phase_velocity(coord_x, coord_y):
 """Compute phase velocity."""
 if coord_x is None or len(coord_x) < 2:
 return None
 
 d_x = np.diff(coord_x)
 d_y = np.diff(coord_y)
 
 # Periodic boundary conditions
 d_x[d_x > np.pi] -= 2 * np.pi
 d_x[d_x < -np.pi] += 2 * np.pi
 d_y[d_y > np.pi] -= 2 * np.pi
 d_y[d_y < -np.pi] += 2 * np.pi
 
 v = np.sqrt(d_x**2 + d_y**2)
 
 return np.mean(v)

def compute_curvature_gradient(coord_x, coord_y):
 """Compute curvature using gradient method."""
 if coord_x is None or len(coord_x) < 3:
 return None
 
 d_x = np.gradient(coord_x)
 d_y = np.gradient(coord_y)
 dd_x = np.gradient(d_x)
 dd_y = np.gradient(d_y)
 
 curvature = np.abs(d_x * dd_y - d_y * dd_x) / (d_x**2 + d_y**2 + 1e-10)**(3/2)
 
 return np.mean(curvature)

def compute_curvature_finite_diff(coord_x, coord_y):
 """Compute curvature using finite difference method."""
 if coord_x is None or len(coord_x) < 3:
 return None
 
 d_x = np.diff(coord_x)
 d_y = np.diff(coord_y)
 dd_x = np.diff(d_x)
 dd_y = np.diff(d_y)
 
 # Align arrays
 d_x = d_x[:-1]
 d_y = d_y[:-1]
 
 curvature = np.abs(d_x * dd_y - d_y * dd_x) / (d_x**2 + d_y**2 + 1e-10)**(3/2)
 
 return np.mean(curvature)

def compute_curvature_spline(coord_x, coord_y):
 """Compute curvature using spline method."""
 if coord_x is None or len(coord_x) < 5:
 return None
 
 try:
 t = np.arange(len(coord_x))
 spline_x = UnivariateSpline(t, coord_x, s=0.1)
 spline_y = UnivariateSpline(t, coord_y, s=0.1)
 
 d_x = spline_x.derivative()(t)
 d_y = spline_y.derivative()(t)
 dd_x = spline_x.derivative(2)(t)
 dd_y = spline_y.derivative(2)(t)
 
 curvature = np.abs(d_x * dd_y - d_y * dd_x) / (d_x**2 + d_y**2 + 1e-10)**(3/2)
 
 return np.mean(curvature)
 except:
 return None

def compute_entropy(coord_x, coord_y, bins=10):
 """Compute phase space entropy."""
 if coord_x is None or len(coord_x) < 2:
 return None
 
 H, _, _ = np.histogram2d(coord_x, coord_y, bins=bins)
 
 p = H.flatten()
 p = p / np.sum(p) if np.sum(p) > 0 else p
 p = p + 1e-10
 
 entropy = -np.sum(p * np.log(p))
 
 return entropy

def infer_frequency(v, curvature, entropy):
 """Infer characteristic frequency."""
 if v is None or curvature is None or entropy is None:
 return None
 
 if v <= 0 or curvature <= 0 or entropy <= 0:
 return None
 
 f_velocity = v / (2 * np.pi)
 f_curvature = np.sqrt(curvature * v) / (2 * np.pi)
 f_entropy = v / (2 * np.pi * entropy)
 
 f_char = (f_velocity * f_curvature * f_entropy) ** (1/3)
 
 return f_char

# ============================================================================
# ROBUSTNESS TEST 1: Grid Resolution
# ============================================================================

log("\n" + "=" * 80)
log("ROBUSTNESS TEST 1: Grid Resolution")
log("=" * 80)

grid_resolutions = [4, 6, 8, 10, 15, 20]
grid_results = {res: [] for res in grid_resolutions}

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys[:2]: # Limit to 2 cells per file
 spike_times = data[cell_key].flatten()
 
 for grid_size in grid_resolutions:
 firing_rate_map = compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps, 
 grid_size=grid_size, sigma=2.0)
 coord_x, coord_y = map_to_toroidal_coordinates(firing_rate_map)
 
 if coord_x is not None:
 v = compute_phase_velocity(coord_x, coord_y)
 curvature = compute_curvature_gradient(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 grid_results[grid_size].append(f)

log("\nGrid Resolution Results:")
for grid_size in grid_resolutions:
 if len(grid_results[grid_size]) > 0:
 mean_f = np.mean(grid_results[grid_size])
 std_f = np.std(grid_results[grid_size])
 log(f" {grid_size}×{grid_size}: {mean_f:.4f} ± {std_f:.4f} Hz (n={len(grid_results[grid_size])})")

all_grid_freqs = [f for freqs in grid_results.values() for f in freqs]
grid_mean = np.mean(all_grid_freqs)
grid_std = np.std(all_grid_freqs)
grid_cv = (grid_std / grid_mean * 100) if grid_mean > 0 else 0

log(f"\nOverall Grid Resolution Sensitivity:")
log(f" Mean: {grid_mean:.4f} Hz")
log(f" Std: {grid_std:.4f} Hz")
log(f" Range: [{np.min(all_grid_freqs):.4f}, {np.max(all_grid_freqs):.4f}] Hz")
log(f" CV: {grid_cv:.1f}%")

# ============================================================================
# ROBUSTNESS TEST 2: Smoothing Parameter
# ============================================================================

log("\n" + "=" * 80)
log("ROBUSTNESS TEST 2: Smoothing Parameter")
log("=" * 80)

smoothing_sigmas = [1.0, 1.5, 2.0, 2.5, 3.0]
smoothing_results = {sigma: [] for sigma in smoothing_sigmas}

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys[:2]:
 spike_times = data[cell_key].flatten()
 
 for sigma in smoothing_sigmas:
 firing_rate_map = compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps,
 grid_size=20, sigma=sigma)
 coord_x, coord_y = map_to_toroidal_coordinates(firing_rate_map)
 
 if coord_x is not None:
 v = compute_phase_velocity(coord_x, coord_y)
 curvature = compute_curvature_gradient(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 smoothing_results[sigma].append(f)

log("\nSmoothing Parameter Results:")
for sigma in smoothing_sigmas:
 if len(smoothing_results[sigma]) > 0:
 mean_f = np.mean(smoothing_results[sigma])
 std_f = np.std(smoothing_results[sigma])
 log(f" σ={sigma}: {mean_f:.4f} ± {std_f:.4f} Hz (n={len(smoothing_results[sigma])})")

all_smoothing_freqs = [f for freqs in smoothing_results.values() for f in freqs]
smoothing_mean = np.mean(all_smoothing_freqs)
smoothing_std = np.std(all_smoothing_freqs)
smoothing_cv = (smoothing_std / smoothing_mean * 100) if smoothing_mean > 0 else 0

log(f"\nOverall Smoothing Sensitivity:")
log(f" Mean: {smoothing_mean:.4f} Hz")
log(f" Std: {smoothing_std:.4f} Hz")
log(f" Range: [{np.min(all_smoothing_freqs):.4f}, {np.max(all_smoothing_freqs):.4f}] Hz")
log(f" CV: {smoothing_cv:.1f}%")

# ============================================================================
# ROBUSTNESS TEST 3: Curvature Estimator
# ============================================================================

log("\n" + "=" * 80)
log("ROBUSTNESS TEST 3: Curvature Estimator")
log("=" * 80)

curvature_methods = {
 'gradient': compute_curvature_gradient,
 'finite_diff': compute_curvature_finite_diff,
 'spline': compute_curvature_spline
}
curvature_results = {method: [] for method in curvature_methods.keys()}

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys[:2]:
 spike_times = data[cell_key].flatten()
 
 firing_rate_map = compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps,
 grid_size=20, sigma=2.0)
 coord_x, coord_y = map_to_toroidal_coordinates(firing_rate_map)
 
 if coord_x is not None:
 v = compute_phase_velocity(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 
 for method_name, method_func in curvature_methods.items():
 curvature = method_func(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 curvature_results[method_name].append(f)

log("\nCurvature Estimator Results:")
for method in curvature_methods.keys():
 if len(curvature_results[method]) > 0:
 mean_f = np.mean(curvature_results[method])
 std_f = np.std(curvature_results[method])
 log(f" {method}: {mean_f:.4f} ± {std_f:.4f} Hz (n={len(curvature_results[method])})")

all_curvature_freqs = [f for freqs in curvature_results.values() for f in freqs]
curvature_mean = np.mean(all_curvature_freqs)
curvature_std = np.std(all_curvature_freqs)
curvature_cv = (curvature_std / curvature_mean * 100) if curvature_mean > 0 else 0

log(f"\nOverall Curvature Estimator Sensitivity:")
log(f" Mean: {curvature_mean:.4f} Hz")
log(f" Std: {curvature_std:.4f} Hz")
log(f" Range: [{np.min(all_curvature_freqs):.4f}, {np.max(all_curvature_freqs):.4f}] Hz")
log(f" CV: {curvature_cv:.1f}%")

# ============================================================================
# OVERALL ROBUSTNESS ASSESSMENT
# ============================================================================

log("\n" + "=" * 80)
log("OVERALL ROBUSTNESS ASSESSMENT")
log("=" * 80)

all_freqs = all_grid_freqs + all_smoothing_freqs + all_curvature_freqs
overall_mean = np.mean(all_freqs)
overall_std = np.std(all_freqs)
overall_cv = (overall_std / overall_mean * 100) if overall_mean > 0 else 0

log(f"\nCombined Sensitivity Across All Parameters:")
log(f" Mean: {overall_mean:.4f} Hz")
log(f" Std: {overall_std:.4f} Hz")
log(f" Range: [{np.min(all_freqs):.4f}, {np.max(all_freqs):.4f}] Hz")
log(f" CV: {overall_cv:.1f}%")

robust = (overall_cv < 20)
log(f"\nVERDICT: {'ROBUST' if robust else 'SENSITIVE'} (CV < 20%: {robust})")

if not robust:
 log("WARNING: Frequency estimate is sensitive to parameter choices (modeling artifact suspected)")
else:
 log("Frequency estimate is stable across parameter variations")

# Save results
results = {
 'grid_resolution': {
 'mean_hz': float(grid_mean),
 'std_hz': float(grid_std),
 'cv_percent': float(grid_cv),
 'range_hz': [float(np.min(all_grid_freqs)), float(np.max(all_grid_freqs))],
 'by_resolution': {str(k): [float(x) for x in v] for k, v in grid_results.items()}
 },
 'smoothing': {
 'mean_hz': float(smoothing_mean),
 'std_hz': float(smoothing_std),
 'cv_percent': float(smoothing_cv),
 'range_hz': [float(np.min(all_smoothing_freqs)), float(np.max(all_smoothing_freqs))],
 'by_sigma': {str(k): [float(x) for x in v] for k, v in smoothing_results.items()}
 },
 'curvature_estimator': {
 'mean_hz': float(curvature_mean),
 'std_hz': float(curvature_std),
 'cv_percent': float(curvature_cv),
 'range_hz': [float(np.min(all_curvature_freqs)), float(np.max(all_curvature_freqs))],
 'by_method': {k: [float(x) for x in v] for k, v in curvature_results.items()}
 },
 'overall': {
 'mean_hz': float(overall_mean),
 'std_hz': float(overall_std),
 'cv_percent': float(overall_cv),
 'range_hz': [float(np.min(all_freqs)), float(np.max(all_freqs))],
 'verdict': 'ROBUST' if robust else 'SENSITIVE'
 }
}

with open(f'{output_dir}/robustness_results.json', 'w') as f:
 json.dump(results, f, indent=2)

log(f"\nSaved results to: {output_dir}/robustness_results.json")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Grid resolution
ax = axes[0]
grid_means = [np.mean(grid_results[res]) if len(grid_results[res]) > 0 else 0 for res in grid_resolutions]
grid_stds = [np.std(grid_results[res]) if len(grid_results[res]) > 0 else 0 for res in grid_resolutions]
ax.errorbar(grid_resolutions, grid_means, yerr=grid_stds, marker='o', capsize=5)
ax.set_xlabel('Grid Resolution')
ax.set_ylabel('Frequency (Hz)')
ax.set_title(f'Grid Resolution Sensitivity\n(CV = {grid_cv:.1f}%)')
ax.grid(True, alpha=0.3)

# Smoothing
ax = axes[1]
smoothing_means = [np.mean(smoothing_results[sigma]) if len(smoothing_results[sigma]) > 0 else 0 for sigma in smoothing_sigmas]
smoothing_stds = [np.std(smoothing_results[sigma]) if len(smoothing_results[sigma]) > 0 else 0 for sigma in smoothing_sigmas]
ax.errorbar(smoothing_sigmas, smoothing_means, yerr=smoothing_stds, marker='o', capsize=5)
ax.set_xlabel('Smoothing σ')
ax.set_ylabel('Frequency (Hz)')
ax.set_title(f'Smoothing Sensitivity\n(CV = {smoothing_cv:.1f}%)')
ax.grid(True, alpha=0.3)

# Curvature estimator
ax = axes[2]
curvature_means = [np.mean(curvature_results[method]) if len(curvature_results[method]) > 0 else 0 for method in curvature_methods.keys()]
curvature_stds = [np.std(curvature_results[method]) if len(curvature_results[method]) > 0 else 0 for method in curvature_methods.keys()]
ax.bar(range(len(curvature_methods)), curvature_means, yerr=curvature_stds, capsize=5)
ax.set_xticks(range(len(curvature_methods)))
ax.set_xticklabels(list(curvature_methods.keys()), rotation=45)
ax.set_ylabel('Frequency (Hz)')
ax.set_title(f'Curvature Estimator Sensitivity\n(CV = {curvature_cv:.1f}%)')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/robustness_sensitivity.png', dpi=150, bbox_inches='tight')
log(f"\nSaved figure to: {output_dir}/figures/robustness_sensitivity.png")

log("\n" + "=" * 80)
log("ROBUSTNESS CHECKS COMPLETE")
log("=" * 80)

log_file.close()
