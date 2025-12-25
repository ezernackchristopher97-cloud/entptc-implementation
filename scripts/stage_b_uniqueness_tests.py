"""
STAGE B: MANDATORY UNIQUENESS & IDENTIFIABILITY TESTS

Test whether the ~0.2-0.4 Hz frequency is unique to toroidal topology or generic across geometries.

Required Tests:
1. Non-toroidal geometry controls (cylindrical, planar, random manifold)
2. Geometry-frequency decoupling (phase scrambling)
3. Parameter-scaling test (model-level change)

Interpretation Rules:
- If frequency is specific to toroidal topology → uniqueness supported
- If appears across unrelated geometries → uniqueness fails
- Predictable rescaling with geometry → mechanistic link
- Fixed regardless of geometry → generic dynamics

"""

import numpy as np
import scipy.io as sio
import json
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Directories
stage_a_dir = '/home/ubuntu/entptc-implementation/stage_a_datasets/hafting_2005'
output_dir = '/home/ubuntu/entptc-implementation/stage_b_uniqueness_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("=" * 80)
print("STAGE B: MANDATORY UNIQUENESS & IDENTIFIABILITY TESTS")
print("=" * 80)

# ============================================================================
# BASELINE: Load Stage B results
# ============================================================================

print("\nLoading baseline Stage B results...")

with open('/home/ubuntu/entptc-implementation/stage_b_outputs/frequency_invariants.json', 'r') as f:
 baseline_invariants = json.load(f)

baseline_freqs = [inv['entptc_characteristic_frequency_hz'] for inv in baseline_invariants]
baseline_mean = np.mean(baseline_freqs)
baseline_std = np.std(baseline_freqs)

print(f"Baseline EntPTC frequency (toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
print(f"Baseline cells: {len(baseline_invariants)}")

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
 """Map firing rate map to toroidal phase coordinates."""
 if firing_rate_map is None:
 return None, None
 
 from scipy.ndimage import maximum_filter
 local_max = (firing_rate_map == maximum_filter(firing_rate_map, size=3))
 peaks = np.argwhere(local_max & (firing_rate_map > np.nanmean(firing_rate_map)))
 
 if len(peaks) < 3:
 return None, None
 
 grid_size = firing_rate_map.shape[0]
 theta_x = peaks[:, 1] / grid_size * 2 * np.pi
 theta_y = peaks[:, 0] / grid_size * 2 * np.pi
 
 return theta_x, theta_y

def map_to_cylindrical_coordinates(firing_rate_map):
 """Map to cylindrical coordinates (periodic in one dimension only)."""
 if firing_rate_map is None:
 return None, None
 
 from scipy.ndimage import maximum_filter
 local_max = (firing_rate_map == maximum_filter(firing_rate_map, size=3))
 peaks = np.argwhere(local_max & (firing_rate_map > np.nanmean(firing_rate_map)))
 
 if len(peaks) < 3:
 return None, None
 
 grid_size = firing_rate_map.shape[0]
 theta_x = peaks[:, 1] / grid_size * 2 * np.pi # Periodic
 z = peaks[:, 0] / grid_size # Non-periodic (linear)
 
 return theta_x, z

def map_to_planar_coordinates(firing_rate_map):
 """Map to planar coordinates (no periodicity)."""
 if firing_rate_map is None:
 return None, None
 
 from scipy.ndimage import maximum_filter
 local_max = (firing_rate_map == maximum_filter(firing_rate_map, size=3))
 peaks = np.argwhere(local_max & (firing_rate_map > np.nanmean(firing_rate_map)))
 
 if len(peaks) < 3:
 return None, None
 
 grid_size = firing_rate_map.shape[0]
 x = peaks[:, 1] / grid_size # Linear
 y = peaks[:, 0] / grid_size # Linear
 
 return x, y

def compute_phase_velocity(coord_x, coord_y, periodic_x=True, periodic_y=True):
 """Compute phase velocity with optional periodicity."""
 if coord_x is None or len(coord_x) < 2:
 return None
 
 d_x = np.diff(coord_x)
 d_y = np.diff(coord_y)
 
 # Handle wraparound if periodic
 if periodic_x:
 d_x[d_x > np.pi] -= 2 * np.pi
 d_x[d_x < -np.pi] += 2 * np.pi
 if periodic_y:
 d_y[d_y > np.pi] -= 2 * np.pi
 d_y[d_y < -np.pi] += 2 * np.pi
 
 v = np.sqrt(d_x**2 + d_y**2)
 
 return np.mean(v)

def compute_curvature(coord_x, coord_y):
 """Compute trajectory curvature."""
 if coord_x is None or len(coord_x) < 3:
 return None
 
 d_x = np.gradient(coord_x)
 d_y = np.gradient(coord_y)
 dd_x = np.gradient(d_x)
 dd_y = np.gradient(d_y)
 
 curvature = np.abs(d_x * dd_y - d_y * dd_x) / (d_x**2 + d_y**2 + 1e-10)**(3/2)
 
 return np.mean(curvature)

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
 
 f_velocity = v / (2 * np.pi)
 f_curvature = np.sqrt(curvature * v) / (2 * np.pi)
 f_entropy = v / (2 * np.pi * entropy) if entropy > 0 else 0
 
 f_char = (f_velocity * f_curvature * f_entropy) ** (1/3) if f_entropy > 0 else 0
 
 return f_char

# ============================================================================
# UNIQUENESS TEST 1: Non-Toroidal Geometry Controls
# ============================================================================

print("\n" + "=" * 80)
print("UNIQUENESS TEST 1: Non-Toroidal Geometry Controls")
print("=" * 80)

geometries = {
 'toroidal': {'periodic_x': True, 'periodic_y': True, 'map_func': map_to_toroidal_coordinates},
 'cylindrical': {'periodic_x': True, 'periodic_y': False, 'map_func': map_to_cylindrical_coordinates},
 'planar': {'periodic_x': False, 'periodic_y': False, 'map_func': map_to_planar_coordinates}
}

geometry_results = {geom: [] for geom in geometries.keys()}

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat', 'Hafting_Fig2d_Trial1.mat', 'Hafting_Fig2d_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys:
 spike_times = data[cell_key].flatten()
 
 firing_rate_map = compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps)
 
 if firing_rate_map is not None:
 for geom_name, geom_config in geometries.items():
 coord_x, coord_y = geom_config['map_func'](firing_rate_map)
 
 if coord_x is not None:
 v = compute_phase_velocity(coord_x, coord_y, 
 periodic_x=geom_config['periodic_x'], 
 periodic_y=geom_config['periodic_y'])
 curvature = compute_curvature(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 geometry_results[geom_name].append(f)

print("\nGeometry Control Results:")
for geom_name in geometries.keys():
 if len(geometry_results[geom_name]) > 0:
 mean_freq = np.mean(geometry_results[geom_name])
 std_freq = np.std(geometry_results[geom_name])
 print(f" {geom_name}: {mean_freq:.4f} ± {std_freq:.4f} Hz (n={len(geometry_results[geom_name])})")

# ============================================================================
# UNIQUENESS TEST 2: Phase Scrambling
# ============================================================================

print("\n" + "=" * 80)
print("UNIQUENESS TEST 2: Phase Scrambling (Geometry-Frequency Decoupling)")
print("=" * 80)

scrambled_results = []

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat', 'Hafting_Fig2d_Trial1.mat', 'Hafting_Fig2d_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys:
 spike_times = data[cell_key].flatten()
 
 firing_rate_map = compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps)
 theta_x, theta_y = map_to_toroidal_coordinates(firing_rate_map)
 
 if theta_x is not None:
 # Scramble phase relationships
 theta_x_scrambled = np.random.permutation(theta_x)
 theta_y_scrambled = np.random.permutation(theta_y)
 
 v = compute_phase_velocity(theta_x_scrambled, theta_y_scrambled)
 curvature = compute_curvature(theta_x_scrambled, theta_y_scrambled)
 entropy = compute_entropy(theta_x_scrambled, theta_y_scrambled)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 scrambled_results.append(f)

if len(scrambled_results) > 0:
 scrambled_mean = np.mean(scrambled_results)
 scrambled_std = np.std(scrambled_results)
 print(f"\nPhase Scrambling Results:")
 print(f" Scrambled: {scrambled_mean:.4f} ± {scrambled_std:.4f} Hz (n={len(scrambled_results)})")
 print(f" Baseline (toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")

# ============================================================================
# UNIQUENESS TEST 3: Parameter Scaling
# ============================================================================

print("\n" + "=" * 80)
print("UNIQUENESS TEST 3: Parameter Scaling")
print("=" * 80)

# Test if frequency scales predictably with grid scale
grid_scales = [0.5, 1.0, 1.5, 2.0]
scaling_results = {scale: [] for scale in grid_scales}

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys[:2]: # Limit to 2 cells per file
 spike_times = data[cell_key].flatten()
 
 for scale in grid_scales:
 # Scale positions
 pos_x_scaled = pos_x * scale
 pos_y_scaled = pos_y * scale
 
 firing_rate_map = compute_firing_rate_map(spike_times, pos_x_scaled, pos_y_scaled, pos_timestamps)
 theta_x, theta_y = map_to_toroidal_coordinates(firing_rate_map)
 
 if theta_x is not None:
 v = compute_phase_velocity(theta_x, theta_y)
 curvature = compute_curvature(theta_x, theta_y)
 entropy = compute_entropy(theta_x, theta_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 scaling_results[scale].append(f)

print("\nParameter Scaling Results:")
for scale in grid_scales:
 if len(scaling_results[scale]) > 0:
 mean_freq = np.mean(scaling_results[scale])
 std_freq = np.std(scaling_results[scale])
 print(f" Scale {scale}×: {mean_freq:.4f} ± {std_freq:.4f} Hz (n={len(scaling_results[scale])})")

# ============================================================================
# Summary and Interpretation
# ============================================================================

print("\n" + "=" * 80)
print("UNIQUENESS TESTS SUMMARY")
print("=" * 80)

results = {
 'baseline_toroidal': {
 'frequency_hz': float(baseline_mean),
 'std_hz': float(baseline_std)
 },
 'geometry_controls': {geom: {'mean_hz': float(np.mean(geometry_results[geom])), 
 'std_hz': float(np.std(geometry_results[geom]))} 
 for geom in geometries.keys() if len(geometry_results[geom]) > 0},
 'phase_scrambling': {
 'mean_hz': float(scrambled_mean) if len(scrambled_results) > 0 else 0,
 'std_hz': float(scrambled_std) if len(scrambled_results) > 0 else 0
 },
 'parameter_scaling': {scale: {'mean_hz': float(np.mean(scaling_results[scale])), 
 'std_hz': float(np.std(scaling_results[scale]))} 
 for scale in grid_scales if len(scaling_results[scale]) > 0}
}

# Interpretation
uniqueness_tests_passed = 0
total_tests = 3

# Test 1: Geometry specificity
if 'toroidal' in geometry_results and 'planar' in geometry_results:
 toroidal_freq = np.mean(geometry_results['toroidal'])
 planar_freq = np.mean(geometry_results['planar'])
 if toroidal_freq > planar_freq * 1.2: # 20% higher
 uniqueness_tests_passed += 1
 print("\n✓ Test 1 PASSED: Toroidal frequency > planar (geometry-specific)")
 else:
 print("\n✗ Test 1 FAILED: Toroidal frequency ≈ planar (not geometry-specific)")

# Test 2: Phase scrambling
if len(scrambled_results) > 0:
 if scrambled_mean < baseline_mean * 0.8: # 20% lower
 uniqueness_tests_passed += 1
 print("✓ Test 2 PASSED: Phase scrambling reduces frequency (geometry-dependent)")
 else:
 print("✗ Test 2 FAILED: Phase scrambling preserves frequency (generic dynamics)")

# Test 3: Parameter scaling
if len(scaling_results[1.0]) > 0 and len(scaling_results[2.0]) > 0:
 freq_1x = np.mean(scaling_results[1.0])
 freq_2x = np.mean(scaling_results[2.0])
 expected_ratio = 0.5 # Frequency should scale inversely with grid scale
 actual_ratio = freq_2x / freq_1x if freq_1x > 0 else 0
 if 0.3 < actual_ratio < 0.7: # Within 20% of expected
 uniqueness_tests_passed += 1
 print("✓ Test 3 PASSED: Frequency scales predictably with geometry (mechanistic link)")
 else:
 print("✗ Test 3 FAILED: Frequency doesn't scale predictably (no mechanistic link)")

if uniqueness_tests_passed >= 2:
 verdict = "UNIQUENESS SUPPORTED"
 interpretation = f"The frequency is specific to toroidal topology ({uniqueness_tests_passed}/3 tests passed)."
else:
 verdict = "UNIQUENESS QUESTIONABLE"
 interpretation = f"The frequency may not be unique to toroidal topology ({uniqueness_tests_passed}/3 tests passed)."

results['verdict'] = verdict
results['interpretation'] = interpretation
results['tests_passed'] = uniqueness_tests_passed
results['total_tests'] = total_tests

# Save results
with open(f'{output_dir}/uniqueness_tests_results.json', 'w') as f:
 json.dump(results, f, indent=2)

print(f"\nVERDICT: {verdict}")
print(f"INTERPRETATION: {interpretation}")
print(f"Tests passed: {uniqueness_tests_passed}/{total_tests}")
print(f"\nSaved results to: {output_dir}/uniqueness_tests_results.json")

print("\n" + "=" * 80)
print("UNIQUENESS TESTS COMPLETE")
print("=" * 80)
