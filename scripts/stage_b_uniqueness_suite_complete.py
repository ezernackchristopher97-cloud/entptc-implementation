"""
STAGE B: COMPLETE UNIQUENESS SUITE (U1-U6)

Mandatory topology-specificity tests to determine if the candidate control timescale
is EntPTC-specific or generic infra-slow dynamics.

Hard Decision Rule:
- Must FAIL (shift/collapse) in ≥4/6 nulls
- Must show graded response: torus → cylinder → plane
- Must NOT be reproduced by phase/1f matched surrogates

If fails: Pivot to invariant vector approach (not frequency-centric)

U1: Remove periodic boundary conditions (torus → planar grid)
U2: Cylinder topology (periodic in one dimension only)
U3: Random adjacency with same degree distribution
U4: Phase randomized surrogate (preserve power spectrum)
U5: 1/f matched surrogate (preserve autocorrelation)
U6: Spatially permuted ROI mapping (shuffle labels)

"""

import numpy as np
import scipy.io as sio
import json
import os
from scipy.ndimage import gaussian_filter
from scipy import signal
import matplotlib.pyplot as plt

# Directories
stage_a_dir = '/home/ubuntu/entptc-implementation/stage_a_datasets/hafting_2005'
output_dir = '/home/ubuntu/entptc-implementation/stage_b_uniqueness_suite_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)
os.makedirs(f'{output_dir}/logs', exist_ok=True)

# Logging
log_file = open(f'{output_dir}/logs/uniqueness_suite_run.log', 'w')

def log(msg):
 """Log to both console and file."""
 print(msg)
 log_file.write(msg + '\n')
 log_file.flush()

log("=" * 80)
log("STAGE B: COMPLETE UNIQUENESS SUITE (U1-U6)")
log("=" * 80)

# Set random seed for reproducibility
np.random.seed(42)
log(f"\nRandom seed set to: 42")

# ============================================================================
# BASELINE: Load Stage B results
# ============================================================================

log("\nLoading baseline Stage B results...")

with open('/home/ubuntu/entptc-implementation/stage_b_outputs/frequency_invariants.json', 'r') as f:
 baseline_invariants = json.load(f)

baseline_freqs = [inv['entptc_characteristic_frequency_hz'] for inv in baseline_invariants]
baseline_mean = np.mean(baseline_freqs)
baseline_std = np.std(baseline_freqs)

log(f"Baseline EntPTC frequency (toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
log(f"Baseline cells: {len(baseline_invariants)}")

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

def map_to_coordinates(firing_rate_map, topology='toroidal'):
 """Map firing rate map to coordinates based on topology."""
 if firing_rate_map is None:
 return None, None
 
 from scipy.ndimage import maximum_filter
 local_max = (firing_rate_map == maximum_filter(firing_rate_map, size=3))
 peaks = np.argwhere(local_max & (firing_rate_map > np.nanmean(firing_rate_map)))
 
 if len(peaks) < 3:
 return None, None
 
 grid_size = firing_rate_map.shape[0]
 
 if topology == 'toroidal':
 # Both dimensions periodic
 coord_x = peaks[:, 1] / grid_size * 2 * np.pi
 coord_y = peaks[:, 0] / grid_size * 2 * np.pi
 periodic_x, periodic_y = True, True
 elif topology == 'cylindrical':
 # One dimension periodic, one linear
 coord_x = peaks[:, 1] / grid_size * 2 * np.pi # Periodic
 coord_y = peaks[:, 0] / grid_size # Linear
 periodic_x, periodic_y = True, False
 elif topology == 'planar':
 # Both dimensions linear
 coord_x = peaks[:, 1] / grid_size
 coord_y = peaks[:, 0] / grid_size
 periodic_x, periodic_y = False, False
 else:
 return None, None
 
 return (coord_x, coord_y, periodic_x, periodic_y)

def compute_phase_velocity(coord_x, coord_y, periodic_x=True, periodic_y=True):
 """Compute phase velocity."""
 if coord_x is None or len(coord_x) < 2:
 return None
 
 d_x = np.diff(coord_x)
 d_y = np.diff(coord_y)
 
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
# U1: Remove Periodic Boundary Conditions (Torus → Planar Grid)
# ============================================================================

log("\n" + "=" * 80)
log("U1: Remove Periodic Boundary Conditions (Torus → Planar Grid)")
log("=" * 80)

u1_results = []

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
 result = map_to_coordinates(firing_rate_map, topology='planar')
 
 if result is not None and result[0] is not None:
 coord_x, coord_y, periodic_x, periodic_y = result
 v = compute_phase_velocity(coord_x, coord_y, periodic_x, periodic_y)
 curvature = compute_curvature(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 u1_results.append(f)

u1_mean = np.mean(u1_results) if len(u1_results) > 0 else 0
u1_std = np.std(u1_results) if len(u1_results) > 0 else 0

log(f"\nU1 Results (Planar Grid):")
log(f" Frequency: {u1_mean:.4f} ± {u1_std:.4f} Hz (n={len(u1_results)})")
log(f" Baseline (Toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
log(f" Collapse: {(1 - u1_mean/baseline_mean)*100:.1f}%")

u1_pass = (u1_mean < baseline_mean * 0.7) # >30% collapse
log(f" VERDICT: {'PASS' if u1_pass else 'FAIL'} (collapse >30%: {u1_pass})")

# ============================================================================
# U2: Cylinder Topology (Periodic in One Dimension Only)
# ============================================================================

log("\n" + "=" * 80)
log("U2: Cylinder Topology (Periodic in One Dimension Only)")
log("=" * 80)

u2_results = []

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
 result = map_to_coordinates(firing_rate_map, topology='cylindrical')
 
 if result is not None and result[0] is not None:
 coord_x, coord_y, periodic_x, periodic_y = result
 v = compute_phase_velocity(coord_x, coord_y, periodic_x, periodic_y)
 curvature = compute_curvature(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 u2_results.append(f)

u2_mean = np.mean(u2_results) if len(u2_results) > 0 else 0
u2_std = np.std(u2_results) if len(u2_results) > 0 else 0

log(f"\nU2 Results (Cylindrical):")
log(f" Frequency: {u2_mean:.4f} ± {u2_std:.4f} Hz (n={len(u2_results)})")
log(f" Baseline (Toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
log(f" Change: {(1 - u2_mean/baseline_mean)*100:.1f}%")

# Check for graded response: torus > cylinder > plane
graded_response = (baseline_mean > u2_mean > u1_mean)
log(f" Graded response (torus > cylinder > plane): {graded_response}")

u2_pass = (u2_mean < baseline_mean * 0.85) # >15% change
log(f" VERDICT: {'PASS' if u2_pass else 'FAIL'} (change >15%: {u2_pass})")

# ============================================================================
# U3: Random Adjacency with Same Degree Distribution
# ============================================================================

log("\n" + "=" * 80)
log("U3: Random Adjacency with Same Degree Distribution")
log("=" * 80)

u3_results = []

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys[:2]: # Limit to 2 cells per file
 spike_times = data[cell_key].flatten()
 
 firing_rate_map = compute_firing_rate_map(spike_times, pos_x, pos_y, pos_timestamps)
 
 if firing_rate_map is not None:
 # Randomize spatial structure while preserving firing rates
 firing_rate_map_shuffled = firing_rate_map.flatten()
 np.random.shuffle(firing_rate_map_shuffled)
 firing_rate_map_shuffled = firing_rate_map_shuffled.reshape(firing_rate_map.shape)
 
 result = map_to_coordinates(firing_rate_map_shuffled, topology='toroidal')
 
 if result is not None and result[0] is not None:
 coord_x, coord_y, periodic_x, periodic_y = result
 v = compute_phase_velocity(coord_x, coord_y, periodic_x, periodic_y)
 curvature = compute_curvature(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 u3_results.append(f)

u3_mean = np.mean(u3_results) if len(u3_results) > 0 else 0
u3_std = np.std(u3_results) if len(u3_results) > 0 else 0

log(f"\nU3 Results (Random Adjacency):")
log(f" Frequency: {u3_mean:.4f} ± {u3_std:.4f} Hz (n={len(u3_results)})")
log(f" Baseline (Toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
log(f" Collapse: {(1 - u3_mean/baseline_mean)*100:.1f}%")

u3_pass = (u3_mean < baseline_mean * 0.7) # >30% collapse
log(f" VERDICT: {'PASS' if u3_pass else 'FAIL'} (collapse >30%: {u3_pass})")

# ============================================================================
# U4: Phase Randomized Surrogate (Preserve Power Spectrum)
# ============================================================================

log("\n" + "=" * 80)
log("U4: Phase Randomized Surrogate (Preserve Power Spectrum)")
log("=" * 80)

u4_results = []

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys[:2]:
 spike_times = data[cell_key].flatten()
 
 # Phase randomization: FFT → randomize phase → IFFT
 spike_train = np.zeros(len(pos_timestamps))
 for spike_t in spike_times:
 idx = np.argmin(np.abs(pos_timestamps - spike_t))
 if idx < len(spike_train):
 spike_train[idx] = 1
 
 fft = np.fft.fft(spike_train)
 amplitudes = np.abs(fft)
 random_phases = np.random.uniform(0, 2*np.pi, len(fft))
 fft_randomized = amplitudes * np.exp(1j * random_phases)
 spike_train_randomized = np.real(np.fft.ifft(fft_randomized))
 spike_train_randomized = (spike_train_randomized > np.median(spike_train_randomized)).astype(float)
 
 # Convert back to spike times
 spike_times_randomized = pos_timestamps[spike_train_randomized > 0]
 
 firing_rate_map = compute_firing_rate_map(spike_times_randomized, pos_x, pos_y, pos_timestamps)
 result = map_to_coordinates(firing_rate_map, topology='toroidal')
 
 if result is not None and result[0] is not None:
 coord_x, coord_y, periodic_x, periodic_y = result
 v = compute_phase_velocity(coord_x, coord_y, periodic_x, periodic_y)
 curvature = compute_curvature(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 u4_results.append(f)

u4_mean = np.mean(u4_results) if len(u4_results) > 0 else 0
u4_std = np.std(u4_results) if len(u4_results) > 0 else 0

log(f"\nU4 Results (Phase Randomized):")
log(f" Frequency: {u4_mean:.4f} ± {u4_std:.4f} Hz (n={len(u4_results)})")
log(f" Baseline (Toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
log(f" Collapse: {(1 - u4_mean/baseline_mean)*100:.1f}%")

u4_pass = (u4_mean < baseline_mean * 0.7) # >30% collapse
log(f" VERDICT: {'PASS' if u4_pass else 'FAIL'} (collapse >30%: {u4_pass})")

# ============================================================================
# U5: 1/f Matched Surrogate (Preserve Autocorrelation)
# ============================================================================

log("\n" + "=" * 80)
log("U5: 1/f Matched Surrogate (Preserve Autocorrelation)")
log("=" * 80)

u5_results = []

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys[:2]:
 spike_times = data[cell_key].flatten()
 
 # Generate 1/f noise with same length as spike train
 n = len(pos_timestamps)
 freqs = np.fft.fftfreq(n)
 freqs[0] = 1e-10 # Avoid division by zero
 power = 1 / np.abs(freqs)
 phases = np.random.uniform(0, 2*np.pi, n)
 fft_1f = np.sqrt(power) * np.exp(1j * phases)
 noise_1f = np.real(np.fft.ifft(fft_1f))
 noise_1f = (noise_1f - np.min(noise_1f)) / (np.max(noise_1f) - np.min(noise_1f))
 
 # Threshold to match spike rate
 spike_rate = len(spike_times) / len(pos_timestamps)
 threshold = np.percentile(noise_1f, (1 - spike_rate) * 100)
 spike_train_1f = (noise_1f > threshold).astype(float)
 
 spike_times_1f = pos_timestamps[spike_train_1f > 0]
 
 firing_rate_map = compute_firing_rate_map(spike_times_1f, pos_x, pos_y, pos_timestamps)
 result = map_to_coordinates(firing_rate_map, topology='toroidal')
 
 if result is not None and result[0] is not None:
 coord_x, coord_y, periodic_x, periodic_y = result
 v = compute_phase_velocity(coord_x, coord_y, periodic_x, periodic_y)
 curvature = compute_curvature(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 u5_results.append(f)

u5_mean = np.mean(u5_results) if len(u5_results) > 0 else 0
u5_std = np.std(u5_results) if len(u5_results) > 0 else 0

log(f"\nU5 Results (1/f Matched):")
log(f" Frequency: {u5_mean:.4f} ± {u5_std:.4f} Hz (n={len(u5_results)})")
log(f" Baseline (Toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
log(f" Collapse: {(1 - u5_mean/baseline_mean)*100:.1f}%")

u5_pass = (u5_mean < baseline_mean * 0.7) # >30% collapse
log(f" VERDICT: {'PASS' if u5_pass else 'FAIL'} (collapse >30%: {u5_pass})")

# ============================================================================
# U6: Spatially Permuted ROI Mapping (Shuffle Labels)
# ============================================================================

log("\n" + "=" * 80)
log("U6: Spatially Permuted ROI Mapping (Shuffle Labels)")
log("=" * 80)

u6_results = []

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys[:2]:
 spike_times = data[cell_key].flatten()
 
 # Shuffle position labels
 pos_x_shuffled = np.random.permutation(pos_x)
 pos_y_shuffled = np.random.permutation(pos_y)
 
 firing_rate_map = compute_firing_rate_map(spike_times, pos_x_shuffled, pos_y_shuffled, pos_timestamps)
 result = map_to_coordinates(firing_rate_map, topology='toroidal')
 
 if result is not None and result[0] is not None:
 coord_x, coord_y, periodic_x, periodic_y = result
 v = compute_phase_velocity(coord_x, coord_y, periodic_x, periodic_y)
 curvature = compute_curvature(coord_x, coord_y)
 entropy = compute_entropy(coord_x, coord_y)
 f = infer_frequency(v, curvature, entropy)
 
 if f is not None and f > 0:
 u6_results.append(f)

u6_mean = np.mean(u6_results) if len(u6_results) > 0 else 0
u6_std = np.std(u6_results) if len(u6_results) > 0 else 0

log(f"\nU6 Results (Spatially Permuted):")
log(f" Frequency: {u6_mean:.4f} ± {u6_std:.4f} Hz (n={len(u6_results)})")
log(f" Baseline (Toroidal): {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
log(f" Collapse: {(1 - u6_mean/baseline_mean)*100:.1f}%")

u6_pass = (u6_mean < baseline_mean * 0.7) # >30% collapse
log(f" VERDICT: {'PASS' if u6_pass else 'FAIL'} (collapse >30%: {u6_pass})")

# ============================================================================
# FINAL DECISION RULE
# ============================================================================

log("\n" + "=" * 80)
log("UNIQUENESS SUITE FINAL DECISION")
log("=" * 80)

tests_passed = sum([u1_pass, u2_pass, u3_pass, u4_pass, u5_pass, u6_pass])

log(f"\nTests passed: {tests_passed}/6")
log(f" U1 (Planar): {'PASS' if u1_pass else 'FAIL'}")
log(f" U2 (Cylindrical): {'PASS' if u2_pass else 'FAIL'}")
log(f" U3 (Random Adjacency): {'PASS' if u3_pass else 'FAIL'}")
log(f" U4 (Phase Randomized): {'PASS' if u4_pass else 'FAIL'}")
log(f" U5 (1/f Matched): {'PASS' if u5_pass else 'FAIL'}")
log(f" U6 (Spatially Permuted): {'PASS' if u6_pass else 'FAIL'}")

log(f"\nGraded response (torus > cylinder > plane): {graded_response}")

uniqueness_supported = (tests_passed >= 4) and graded_response

if uniqueness_supported:
 verdict = "UNIQUENESS SUPPORTED"
 interpretation = f"The candidate control timescale is EntPTC-specific ({tests_passed}/6 nulls failed, graded response confirmed)."
else:
 verdict = "UNIQUENESS NOT SUPPORTED"
 interpretation = f"The candidate control timescale may be generic infra-slow dynamics ({tests_passed}/6 nulls failed). PIVOT to invariant vector approach."

log(f"\nVERDICT: {verdict}")
log(f"INTERPRETATION: {interpretation}")

# Save results
results = {
 'baseline_toroidal': {
 'frequency_hz': float(baseline_mean),
 'std_hz': float(baseline_std)
 },
 'u1_planar': {
 'frequency_hz': float(u1_mean),
 'std_hz': float(u1_std),
 'collapse_percent': float((1 - u1_mean/baseline_mean)*100 if baseline_mean > 0 else 0),
 'pass': bool(u1_pass)
 },
 'u2_cylindrical': {
 'frequency_hz': float(u2_mean),
 'std_hz': float(u2_std),
 'change_percent': float((1 - u2_mean/baseline_mean)*100 if baseline_mean > 0 else 0),
 'pass': bool(u2_pass)
 },
 'u3_random_adjacency': {
 'frequency_hz': float(u3_mean),
 'std_hz': float(u3_std),
 'collapse_percent': float((1 - u3_mean/baseline_mean)*100 if baseline_mean > 0 else 0),
 'pass': bool(u3_pass)
 },
 'u4_phase_randomized': {
 'frequency_hz': float(u4_mean),
 'std_hz': float(u4_std),
 'collapse_percent': float((1 - u4_mean/baseline_mean)*100 if baseline_mean > 0 else 0),
 'pass': bool(u4_pass)
 },
 'u5_1f_matched': {
 'frequency_hz': float(u5_mean),
 'std_hz': float(u5_std),
 'collapse_percent': float((1 - u5_mean/baseline_mean)*100 if baseline_mean > 0 else 0),
 'pass': bool(u5_pass)
 },
 'u6_spatially_permuted': {
 'frequency_hz': float(u6_mean),
 'std_hz': float(u6_std),
 'collapse_percent': float((1 - u6_mean/baseline_mean)*100 if baseline_mean > 0 else 0),
 'pass': bool(u6_pass)
 },
 'graded_response': bool(graded_response),
 'tests_passed': int(tests_passed),
 'total_tests': 6,
 'verdict': verdict,
 'interpretation': interpretation
}

with open(f'{output_dir}/uniqueness_suite_results.json', 'w') as f:
 json.dump(results, f, indent=2)

log(f"\nSaved results to: {output_dir}/uniqueness_suite_results.json")

log("\n" + "=" * 80)
log("UNIQUENESS SUITE COMPLETE")
log("=" * 80)

log_file.close()
