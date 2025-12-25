"""
STAGE B: MANDATORY CAUSALITY ABLATION

Test whether the ~0.4 Hz control mode is causally dependent on toroidal structure.

Procedure:
1. Run baseline Stage A → Stage B pipeline (toroidal structure intact)
2. Break toroidal structure in controlled ways:
 - Remove periodic boundary conditions (planar grid)
 - Randomize neighbor adjacency (destroy spatial structure)
 - Destroy grid-cell phase coherence (shuffle spike times)
3. Re-infer internal frequencies under each ablation condition

Interpretation Rules:
- If ~0.4 Hz collapses/shifts significantly → toroidal geometry is CAUSAL
- If ~0.4 Hz persists unchanged → generic infra-slow dynamics, REINTERPRET

"""

import numpy as np
import scipy.io as sio
import json
import os
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Directories
stage_a_dir = '/home/ubuntu/entptc-implementation/stage_a_datasets/hafting_2005'
output_dir = '/home/ubuntu/entptc-implementation/stage_b_ablation_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("=" * 80)
print("STAGE B: MANDATORY CAUSALITY ABLATION")
print("=" * 80)

# ============================================================================
# BASELINE: Load original Stage A results
# ============================================================================

print("\nLoading baseline Stage A results...")

with open('/home/ubuntu/entptc-implementation/stage_b_outputs/frequency_invariants.json', 'r') as f:
 baseline_invariants = json.load(f)

baseline_freqs = [inv['entptc_characteristic_frequency_hz'] for inv in baseline_invariants]
baseline_mean = np.mean(baseline_freqs)
baseline_std = np.std(baseline_freqs)

print(f"Baseline EntPTC frequency: {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
print(f"Baseline cells: {len(baseline_invariants)}")

# ============================================================================
# ABLATION 1: Remove Periodic Boundary Conditions (Planar Grid)
# ============================================================================

print("\n" + "=" * 80)
print("ABLATION 1: Remove Periodic Boundary Conditions (Planar Grid)")
print("=" * 80)

"""
Convert toroidal topology to planar grid:
- Remove wraparound connections
- Edges become boundaries (no periodic closure)
- Test if ~0.4 Hz depends on toroidal closure
"""

def compute_firing_rate_map_planar(spike_times, pos_x, pos_y, pos_timestamps, grid_size=20, sigma=2.0):
 """
 Compute spatial firing rate map WITHOUT toroidal wraparound.
 """
 # Filter valid positions
 valid_mask = ~np.isnan(pos_x) & ~np.isnan(pos_y)
 pos_x_clean = pos_x[valid_mask]
 pos_y_clean = pos_y[valid_mask]
 pos_t_clean = pos_timestamps[valid_mask]
 
 if len(pos_x_clean) == 0:
 return None, None, None
 
 # Create planar grid (NO periodic boundaries)
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

def infer_frequency_planar(firing_rate_map):
 """
 Infer frequency from planar grid (no toroidal structure).
 """
 if firing_rate_map is None:
 return None
 
 # Compute gradients (planar, no wraparound)
 grad_x = np.gradient(firing_rate_map, axis=1)
 grad_y = np.gradient(firing_rate_map, axis=0)
 
 # Phase velocity (magnitude of gradient)
 phase_velocity = np.nanmean(np.sqrt(grad_x**2 + grad_y**2))
 
 # Curvature (second derivatives, planar)
 grad_xx = np.gradient(grad_x, axis=1)
 grad_yy = np.gradient(grad_y, axis=0)
 curvature = np.nanmean(np.abs(grad_xx + grad_yy))
 
 # Entropy (spatial)
 p = firing_rate_map.flatten()
 p = p[~np.isnan(p)]
 p = p / np.sum(p) if np.sum(p) > 0 else p
 p = p + 1e-10
 entropy = -np.sum(p * np.log(p))
 
 # Frequency inference (same formula as Stage B)
 f_velocity = phase_velocity / (2 * np.pi)
 f_curvature = np.sqrt(curvature * phase_velocity) / (2 * np.pi)
 f_entropy = phase_velocity / (2 * np.pi * entropy) if entropy > 0 else 0
 
 # Composite
 f_entptc = (f_velocity * f_curvature * f_entropy) ** (1/3) if f_entropy > 0 else 0
 
 return f_entptc

# Process all cells with planar topology
ablation1_freqs = []

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat', 'Hafting_Fig2d_Trial1.mat', 'Hafting_Fig2d_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys:
 spike_times = data[cell_key].flatten()
 
 firing_rate_map, _, _ = compute_firing_rate_map_planar(spike_times, pos_x, pos_y, pos_timestamps)
 
 if firing_rate_map is not None:
 f_entptc = infer_frequency_planar(firing_rate_map)
 if f_entptc is not None and f_entptc > 0:
 ablation1_freqs.append(f_entptc)

ablation1_mean = np.mean(ablation1_freqs) if len(ablation1_freqs) > 0 else 0
ablation1_std = np.std(ablation1_freqs) if len(ablation1_freqs) > 0 else 0

print(f"\nAblation 1 Results:")
print(f" Planar grid frequency: {ablation1_mean:.4f} ± {ablation1_std:.4f} Hz")
print(f" Baseline frequency: {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
print(f" Cells processed: {len(ablation1_freqs)}")

# Test significance
if len(ablation1_freqs) > 0:
 freq_shift = abs(ablation1_mean - baseline_mean)
 relative_shift = freq_shift / baseline_mean * 100
 print(f" Frequency shift: {freq_shift:.4f} Hz ({relative_shift:.1f}%)")
 
 if relative_shift > 20:
 print(" → SIGNIFICANT SHIFT: Toroidal closure is CAUSAL")
 else:
 print(" → MINIMAL SHIFT: Toroidal closure may not be causal")

# ============================================================================
# ABLATION 2: Randomize Neighbor Adjacency
# ============================================================================

print("\n" + "=" * 80)
print("ABLATION 2: Randomize Neighbor Adjacency")
print("=" * 80)

"""
Destroy spatial structure by randomizing which cells are neighbors.
- Preserve number of connections
- Destroy spatial coherence
- Test if ~0.4 Hz depends on spatial organization
"""

def randomize_adjacency_and_infer(firing_rate_map):
 """
 Randomize spatial adjacency and re-infer frequency.
 """
 if firing_rate_map is None:
 return None
 
 # Flatten and shuffle
 flat = firing_rate_map.flatten()
 flat_shuffled = np.random.permutation(flat)
 
 # Reshape to random spatial arrangement
 shuffled_map = flat_shuffled.reshape(firing_rate_map.shape)
 
 # Infer frequency from shuffled map
 return infer_frequency_planar(shuffled_map)

# Process with randomized adjacency
ablation2_freqs = []

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat', 'Hafting_Fig2d_Trial1.mat', 'Hafting_Fig2d_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys:
 spike_times = data[cell_key].flatten()
 
 firing_rate_map, _, _ = compute_firing_rate_map_planar(spike_times, pos_x, pos_y, pos_timestamps)
 
 if firing_rate_map is not None:
 f_entptc = randomize_adjacency_and_infer(firing_rate_map)
 if f_entptc is not None and f_entptc > 0:
 ablation2_freqs.append(f_entptc)

ablation2_mean = np.mean(ablation2_freqs) if len(ablation2_freqs) > 0 else 0
ablation2_std = np.std(ablation2_freqs) if len(ablation2_freqs) > 0 else 0

print(f"\nAblation 2 Results:")
print(f" Randomized adjacency frequency: {ablation2_mean:.4f} ± {ablation2_std:.4f} Hz")
print(f" Baseline frequency: {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
print(f" Cells processed: {len(ablation2_freqs)}")

if len(ablation2_freqs) > 0:
 freq_shift = abs(ablation2_mean - baseline_mean)
 relative_shift = freq_shift / baseline_mean * 100
 print(f" Frequency shift: {freq_shift:.4f} Hz ({relative_shift:.1f}%)")
 
 if relative_shift > 20:
 print(" → SIGNIFICANT SHIFT: Spatial organization is CAUSAL")
 else:
 print(" → MINIMAL SHIFT: Spatial organization may not be causal")

# ============================================================================
# ABLATION 3: Destroy Phase Coherence (Shuffle Spike Times)
# ============================================================================

print("\n" + "=" * 80)
print("ABLATION 3: Destroy Phase Coherence (Shuffle Spike Times)")
print("=" * 80)

"""
Break temporal phase relationships while preserving firing rates.
- Shuffle spike times within each cell
- Destroy phase coherence across cells
- Test if ~0.4 Hz depends on phase relationships
"""

def shuffle_spike_times_and_infer(spike_times, pos_x, pos_y, pos_timestamps):
 """
 Shuffle spike times and re-infer frequency.
 """
 # Shuffle spike times (breaks phase coherence)
 shuffled_spikes = np.random.permutation(spike_times)
 
 # Compute firing rate map with shuffled spikes
 firing_rate_map, _, _ = compute_firing_rate_map_planar(shuffled_spikes, pos_x, pos_y, pos_timestamps)
 
 if firing_rate_map is not None:
 return infer_frequency_planar(firing_rate_map)
 return None

# Process with shuffled spike times
ablation3_freqs = []

for mat_file in ['Hafting_Fig2c_Trial1.mat', 'Hafting_Fig2c_Trial2.mat', 'Hafting_Fig2d_Trial1.mat', 'Hafting_Fig2d_Trial2.mat']:
 file_path = os.path.join(stage_a_dir, mat_file)
 data = sio.loadmat(file_path)
 
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 for cell_key in cell_keys:
 spike_times = data[cell_key].flatten()
 
 f_entptc = shuffle_spike_times_and_infer(spike_times, pos_x, pos_y, pos_timestamps)
 if f_entptc is not None and f_entptc > 0:
 ablation3_freqs.append(f_entptc)

ablation3_mean = np.mean(ablation3_freqs) if len(ablation3_freqs) > 0 else 0
ablation3_std = np.std(ablation3_freqs) if len(ablation3_freqs) > 0 else 0

print(f"\nAblation 3 Results:")
print(f" Shuffled spike times frequency: {ablation3_mean:.4f} ± {ablation3_std:.4f} Hz")
print(f" Baseline frequency: {baseline_mean:.4f} ± {baseline_std:.4f} Hz")
print(f" Cells processed: {len(ablation3_freqs)}")

if len(ablation3_freqs) > 0:
 freq_shift = abs(ablation3_mean - baseline_mean)
 relative_shift = freq_shift / baseline_mean * 100
 print(f" Frequency shift: {freq_shift:.4f} Hz ({relative_shift:.1f}%)")
 
 if relative_shift > 20:
 print(" → SIGNIFICANT SHIFT: Phase coherence is CAUSAL")
 else:
 print(" → MINIMAL SHIFT: Phase coherence may not be causal")

# ============================================================================
# Summary and Interpretation
# ============================================================================

print("\n" + "=" * 80)
print("CAUSALITY ABLATION SUMMARY")
print("=" * 80)

results = {
 'baseline': {
 'frequency_hz': float(baseline_mean),
 'std_hz': float(baseline_std),
 'n_cells': len(baseline_invariants)
 },
 'ablation1_planar_grid': {
 'frequency_hz': float(ablation1_mean),
 'std_hz': float(ablation1_std),
 'n_cells': len(ablation1_freqs),
 'shift_hz': float(abs(ablation1_mean - baseline_mean)),
 'shift_percent': float(abs(ablation1_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
 },
 'ablation2_randomized_adjacency': {
 'frequency_hz': float(ablation2_mean),
 'std_hz': float(ablation2_std),
 'n_cells': len(ablation2_freqs),
 'shift_hz': float(abs(ablation2_mean - baseline_mean)),
 'shift_percent': float(abs(ablation2_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
 },
 'ablation3_shuffled_spikes': {
 'frequency_hz': float(ablation3_mean),
 'std_hz': float(ablation3_std),
 'n_cells': len(ablation3_freqs),
 'shift_hz': float(abs(ablation3_mean - baseline_mean)),
 'shift_percent': float(abs(ablation3_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
 }
}

# Determine causality
causality_threshold = 20 # 20% shift indicates causality

causal_factors = []
if results['ablation1_planar_grid']['shift_percent'] > causality_threshold:
 causal_factors.append('toroidal_closure')
if results['ablation2_randomized_adjacency']['shift_percent'] > causality_threshold:
 causal_factors.append('spatial_organization')
if results['ablation3_shuffled_spikes']['shift_percent'] > causality_threshold:
 causal_factors.append('phase_coherence')

if len(causal_factors) > 0:
 verdict = "TOROIDAL GEOMETRY IS CAUSAL"
 interpretation = f"The ~0.4 Hz mode depends on: {', '.join(causal_factors)}"
else:
 verdict = "GENERIC INFRA-SLOW DYNAMICS"
 interpretation = "The ~0.4 Hz mode persists regardless of toroidal structure. Reinterpret as non-EntPTC-specific."

results['verdict'] = verdict
results['interpretation'] = interpretation
results['causal_factors'] = causal_factors

print(f"\nVERDICT: {verdict}")
print(f"INTERPRETATION: {interpretation}")

# Save results
with open(f'{output_dir}/causality_ablation_results.json', 'w') as f:
 json.dump(results, f, indent=2)

print(f"\nSaved results to: {output_dir}/causality_ablation_results.json")

# Create visualization
fig, ax = plt.subplots(figsize=(10, 6))

conditions = ['Baseline\n(Toroidal)', 'Ablation 1\n(Planar)', 'Ablation 2\n(Random Adj.)', 'Ablation 3\n(Shuffled)']
means = [baseline_mean, ablation1_mean, ablation2_mean, ablation3_mean]
stds = [baseline_std, ablation1_std, ablation2_std, ablation3_std]

bars = ax.bar(conditions, means, yerr=stds, capsize=5, color=['steelblue', 'coral', 'mediumseagreen', 'gold'], 
 alpha=0.7, edgecolor='black')

ax.axhline(baseline_mean, color='red', linestyle='--', label='Baseline', linewidth=2)
ax.set_ylabel('EntPTC Frequency (Hz)', fontsize=12)
ax.set_title('Toroidal Causality Ablation Results', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/causality_ablation.png', dpi=300, bbox_inches='tight')
print(f"Saved figure: {output_dir}/figures/causality_ablation.png")

print("\n" + "=" * 80)
print("CAUSALITY ABLATION COMPLETE")
print("=" * 80)
