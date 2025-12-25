"""
STAGE B: Internal Frequency Inference from Geometry-Driven Dynamics

Infer internal frequencies from toroidal geometry and dynamics extracted in Stage A.
This is NOT direct frequency measurement from EEG - it's geometry → dynamics → frequency.

The EntPTC model posits that experiential coherence is reflected in the dynamics
on T³, which naturally give rise to characteristic frequencies through:
1. Phase velocity on the torus
2. Trajectory curvature (geodesic deviation)
3. Entropy flow dynamics

These frequencies are MODALITY-AGNOSTIC - they emerge from the geometry,
not from the recording modality.

"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt

# Load Stage A results
stage_a_dir = '/home/ubuntu/entptc-implementation/stage_a_outputs'
output_dir = '/home/ubuntu/entptc-implementation/stage_b_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("=" * 80)
print("STAGE B: INTERNAL FREQUENCY INFERENCE FROM GEOMETRY")
print("=" * 80)

# ============================================================================
# STEP 1: Load Stage A geometry-based invariants
# ============================================================================

print("\nSTEP 1: Loading Stage A invariants...")

with open(f'{stage_a_dir}/grid_cell_invariants.json', 'r') as f:
 invariants = json.load(f)

print(f"Loaded {len(invariants)} grid cells with valid toroidal structure")

# ============================================================================
# STEP 2: Infer frequencies from phase velocity
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Inferring frequencies from phase velocity on T²")
print("=" * 80)

"""
Frequency inference from phase velocity:

For a trajectory on T² with phase coordinates (φ_x, φ_y), the phase velocity
is v_φ = sqrt((dφ_x/dt)² + (dφ_y/dt)²).

The characteristic frequency is:
f = v_φ / (2π)

This frequency represents how fast the system moves through phase space,
which in the EntPTC model corresponds to the rate of experiential state updates.
"""

for inv in invariants:
 # Extract phase velocity (rad/s)
 v_phi = inv['mean_phase_velocity']
 
 # Convert to frequency (Hz)
 # f = v_φ / (2π)
 frequency_hz = v_phi / (2 * np.pi)
 
 inv['inferred_frequency_hz'] = frequency_hz
 
 print(f"{inv['cell_id']}:")
 print(f" Phase velocity: {v_phi:.4f} rad/s")
 print(f" Inferred frequency: {frequency_hz:.4f} Hz")

# ============================================================================
# STEP 3: Compute frequency from curvature dynamics
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Computing frequency from trajectory curvature")
print("=" * 80)

"""
Curvature-based frequency:

The curvature κ of a trajectory on T² measures how quickly the direction changes.
High curvature → rapid directional changes → higher characteristic frequency.

We can relate curvature to frequency through:
f_κ = sqrt(κ * v_φ) / (2π)

This captures the "oscillatory" nature of curved trajectories.
"""

for inv in invariants:
 v_phi = inv['mean_phase_velocity']
 kappa = inv['mean_curvature']
 
 # Curvature-based frequency
 f_kappa = np.sqrt(kappa * v_phi) / (2 * np.pi)
 
 inv['curvature_frequency_hz'] = f_kappa
 
 print(f"{inv['cell_id']}:")
 print(f" Curvature: {kappa:.4f}")
 print(f" Curvature-based frequency: {f_kappa:.4f} Hz")

# ============================================================================
# STEP 4: Entropy flow frequency
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Computing frequency from entropy flow dynamics")
print("=" * 80)

"""
Entropy flow frequency:

The phase entropy H measures how uniformly the trajectory covers T².
Changes in entropy over time (entropy flow) indicate transitions between
different dynamical regimes.

For a system with entropy H and phase velocity v_φ:
f_H = v_φ / (2π * H)

High entropy → slower effective frequency (more dispersed dynamics)
Low entropy → faster effective frequency (more focused dynamics)
"""

for inv in invariants:
 v_phi = inv['mean_phase_velocity']
 H = inv['phase_entropy']
 
 # Entropy-modulated frequency
 f_entropy = v_phi / (2 * np.pi * H)
 
 inv['entropy_frequency_hz'] = f_entropy
 
 print(f"{inv['cell_id']}:")
 print(f" Phase entropy: {H:.4f}")
 print(f" Entropy-modulated frequency: {f_entropy:.4f} Hz")

# ============================================================================
# STEP 5: Composite frequency (EntPTC characteristic frequency)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Computing composite EntPTC characteristic frequency")
print("=" * 80)

"""
EntPTC Characteristic Frequency:

The model predicts a characteristic frequency that emerges from the interplay
of phase velocity, curvature, and entropy. Computed as a weighted
geometric mean:

f_EntPTC = (f_velocity * f_curvature * f_entropy)^(1/3)

This is the MODALITY-AGNOSTIC frequency that should persist across
different recording modalities (grid cells, EEG, fMRI, etc.)
"""

for inv in invariants:
 f_v = inv['inferred_frequency_hz']
 f_k = inv['curvature_frequency_hz']
 f_h = inv['entropy_frequency_hz']
 
 # Geometric mean (to avoid any single component dominating)
 f_entptc = (f_v * f_k * f_h) ** (1/3)
 
 inv['entptc_characteristic_frequency_hz'] = f_entptc
 
 print(f"{inv['cell_id']}:")
 print(f" f_velocity: {f_v:.4f} Hz")
 print(f" f_curvature: {f_k:.4f} Hz")
 print(f" f_entropy: {f_h:.4f} Hz")
 print(f" f_EntPTC: {f_entptc:.4f} Hz")

# ============================================================================
# STEP 6: Frequency band classification
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Classifying frequencies into neural bands")
print("=" * 80)

"""
Map inferred frequencies to standard neural frequency bands:
- Delta: 0.5-4 Hz
- Theta: 4-8 Hz
- Alpha: 8-13 Hz
- Beta: 13-30 Hz
- Gamma: 30-100 Hz
"""

def classify_frequency_band(freq_hz):
 if freq_hz < 0.5:
 return "Sub-delta"
 elif freq_hz < 4:
 return "Delta"
 elif freq_hz < 8:
 return "Theta"
 elif freq_hz < 13:
 return "Alpha"
 elif freq_hz < 30:
 return "Beta"
 elif freq_hz < 100:
 return "Gamma"
 else:
 return "High-gamma"

for inv in invariants:
 f_entptc = inv['entptc_characteristic_frequency_hz']
 band = classify_frequency_band(f_entptc)
 
 inv['frequency_band'] = band
 
 print(f"{inv['cell_id']}: {f_entptc:.4f} Hz → {band}")

# ============================================================================
# STEP 7: Summary statistics
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Summary statistics")
print("=" * 80)

frequencies = [inv['entptc_characteristic_frequency_hz'] for inv in invariants]

print(f"\nEntPTC Characteristic Frequency:")
print(f" Mean: {np.mean(frequencies):.4f} Hz")
print(f" Std: {np.std(frequencies):.4f} Hz")
print(f" Range: [{np.min(frequencies):.4f}, {np.max(frequencies):.4f}] Hz")

# Count frequency bands
bands = [inv['frequency_band'] for inv in invariants]
unique_bands = list(set(bands))
print(f"\nFrequency band distribution:")
for band in unique_bands:
 count = bands.count(band)
 print(f" {band}: {count}/{len(bands)}")

# ============================================================================
# STEP 8: Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Creating visualizations...")
print("=" * 80)

# Figure 1: Frequency components
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

cell_ids = [inv['cell_id'] for inv in invariants]
f_velocity = [inv['inferred_frequency_hz'] for inv in invariants]
f_curvature = [inv['curvature_frequency_hz'] for inv in invariants]
f_entropy = [inv['entropy_frequency_hz'] for inv in invariants]
f_entptc = [inv['entptc_characteristic_frequency_hz'] for inv in invariants]

x = np.arange(len(cell_ids))

axes[0, 0].bar(x, f_velocity, color='steelblue', alpha=0.7)
axes[0, 0].set_ylabel('Frequency (Hz)')
axes[0, 0].set_title('Velocity-based Frequency')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(cell_ids, rotation=45, ha='right')

axes[0, 1].bar(x, f_curvature, color='coral', alpha=0.7)
axes[0, 1].set_ylabel('Frequency (Hz)')
axes[0, 1].set_title('Curvature-based Frequency')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(cell_ids, rotation=45, ha='right')

axes[1, 0].bar(x, f_entropy, color='mediumseagreen', alpha=0.7)
axes[1, 0].set_ylabel('Frequency (Hz)')
axes[1, 0].set_title('Entropy-modulated Frequency')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(cell_ids, rotation=45, ha='right')

axes[1, 1].bar(x, f_entptc, color='purple', alpha=0.7)
axes[1, 1].set_ylabel('Frequency (Hz)')
axes[1, 1].set_title('EntPTC Characteristic Frequency')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(cell_ids, rotation=45, ha='right')

plt.tight_layout()
plt.savefig(f'{output_dir}/figures/frequency_components.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/figures/frequency_components.png")
plt.close()

# Figure 2: Frequency distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(f_entptc, bins=10, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(f_entptc), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f_entptc):.2f} Hz')
ax.set_xlabel('EntPTC Characteristic Frequency (Hz)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Inferred Frequencies from Grid Cell Geometry')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/figures/frequency_distribution.png', dpi=300, bbox_inches='tight')
print(f"Saved: {output_dir}/figures/frequency_distribution.png")
plt.close()

# ============================================================================
# STEP 9: Save results
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Saving results...")
print("=" * 80)

# Save updated invariants with frequencies
output_file = f'{output_dir}/frequency_invariants.json'
with open(output_file, 'w') as f:
 json.dump(invariants, f, indent=2)
print(f"Saved: {output_file}")

# Save summary
summary_file = f'{output_dir}/stage_b_summary.txt'
with open(summary_file, 'w') as f:
 f.write("STAGE B: Internal Frequency Inference - Summary\n")
 f.write("=" * 80 + "\n\n")
 f.write(f"Grid cells analyzed: {len(invariants)}\n\n")
 f.write("EntPTC Characteristic Frequency:\n")
 f.write(f" Mean: {np.mean(frequencies):.4f} Hz\n")
 f.write(f" Std: {np.std(frequencies):.4f} Hz\n")
 f.write(f" Range: [{np.min(frequencies):.4f}, {np.max(frequencies):.4f}] Hz\n\n")
 f.write("Frequency band distribution:\n")
 for band in unique_bands:
 count = bands.count(band)
 f.write(f" {band}: {count}/{len(bands)}\n")
 f.write("\nKey insight:\n")
 f.write("These frequencies are MODALITY-AGNOSTIC - they emerge from toroidal\n")
 f.write("geometry and dynamics, NOT from the recording modality. They should\n")
 f.write("persist when projected into EEG/fMRI in Stage C.\n")
print(f"Saved: {summary_file}")

print("\n" + "=" * 80)
print("STAGE B COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {output_dir}")
print("\nNext: STAGE C - Project invariants into EEG/fMRI and test cross-modal persistence")
