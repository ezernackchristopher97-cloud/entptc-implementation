"""
STAGE C: Projection of Toroidal Invariants into EEG

Project the geometry-derived invariants from Stages A-B into EEG data
and test for cross-modal persistence of the ~0.4 Hz control mode.

CRITICAL: EEG is treated as a PROJECTION SPACE, not a generator.
NOT re-estimating frequencies from EEG. Testing whether
the geometry-derived invariants persist when projected into EEG coordinates.

"""

import numpy as np
import scipy.io as sio
import json
import os
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import h5py

# Load Stage B frequency invariants
stage_b_dir = '/home/ubuntu/entptc-implementation/stage_b_outputs'
output_dir = '/home/ubuntu/entptc-implementation/stage_c_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("=" * 80)
print("STAGE C: PROJECTION OF TOROIDAL INVARIANTS INTO EEG")
print("=" * 80)

# ============================================================================
# STEP 1: Load Stage B invariants (target for projection)
# ============================================================================

print("\nSTEP 1: Loading Stage B invariants...")

with open(f'{stage_b_dir}/frequency_invariants.json', 'r') as f:
 invariants = json.load(f)

# Extract target frequency
target_freq = np.mean([inv['entptc_characteristic_frequency_hz'] for inv in invariants])
target_freq_std = np.std([inv['entptc_characteristic_frequency_hz'] for inv in invariants])

print(f"Target EntPTC frequency: {target_freq:.4f} ± {target_freq_std:.4f} Hz")
print(f"Target band: Sub-delta (~0.4 Hz)")

# ============================================================================
# STEP 2: Load EEG data (Dataset Set 2)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Loading EEG data (Dataset Set 2 - PhysioNet Motor Movement)")
print("=" * 80)

eeg_data_dir = '/home/ubuntu/entptc-implementation/data/dataset_set_2'
eeg_files = [f for f in os.listdir(eeg_data_dir) if f.endswith('.mat')]
eeg_files.sort()

print(f"Found {len(eeg_files)} EEG recordings")

# ============================================================================
# STEP 3: Define projection mapping
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Defining projection mapping")
print("=" * 80)

"""
PROJECTION MAPPING: Toroidal Invariants → EEG Observables

From Stage B, available:
- Phase velocity v_φ (rad/s)
- Trajectory curvature κ
- Phase entropy H
- EntPTC characteristic frequency f_EntPTC ≈ 0.4 Hz

Projection into EEG:
1. Extract ~0.4 Hz component from EEG (slow envelope)
2. Test for phase organization at this frequency
3. Test for modulation of higher-frequency activity
4. Compute cross-frequency coupling with target frequency

This tests whether the geometry-derived control mode appears in EEG.
"""

def butter_bandpass(lowcut, highcut, fs, order=4):
 nyq = 0.5 * fs
 low = lowcut / nyq
 high = highcut / nyq
 b, a = butter(order, [low, high], btype='band')
 return b, a

def extract_slow_envelope(eeg_signal, fs, target_freq, bandwidth=0.1):
 """
 Extract slow envelope around target frequency.
 
 Args:
 eeg_signal: EEG time series (channels × time)
 fs: Sampling frequency
 target_freq: Target frequency (Hz)
 bandwidth: Bandwidth around target (Hz)
 
 Returns:
 envelope: Slow envelope signal
 phase: Instantaneous phase
 """
 lowcut = max(0.05, target_freq - bandwidth)
 highcut = target_freq + bandwidth
 
 b, a = butter_bandpass(lowcut, highcut, fs, order=4)
 
 # Filter each channel
 filtered = np.zeros_like(eeg_signal)
 for ch in range(eeg_signal.shape[0]):
 filtered[ch, :] = filtfilt(b, a, eeg_signal[ch, :])
 
 # Compute analytic signal
 analytic = hilbert(filtered, axis=1)
 envelope = np.abs(analytic)
 phase = np.angle(analytic)
 
 return envelope, phase

def compute_phase_locking_value(phases):
 """
 Compute phase locking value across channels.
 
 High PLV indicates phase organization at the target frequency.
 """
 # Average phase across channels
 mean_phase = np.angle(np.mean(np.exp(1j * phases), axis=0))
 
 # PLV for each channel relative to mean
 plv = np.abs(np.mean(np.exp(1j * (phases - mean_phase)), axis=1))
 
 return np.mean(plv)

def compute_cross_frequency_coupling(low_freq_phase, high_freq_amp):
 """
 Compute modulation index: how much low frequency modulates high frequency.
 
 This tests whether ~0.4 Hz acts as a control signal for higher frequencies.
 """
 # Bin high-freq amplitude by low-freq phase
 n_bins = 18
 phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
 
 amp_by_phase = []
 for i in range(n_bins):
 mask = (low_freq_phase >= phase_bins[i]) & (low_freq_phase < phase_bins[i+1])
 if np.sum(mask) > 0:
 amp_by_phase.append(np.mean(high_freq_amp[mask]))
 else:
 amp_by_phase.append(0)
 
 amp_by_phase = np.array(amp_by_phase)
 
 # Modulation index (normalized entropy)
 p = amp_by_phase / np.sum(amp_by_phase) if np.sum(amp_by_phase) > 0 else np.ones(n_bins) / n_bins
 p = p + 1e-10 # Avoid log(0)
 H = -np.sum(p * np.log(p))
 H_max = np.log(n_bins)
 MI = (H_max - H) / H_max
 
 return MI

# ============================================================================
# STEP 4: Project invariants into EEG
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Projecting invariants into EEG...")
print("=" * 80)

results = []

for i, eeg_file in enumerate(eeg_files[:10]): # Process first 10 for speed
 print(f"\n[{i+1}/{min(10, len(eeg_files))}] Processing {eeg_file}...")
 
 file_path = os.path.join(eeg_data_dir, eeg_file)
 
 try:
 # Load EEG data
 data = sio.loadmat(file_path)
 eeg_signal = data['eeg_data'] # 16 ROIs × time
 fs = int(data['fs'][0, 0]) # Sampling frequency
 
 # Transpose if needed
 if eeg_signal.shape[0] > eeg_signal.shape[1]:
 eeg_signal = eeg_signal.T
 
 print(f" EEG shape: {eeg_signal.shape}")
 print(f" Duration: {eeg_signal.shape[1] / fs:.1f} seconds")
 
 # Extract slow envelope at target frequency
 envelope, phase = extract_slow_envelope(eeg_signal, fs, target_freq, bandwidth=0.1)
 
 # Test 1: Phase organization
 plv = compute_phase_locking_value(phase)
 print(f" Phase Locking Value (PLV): {plv:.4f}")
 
 # Test 2: Envelope coherence across channels
 envelope_corr = np.corrcoef(envelope)
 mean_envelope_corr = np.mean(envelope_corr[np.triu_indices_from(envelope_corr, k=1)])
 print(f" Mean envelope correlation: {mean_envelope_corr:.4f}")
 
 # Test 3: Cross-frequency coupling with higher frequencies
 # Extract alpha band (8-13 Hz) as example
 alpha_env, _ = extract_slow_envelope(eeg_signal, fs, 10.5, bandwidth=2.5)
 
 # Compute CFC between ~0.4 Hz phase and alpha amplitude
 cfc_scores = []
 for ch in range(eeg_signal.shape[0]):
 mi = compute_cross_frequency_coupling(phase[ch, :], alpha_env[ch, :])
 cfc_scores.append(mi)
 
 mean_cfc = np.mean(cfc_scores)
 print(f" Cross-frequency coupling (0.4 Hz → Alpha): {mean_cfc:.4f}")
 
 # Test 4: Slow envelope power
 envelope_power = np.mean(np.var(envelope, axis=1))
 print(f" Slow envelope power: {envelope_power:.4f}")
 
 results.append({
 'file': eeg_file,
 'plv': plv,
 'envelope_correlation': mean_envelope_corr,
 'cross_frequency_coupling': mean_cfc,
 'envelope_power': envelope_power
 })
 
 except Exception as e:
 print(f" Error: {e}")
 continue

# ============================================================================
# STEP 5: Assess persistence
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Assessing cross-modal persistence")
print("=" * 80)

if len(results) > 0:
 plv_values = [r['plv'] for r in results]
 env_corr_values = [r['envelope_correlation'] for r in results]
 cfc_values = [r['cross_frequency_coupling'] for r in results]
 env_power_values = [r['envelope_power'] for r in results]
 
 print(f"\nPhase Locking Value:")
 print(f" Mean: {np.mean(plv_values):.4f} ± {np.std(plv_values):.4f}")
 print(f" Range: [{np.min(plv_values):.4f}, {np.max(plv_values):.4f}]")
 
 print(f"\nEnvelope Correlation:")
 print(f" Mean: {np.mean(env_corr_values):.4f} ± {np.std(env_corr_values):.4f}")
 print(f" Range: [{np.min(env_corr_values):.4f}, {np.max(env_corr_values):.4f}]")
 
 print(f"\nCross-Frequency Coupling:")
 print(f" Mean: {np.mean(cfc_values):.4f} ± {np.std(cfc_values):.4f}")
 print(f" Range: [{np.min(cfc_values):.4f}, {np.max(cfc_values):.4f}]")
 
 print(f"\nEnvelope Power:")
 print(f" Mean: {np.mean(env_power_values):.4f} ± {np.std(env_power_values):.4f}")
 
 # Persistence assessment
 print("\n" + "=" * 80)
 print("PERSISTENCE ASSESSMENT")
 print("=" * 80)
 
 # Criteria for persistence:
 # 1. PLV > 0.3 (moderate phase organization)
 # 2. Envelope correlation > 0.2 (cross-channel coherence)
 # 3. CFC > 0.1 (modulation of higher frequencies)
 
 plv_persist = np.mean(plv_values) > 0.3
 env_persist = np.mean(env_corr_values) > 0.2
 cfc_persist = np.mean(cfc_values) > 0.1
 
 persistence_score = sum([plv_persist, env_persist, cfc_persist])
 
 print(f"\nPersistence Criteria:")
 print(f" Phase organization (PLV > 0.3): {'✓' if plv_persist else '✗'}")
 print(f" Envelope coherence (corr > 0.2): {'✓' if env_persist else '✗'}")
 print(f" Cross-frequency coupling (CFC > 0.1): {'✓' if cfc_persist else '✗'}")
 print(f"\nPersistence Score: {persistence_score}/3")
 
 if persistence_score >= 2:
 verdict = "STRONG PERSISTENCE"
 elif persistence_score == 1:
 verdict = "PARTIAL PERSISTENCE"
 else:
 verdict = "WEAK PERSISTENCE"
 
 print(f"\nVERDICT: {verdict}")
 
 # Save results (convert numpy types to Python types)
 results_file = f'{output_dir}/stage_c_projection_results.json'
 with open(results_file, 'w') as f:
 json.dump({
 'target_frequency': float(target_freq),
 'persistence_score': int(persistence_score),
 'verdict': verdict,
 'results': [{k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in r.items()} for r in results],
 'summary': {
 'mean_plv': float(np.mean(plv_values)),
 'mean_envelope_correlation': float(np.mean(env_corr_values)),
 'mean_cfc': float(np.mean(cfc_values)),
 'mean_envelope_power': float(np.mean(env_power_values))
 }
 }, f, indent=2)
 print(f"\nSaved results to: {results_file}")
 
 # Create visualization
 fig, axes = plt.subplots(2, 2, figsize=(12, 10))
 
 axes[0, 0].hist(plv_values, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
 axes[0, 0].axvline(0.3, color='red', linestyle='--', label='Threshold')
 axes[0, 0].set_xlabel('Phase Locking Value')
 axes[0, 0].set_ylabel('Count')
 axes[0, 0].set_title('Phase Organization at ~0.4 Hz')
 axes[0, 0].legend()
 
 axes[0, 1].hist(env_corr_values, bins=10, color='coral', alpha=0.7, edgecolor='black')
 axes[0, 1].axvline(0.2, color='red', linestyle='--', label='Threshold')
 axes[0, 1].set_xlabel('Envelope Correlation')
 axes[0, 1].set_ylabel('Count')
 axes[0, 1].set_title('Cross-Channel Coherence')
 axes[0, 1].legend()
 
 axes[1, 0].hist(cfc_values, bins=10, color='mediumseagreen', alpha=0.7, edgecolor='black')
 axes[1, 0].axvline(0.1, color='red', linestyle='--', label='Threshold')
 axes[1, 0].set_xlabel('Modulation Index')
 axes[1, 0].set_ylabel('Count')
 axes[1, 0].set_title('Cross-Frequency Coupling (0.4 Hz → Alpha)')
 axes[1, 0].legend()
 
 axes[1, 1].text(0.5, 0.5, f"VERDICT:\n{verdict}\n\nScore: {persistence_score}/3",
 ha='center', va='center', fontsize=20, fontweight='bold',
 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
 axes[1, 1].axis('off')
 
 plt.tight_layout()
 plt.savefig(f'{output_dir}/figures/persistence_assessment.png', dpi=300, bbox_inches='tight')
 print(f"Saved figure: {output_dir}/figures/persistence_assessment.png")
 
 print("\n" + "=" * 80)
 print("STAGE C (EEG PROJECTION) COMPLETE")
 print("=" * 80)
 
 if persistence_score < 2:
 print("\nWEAK PERSISTENCE DETECTED")
 print("Proceeding to Alternative Path B: Control-frequency reinterpretation")
 else:
 print("\nSTRONG/PARTIAL PERSISTENCE CONFIRMED")
 print("Geometry-derived control mode projects into EEG observables")

else:
 print("\nERROR: No EEG files processed successfully")
 print("Proceeding to Alternative Path A: fMRI-first projection")
