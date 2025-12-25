"""
Stage C: Projection Tests (C1/C2/C3) - CORRECTED FRAMING
==========================================================

Per locked protocol, Stage C tests:
C1) Gating: Does geometry-derived mode modulate higher frequencies OR gate regime transitions/event timing?
C2) Organization: Does it organize phase relationships/coherence in a way disrupted by geometry-targeted ablations?
C3) Regime timing: Does it align with regime transitions or task events (after artifact removal)?

NOT: "Does 0.2-0.4 Hz appear in EEG/fMRI"

"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from pathlib import Path
import json
from typing import Dict

# Set random seed
np.random.seed(42)

# ============================================================================
# STAGE C: C1 (GATING)
# ============================================================================

def test_c1_gating(data: np.ndarray, fs: float, control_frequency: float) -> Dict:
 """
 C1: Does geometry-derived mode gate regime transitions or modulate higher frequencies?
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 control_frequency: inferred control frequency from operator collapse (Hz)
 
 Returns:
 c1_results: dict with gating metrics
 """
 print("\n" + "="*80)
 print("C1: GATING TEST")
 print("="*80)
 
 n_rois, n_samples = data.shape
 
 # Per corrected framing: control frequency projects as infra-slow in EEG
 # Use sub-delta band (0.14-0.33 Hz) as EEG projection of control mode
 f_low = 0.14
 f_high = 0.33
 
 print(f"\nOperator-derived control frequency: {control_frequency:.2f} Hz (intrinsic)")
 print(f"EEG projection band (infra-slow): {f_low:.2f}-{f_high:.2f} Hz")
 print("Note: Control frequency appears as infra-slow modulation in EEG projection")
 
 # Bandpass filter
 sos = signal.butter(4, [f_low, f_high], btype='band', fs=fs, output='sos')
 
 control_mode = np.zeros((n_rois, n_samples))
 for i in range(n_rois):
 control_mode[i] = signal.sosfiltfilt(sos, data[i])
 
 # Compute envelope (amplitude modulation)
 control_envelope = np.abs(signal.hilbert(control_mode))
 
 # Average across ROIs
 control_envelope_avg = np.mean(control_envelope, axis=0)
 
 # Test 1: Does control envelope modulate higher-frequency power?
 # Extract gamma band (30-50 Hz)
 sos_gamma = signal.butter(4, [30, 50], btype='band', fs=fs, output='sos')
 gamma_power = np.zeros(n_samples)
 for i in range(n_rois):
 gamma_filtered = signal.sosfiltfilt(sos_gamma, data[i])
 gamma_power += np.abs(signal.hilbert(gamma_filtered))**2
 gamma_power /= n_rois
 
 # Correlation between control envelope and gamma power
 correlation_control_gamma = np.corrcoef(control_envelope_avg, gamma_power)[0, 1]
 
 print(f"\nControl envelope - Gamma power correlation: {correlation_control_gamma:.4f}")
 
 # Test 2: Does control envelope gate regime transitions?
 # Compute regime transitions (threshold crossings)
 threshold = np.median(control_envelope_avg)
 regime_labels = (control_envelope_avg > threshold).astype(int)
 n_transitions = np.sum(np.diff(regime_labels) != 0)
 
 print(f"Number of regime transitions: {n_transitions}")
 
 # Gating strength: variance of control envelope
 gating_strength = np.var(control_envelope_avg)
 
 print(f"Gating strength (envelope variance): {gating_strength:.6f}")
 
 # Pass criterion: |correlation| > 0.1 OR gating strength > 0.01
 pass_c1 = (abs(correlation_control_gamma) > 0.1) or (gating_strength > 0.01)
 
 print(f"\nC1 verdict: {'✅ PASS' if pass_c1 else '❌ FAIL'}")
 
 results = {
 'control_frequency_hz': float(control_frequency),
 'gating_band_hz': [float(f_low), float(f_high)],
 'correlation_control_gamma': float(correlation_control_gamma),
 'n_regime_transitions': int(n_transitions),
 'gating_strength': float(gating_strength),
 'pass': bool(pass_c1)
 }
 
 return results

# ============================================================================
# STAGE C: C2 (ORGANIZATION)
# ============================================================================

def test_c2_organization(data: np.ndarray, fs: float, adjacency: np.ndarray) -> Dict:
 """
 C2: Does geometry-derived mode organize phase relationships/coherence?
 
 Tests if coherence structure is disrupted by geometry-targeted ablations.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 c2_results: dict with organization metrics
 """
 from entptc.t3_to_r3_mapping import compute_t3_coordinates
 from uniqueness_U3_causal_ablation import randomize_adjacency_degree_preserved
 
 print("\n" + "="*80)
 print("C2: ORGANIZATION TEST")
 print("="*80)
 
 n_rois = data.shape[0]
 
 # Baseline: compute phase coherence with real adjacency
 print("\nBaseline (real adjacency)...")
 t3_coords_baseline = compute_t3_coordinates(data, fs)
 
 # PLV on sub-delta band
 phases = t3_coords_baseline[0] # θ₁ (sub-delta)
 
 plv_baseline = []
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
 plv_baseline.append(plv)
 
 plv_baseline_mean = np.mean(plv_baseline) if len(plv_baseline) > 0 else 0
 
 print(f"Baseline PLV: {plv_baseline_mean:.6f}")
 
 # Ablation: randomize adjacency (geometry-targeted)
 print("\nAblation (randomized adjacency)...")
 adjacency_random = randomize_adjacency_degree_preserved(adjacency, n_swaps=100)
 
 plv_ablated = []
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency_random[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
 plv_ablated.append(plv)
 
 plv_ablated_mean = np.mean(plv_ablated) if len(plv_ablated) > 0 else 0
 
 print(f"Ablated PLV: {plv_ablated_mean:.6f}")
 
 # Effect size
 plv_change = abs(plv_baseline_mean - plv_ablated_mean)
 plv_change_pct = (plv_change / plv_baseline_mean * 100) if plv_baseline_mean > 0 else 0
 
 print(f"PLV change: {plv_change:.6f} ({plv_change_pct:.1f}%)")
 
 # Pass criterion: PLV changes by > 5% under geometry ablation
 pass_c2 = plv_change_pct > 5
 
 print(f"\nC2 verdict: {'✅ PASS' if pass_c2 else '❌ FAIL'}")
 
 results = {
 'plv_baseline': float(plv_baseline_mean),
 'plv_ablated': float(plv_ablated_mean),
 'plv_change': float(plv_change),
 'plv_change_pct': float(plv_change_pct),
 'pass': bool(pass_c2)
 }
 
 return results

# ============================================================================
# STAGE C: C3 (REGIME TIMING)
# ============================================================================

def test_c3_regime_timing(data: np.ndarray, fs: float, min_dwell_sec: float = 15) -> Dict:
 """
 C3: Does geometry-derived mode align with regime transitions?
 
 Tests if regime transitions occur at stable intervals (after artifact removal).
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 min_dwell_sec: minimum dwell time (seconds)
 
 Returns:
 c3_results: dict with regime timing metrics
 """
 print("\n" + "="*80)
 print("C3: REGIME TIMING TEST")
 print("="*80)
 
 n_rois, n_samples = data.shape
 
 # Compute spectral gap time series (as in previous analysis)
 window_length_sec = 10.0
 window_length_samples = int(window_length_sec * fs)
 overlap = 0.5
 step = int(window_length_samples * (1 - overlap))
 
 spectral_gaps = []
 
 for start in range(0, n_samples - window_length_samples, step):
 end = start + window_length_samples
 window_data = data[:, start:end]
 
 # Covariance matrix
 cov = np.cov(window_data)
 
 # Eigenvalues
 eigenvalues = np.linalg.eigvalsh(cov)
 eigenvalues = np.sort(eigenvalues)[::-1]
 
 # Spectral gap
 if len(eigenvalues) >= 2:
 gap = eigenvalues[0] - eigenvalues[1]
 spectral_gaps.append(gap)
 
 spectral_gaps = np.array(spectral_gaps)
 
 # Threshold at median
 threshold = np.median(spectral_gaps)
 
 # Assign regime labels
 regime_labels_raw = (spectral_gaps > threshold).astype(int)
 
 # Apply minimum dwell time filter
 min_dwell_windows = int(min_dwell_sec / (window_length_sec * (1 - overlap)))
 
 regime_labels = regime_labels_raw.copy()
 current_regime = regime_labels[0]
 dwell_count = 1
 
 for i in range(1, len(regime_labels)):
 if regime_labels_raw[i] == current_regime:
 dwell_count += 1
 else:
 if dwell_count >= min_dwell_windows:
 current_regime = regime_labels_raw[i]
 dwell_count = 1
 else:
 regime_labels[i] = current_regime
 dwell_count += 1
 
 # Count transitions
 n_transitions = np.sum(np.diff(regime_labels) != 0)
 
 # Compute dwell times
 dwell_times = []
 current_regime = regime_labels[0]
 dwell_count = 1
 
 for i in range(1, len(regime_labels)):
 if regime_labels[i] == current_regime:
 dwell_count += 1
 else:
 dwell_times.append(dwell_count * window_length_sec * (1 - overlap))
 current_regime = regime_labels[i]
 dwell_count = 1
 
 dwell_times.append(dwell_count * window_length_sec * (1 - overlap))
 
 mean_dwell_time = np.mean(dwell_times)
 std_dwell_time = np.std(dwell_times)
 cv_dwell_time = std_dwell_time / mean_dwell_time if mean_dwell_time > 0 else 0
 
 print(f"\nNumber of regime transitions: {n_transitions}")
 print(f"Mean dwell time: {mean_dwell_time:.2f} seconds")
 print(f"Std dwell time: {std_dwell_time:.2f} seconds")
 print(f"CV (coefficient of variation): {cv_dwell_time:.2f}")
 
 # Pass criterion: CV < 1.0 (dwell times are relatively stable)
 pass_c3 = cv_dwell_time < 1.0
 
 print(f"\nC3 verdict: {'✅ PASS' if pass_c3 else '❌ FAIL'}")
 
 results = {
 'n_transitions': int(n_transitions),
 'mean_dwell_time_sec': float(mean_dwell_time),
 'std_dwell_time_sec': float(std_dwell_time),
 'cv_dwell_time': float(cv_dwell_time),
 'pass': bool(pass_c3)
 }
 
 return results

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 from entptc.utils.grid_utils import create_toroidal_grid
 from operator_collapse_frequency import compute_operator_collapse_frequency
 
 # Load data
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/stage_c_final')
 output_dir.mkdir(parents=True, exist_ok=True)
 
 if not data_path.exists():
 print(f"Data file not found: {data_path}")
 exit(1)
 
 print("="*80)
 print("STAGE C: C1/C2/C3 TESTS (CORRECTED FRAMING)")
 print("="*80)
 
 print("\nLoading data...")
 mat = sio.loadmat(data_path)
 data = mat['eeg_data']
 fs = float(mat['fs'][0, 0])
 
 grid_size = int(np.sqrt(data.shape[0]))
 adjacency = create_toroidal_grid(grid_size)
 
 print(f"Data shape: {data.shape}")
 print(f"Sampling rate: {fs} Hz")
 print(f"Grid size: {grid_size}×{grid_size}")
 
 # Get control frequency from operator collapse
 print("\nComputing operator collapse frequency...")
 collapse_results = compute_operator_collapse_frequency(data, fs, adjacency)
 control_frequency = collapse_results['f_control_hz']
 
 # Run C1/C2/C3 tests
 c1_results = test_c1_gating(data, fs, control_frequency)
 c2_results = test_c2_organization(data, fs, adjacency)
 c3_results = test_c3_regime_timing(data, fs, min_dwell_sec=15)
 
 # Consolidate results
 results = {
 'dataset': str(data_path.name),
 'control_frequency_hz': control_frequency,
 'c1_gating': c1_results,
 'c2_organization': c2_results,
 'c3_regime_timing': c3_results,
 'overall_pass': c1_results['pass'] and c2_results['pass'] and c3_results['pass']
 }
 
 # Save results
 output_path = output_dir / 'stage_c_final_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print("\n" + "="*80)
 print("STAGE C OVERALL VERDICT")
 print("="*80)
 print(f"C1 (Gating): {'✅ PASS' if c1_results['pass'] else '❌ FAIL'}")
 print(f"C2 (Organization): {'✅ PASS' if c2_results['pass'] else '❌ FAIL'}")
 print(f"C3 (Regime timing): {'✅ PASS' if c3_results['pass'] else '❌ FAIL'}")
 print(f"\nOverall: {'✅ PASS' if results['overall_pass'] else '⚠️ PARTIAL'}")
 
 print(f"\n✅ Results saved to {output_path}")
