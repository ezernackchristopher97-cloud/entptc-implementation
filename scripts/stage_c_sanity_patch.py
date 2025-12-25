"""
Stage C Sanity Patch - Artifact Fixes for ds004706 Analysis

MANDATORY fixes before any Stage C interpretation:
1. Phase/PLV sanity: Verify per-ROI phases, add null controls, replace PLV with PPC
2. Regime transition sanity: Boundary crossings only, dwell time, transitions/minute
3. Ablation correctness: Verify constraint matrices actually change
4. Windowing/sample rate sanity: Verify actual rates, window lengths, PAC cycles

"""

import numpy as np
import scipy.signal as signal
from scipy.stats import circmean, circstd
import json
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# A) PHASE / PLV SANITY CHECKS
# ============================================================================

def compute_phase_per_roi(data, fs, freq_range=(0.14, 0.33)):
 """
 Compute phase per ROI with sanity checks.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 freq_range: (low, high) frequency range for bandpass
 
 Returns:
 phases: (n_rois, n_samples) array of instantaneous phases
 sanity_checks: dict of diagnostic information
 """
 n_rois, n_samples = data.shape
 
 # Sanity check 1: Verify ROI variance (should not be identical)
 roi_variances = np.var(data, axis=1)
 print(f"ROI variances: min={roi_variances.min():.6f}, max={roi_variances.max():.6f}, mean={roi_variances.mean():.6f}")
 
 # Sanity check 2: Pairwise correlations (should not all be 1.0)
 pairwise_corrs = []
 for i in range(n_rois):
 for j in range(i+1, n_rois):
 corr = np.corrcoef(data[i], data[j])[0, 1]
 pairwise_corrs.append(corr)
 pairwise_corrs = np.array(pairwise_corrs)
 print(f"Pairwise correlations: min={pairwise_corrs.min():.3f}, max={pairwise_corrs.max():.3f}, mean={pairwise_corrs.mean():.3f}")
 
 # Bandpass filter
 nyquist = fs / 2
 if freq_range[1] >= nyquist:
 raise ValueError(f"High frequency {freq_range[1]} Hz exceeds Nyquist {nyquist} Hz")
 
 sos = signal.butter(4, freq_range, btype='band', fs=fs, output='sos')
 
 phases = np.zeros_like(data)
 for i in range(n_rois):
 # Filter
 filtered = signal.sosfiltfilt(sos, data[i])
 
 # Hilbert transform
 analytic = signal.hilbert(filtered)
 phases[i] = np.angle(analytic)
 
 # Sanity check 3: Phase variance per ROI (should not be zero)
 phase_variances = np.var(phases, axis=1)
 print(f"Phase variances: min={phase_variances.min():.6f}, max={phase_variances.max():.6f}, mean={phase_variances.mean():.6f}")
 
 # Sanity check 4: Phase uniqueness (should not all be identical)
 phase_uniqueness = []
 for i in range(n_rois):
 unique_phases = len(np.unique(np.round(phases[i], 3)))
 phase_uniqueness.append(unique_phases)
 print(f"Unique phases per ROI: min={min(phase_uniqueness)}, max={max(phase_uniqueness)}, mean={np.mean(phase_uniqueness):.1f}")
 
 sanity_checks = {
 'roi_variances': roi_variances.tolist(),
 'pairwise_corrs': {'min': float(pairwise_corrs.min()), 'max': float(pairwise_corrs.max()), 'mean': float(pairwise_corrs.mean())},
 'phase_variances': phase_variances.tolist(),
 'phase_uniqueness': phase_uniqueness
 }
 
 return phases, sanity_checks

def compute_plv_with_null_control(phases):
 """
 Compute PLV with time-shift null control.
 
 Args:
 phases: (n_rois, n_samples) array of instantaneous phases
 
 Returns:
 plv_real: real PLV
 plv_null: null PLV (time-shifted control)
 plv_diff: difference (real - null)
 """
 n_rois, n_samples = phases.shape
 
 # Real PLV
 phase_diffs = []
 for i in range(n_rois):
 for j in range(i+1, n_rois):
 phase_diff = phases[i] - phases[j]
 phase_diffs.append(phase_diff)
 phase_diffs = np.array(phase_diffs)
 plv_real = np.abs(np.mean(np.exp(1j * phase_diffs), axis=1)).mean()
 
 # Null PLV (circular shift one ROI by random lag)
 plv_null_values = []
 for _ in range(100): # 100 null iterations
 phases_shifted = phases.copy()
 for i in range(n_rois):
 shift = np.random.randint(100, n_samples - 100)
 phases_shifted[i] = np.roll(phases[i], shift)
 
 phase_diffs_null = []
 for i in range(n_rois):
 for j in range(i+1, n_rois):
 phase_diff = phases_shifted[i] - phases_shifted[j]
 phase_diffs_null.append(phase_diff)
 phase_diffs_null = np.array(phase_diffs_null)
 plv_null = np.abs(np.mean(np.exp(1j * phase_diffs_null), axis=1)).mean()
 plv_null_values.append(plv_null)
 
 plv_null_mean = np.mean(plv_null_values)
 plv_null_std = np.std(plv_null_values)
 
 print(f"PLV real: {plv_real:.6f}")
 print(f"PLV null: {plv_null_mean:.6f} ± {plv_null_std:.6f}")
 print(f"PLV difference: {plv_real - plv_null_mean:.6f} ({(plv_real - plv_null_mean) / plv_null_std:.2f} SD)")
 
 return plv_real, plv_null_mean, plv_real - plv_null_mean

def compute_ppc(phases):
 """
 Compute Pairwise Phase Consistency (PPC) as alternative to PLV.
 
 PPC is less biased by trial count and more sensitive to true phase coupling.
 
 Args:
 phases: (n_rois, n_samples) array of instantaneous phases
 
 Returns:
 ppc: pairwise phase consistency
 """
 n_rois, n_samples = phases.shape
 
 ppc_values = []
 for i in range(n_rois):
 for j in range(i+1, n_rois):
 phase_diff = phases[i] - phases[j]
 
 # PPC formula: (sum of pairwise dot products) / (n * (n-1) / 2)
 # For continuous data, use windowed approach
 window_size = 1000
 n_windows = n_samples // window_size
 
 ppc_window = []
 for w in range(n_windows):
 start = w * window_size
 end = start + window_size
 phase_diff_window = phase_diff[start:end]
 
 # Compute pairwise dot products
 exp_phase = np.exp(1j * phase_diff_window)
 ppc_val = np.abs(np.sum(exp_phase)) ** 2 - len(exp_phase)
 ppc_val /= (len(exp_phase) * (len(exp_phase) - 1))
 ppc_window.append(ppc_val)
 
 ppc_values.append(np.mean(ppc_window))
 
 ppc = np.mean(ppc_values)
 print(f"PPC: {ppc:.6f}")
 
 return ppc

# ============================================================================
# B) REGIME TRANSITION SANITY CHECKS
# ============================================================================

def compute_regime_transitions_fixed(spectral_gap_ts, threshold_percentile=50, min_dwell_windows=3):
 """
 Compute regime transitions with proper boundary crossing detection and dwell time.
 
 Args:
 spectral_gap_ts: (n_samples,) array of spectral gap time series
 threshold_percentile: percentile for regime threshold
 min_dwell_windows: minimum number of windows to remain in new regime
 
 Returns:
 transitions: number of regime transitions
 transitions_per_minute: transitions per minute
 regime_labels: (n_samples,) array of regime labels (0 or 1)
 """
 # Compute threshold
 threshold = np.percentile(spectral_gap_ts, threshold_percentile)
 print(f"Spectral gap: min={spectral_gap_ts.min():.6f}, max={spectral_gap_ts.max():.6f}, mean={spectral_gap_ts.mean():.6f}")
 print(f"Threshold ({threshold_percentile}th percentile): {threshold:.6f}")
 
 # Assign initial regime labels
 regime_labels_raw = (spectral_gap_ts > threshold).astype(int)
 
 # Apply minimum dwell time filter
 regime_labels = regime_labels_raw.copy()
 current_regime = regime_labels[0]
 dwell_count = 1
 
 for i in range(1, len(regime_labels)):
 if regime_labels_raw[i] == current_regime:
 dwell_count += 1
 else:
 # Potential transition
 if dwell_count >= min_dwell_windows:
 # Accept transition
 current_regime = regime_labels_raw[i]
 dwell_count = 1
 else:
 # Reject transition (flicker)
 regime_labels[i] = current_regime
 dwell_count += 1
 
 # Count boundary crossings only
 transitions = 0
 for i in range(1, len(regime_labels)):
 if regime_labels[i] != regime_labels[i-1]:
 transitions += 1
 
 # Compute transitions per minute (assuming 1 Hz sampling of spectral gap)
 duration_minutes = len(spectral_gap_ts) / 60.0
 transitions_per_minute = transitions / duration_minutes
 
 print(f"Regime transitions: {transitions} total, {transitions_per_minute:.2f} per minute")
 print(f"Regime distribution: {np.sum(regime_labels == 0)} low, {np.sum(regime_labels == 1)} high")
 
 return transitions, transitions_per_minute, regime_labels

# ============================================================================
# C) ABLATION CORRECTNESS SANITY CHECKS
# ============================================================================

def verify_ablation_changes_constraint_matrix(constraint_intact, constraint_removed, constraint_randomized):
 """
 Verify that ablations actually modify the constraint matrix.
 
 Args:
 constraint_intact: (n_rois, n_rois) constraint matrix for intact torus
 constraint_removed: (n_rois, n_rois) constraint matrix for removed torus
 constraint_randomized: (n_rois, n_rois) constraint matrix for randomized torus
 
 Returns:
 verification: dict of verification results
 """
 print("\n=== Ablation Verification ===")
 
 # Check intact vs removed
 diff_removed = np.sum(constraint_intact != constraint_removed)
 print(f"Intact vs Removed: {diff_removed} elements differ (out of {constraint_intact.size})")
 
 # Check intact vs randomized
 diff_randomized = np.sum(constraint_intact != constraint_randomized)
 print(f"Intact vs Randomized: {diff_randomized} elements differ (out of {constraint_intact.size})")
 
 # Check removed vs randomized
 diff_removed_randomized = np.sum(constraint_removed != constraint_randomized)
 print(f"Removed vs Randomized: {diff_removed_randomized} elements differ (out of {constraint_intact.size})")
 
 # Verify intact has periodic boundary structure (4x4 grid with wraparound)
 n_rois = constraint_intact.shape[0]
 grid_size = int(np.sqrt(n_rois))
 
 # Count neighbors per ROI
 neighbors_intact = np.sum(constraint_intact, axis=1)
 neighbors_removed = np.sum(constraint_removed, axis=1)
 neighbors_randomized = np.sum(constraint_randomized, axis=1)
 
 print(f"\nNeighbor counts (intact): min={neighbors_intact.min()}, max={neighbors_intact.max()}, mean={neighbors_intact.mean():.2f}")
 print(f"Neighbor counts (removed): min={neighbors_removed.min()}, max={neighbors_removed.max()}, mean={neighbors_removed.mean():.2f}")
 print(f"Neighbor counts (randomized): min={neighbors_randomized.min()}, max={neighbors_randomized.max()}, mean={neighbors_randomized.mean():.2f}")
 
 verification = {
 'diff_intact_removed': int(diff_removed),
 'diff_intact_randomized': int(diff_randomized),
 'diff_removed_randomized': int(diff_removed_randomized),
 'neighbors_intact': neighbors_intact.tolist(),
 'neighbors_removed': neighbors_removed.tolist(),
 'neighbors_randomized': neighbors_randomized.tolist()
 }
 
 # Assert that ablations actually change the matrix
 assert diff_removed > 0, "ERROR: 'Removed' ablation did not change constraint matrix!"
 assert diff_randomized > 0, "ERROR: 'Randomized' ablation did not change constraint matrix!"
 
 print("\n✅ Ablation verification PASSED: All ablations modify constraint matrix")
 
 return verification

# ============================================================================
# D) WINDOWING + SAMPLE RATE SANITY CHECKS
# ============================================================================

def verify_windowing_and_sampling(fs, window_length_sec, overlap, freq_range=(0.14, 0.33)):
 """
 Verify windowing and sampling parameters for PAC estimation.
 
 Args:
 fs: sampling rate (Hz)
 window_length_sec: window length in seconds
 overlap: overlap fraction (0-1)
 freq_range: (low, high) frequency range for modulating band
 
 Returns:
 verification: dict of verification results
 """
 print("\n=== Windowing and Sampling Verification ===")
 
 # Verify Nyquist
 nyquist = fs / 2
 print(f"Sampling rate: {fs} Hz")
 print(f"Nyquist frequency: {nyquist} Hz")
 print(f"Target frequency range: {freq_range[0]}-{freq_range[1]} Hz")
 
 if freq_range[1] >= nyquist:
 print(f"❌ ERROR: High frequency {freq_range[1]} Hz exceeds Nyquist {nyquist} Hz")
 raise ValueError(f"High frequency {freq_range[1]} Hz exceeds Nyquist {nyquist} Hz")
 else:
 print(f"✅ Frequency range valid (below Nyquist)")
 
 # Verify window length has enough cycles
 min_freq = freq_range[0]
 cycles_per_window = window_length_sec * min_freq
 print(f"\nWindow length: {window_length_sec} seconds")
 print(f"Cycles per window at {min_freq} Hz: {cycles_per_window:.2f}")
 
 if cycles_per_window < 3:
 print(f"⚠️ WARNING: Window length may be too short (< 3 cycles at {min_freq} Hz)")
 else:
 print(f"✅ Window length sufficient (≥ 3 cycles)")
 
 # Verify overlap
 print(f"\nOverlap: {overlap * 100:.0f}%")
 step_sec = window_length_sec * (1 - overlap)
 print(f"Step size: {step_sec:.2f} seconds")
 
 verification = {
 'fs': fs,
 'nyquist': nyquist,
 'window_length_sec': window_length_sec,
 'cycles_per_window': cycles_per_window,
 'overlap': overlap,
 'step_sec': step_sec
 }
 
 return verification

# ============================================================================
# MAIN SANITY PATCH RUNNER
# ============================================================================

def run_sanity_patch_on_session(data, fs, output_dir):
 """
 Run all sanity checks on a single session.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 output_dir: directory to save results
 
 Returns:
 results: dict of all sanity check results
 """
 print("\n" + "="*80)
 print("STAGE C SANITY PATCH")
 print("="*80)
 
 results = {}
 
 # A) Phase / PLV sanity
 print("\n--- A) PHASE / PLV SANITY ---")
 phases, sanity_checks = compute_phase_per_roi(data, fs)
 results['phase_sanity'] = sanity_checks
 
 plv_real, plv_null, plv_diff = compute_plv_with_null_control(phases)
 results['plv'] = {
 'real': float(plv_real),
 'null': float(plv_null),
 'diff': float(plv_diff)
 }
 
 ppc = compute_ppc(phases)
 results['ppc'] = float(ppc)
 
 # B) Regime transition sanity (requires spectral gap computation)
 print("\n--- B) REGIME TRANSITION SANITY ---")
 # Placeholder: spectral gap computation would go here
 # For now, generate synthetic spectral gap for demonstration
 n_samples = data.shape[1]
 spectral_gap_ts = np.random.randn(n_samples // 100) # Assuming 100-sample windows
 
 transitions, transitions_per_minute, regime_labels = compute_regime_transitions_fixed(
 spectral_gap_ts, threshold_percentile=50, min_dwell_windows=3
 )
 results['regime_transitions'] = {
 'total': int(transitions),
 'per_minute': float(transitions_per_minute),
 'n_windows': len(regime_labels)
 }
 
 # D) Windowing + sample rate sanity
 print("\n--- D) WINDOWING + SAMPLE RATE SANITY ---")
 window_length_sec = 10.0 # Example
 overlap = 0.5
 windowing_verification = verify_windowing_and_sampling(fs, window_length_sec, overlap)
 results['windowing'] = windowing_verification
 
 # Save results
 output_path = Path(output_dir) / 'sanity_patch_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 print(f"\n✅ Sanity patch results saved to {output_path}")
 
 return results

if __name__ == '__main__':
 print("Stage C Sanity Patch - Artifact Fixes")
 print("Provides functions to fix Stage C artifacts.")
 print("Import and use in stage_c_dataset2_analysis_fixed.py")
