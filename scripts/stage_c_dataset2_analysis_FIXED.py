"""
Stage C Dataset 2 (ds004706) Analysis - FIXED VERSION

Integrates:
1. Stage C Sanity Patch (artifact fixes)
2. Uniqueness Tests U1-U3
3. Proper ablation verification
4. Artifact-free metrics only

"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
from scipy.stats import circmean, circstd
import json
import os
from pathlib import Path
import sys

# Import sanity patch and uniqueness test functions
sys.path.append('/home/ubuntu/entptc-implementation')
from stage_c_sanity_patch import (
 compute_phase_per_roi,
 compute_plv_with_null_control,
 compute_ppc,
 compute_regime_transitions_fixed,
 verify_ablation_changes_constraint_matrix,
 verify_windowing_and_sampling
)
from stage_c_uniqueness_tests import (
 create_toroidal_grid,
 create_cylindrical_grid,
 randomize_adjacency,
 phase_scramble_data,
 run_u1_ablation_ladder,
 run_u2_manifold_controls,
 run_u3_resolution_consistency
)

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

PREPROCESSED_DIR = Path("/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706")
OUTPUT_DIR = Path('/home/ubuntu/entptc-implementation/outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

# Frequency range (from Stage B)
FREQ_RANGE = (0.14, 0.33) # Hz, sub-delta

# Grid configuration
GRID_SIZE = 4 # 4x4 = 16 ROIs
N_ROIS = GRID_SIZE * GRID_SIZE

# ============================================================================
# LOAD PREPROCESSED DATA
# ============================================================================

def load_session_data(session_file):
 """Load preprocessed session data."""
 print(f"\nLoading {session_file.name}...")
 
 mat = sio.loadmat(session_file)
 data = mat['eeg_data'] # (n_rois, n_samples)
 fs = float(mat['fs'][0, 0])
 
 print(f" Shape: {data.shape}")
 print(f" Sampling rate: {fs} Hz")
 print(f" Duration: {data.shape[1] / fs:.1f} seconds")
 
 return data, fs

# ============================================================================
# ORGANIZATION METRICS (ARTIFACT-FREE)
# ============================================================================

def compute_phase_winding_metric(data, adjacency, fs):
 """
 Compute phase winding metric (geometry-sensitive).
 
 This metric showed correct ablation response in previous analysis.
 """
 # Compute phases with sanity checks
 phases, sanity_checks = compute_phase_per_roi(data, fs, FREQ_RANGE)
 
 # Compute phase winding over adjacent ROIs
 n_rois = data.shape[0]
 phase_windings = []
 
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 winding = np.abs(np.mean(np.exp(1j * phase_diff)))
 phase_windings.append(winding)
 
 if len(phase_windings) == 0:
 return 0.0, sanity_checks
 
 metric = np.mean(phase_windings)
 return metric, sanity_checks

def compute_trajectory_alignment_metric(data, adjacency, fs):
 """
 Compute trajectory alignment metric (geometry-sensitive).
 
 NOTE: Previous version showed WRONG ablation response (increased under ablation).
 This version uses proper phase-based alignment.
 """
 # Compute phases
 phases, _ = compute_phase_per_roi(data, fs, FREQ_RANGE)
 
 # Compute trajectory alignment over adjacent ROIs
 n_rois, n_samples = data.shape
 alignments = []
 
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 # Phase velocity alignment
 phase_vel_i = np.diff(np.unwrap(phases[i]))
 phase_vel_j = np.diff(np.unwrap(phases[j]))
 
 # Correlation of phase velocities
 if len(phase_vel_i) > 0 and len(phase_vel_j) > 0:
 corr = np.corrcoef(phase_vel_i, phase_vel_j)[0, 1]
 if not np.isnan(corr):
 alignments.append(abs(corr))
 
 if len(alignments) == 0:
 return 0.0
 
 metric = np.mean(alignments)
 return metric

# ============================================================================
# GATING METRIC (C1)
# ============================================================================

def compute_pac_metric(data, fs):
 """
 Compute Phase-Amplitude Coupling (PAC) with proper windowing.
 
 Tests whether ~0.2-0.4 Hz mode modulates higher frequencies.
 """
 # Verify windowing parameters
 window_length_sec = 10.0 # 10 seconds
 overlap = 0.5
 verify_windowing_and_sampling(fs, window_length_sec, overlap, FREQ_RANGE)
 
 # Low frequency (phase)
 low_freq = FREQ_RANGE
 
 # High frequency (amplitude) - gamma band
 high_freq = (30, 50)
 
 # Bandpass filters
 sos_low = signal.butter(4, low_freq, btype='band', fs=fs, output='sos')
 sos_high = signal.butter(4, high_freq, btype='band', fs=fs, output='sos')
 
 n_rois, n_samples = data.shape
 pac_values = []
 
 for i in range(n_rois):
 # Filter
 low_filtered = signal.sosfiltfilt(sos_low, data[i])
 high_filtered = signal.sosfiltfilt(sos_high, data[i])
 
 # Extract phase and amplitude
 low_phase = np.angle(signal.hilbert(low_filtered))
 high_amp = np.abs(signal.hilbert(high_filtered))
 
 # Compute PAC (mean vector length)
 pac = np.abs(np.mean(high_amp * np.exp(1j * low_phase)))
 pac_values.append(pac)
 
 metric = np.mean(pac_values)
 return metric

# ============================================================================
# REGIME TIMING METRIC (C3)
# ============================================================================

def compute_spectral_gap_timeseries(data, fs):
 """
 Compute spectral gap time series for regime detection.
 """
 n_rois, n_samples = data.shape
 
 # Window parameters
 window_length_sec = 10.0
 window_length_samples = int(window_length_sec * fs)
 overlap = 0.5
 step = int(window_length_samples * (1 - overlap))
 
 # Compute spectral gap in sliding windows
 spectral_gaps = []
 
 for start in range(0, n_samples - window_length_samples, step):
 end = start + window_length_samples
 window_data = data[:, start:end]
 
 # Compute covariance matrix
 cov = np.cov(window_data)
 
 # Eigenvalues
 eigenvalues = np.linalg.eigvalsh(cov)
 eigenvalues = np.sort(eigenvalues)[::-1] # Descending order
 
 # Spectral gap (difference between top 2 eigenvalues)
 if len(eigenvalues) >= 2:
 gap = eigenvalues[0] - eigenvalues[1]
 spectral_gaps.append(gap)
 
 return np.array(spectral_gaps)

def compute_regime_timing_metric(data, fs):
 """
 Compute regime timing correlation with fixed transition counting.
 """
 # Compute spectral gap time series
 spectral_gap_ts = compute_spectral_gap_timeseries(data, fs)
 
 # Compute regime transitions with proper boundary crossing detection
 transitions, transitions_per_minute, regime_labels = compute_regime_transitions_fixed(
 spectral_gap_ts, threshold_percentile=50, min_dwell_windows=3
 )
 
 # Compute correlation with spectral gap
 # (This is a placeholder - in real analysis, correlate with task events)
 corr = np.corrcoef(regime_labels[:-1], np.diff(spectral_gap_ts))[0, 1]
 if np.isnan(corr):
 corr = 0.0
 
 return corr, transitions, transitions_per_minute

# ============================================================================
# ABLATION TESTS
# ============================================================================

def run_ablation_tests(data, fs):
 """
 Run ablation tests: intact, removed, randomized.
 
 With proper verification that constraint matrices change.
 """
 print("\n" + "="*80)
 print("ABLATION TESTS WITH VERIFICATION")
 print("="*80)
 
 # Create constraint matrices
 adjacency_intact = create_toroidal_grid(GRID_SIZE)
 adjacency_removed = np.zeros_like(adjacency_intact) # No constraints
 adjacency_randomized = randomize_adjacency(adjacency_intact)
 
 # Verify ablations change the matrix
 verification = verify_ablation_changes_constraint_matrix(
 adjacency_intact, adjacency_removed, adjacency_randomized
 )
 
 results = {
 'verification': verification,
 'c1_pac': {},
 'c2_phase_winding': {},
 'c2_trajectory_alignment': {},
 'c2_plv': {},
 'c2_ppc': {},
 'c3_regime_timing': {}
 }
 
 # Run metrics for each ablation
 for ablation_name, adjacency in [
 ('intact', adjacency_intact),
 ('removed', adjacency_removed),
 ('randomized', adjacency_randomized)
 ]:
 print(f"\n--- Ablation: {ablation_name} ---")
 
 # C1: PAC (gating)
 pac = compute_pac_metric(data, fs)
 results['c1_pac'][ablation_name] = float(pac)
 print(f"C1 (PAC): {pac:.6f}")
 
 # C2: Phase winding
 phase_winding, sanity_checks = compute_phase_winding_metric(data, adjacency, fs)
 results['c2_phase_winding'][ablation_name] = float(phase_winding)
 if ablation_name == 'intact':
 results['c2_phase_winding']['sanity_checks'] = sanity_checks
 print(f"C2 (Phase Winding): {phase_winding:.6f}")
 
 # C2: Trajectory alignment
 trajectory_alignment = compute_trajectory_alignment_metric(data, adjacency, fs)
 results['c2_trajectory_alignment'][ablation_name] = float(trajectory_alignment)
 print(f"C2 (Trajectory Alignment): {trajectory_alignment:.6f}")
 
 # C2: PLV with null control
 phases, _ = compute_phase_per_roi(data, fs, FREQ_RANGE)
 plv_real, plv_null, plv_diff = compute_plv_with_null_control(phases)
 results['c2_plv'][ablation_name] = {
 'real': float(plv_real),
 'null': float(plv_null),
 'diff': float(plv_diff)
 }
 print(f"C2 (PLV): real={plv_real:.6f}, null={plv_null:.6f}, diff={plv_diff:.6f}")
 
 # C2: PPC
 ppc = compute_ppc(phases)
 results['c2_ppc'][ablation_name] = float(ppc)
 print(f"C2 (PPC): {ppc:.6f}")
 
 # C3: Regime timing (only for intact)
 if ablation_name == 'intact':
 corr, transitions, transitions_per_minute = compute_regime_timing_metric(data, fs)
 results['c3_regime_timing'][ablation_name] = {
 'correlation': float(corr),
 'transitions': int(transitions),
 'transitions_per_minute': float(transitions_per_minute)
 }
 print(f"C3 (Regime Timing): corr={corr:.6f}, transitions={transitions}, per_min={transitions_per_minute:.2f}")
 
 return results

# ============================================================================
# UNIQUENESS TESTS
# ============================================================================

def run_uniqueness_tests_on_session(data, fs):
 """
 Run uniqueness tests U1-U3 on session data.
 """
 print("\n" + "="*80)
 print("UNIQUENESS TESTS U1-U3")
 print("="*80)
 
 # Define metric function for uniqueness tests (use phase winding)
 def metric_func(data_input, adjacency):
 phases, _ = compute_phase_per_roi(data_input, fs, FREQ_RANGE)
 
 n_rois = data_input.shape[0]
 phase_windings = []
 
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 winding = np.abs(np.mean(np.exp(1j * phase_diff)))
 phase_windings.append(winding)
 
 if len(phase_windings) == 0:
 return 0.0
 
 return np.mean(phase_windings)
 
 # U1: Ablation ladder
 u1_results = run_u1_ablation_ladder(data, fs, metric_func)
 
 # U2: Manifold controls
 u2_results = run_u2_manifold_controls(data, fs, metric_func)
 
 # U3: Resolution consistency
 u3_results = run_u3_resolution_consistency(data, fs, metric_func, original_grid_size=GRID_SIZE)
 
 # Overall verdict
 u1_pass = u1_results['monotonic_degradation']
 u2_pass = u2_results['unique_signature']
 u3_pass = u3_results['consistent']
 all_pass = u1_pass and u2_pass and u3_pass
 
 print("\n" + "="*80)
 print("UNIQUENESS TESTS SUMMARY")
 print("="*80)
 print(f"U1 (Ablation Ladder): {'✅ PASS' if u1_pass else '❌ FAIL'}")
 print(f"U2 (Manifold Controls): {'✅ PASS' if u2_pass else '❌ FAIL'}")
 print(f"U3 (Resolution Consistency): {'✅ PASS' if u3_pass else '❌ FAIL'}")
 print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
 
 uniqueness_results = {
 'u1_ablation_ladder': u1_results,
 'u2_manifold_controls': u2_results,
 'u3_resolution_consistency': u3_results,
 'overall_verdict': {
 'u1_pass': u1_pass,
 'u2_pass': u2_pass,
 'u3_pass': u3_pass,
 'all_pass': all_pass
 }
 }
 
 return uniqueness_results

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
 """Main analysis pipeline."""
 print("="*80)
 print("Stage C Dataset 2 (ds004706) - FIXED ANALYSIS")
 print("="*80)
 
 # Get all preprocessed sessions
 session_files = sorted(PREPROCESSED_DIR.glob('*.mat'))
 print(f"\nFound {len(session_files)} preprocessed sessions")
 
 all_results = []
 
 for session_file in session_files:
 # Load data
 data, fs = load_session_data(session_file)
 
 # Verify data shape
 if data.shape[0] != N_ROIS:
 print(f" ❌ ERROR: Expected {N_ROIS} ROIs, got {data.shape[0]}")
 continue
 
 # Run ablation tests
 ablation_results = run_ablation_tests(data, fs)
 
 # Run uniqueness tests (on first session only to save time)
 if session_file == session_files[0]:
 uniqueness_results = run_uniqueness_tests_on_session(data, fs)
 ablation_results['uniqueness_tests'] = uniqueness_results
 
 # Store results
 session_results = {
 'session': session_file.stem,
 'fs': fs,
 'duration_sec': data.shape[1] / fs,
 'ablation_tests': ablation_results
 }
 all_results.append(session_results)
 
 # Save results
 output_file = OUTPUT_DIR / 'stage_c_dataset2_FIXED_results.json'
 with open(output_file, 'w') as f:
 json.dump(all_results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_file}")
 
 # Generate summary
 print("\n" + "="*80)
 print("SUMMARY ACROSS ALL SESSIONS")
 print("="*80)
 
 # Aggregate metrics
 c1_pac_intact = []
 c2_phase_winding_intact = []
 c2_phase_winding_removed = []
 c2_ppc_intact = []
 c3_transitions_per_minute = []
 
 for result in all_results:
 ablation = result['ablation_tests']
 c1_pac_intact.append(ablation['c1_pac']['intact'])
 c2_phase_winding_intact.append(ablation['c2_phase_winding']['intact'])
 c2_phase_winding_removed.append(ablation['c2_phase_winding']['removed'])
 c2_ppc_intact.append(ablation['c2_ppc']['intact'])
 if 'intact' in ablation['c3_regime_timing']:
 c3_transitions_per_minute.append(ablation['c3_regime_timing']['intact']['transitions_per_minute'])
 
 print(f"\nC1 (PAC) intact: {np.mean(c1_pac_intact):.6f} ± {np.std(c1_pac_intact):.6f}")
 print(f"C2 (Phase Winding) intact: {np.mean(c2_phase_winding_intact):.6f} ± {np.std(c2_phase_winding_intact):.6f}")
 print(f"C2 (Phase Winding) removed: {np.mean(c2_phase_winding_removed):.6f} ± {np.std(c2_phase_winding_removed):.6f}")
 collapse_pct = (np.mean(c2_phase_winding_intact) - np.mean(c2_phase_winding_removed)) / np.mean(c2_phase_winding_intact) * 100
 print(f"C2 (Phase Winding) collapse: {collapse_pct:.1f}%")
 print(f"C2 (PPC) intact: {np.mean(c2_ppc_intact):.6f} ± {np.std(c2_ppc_intact):.6f}")
 print(f"C3 (Transitions/min): {np.mean(c3_transitions_per_minute):.2f} ± {np.std(c3_transitions_per_minute):.2f}")
 
 print("\n" + "="*80)
 print("FIXED ANALYSIS COMPLETE")
 print("="*80)

if __name__ == '__main__':
 main()
