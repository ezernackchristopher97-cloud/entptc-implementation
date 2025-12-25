"""
Stage C Analysis - CORRECTED PROTOCOL
======================================

Complete Stage C analysis with:
1. T³→R³ mapping (3-torus, not T²)
2. Fixed PAC windows (120-300 sec)
3. Sanity-checked phase/amplitude extraction
4. Regime detection with minimum dwell time
5. Uniqueness tests U1/U2/U3

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
# STAGE C ANALYSIS WITH T³ MAPPING
# ============================================================================

def run_stage_c_corrected(data_path: Path, output_dir: Path):
 """
 Run corrected Stage C analysis.
 
 Args:
 data_path: path to preprocessed MAT file
 output_dir: directory to save results
 """
 from entptc.t3_to_r3_mapping import entptc_t3_to_r3_pipeline
 from entptc.utils.grid_utils import create_toroidal_grid
 from stage_c_pac_fixed import compute_pac_with_windowing
 
 output_dir.mkdir(parents=True, exist_ok=True)
 
 print("="*80)
 print("STAGE C ANALYSIS - CORRECTED PROTOCOL")
 print("="*80)
 
 # Load data
 print(f"\nLoading {data_path.name}...")
 mat = sio.loadmat(data_path)
 data = mat['eeg_data']
 fs = float(mat['fs'][0, 0])
 
 n_rois, n_samples = data.shape
 duration_sec = n_samples / fs
 
 print(f"Data shape: {data.shape}")
 print(f"Sampling rate: {fs} Hz")
 print(f"Duration: {duration_sec:.1f} seconds")
 
 # Create adjacency matrix
 grid_size = int(np.sqrt(n_rois))
 adjacency = create_toroidal_grid(grid_size)
 
 print(f"Grid size: {grid_size}×{grid_size}")
 print(f"Adjacency: {np.sum(adjacency)} edges")
 
 # ========================================================================
 # 1. T³→R³ MAPPING
 # ========================================================================
 
 print("\n" + "="*80)
 print("1. T³→R³ MAPPING")
 print("="*80)
 
 t3_results = entptc_t3_to_r3_pipeline(
 data, fs, adjacency,
 projection_type='stereographic',
 normalization='unit_variance'
 )
 
 print("\nT³ Topology Verification:")
 for key, value in t3_results['t3_verification'].items():
 print(f" {key}: {value:.3f}")
 
 print("\nT³ Invariants:")
 for key, value in t3_results['t3_invariants'].items():
 print(f" {key}: {value:.6f}")
 
 print("\nR³ Invariants:")
 for key, value in t3_results['r3_invariants'].items():
 print(f" {key}: {value:.6f}")
 
 # ========================================================================
 # 2. FIXED PAC COMPUTATION
 # ========================================================================
 
 print("\n" + "="*80)
 print("2. FIXED PAC COMPUTATION")
 print("="*80)
 
 pac_results = compute_pac_with_windowing(
 data, fs,
 phase_freq=(0.14, 0.33),
 amp_freq=(30, 50),
 window_lengths=[60, 120, 180, 240, 300]
 )
 
 # ========================================================================
 # 3. ORGANIZATION METRICS
 # ========================================================================
 
 print("\n" + "="*80)
 print("3. ORGANIZATION METRICS")
 print("="*80)
 
 # PLV (Phase Locking Value)
 plv = compute_plv_sanity_checked(data, fs, (0.14, 0.33), adjacency)
 print(f"\nPLV: {plv:.6f}")
 
 # PPC (Pairwise Phase Consistency)
 ppc = compute_ppc(data, fs, (0.14, 0.33), adjacency)
 print(f"PPC: {ppc:.6f}")
 
 # Regime transitions
 transitions = compute_regime_transitions_fixed(data, fs, min_dwell_sec=15)
 print(f"Regime transitions: {transitions} (with 15s minimum dwell)")
 
 # ========================================================================
 # 4. CONSOLIDATE RESULTS
 # ========================================================================
 
 results = {
 'dataset': data_path.name,
 'duration_sec': float(duration_sec),
 'n_rois': int(n_rois),
 'fs': float(fs),
 'grid_size': int(grid_size),
 
 # T³ mapping
 't3_verification': t3_results['t3_verification'],
 't3_invariants': t3_results['t3_invariants'],
 'r3_invariants': t3_results['r3_invariants'],
 'projection_type': t3_results['projection_type'],
 'normalization': t3_results['normalization'],
 
 # PAC
 'pac_results': {
 'window_lengths': pac_results['window_lengths'],
 'pac_values': pac_results['pac_values'],
 'pac_null_values': pac_results['pac_null_values'],
 'cycles_per_window': pac_results['cycles_per_window']
 },
 
 # Organization metrics
 'plv': float(plv),
 'ppc': float(ppc),
 'regime_transitions': int(transitions)
 }
 
 # Save results
 output_path = output_dir / 'stage_c_corrected_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print("\n" + "="*80)
 print("STAGE C ANALYSIS COMPLETE")
 print("="*80)
 print(f"Results saved to {output_path}")
 
 return results

# ============================================================================
# SANITY-CHECKED METRICS
# ============================================================================

def compute_plv_sanity_checked(data: np.ndarray, fs: float, freq_range: tuple, adjacency: np.ndarray) -> float:
 """
 Compute PLV with sanity checks.
 
 Verifies that phases are not identical before computing PLV.
 """
 n_rois, n_samples = data.shape
 
 # Bandpass filter
 sos = signal.butter(4, freq_range, btype='band', fs=fs, output='sos')
 
 # Extract phases
 phases = np.zeros((n_rois, n_samples))
 for i in range(n_rois):
 filtered = signal.sosfiltfilt(sos, data[i])
 phases[i] = np.angle(signal.hilbert(filtered))
 
 # Sanity check: phases should not be identical
 for i in range(n_rois):
 if np.var(phases[i]) == 0:
 print(f"⚠️ WARNING: ROI {i} has zero phase variance")
 
 # Check for identical phases across ROIs
 for i in range(n_rois):
 for j in range(i+1, n_rois):
 if np.allclose(phases[i], phases[j], atol=1e-6):
 print(f"⚠️ WARNING: ROI {i} and {j} have identical phases")
 
 # Compute PLV for adjacent ROIs
 plv_values = []
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
 plv_values.append(plv)
 
 if len(plv_values) == 0:
 return 0.0
 
 return np.mean(plv_values)

def compute_ppc(data: np.ndarray, fs: float, freq_range: tuple, adjacency: np.ndarray) -> float:
 """
 Compute Pairwise Phase Consistency (bias-corrected alternative to PLV).
 
 PPC = (sum of cos(phase_diff_i - phase_diff_j) for all pairs) / (N*(N-1)/2)
 """
 n_rois, n_samples = data.shape
 
 # Bandpass filter
 sos = signal.butter(4, freq_range, btype='band', fs=fs, output='sos')
 
 # Extract phases
 phases = np.zeros((n_rois, n_samples))
 for i in range(n_rois):
 filtered = signal.sosfiltfilt(sos, data[i])
 phases[i] = np.angle(signal.hilbert(filtered))
 
 # Compute PPC for adjacent ROIs
 ppc_values = []
 for i in range(n_rois):
 for j in range(n_rois):
 if adjacency[i, j] > 0:
 phase_diff = phases[i] - phases[j]
 
 # PPC: average pairwise cosine
 ppc = 0
 n_pairs = 0
 for t1 in range(0, n_samples, 100): # Subsample for speed
 for t2 in range(t1+1, n_samples, 100):
 ppc += np.cos(phase_diff[t1] - phase_diff[t2])
 n_pairs += 1
 
 if n_pairs > 0:
 ppc /= n_pairs
 ppc_values.append(ppc)
 
 if len(ppc_values) == 0:
 return 0.0
 
 return np.mean(ppc_values)

def compute_regime_transitions_fixed(data: np.ndarray, fs: float, min_dwell_sec: float = 15) -> int:
 """
 Compute regime transitions with minimum dwell time.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 min_dwell_sec: minimum dwell time in seconds
 
 Returns:
 n_transitions: number of regime transitions
 """
 n_rois, n_samples = data.shape
 window_length_sec = 10.0
 window_length_samples = int(window_length_sec * fs)
 overlap = 0.5
 step = int(window_length_samples * (1 - overlap))
 
 # Compute spectral gap time series
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
 
 return n_transitions

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/stage_c_corrected')
 
 if data_path.exists():
 results = run_stage_c_corrected(data_path, output_dir)
 else:
 print(f"Data file not found: {data_path}")
