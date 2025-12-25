#!/usr/bin/env python3.11
"""
Stage C: Dataset 2 (ds004706 Spatial Navigation) Analysis
==========================================================

Geometry-anchored task EEG projection test with strict C1-C3 criteria:

C1 - Gating: Does geometry-derived slow mode modulate higher frequencies or event structure?
C2 - Organization: Does it organize phase relationships (geometry-sensitive observables)?
C3 - Regime Timing: Does it correlate with task events and regime transitions?

All tests include topology ablations (intact, removed, randomized).

"""

import numpy as np
import scipy.io
import scipy.signal
import scipy.stats
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

# Paths
DATA_DIR = Path("/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706")
OUTPUT_DIR = Path("/home/ubuntu/entptc-implementation/stage_c_outputs_dataset2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load toroidal topology
TOPO_DIR = Path("/home/ubuntu/entptc-implementation/outputs/toroidal_topology")
TOROIDAL_ADJ = np.load(TOPO_DIR / "toroidal_adjacency_matrix.npy")
TOROIDAL_DIST = np.load(TOPO_DIR / "toroidal_distance_matrix.npy")

# Parameters from Stage B
CONTROL_FREQ_RANGE = (0.14, 0.33) # Hz (from Stage B robustness checks)
ALPHA_RANGE = (8, 13) # Hz (for PAC testing)
TARGET_FS = 160 # Hz

def load_eeg_data(mat_path):
 """Load preprocessed EEG data"""
 data = scipy.io.loadmat(mat_path)
 eeg = data['eeg_data'] # Shape: (16 ROIs, n_samples)
 fs = float(data['fs'][0, 0])
 duration = float(data['duration_seconds'][0, 0])
 
 # Transpose to (n_samples, 16 ROIs)
 eeg = eeg.T
 
 return eeg, fs, duration

def extract_control_mode(eeg, fs):
 """Extract geometry-derived control mode (0.14-0.33 Hz)"""
 # Bandpass filter
 sos = scipy.signal.butter(4, CONTROL_FREQ_RANGE, 'bandpass', fs=fs, output='sos')
 control_mode = scipy.signal.sosfiltfilt(sos, eeg, axis=0)
 
 # Hilbert transform for phase and envelope
 analytic = scipy.signal.hilbert(control_mode, axis=0)
 phase = np.angle(analytic)
 envelope = np.abs(analytic)
 
 return control_mode, phase, envelope

def extract_alpha_band(eeg, fs):
 """Extract alpha band (8-13 Hz) for PAC testing"""
 sos = scipy.signal.butter(4, ALPHA_RANGE, 'bandpass', fs=fs, output='sos')
 alpha = scipy.signal.sosfiltfilt(sos, eeg, axis=0)
 
 # Envelope
 analytic = scipy.signal.hilbert(alpha, axis=0)
 alpha_envelope = np.abs(analytic)
 
 return alpha, alpha_envelope

def compute_pac(phase_low, amp_high):
 """Compute phase-amplitude coupling (Mean Vector Length method)"""
 # Bin phases
 n_bins = 18
 phase_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
 
 pac_values = []
 for roi in range(phase_low.shape[1]):
 # Get phase and amplitude for this ROI
 phase = phase_low[:, roi]
 amp = amp_high[:, roi]
 
 # Compute mean amplitude in each phase bin
 bin_amps = []
 for i in range(n_bins):
 mask = (phase >= phase_bins[i]) & (phase < phase_bins[i+1])
 if np.sum(mask) > 0:
 bin_amps.append(np.mean(amp[mask]))
 else:
 bin_amps.append(0)
 
 # Normalize
 bin_amps = np.array(bin_amps)
 if np.sum(bin_amps) > 0:
 bin_amps = bin_amps / np.sum(bin_amps)
 
 # Compute modulation index (MVL)
 angles = np.linspace(-np.pi, np.pi, n_bins, endpoint=False) + np.pi/n_bins
 mvl = np.abs(np.sum(bin_amps * np.exp(1j * angles)))
 pac_values.append(mvl)
 
 return np.mean(pac_values)

def compute_phase_locking(phase1, phase2):
 """Compute phase locking value between two signals"""
 phase_diff = phase1 - phase2
 plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
 return np.mean(plv)

def compute_trajectory_alignment(eeg, toroidal_adj):
 """Geometry-sensitive: Measure trajectory alignment with toroidal structure"""
 # Compute pairwise correlations
 corr_matrix = np.corrcoef(eeg.T)
 
 # Weight by toroidal adjacency
 weighted_corr = corr_matrix * toroidal_adj
 
 # Mean alignment
 alignment = np.sum(weighted_corr) / np.sum(toroidal_adj)
 
 return alignment

def compute_phase_winding(phase, toroidal_adj):
 """Geometry-sensitive: Measure phase winding around torus"""
 # Compute phase differences along toroidal edges
 n_rois = phase.shape[1]
 winding = []
 
 for i in range(n_rois):
 for j in range(i+1, n_rois):
 if toroidal_adj[i, j] > 0: # Adjacent in torus
 phase_diff = np.abs(phase[:, i] - phase[:, j])
 # Wrap to [-pi, pi]
 phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
 winding.append(np.std(phase_diff))
 
 return np.mean(winding) if winding else 0.0

def detect_regime_transitions(entropy_series, threshold=0.5):
 """Detect regime transitions based on entropy changes"""
 # Compute entropy derivative
 entropy_diff = np.diff(entropy_series)
 
 # Find transitions (large changes)
 transitions = np.where(np.abs(entropy_diff) > threshold)[0]
 
 return transitions

def compute_entropy_timeseries(eeg, window_size=160):
 """Compute entropy over sliding windows"""
 n_samples = eeg.shape[0]
 n_windows = n_samples // window_size
 
 entropy = []
 for i in range(n_windows):
 window = eeg[i*window_size:(i+1)*window_size, :]
 # Shannon entropy of amplitude distribution
 hist, _ = np.histogram(window.flatten(), bins=50, density=True)
 hist = hist[hist > 0]
 ent = -np.sum(hist * np.log(hist))
 entropy.append(ent)
 
 return np.array(entropy)

def apply_topology_ablation(eeg, ablation_type):
 """Apply topology ablation to EEG data"""
 if ablation_type == 'intact':
 return eeg, TOROIDAL_ADJ
 
 elif ablation_type == 'removed':
 # Remove toroidal structure (use identity adjacency)
 adj = np.eye(16)
 return eeg, adj
 
 elif ablation_type == 'randomized':
 # Randomize adjacency (preserve degree distribution)
 adj = TOROIDAL_ADJ.copy()
 np.random.seed(42)
 for i in range(16):
 neighbors = np.where(adj[i, :] > 0)[0]
 if len(neighbors) > 0:
 np.random.shuffle(neighbors)
 adj[i, :] = 0
 adj[i, neighbors] = 1
 return eeg, adj
 
 else:
 raise ValueError(f"Unknown ablation type: {ablation_type}")

def analyze_single_file(mat_path, ablation_type='intact'):
 """Analyze a single EEG file with specified topology ablation"""
 # Load data
 eeg, fs, duration = load_eeg_data(mat_path)
 
 # Apply ablation
 eeg_abl, adj_abl = apply_topology_ablation(eeg, ablation_type)
 
 # Extract control mode
 control_mode, control_phase, control_envelope = extract_control_mode(eeg_abl, fs)
 
 # Extract alpha band
 alpha, alpha_envelope = extract_alpha_band(eeg_abl, fs)
 
 # C1: Gating
 pac = compute_pac(control_phase, alpha_envelope)
 
 # C2: Organization
 plv = compute_phase_locking(control_phase, control_phase)
 trajectory_alignment = compute_trajectory_alignment(eeg_abl, adj_abl)
 phase_winding = compute_phase_winding(control_phase, adj_abl)
 
 # C3: Regime Timing
 entropy_ts = compute_entropy_timeseries(eeg_abl)
 transitions = detect_regime_transitions(entropy_ts)
 
 # Compute correlation between control envelope and entropy
 # Downsample control envelope to match entropy timeseries
 window_size = 160
 n_windows = len(entropy_ts)
 control_env_downsampled = []
 for i in range(n_windows):
 window = control_envelope[i*window_size:(i+1)*window_size, :]
 control_env_downsampled.append(np.mean(window))
 control_env_downsampled = np.array(control_env_downsampled)
 
 if len(control_env_downsampled) == len(entropy_ts):
 regime_corr = np.corrcoef(control_env_downsampled, entropy_ts)[0, 1]
 else:
 regime_corr = 0.0
 
 return {
 'filename': mat_path.name,
 'ablation': ablation_type,
 'duration': duration,
 'c1_pac': pac,
 'c2_plv': plv,
 'c2_trajectory_alignment': trajectory_alignment,
 'c2_phase_winding': phase_winding,
 'c3_regime_corr': regime_corr,
 'c3_n_transitions': len(transitions)
 }

def main():
 print("="*80)
 print("STAGE C: DATASET 2 (DS004706 SPATIAL NAVIGATION) ANALYSIS")
 print("="*80)
 
 # Find all MAT files
 mat_files = sorted(DATA_DIR.glob("*.mat"))
 print(f"\nFound {len(mat_files)} MAT files\n")
 
 # Analyze with all three ablation types
 results = []
 
 for ablation in ['intact', 'removed', 'randomized']:
 print(f"\n{'='*80}")
 print(f"ABLATION: {ablation.upper()}")
 print(f"{'='*80}\n")
 
 for mat_file in tqdm(mat_files, desc=f"Processing ({ablation})"):
 try:
 result = analyze_single_file(mat_file, ablation)
 results.append(result)
 except Exception as e:
 print(f"Error processing {mat_file.name}: {e}")
 
 # Convert to DataFrame
 df = pd.DataFrame(results)
 
 # Save results
 csv_path = OUTPUT_DIR / "dataset2_results.csv"
 df.to_csv(csv_path, index=False)
 print(f"\n✅ Results saved to {csv_path}")
 
 # Compute summary statistics
 summary = {}
 for ablation in ['intact', 'removed', 'randomized']:
 df_abl = df[df['ablation'] == ablation]
 summary[ablation] = {
 'c1_pac_mean': float(df_abl['c1_pac'].mean()),
 'c1_pac_std': float(df_abl['c1_pac'].std()),
 'c2_plv_mean': float(df_abl['c2_plv'].mean()),
 'c2_plv_std': float(df_abl['c2_plv'].std()),
 'c2_trajectory_alignment_mean': float(df_abl['c2_trajectory_alignment'].mean()),
 'c2_trajectory_alignment_std': float(df_abl['c2_trajectory_alignment'].std()),
 'c2_phase_winding_mean': float(df_abl['c2_phase_winding'].mean()),
 'c2_phase_winding_std': float(df_abl['c2_phase_winding'].std()),
 'c3_regime_corr_mean': float(df_abl['c3_regime_corr'].mean()),
 'c3_regime_corr_std': float(df_abl['c3_regime_corr'].std()),
 'c3_n_transitions_mean': float(df_abl['c3_n_transitions'].mean()),
 'c3_n_transitions_std': float(df_abl['c3_n_transitions'].std())
 }
 
 # Save summary
 json_path = OUTPUT_DIR / "dataset2_summary.json"
 with open(json_path, 'w') as f:
 json.dump(summary, f, indent=2)
 print(f"✅ Summary saved to {json_path}")
 
 # Print summary
 print("\n" + "="*80)
 print("SUMMARY")
 print("="*80)
 for ablation in ['intact', 'removed', 'randomized']:
 print(f"\n{ablation.upper()}:")
 s = summary[ablation]
 print(f" C1 (PAC): {s['c1_pac_mean']:.3f} ± {s['c1_pac_std']:.3f}")
 print(f" C2 (PLV): {s['c2_plv_mean']:.3f} ± {s['c2_plv_std']:.3f}")
 print(f" C2 (Trajectory Alignment): {s['c2_trajectory_alignment_mean']:.3f} ± {s['c2_trajectory_alignment_std']:.3f}")
 print(f" C2 (Phase Winding): {s['c2_phase_winding_mean']:.3f} ± {s['c2_phase_winding_std']:.3f}")
 print(f" C3 (Regime Corr): {s['c3_regime_corr_mean']:.3f} ± {s['c3_regime_corr_std']:.3f}")
 print(f" C3 (Transitions): {s['c3_n_transitions_mean']:.1f} ± {s['c3_n_transitions_std']:.1f}")
 
 print("\n" + "="*80)

if __name__ == "__main__":
 main()
