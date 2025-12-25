#!/usr/bin/env python3.11
"""
Stage C: Reframed Analysis with Proper Constraints

C1) Gating: Does geometry-derived slow mode modulate higher frequency power?
C2) Organization: Does it organize phase relationships beyond null controls?
C3) Regime Timing: Does it time regime transitions, and does this vanish when torus is ablated?

Dataset: PhysioNet Motor Movement EEG (30 files, 15 subjects, eyes-open + eyes-closed)
"""

import numpy as np
import scipy.io
import scipy.signal
from scipy.stats import ttest_ind, pearsonr
import json
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

# Paths
DATA_DIR = Path("/home/ubuntu/entptc-implementation/data/dataset_set_2")
OUTPUT_DIR = Path("/home/ubuntu/entptc-implementation/stage_c_outputs_reframed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load toroidal adjacency matrix
TORO_ADJ = np.load("/home/ubuntu/entptc-implementation/outputs/toroidal_topology/toroidal_adjacency_matrix.npy")

# EntPTC frequency range from Stage B
CONTROL_FREQ_MIN = 0.14 # Hz
CONTROL_FREQ_MAX = 0.33 # Hz

def load_eeg_data(mat_file):
 """Load EEG data from MAT file"""
 try:
 data = scipy.io.loadmat(mat_file)
 eeg = data['eeg_data'] # Shape: (16 ROIs, n_samples)
 eeg = eeg.T # Transpose to (n_samples, 16 ROIs)
 fs = 160 # Hz
 return eeg, fs
 except Exception as e:
 print(f"Error loading {mat_file}: {e}")
 return None, None

def aggregate_to_rois(eeg_64ch, fs=160):
 """Aggregate 64 channels to 16 ROIs"""
 # Simple spatial averaging (8 ROI groups × 2 hemispheres = 16 ROIs)
 roi_map = {
 0: list(range(0, 4)), # Frontal L
 1: list(range(4, 8)), # Frontal R
 2: list(range(8, 12)), # Central L
 3: list(range(12, 16)), # Central R
 4: list(range(16, 20)), # Parietal L
 5: list(range(20, 24)), # Parietal R
 6: list(range(24, 28)), # Occipital L
 7: list(range(28, 32)), # Occipital R
 8: list(range(32, 36)), # Temporal L
 9: list(range(36, 40)), # Temporal R
 10: list(range(40, 44)), # Frontal-Central L
 11: list(range(44, 48)), # Frontal-Central R
 12: list(range(48, 52)), # Central-Parietal L
 13: list(range(52, 56)), # Central-Parietal R
 14: list(range(56, 60)), # Parietal-Occipital L
 15: list(range(60, 64)), # Parietal-Occipital R
 }
 
 n_samples = eeg_64ch.shape[0]
 eeg_16roi = np.zeros((n_samples, 16))
 
 for roi_idx, ch_indices in roi_map.items():
 eeg_16roi[:, roi_idx] = np.mean(eeg_64ch[:, ch_indices], axis=1)
 
 return eeg_16roi

def extract_control_mode(eeg, fs, freq_min, freq_max):
 """Extract control mode in specified frequency range"""
 # Bandpass filter (use order 2 for short data)
 sos = scipy.signal.butter(2, [freq_min, freq_max], btype='band', fs=fs, output='sos')
 control_mode = scipy.signal.sosfiltfilt(sos, eeg, axis=0)
 return control_mode

def extract_higher_freq_power(eeg, fs, band_name='alpha'):
 """Extract power in higher frequency bands"""
 bands = {
 'theta': (4, 8),
 'alpha': (8, 13),
 'beta': (13, 30),
 'gamma': (30, 50)
 }
 
 freq_range = bands[band_name]
 sos = scipy.signal.butter(2, freq_range, btype='band', fs=fs, output='sos')
 filtered = scipy.signal.sosfiltfilt(sos, eeg, axis=0)
 
 # Hilbert envelope
 analytic = scipy.signal.hilbert(filtered, axis=0)
 envelope = np.abs(analytic)
 
 return envelope

def compute_pac(control_mode, higher_freq_envelope):
 """
 C1 Gating: Phase-Amplitude Coupling
 
 Does the phase of control mode modulate the amplitude of higher frequencies?
 """
 # Extract phase of control mode
 analytic_control = scipy.signal.hilbert(control_mode, axis=0)
 phase_control = np.angle(analytic_control)
 
 # Compute PAC using Mean Vector Length (MVL)
 pac_values = []
 
 for roi in range(control_mode.shape[1]):
 phase = phase_control[:, roi]
 amplitude = higher_freq_envelope[:, roi]
 
 # Complex representation: amplitude * exp(i*phase)
 complex_pac = amplitude * np.exp(1j * phase)
 mvl = np.abs(np.mean(complex_pac))
 pac_values.append(mvl)
 
 return np.array(pac_values)

def compute_plv(control_mode):
 """
 C2 Organization: Phase Locking Value
 
 Does control mode organize phase relationships across ROIs?
 """
 # Extract phase
 analytic = scipy.signal.hilbert(control_mode, axis=0)
 phase = np.angle(analytic)
 
 n_rois = phase.shape[1]
 plv_matrix = np.zeros((n_rois, n_rois))
 
 for i in range(n_rois):
 for j in range(i+1, n_rois):
 phase_diff = phase[:, i] - phase[:, j]
 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
 plv_matrix[i, j] = plv
 plv_matrix[j, i] = plv
 
 return plv_matrix

def compute_regime_transitions(eeg, fs, window_sec=10):
 """
 C3 Regime Timing: Detect regime transitions
 
 Use entropy as regime indicator (high entropy = Regime I, low = Regime II/III)
 """
 window_samples = int(window_sec * fs)
 n_windows = eeg.shape[0] // window_samples
 
 entropy_trajectory = []
 
 for w in range(n_windows):
 start = w * window_samples
 end = start + window_samples
 window_data = eeg[start:end, :]
 
 # Compute entropy (Shannon entropy of amplitude distribution)
 hist, _ = np.histogram(window_data.flatten(), bins=50, density=True)
 hist = hist[hist > 0] # Remove zeros
 entropy = -np.sum(hist * np.log(hist))
 entropy_trajectory.append(entropy)
 
 # Detect transitions (large changes in entropy)
 entropy_trajectory = np.array(entropy_trajectory)
 entropy_diff = np.abs(np.diff(entropy_trajectory))
 
 # Transition = entropy change > 1 std
 threshold = np.mean(entropy_diff) + np.std(entropy_diff)
 transitions = entropy_diff > threshold
 
 return transitions, entropy_trajectory

def test_regime_timing_correlation(control_mode, transitions, fs, window_sec=10):
 """
 C3 Regime Timing: Test if control mode correlates with regime transitions
 """
 window_samples = int(window_sec * fs)
 n_windows = len(transitions) + 1
 
 # Compute control mode power in each window
 control_power = []
 
 for w in range(n_windows):
 start = w * window_samples
 end = start + window_samples
 if end > control_mode.shape[0]:
 break
 window_data = control_mode[start:end, :]
 power = np.mean(np.abs(window_data)**2)
 control_power.append(power)
 
 control_power = np.array(control_power[:-1]) # Match transitions length
 
 # Correlate control power with transition probability
 if len(control_power) > 1 and len(transitions) > 1:
 corr, pval = pearsonr(control_power, transitions.astype(float))
 return corr, pval
 else:
 return 0.0, 1.0

def apply_toroidal_ablation(eeg, ablation_type='remove_closure'):
 """
 Topology Ablation: Break toroidal structure
 
 ablation_type:
 - 'remove_closure': Remove periodic boundaries (torus → planar grid)
 - 'randomize': Randomize spatial adjacency
 - 'destroy_coherence': Phase scramble
 """
 if ablation_type == 'remove_closure':
 # Zero out wraparound connections in adjacency matrix
 # This is conceptual - in practice, affects downstream analysis
 return eeg # Placeholder
 
 elif ablation_type == 'randomize':
 # Spatially permute ROIs
 perm = np.random.permutation(eeg.shape[1])
 return eeg[:, perm]
 
 elif ablation_type == 'destroy_coherence':
 # Phase scramble each ROI independently
 scrambled = np.zeros_like(eeg)
 for roi in range(eeg.shape[1]):
 fft = np.fft.fft(eeg[:, roi])
 phases = np.angle(fft)
 scrambled_phases = np.random.permutation(phases)
 scrambled_fft = np.abs(fft) * np.exp(1j * scrambled_phases)
 scrambled[:, roi] = np.real(np.fft.ifft(scrambled_fft))
 return scrambled
 
 return eeg

def analyze_single_file(mat_file):
 """Run complete C1-C3 analysis on single file"""
 results = {
 'file': mat_file.name,
 'subject': mat_file.stem.split('_')[0],
 'condition': 'eyes_open' if 'R01' in mat_file.name or 'R02' in mat_file.name else 'eyes_closed'
 }
 
 # Load data
 eeg, fs = load_eeg_data(mat_file)
 if eeg is None:
 return None
 
 # Extract control mode
 control_mode = extract_control_mode(eeg, fs, CONTROL_FREQ_MIN, CONTROL_FREQ_MAX)
 
 # C1: Gating (PAC with alpha band)
 alpha_envelope = extract_higher_freq_power(eeg, fs, band_name='alpha')
 pac_values = compute_pac(control_mode, alpha_envelope)
 results['c1_gating_pac_mean'] = float(np.mean(pac_values))
 results['c1_gating_pac_std'] = float(np.std(pac_values))
 
 # C2: Organization (PLV)
 plv_matrix = compute_plv(control_mode)
 # Mean PLV excluding diagonal
 plv_off_diag = plv_matrix[np.triu_indices_from(plv_matrix, k=1)]
 results['c2_organization_plv_mean'] = float(np.mean(plv_off_diag))
 results['c2_organization_plv_std'] = float(np.std(plv_off_diag))
 
 # C3: Regime Timing
 transitions, entropy_traj = compute_regime_transitions(eeg, fs)
 corr, pval = test_regime_timing_correlation(control_mode, transitions, fs)
 results['c3_regime_timing_corr'] = float(corr)
 results['c3_regime_timing_pval'] = float(pval)
 results['c3_n_transitions'] = int(np.sum(transitions))
 
 # Topology Ablations
 for ablation_type in ['randomize', 'destroy_coherence']:
 eeg_ablated = apply_toroidal_ablation(eeg, ablation_type)
 control_ablated = extract_control_mode(eeg_ablated, fs, CONTROL_FREQ_MIN, CONTROL_FREQ_MAX)
 
 # Re-run C2 (Organization) under ablation
 plv_ablated = compute_plv(control_ablated)
 plv_ablated_off_diag = plv_ablated[np.triu_indices_from(plv_ablated, k=1)]
 results[f'c2_ablation_{ablation_type}_plv_mean'] = float(np.mean(plv_ablated_off_diag))
 
 # Re-run C3 (Regime Timing) under ablation
 transitions_abl, _ = compute_regime_transitions(eeg_ablated, fs)
 corr_abl, pval_abl = test_regime_timing_correlation(control_ablated, transitions_abl, fs)
 results[f'c3_ablation_{ablation_type}_corr'] = float(corr_abl)
 results[f'c3_ablation_{ablation_type}_pval'] = float(pval_abl)
 
 return results

def main():
 """Run Stage C reframed analysis on all PhysioNet EEG files"""
 print("="*80)
 print("STAGE C: REFRAMED ANALYSIS (C1-C3 + Topology Ablations)")
 print("="*80)
 print(f"Dataset: PhysioNet Motor Movement EEG")
 print(f"Control frequency range: {CONTROL_FREQ_MIN}-{CONTROL_FREQ_MAX} Hz")
 print(f"Output directory: {OUTPUT_DIR}")
 print()
 
 # Find all MAT files
 mat_files = sorted(DATA_DIR.glob("*.mat"))
 print(f"Found {len(mat_files)} MAT files")
 print()
 
 all_results = []
 
 for i, mat_file in enumerate(mat_files, 1):
 print(f"[{i}/{len(mat_files)}] Processing {mat_file.name}...")
 results = analyze_single_file(mat_file)
 if results:
 all_results.append(results)
 
 # Save results
 df = pd.DataFrame(all_results)
 csv_path = OUTPUT_DIR / "stage_c_reframed_results.csv"
 df.to_csv(csv_path, index=False)
 print(f"\n✅ Results saved to {csv_path}")
 
 # Compute summary statistics
 print("\n" + "="*80)
 print("STAGE C SUMMARY")
 print("="*80)
 
 # C1: Gating
 print("\n### C1: GATING (Phase-Amplitude Coupling)")
 print(f"PAC Mean: {df['c1_gating_pac_mean'].mean():.4f} ± {df['c1_gating_pac_mean'].std():.4f}")
 
 # Test eyes-open vs eyes-closed
 eo = df[df['condition'] == 'eyes_open']['c1_gating_pac_mean']
 ec = df[df['condition'] == 'eyes_closed']['c1_gating_pac_mean']
 if len(eo) > 0 and len(ec) > 0:
 t_stat, p_val = ttest_ind(eo, ec)
 print(f"Eyes-Open vs Eyes-Closed: t={t_stat:.3f}, p={p_val:.4f}")
 
 # C2: Organization
 print("\n### C2: ORGANIZATION (Phase Locking Value)")
 print(f"PLV Mean: {df['c2_organization_plv_mean'].mean():.4f} ± {df['c2_organization_plv_mean'].std():.4f}")
 
 # Test ablation effect
 for ablation in ['randomize', 'destroy_coherence']:
 intact = df['c2_organization_plv_mean']
 ablated = df[f'c2_ablation_{ablation}_plv_mean']
 collapse = ((intact.mean() - ablated.mean()) / intact.mean()) * 100
 print(f"Ablation ({ablation}): {collapse:.1f}% collapse")
 
 # C3: Regime Timing
 print("\n### C3: REGIME TIMING (Transition Correlation)")
 print(f"Correlation: {df['c3_regime_timing_corr'].mean():.4f} ± {df['c3_regime_timing_corr'].std():.4f}")
 print(f"Mean transitions per recording: {df['c3_n_transitions'].mean():.1f}")
 
 # Test ablation effect
 for ablation in ['randomize', 'destroy_coherence']:
 intact = df['c3_regime_timing_corr']
 ablated = df[f'c3_ablation_{ablation}_corr']
 collapse = ((np.abs(intact.mean()) - np.abs(ablated.mean())) / np.abs(intact.mean())) * 100
 print(f"Ablation ({ablation}): {collapse:.1f}% collapse")
 
 # Save summary
 summary = {
 'c1_gating_pac_mean': float(df['c1_gating_pac_mean'].mean()),
 'c1_gating_pac_std': float(df['c1_gating_pac_mean'].std()),
 'c2_organization_plv_mean': float(df['c2_organization_plv_mean'].mean()),
 'c2_organization_plv_std': float(df['c2_organization_plv_mean'].std()),
 'c3_regime_timing_corr_mean': float(df['c3_regime_timing_corr'].mean()),
 'c3_regime_timing_corr_std': float(df['c3_regime_timing_corr'].std()),
 'c3_n_transitions_mean': float(df['c3_n_transitions'].mean()),
 }
 
 # Add ablation effects
 for ablation in ['randomize', 'destroy_coherence']:
 intact_plv = df['c2_organization_plv_mean'].mean()
 ablated_plv = df[f'c2_ablation_{ablation}_plv_mean'].mean()
 summary[f'c2_ablation_{ablation}_collapse_pct'] = float(((intact_plv - ablated_plv) / intact_plv) * 100)
 
 intact_corr = np.abs(df['c3_regime_timing_corr'].mean())
 ablated_corr = np.abs(df[f'c3_ablation_{ablation}_corr'].mean())
 summary[f'c3_ablation_{ablation}_collapse_pct'] = float(((intact_corr - ablated_corr) / intact_corr) * 100)
 
 json_path = OUTPUT_DIR / "stage_c_reframed_summary.json"
 with open(json_path, 'w') as f:
 json.dump(summary, f, indent=2)
 print(f"\n✅ Summary saved to {json_path}")
 
 print("\n" + "="*80)
 print("STAGE C REFRAMED ANALYSIS COMPLETE")
 print("="*80)

if __name__ == "__main__":
 main()
