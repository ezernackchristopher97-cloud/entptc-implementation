#!/usr/bin/env python3
"""
Fast EntPTC Analysis - Optimized Version
Removes slow geodesic computations, focuses on core metrics
"""

import numpy as np
import h5py
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/ubuntu/entptc-implementation')

from entptc.core.progenitor import ProgenitorMatrix
from entptc.core.perron_frobenius import PerronFrobeniusOperator
from entptc.core.entropy import EntropyField
from entptc.core.clifford import CliffordElement
from entptc.core.quaternion import Quaternion
from entptc.analysis.absurdity_gap import AbsurdityGap

# ROI aggregation map
ROI_MAP = {
 0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15],
 4: [16, 17, 18, 19], 5: [20, 21, 22, 23], 6: [24, 25, 26, 27], 7: [28, 29, 30, 31],
 8: [32, 33, 34, 35], 9: [36, 37, 38, 39], 10: [40, 41, 42, 43], 11: [44, 45, 46, 47],
 12: [48, 49, 50, 51], 13: [52, 53, 54, 55], 14: [56, 57, 58, 59], 15: [60, 61, 62, 63]
}

def aggregate_to_rois(eeg_data, roi_map):
 """Aggregate 64 channels to 16 ROIs."""
 assert eeg_data.shape[0] == 64, f"Expected 64 channels, got {eeg_data.shape[0]}"
 n_rois = len(roi_map)
 n_timepoints = eeg_data.shape[1]
 roi_data = np.zeros((n_rois, n_timepoints))
 for roi_idx, channel_indices in roi_map.items():
 roi_data[roi_idx, :] = np.mean(eeg_data[channel_indices, :], axis=0)
 return roi_data

def classify_regime(spectral_gap):
 """Classify regime based on spectral gap."""
 if spectral_gap > 2.0:
 return "Regime_I_Local_Stabilized"
 elif 1.2 < spectral_gap <= 2.0:
 return "Regime_II_Transitional"
 elif spectral_gap <= 1.5:
 return "Regime_III_Global_Experience"
 else:
 return "Regime_Undefined"

def extract_thz_structural_invariants(eigenvalues):
 """Extract THz structural invariants (dimensionless ratios only)."""
 eigs_sorted = np.sort(np.abs(eigenvalues))[::-1]
 invariants = {}
 
 # Eigenvalue ratios
 ratios = []
 for i in range(len(eigs_sorted) - 1):
 if eigs_sorted[i+1] != 0:
 ratio = eigs_sorted[i] / eigs_sorted[i+1]
 ratios.append(ratio)
 invariants[f'lambda_ratio_{i}_{i+1}'] = ratio
 
 invariants['eigenvalue_ratios_mean'] = np.mean(ratios) if ratios else np.nan
 invariants['eigenvalue_ratios_std'] = np.std(ratios) if ratios else np.nan
 
 # Spectral decay slope
 if len(eigs_sorted) > 1:
 decay_diffs = np.diff(eigs_sorted)
 invariants['spectral_decay_slope_mean'] = np.mean(decay_diffs)
 invariants['spectral_decay_slope_std'] = np.std(decay_diffs)
 else:
 invariants['spectral_decay_slope_mean'] = np.nan
 invariants['spectral_decay_slope_std'] = np.nan
 
 # Spectral radius
 invariants['spectral_radius'] = eigs_sorted[0] if len(eigs_sorted) > 0 else np.nan
 
 # Symmetry breaking
 if len(eigs_sorted) > 2:
 invariants['symmetry_breaking'] = (eigs_sorted[0] - eigs_sorted[1]) / (eigs_sorted[1] - eigs_sorted[2] + 1e-12)
 else:
 invariants['symmetry_breaking'] = np.nan
 
 return invariants

def extract_features(eeg_data, subject_id, session, task, timepoint):
 """Extract EntPTC features."""
 features = {
 'subject_id': subject_id,
 'session': session,
 'task': task,
 'timepoint': timepoint
 }
 
 assert eeg_data.shape[0] == 64, f"Expected 64 channels, got {eeg_data.shape[0]}"
 
 # Aggregate to 16 ROIs
 roi_data = aggregate_to_rois(eeg_data, ROI_MAP)
 n_rois = roi_data.shape[0]
 
 # Build Progenitor Matrix
 progenitor = ProgenitorMatrix()
 P = progenitor.construct_from_eeg_data(roi_data)
 assert P.shape == (16, 16), f"Progenitor Matrix must be 16×16, got {P.shape}"
 
 # Perron-Frobenius operator
 pf_operator = PerronFrobeniusOperator()
 eigenvalues, eigenvectors = pf_operator.compute_eigendecomposition(P)
 assert len(eigenvalues) == 16, f"Expected 16 eigenvalues, got {len(eigenvalues)}"
 
 # Sort eigenvalues
 idx = np.argsort(np.abs(eigenvalues))[::-1]
 eigenvalues = eigenvalues[idx]
 eigenvectors = eigenvectors[:, idx]
 
 # Store eigenvalues
 features['lambda_max'] = np.abs(eigenvalues[0])
 for i in range(16):
 features[f'eigenvalue_{i}'] = np.abs(eigenvalues[i])
 
 # Spectral gap
 if np.abs(eigenvalues[1]) > 1e-12:
 spectral_gap = np.abs(eigenvalues[0]) / np.abs(eigenvalues[1])
 else:
 spectral_gap = np.inf
 features['spectral_gap'] = spectral_gap
 
 # Regime classification
 regime = classify_regime(spectral_gap)
 features['regime'] = regime
 
 # THz structural invariants
 thz_invariants = extract_thz_structural_invariants(eigenvalues)
 features.update(thz_invariants)
 
 # Entropy field
 shannon_entropies = []
 for i in range(n_rois):
 signal = roi_data[i, :]
 hist, _ = np.histogram(signal, bins=50, density=True)
 hist = hist / (np.sum(hist) + 1e-12)
 entropy = -np.sum(hist * np.log(hist + 1e-12))
 shannon_entropies.append(entropy)
 
 features['entropy_mean'] = np.mean(shannon_entropies)
 features['entropy_std'] = np.std(shannon_entropies)
 
 # Absurdity Gap (post-operator)
 psi_initial = roi_data[:, 0]
 psi_collapsed = eigenvectors[:, 0].real
 
 psi_initial = psi_initial / (np.linalg.norm(psi_initial) + 1e-12)
 psi_collapsed = psi_collapsed / (np.linalg.norm(psi_collapsed) + 1e-12)
 
 absurdity_gap = AbsurdityGap()
 gap_components = absurdity_gap.compute_gap_components(psi_initial, psi_collapsed)
 
 features['absurdity_gap_L1'] = gap_components.get('gap_L1', np.nan)
 features['absurdity_gap_L2'] = gap_components.get('gap_L2', np.nan)
 features['absurdity_gap_Linf'] = gap_components.get('gap_Linf', np.nan)
 features['absurdity_gap_overlap'] = gap_components.get('overlap', np.nan)
 features['absurdity_gap_info_loss'] = gap_components.get('info_loss', np.nan)
 features['absurdity_gap_entropy_change'] = gap_components.get('entropy_change', np.nan)
 
 # Quaternion operations
 if len(eigenvalues) >= 4:
 q = Quaternion(
 a=eigenvalues[0].real,
 b=eigenvalues[1].real,
 c=eigenvalues[2].real,
 d=eigenvalues[3].real
 )
 features['quaternion_norm'] = q.norm()
 features['quaternion_scalar'] = q.a
 features['quaternion_vector_norm'] = np.sqrt(q.b**2 + q.c**2 + q.d**2)
 else:
 features['quaternion_norm'] = np.nan
 features['quaternion_scalar'] = np.nan
 features['quaternion_vector_norm'] = np.nan
 
 # Clifford algebra operations
 if len(eigenvalues) >= 3:
 mv = CliffordElement(
 e1=eigenvalues[0].real,
 e2=eigenvalues[1].real,
 e3=eigenvalues[2].real
 )
 features['clifford_multivector_norm'] = np.linalg.norm(mv.to_array())
 else:
 features['clifford_multivector_norm'] = np.nan
 
 return features

def main():
 """Main execution."""
 print("=" * 80)
 print("FAST EntPTC ANALYSIS - FULL DATASET")
 print("=" * 80)
 
 data_dir = Path('/home/ubuntu/entptc-implementation/data')
 
 if not data_dir.exists():
 print(f"ERROR: Data directory not found: {data_dir}")
 sys.exit(1)
 
 mat_files = sorted(list(data_dir.glob('*.mat')))
 print(f"\nFound {len(mat_files)} .mat files")
 
 if len(mat_files) == 0:
 print("ERROR: No .mat files found!")
 sys.exit(1)
 
 results = []
 errors = []
 
 for mat_file in tqdm(mat_files, desc="Processing MAT files"):
 try:
 with h5py.File(mat_file, 'r') as f:
 data_matrix = np.array(f['data_matrix'])
 
 if data_matrix.shape[0] > data_matrix.shape[1]:
 data_matrix = data_matrix.T
 
 if data_matrix.shape[0] != 64:
 raise ValueError(f"Expected 64 channels, got {data_matrix.shape[0]}")
 
 filename = mat_file.stem
 parts = filename.split('_')
 
 subject_id = parts[0]
 session = parts[1]
 task = parts[2].replace('task-', '')
 timepoint = parts[3].replace('acq-', '')
 
 features = extract_features(data_matrix, subject_id, session, task, timepoint)
 features['filename'] = filename
 
 results.append(features)
 
 except Exception as e:
 error_msg = f"ERROR processing {mat_file.name}: {str(e)}"
 print(f"\n{error_msg}")
 errors.append({'filename': mat_file.name, 'error': str(e)})
 continue
 
 df = pd.DataFrame(results)
 
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs')
 output_dir.mkdir(exist_ok=True)
 
 master_csv_path = output_dir / 'master_results.csv'
 df.to_csv(master_csv_path, index=False)
 
 print(f"\n{'=' * 80}")
 print(f"ANALYSIS COMPLETE!")
 print(f"{'=' * 80}")
 print(f"Successfully processed: {len(df)} files")
 print(f"Errors encountered: {len(errors)}")
 print(f"Master CSV: {master_csv_path}")
 
 print("\n" + "=" * 80)
 print("SUMMARY STATISTICS")
 print("=" * 80)
 
 print(f"\nTotal subjects: {df['subject_id'].nunique()}")
 print(f"Total sessions: {len(df)}")
 
 print(f"\nBy task:")
 print(df['task'].value_counts())
 
 print(f"\nBy timepoint:")
 print(df['timepoint'].value_counts())
 
 print(f"\nBy regime:")
 print(df['regime'].value_counts())
 
 print("\n" + "=" * 80)
 print("KEY METRICS (Mean ± Std)")
 print("=" * 80)
 
 print(f"\nλ_max: {df['lambda_max'].mean():.6f} ± {df['lambda_max'].std():.6f}")
 print(f"Spectral Gap: {df['spectral_gap'].mean():.6f} ± {df['spectral_gap'].std():.6f}")
 print(f"Entropy: {df['entropy_mean'].mean():.6f} ± {df['entropy_mean'].std():.6f}")
 print(f"Absurdity Gap (L2): {df['absurdity_gap_L2'].mean():.6f} ± {df['absurdity_gap_L2'].std():.6f}")
 
 # Falsifiability test
 print("\n" + "=" * 80)
 print("FALSIFIABILITY TEST")
 print("=" * 80)
 
 eyes_closed = df[df['task'] == 'EyesClosed']['absurdity_gap_L2'].dropna()
 eyes_open = df[df['task'] == 'EyesOpen']['absurdity_gap_L2'].dropna()
 
 if len(eyes_closed) > 0 and len(eyes_open) > 0:
 from scipy import stats
 
 t_stat, p_value = stats.ttest_ind(eyes_closed, eyes_open)
 
 print(f"\nAbsurdity Gap Comparison:")
 print(f" Eyes Closed: {eyes_closed.mean():.6f} ± {eyes_closed.std():.6f} (n={len(eyes_closed)})")
 print(f" Eyes Open: {eyes_open.mean():.6f} ± {eyes_open.std():.6f} (n={len(eyes_open)})")
 print(f" t-statistic: {t_stat:.6f}")
 print(f" p-value: {p_value:.6f}")
 
 if p_value < 0.05:
 print(f"\n✓ FALSIFIABILITY TEST PASSED (p < 0.05)")
 else:
 print(f"\n✗ FALSIFIABILITY TEST FAILED (p >= 0.05)")
 
 if errors:
 error_df = pd.DataFrame(errors)
 error_log_path = output_dir / 'error_log.csv'
 error_df.to_csv(error_log_path, index=False)
 print(f"\nError log: {error_log_path}")

if __name__ == '__main__':
 main()
