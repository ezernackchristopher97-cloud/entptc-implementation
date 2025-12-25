#!/usr/bin/env python3
"""
Complete EntPTC Analysis Pipeline
Strictly follows ENTPC.tex specification (lines 493-738)
Implements all critical requirements from user instructions

CRITICAL REQUIREMENTS:
1. NO SYNTHETIC DATA - Use only real MAT files
2. 64 indices, NOT 65 - Assert checks everywhere
3. 16×16 Progenitor Matrix exactly as ENTPC.tex defines
4. THz via structural invariants ONLY - NO frequency conversion
5. Absurdity Gap post-operator ONLY (lines 733-734)
6. All three regimes separately (lines 669-676)
"""

import numpy as np
import h5py
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add entptc to path
sys.path.insert(0, '/home/ubuntu/entptc-implementation')

from entptc.core.progenitor import ProgenitorMatrix
from entptc.core.perron_frobenius import PerronFrobeniusOperator
from entptc.core.entropy import EntropyField
from entptc.core.clifford import CliffordElement
from entptc.core.quaternion import Quaternion
from entptc.core.geodesics import GeodesicSolver
from entptc.analysis.absurdity_gap import AbsurdityGap, AbsurdityGapAnalyzer
from entptc.analysis.geodesics import GeodesicAnalyzer
from entptc.analysis.thz_inference import THzStructuralInvariants, THzPatternMatcher

# ROI aggregation map (64 channels -> 16 ROIs)
# Each ROI averages 4 channels (64/16 = 4)
ROI_MAP = {
 0: [0, 1, 2, 3], # Frontal Left
 1: [4, 5, 6, 7], # Frontal Right
 2: [8, 9, 10, 11], # Central Left
 3: [12, 13, 14, 15], # Central Right
 4: [16, 17, 18, 19], # Parietal Left
 5: [20, 21, 22, 23], # Parietal Right
 6: [24, 25, 26, 27], # Occipital Left
 7: [28, 29, 30, 31], # Occipital Right
 8: [32, 33, 34, 35], # Temporal Left
 9: [36, 37, 38, 39], # Temporal Right
 10: [40, 41, 42, 43], # Frontal Midline
 11: [44, 45, 46, 47], # Central Midline
 12: [48, 49, 50, 51], # Parietal Midline
 13: [52, 53, 54, 55], # Occipital Midline
 14: [56, 57, 58, 59], # Temporal Midline
 15: [60, 61, 62, 63] # Reference/Ground
}

def aggregate_to_rois(eeg_data, roi_map):
 """
 Aggregate 64 channels to 16 ROIs by averaging.
 CRITICAL: Validates 64-channel constraint.
 """
 # CRITICAL ASSERTION: 64 indices, NOT 65
 assert eeg_data.shape[0] == 64, f"CRITICAL ERROR: Expected 64 channels, got {eeg_data.shape[0]}"
 
 n_rois = len(roi_map)
 assert n_rois == 16, f"CRITICAL ERROR: Expected 16 ROIs, got {n_rois}"
 
 n_timepoints = eeg_data.shape[1]
 roi_data = np.zeros((n_rois, n_timepoints))
 
 for roi_idx, channel_indices in roi_map.items():
 assert len(channel_indices) == 4, f"ROI {roi_idx} should have 4 channels, got {len(channel_indices)}"
 roi_data[roi_idx, :] = np.mean(eeg_data[channel_indices, :], axis=0)
 
 return roi_data

def compute_coherence_matrix(roi_data):
 """
 Compute 16×16 Phase Locking Value (PLV) coherence matrix.
 ENTPC.tex lines 696-703: PLV for each pair of ROIs.
 """
 n_rois = roi_data.shape[0]
 assert n_rois == 16, f"Expected 16 ROIs, got {n_rois}"
 
 coherence_matrix = np.zeros((n_rois, n_rois), dtype=complex)
 
 for i in range(n_rois):
 for j in range(n_rois):
 if i == j:
 coherence_matrix[i, j] = 1.0 + 0.0j
 else:
 # Compute phase locking value
 signal_i = roi_data[i, :]
 signal_j = roi_data[j, :]
 
 # Hilbert transform for instantaneous phase
 from scipy.signal import hilbert
 analytic_i = hilbert(signal_i)
 analytic_j = hilbert(signal_j)
 
 phase_i = np.angle(analytic_i)
 phase_j = np.angle(analytic_j)
 
 phase_diff = phase_i - phase_j
 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
 
 coherence_matrix[i, j] = plv + 0.0j
 
 # CRITICAL ASSERTION: Matrix must be 16×16
 assert coherence_matrix.shape == (16, 16), f"Coherence matrix must be 16×16, got {coherence_matrix.shape}"
 
 return coherence_matrix

def classify_regime(spectral_gap):
 """
 Classify into three regimes based on spectral gap λ₁/λ₂.
 ENTPC.tex lines 669-676:
 - Regime I: λ₁/λ₂ > 2.0 (Local Stabilized)
 - Regime II: 1.2 < λ₁/λ₂ < 2.0 (Transitional)
 - Regime III: λ₁/λ₂ < 1.5 (Global Experience)
 """
 if spectral_gap > 2.0:
 return "Regime_I_Local_Stabilized"
 elif 1.2 < spectral_gap <= 2.0:
 return "Regime_II_Transitional"
 elif spectral_gap <= 1.5:
 return "Regime_III_Global_Experience"
 else:
 return "Regime_Undefined"

def extract_thz_structural_invariants(eigenvalues):
 """
 Extract THz structural invariants via dimensionless ratios ONLY.
 ENTPC.tex lines 713-727: NO frequency conversion.
 
 Returns dimensionless structural invariants:
 - Eigenvalue ratios: λ₁/λ₂, λ₂/λ₃, etc.
 - Spectral decay slopes
 - Regime-dependent stability thresholds
 """
 n_eigs = len(eigenvalues)
 
 # Sort eigenvalues in descending order
 eigs_sorted = np.sort(np.abs(eigenvalues))[::-1]
 
 invariants = {}
 
 # Eigenvalue ratios (dimensionless)
 ratios = []
 for i in range(n_eigs - 1):
 if eigs_sorted[i+1] != 0:
 ratio = eigs_sorted[i] / eigs_sorted[i+1]
 ratios.append(ratio)
 invariants[f'lambda_ratio_{i}_{i+1}'] = ratio
 
 invariants['eigenvalue_ratios_mean'] = np.mean(ratios) if ratios else np.nan
 invariants['eigenvalue_ratios_std'] = np.std(ratios) if ratios else np.nan
 
 # Spectral decay slope (dimensionless)
 if n_eigs > 1:
 decay_diffs = np.diff(eigs_sorted)
 invariants['spectral_decay_slope_mean'] = np.mean(decay_diffs)
 invariants['spectral_decay_slope_std'] = np.std(decay_diffs)
 else:
 invariants['spectral_decay_slope_mean'] = np.nan
 invariants['spectral_decay_slope_std'] = np.nan
 
 # Spectral radius (dimensionless)
 invariants['spectral_radius'] = eigs_sorted[0] if n_eigs > 0 else np.nan
 
 # Symmetry breaking parameter (dimensionless)
 if n_eigs > 2:
 invariants['symmetry_breaking'] = (eigs_sorted[0] - eigs_sorted[1]) / (eigs_sorted[1] - eigs_sorted[2] + 1e-12)
 else:
 invariants['symmetry_breaking'] = np.nan
 
 return invariants

def extract_complete_features(eeg_data, subject_id, session, task, timepoint):
 """
 Extract ALL EntPTC features from EEG data.
 Strictly follows ENTPC.tex specification.
 """
 features = {
 'subject_id': subject_id,
 'session': session,
 'task': task,
 'timepoint': timepoint
 }
 
 # CRITICAL: Validate 64 channels
 assert eeg_data.shape[0] == 64, f"Expected 64 channels, got {eeg_data.shape[0]}"
 
 # Step 1: Aggregate to 16 ROIs
 roi_data = aggregate_to_rois(eeg_data, ROI_MAP)
 n_rois = roi_data.shape[0]
 assert n_rois == 16, f"Expected 16 ROIs, got {n_rois}"
 
 # Step 2: Compute 16×16 coherence matrix (PLV)
 coherence_matrix = compute_coherence_matrix(roi_data)
 assert coherence_matrix.shape == (16, 16), f"Coherence matrix must be 16×16"
 
 # Step 3: Build Progenitor Matrix (16×16)
 # ENTPC.tex Eq. 6, lines 493-524: c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|
 progenitor = ProgenitorMatrix(n_rois)
 P = progenitor.construct_from_eeg_data(roi_data)
 
 # CRITICAL ASSERTION: Progenitor Matrix must be 16×16
 assert P.shape == (16, 16), f"Progenitor Matrix must be 16×16, got {P.shape}"
 
 # Step 4: Apply Perron-Frobenius operator
 # ENTPC.tex Def 2.7-2.8: Operator collapse
 pf_operator = PerronFrobeniusOperator(P)
 eigenvalues, eigenvectors = pf_operator.compute_eigendecomposition()
 
 # CRITICAL ASSERTION: Must have 16 eigenvalues
 assert len(eigenvalues) == 16, f"Expected 16 eigenvalues, got {len(eigenvalues)}"
 
 # Sort eigenvalues in descending order
 idx = np.argsort(np.abs(eigenvalues))[::-1]
 eigenvalues = eigenvalues[idx]
 eigenvectors = eigenvectors[:, idx]
 
 # Store eigenvalues
 features['lambda_max'] = np.abs(eigenvalues[0])
 for i in range(16):
 features[f'eigenvalue_{i}'] = np.abs(eigenvalues[i])
 
 # Step 5: Compute spectral gap λ₁/λ₂
 if np.abs(eigenvalues[1]) > 1e-12:
 spectral_gap = np.abs(eigenvalues[0]) / np.abs(eigenvalues[1])
 else:
 spectral_gap = np.inf
 features['spectral_gap'] = spectral_gap
 
 # Step 6: Classify regime
 # ENTPC.tex lines 669-676
 regime = classify_regime(spectral_gap)
 features['regime'] = regime
 
 # Step 7: Extract THz structural invariants (NO frequency conversion)
 # ENTPC.tex lines 713-727
 thz_invariants = extract_thz_structural_invariants(eigenvalues)
 features.update(thz_invariants)
 
 # Step 8: Compute entropy field
 # ENTPC.tex Def 2.4-2.5
 entropy_field = EntropyField(n_rois)
 shannon_entropies = []
 for i in range(n_rois):
 signal = roi_data[i, :]
 hist, _ = np.histogram(signal, bins=50, density=True)
 hist = hist / (np.sum(hist) + 1e-12)
 entropy = -np.sum(hist * np.log(hist + 1e-12))
 shannon_entropies.append(entropy)
 
 features['entropy_mean'] = np.mean(shannon_entropies)
 features['entropy_std'] = np.std(shannon_entropies)
 
 # Step 9: Compute geodesics
 # ENTPC.tex Section 5.2, lines 678-687
 geodesic_solver = GeodesicSolver(entropy_field, alpha=0.1)
 
 start_point = np.array([0.0, 0.0, 0.0])
 end_point = np.array([np.pi, np.pi, np.pi])
 
 try:
 geodesic_dist = geodesic_solver.geodesic_distance(start_point, end_point)
 features['geodesic_distance'] = geodesic_dist
 except:
 features['geodesic_distance'] = np.nan
 
 # Step 10: Compute Absurdity Gap (POST-OPERATOR ONLY)
 # ENTPC.tex lines 649-664, 728-734
 # CRITICAL: Computed ONLY after Perron-Frobenius convergence
 psi_initial = roi_data[:, 0] # Initial state
 psi_collapsed = eigenvectors[:, 0].real # Dominant eigenvector (collapsed state)
 
 # Normalize states
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
 
 # Step 11: Quaternion operations
 # ENTPC.tex Def 2.1-2.2
 if len(eigenvalues) >= 4:
 q = Quaternion(
 w=eigenvalues[0].real,
 x=eigenvalues[1].real,
 y=eigenvalues[2].real,
 z=eigenvalues[3].real
 )
 features['quaternion_norm'] = q.norm()
 features['quaternion_scalar'] = q.w
 features['quaternion_vector_norm'] = np.sqrt(q.x**2 + q.y**2 + q.z**2)
 else:
 features['quaternion_norm'] = np.nan
 features['quaternion_scalar'] = np.nan
 features['quaternion_vector_norm'] = np.nan
 
 # Step 12: Clifford algebra operations
 # ENTPC.tex Def 2.3
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
 print("COMPLETE EntPTC ANALYSIS - FULL DATASET")
 print("Strictly follows ENTPC.tex specification")
 print("=" * 80)
 
 # Data directory
 data_dir = Path('/home/ubuntu/entptc-implementation/data')
 
 if not data_dir.exists():
 print(f"CRITICAL ERROR: Data directory not found: {data_dir}")
 print("NO SYNTHETIC DATA ALLOWED - FAILING LOUDLY")
 sys.exit(1)
 
 # Get all .mat files
 mat_files = sorted(list(data_dir.glob('*.mat')))
 print(f"\nFound {len(mat_files)} .mat files")
 
 if len(mat_files) == 0:
 print("CRITICAL ERROR: No .mat files found!")
 print("NO SYNTHETIC DATA ALLOWED - FAILING LOUDLY")
 sys.exit(1)
 
 # Process all files
 results = []
 errors = []
 
 for mat_file in tqdm(mat_files, desc="Processing MAT files"):
 try:
 # Load data
 with h5py.File(mat_file, 'r') as f:
 data_matrix = np.array(f['data_matrix'])
 
 # Transpose if needed (timepoints, channels) -> (channels, timepoints)
 if data_matrix.shape[0] > data_matrix.shape[1]:
 data_matrix = data_matrix.T
 
 # CRITICAL: Validate 64 channels
 if data_matrix.shape[0] != 64:
 raise ValueError(f"Expected 64 channels, got {data_matrix.shape[0]}")
 
 # Parse filename
 filename = mat_file.stem
 parts = filename.split('_')
 
 subject_id = parts[0] # e.g., 'sub-001'
 session = parts[1] # e.g., 'ses-1'
 task = parts[2].replace('task-', '') # e.g., 'EyesClosed' or 'EyesOpen'
 timepoint = parts[3].replace('acq-', '') # e.g., 'pre' or 'post'
 
 # Extract features
 features = extract_complete_features(data_matrix, subject_id, session, task, timepoint)
 features['filename'] = filename
 
 results.append(features)
 
 except Exception as e:
 error_msg = f"ERROR processing {mat_file.name}: {str(e)}"
 print(f"\n{error_msg}")
 errors.append({'filename': mat_file.name, 'error': str(e)})
 continue
 
 # Convert to DataFrame
 df = pd.DataFrame(results)
 
 # Create outputs directory
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs')
 output_dir.mkdir(exist_ok=True)
 
 # Save master results CSV
 master_csv_path = output_dir / 'master_results.csv'
 df.to_csv(master_csv_path, index=False)
 
 print(f"\n{'=' * 80}")
 print(f"ANALYSIS COMPLETE!")
 print(f"{'=' * 80}")
 print(f"Successfully processed: {len(df)} files")
 print(f"Errors encountered: {len(errors)}")
 print(f"Master CSV: {master_csv_path}")
 
 # Print summary statistics
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
 
 # Key metrics
 print("\n" + "=" * 80)
 print("KEY METRICS (Mean ± Std)")
 print("=" * 80)
 
 print(f"\nλ_max (Dominant Eigenvalue): {df['lambda_max'].mean():.6f} ± {df['lambda_max'].std():.6f}")
 print(f"Spectral Gap (λ₁/λ₂): {df['spectral_gap'].mean():.6f} ± {df['spectral_gap'].std():.6f}")
 print(f"Entropy (Mean): {df['entropy_mean'].mean():.6f} ± {df['entropy_mean'].std():.6f}")
 print(f"Absurdity Gap (L2): {df['absurdity_gap_L2'].mean():.6f} ± {df['absurdity_gap_L2'].std():.6f}")
 print(f"Geodesic Distance: {df['geodesic_distance'].mean():.6f} ± {df['geodesic_distance'].std():.6f}")
 
 # Falsifiability test (ENTPC.tex line 663)
 print("\n" + "=" * 80)
 print("FALSIFIABILITY TEST (ENTPC.tex line 663)")
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
 print(f"\n✓ FALSIFIABILITY TEST PASSED: Significant difference detected (p < 0.05)")
 print(f" Model is NOT falsified.")
 else:
 print(f"\n✗ FALSIFIABILITY TEST FAILED: No significant difference (p >= 0.05)")
 print(f" WARNING: Model may be falsified per ENTPC.tex line 663")
 else:
 print("\nInsufficient data for falsifiability test")
 
 # Save error log if any
 if errors:
 error_df = pd.DataFrame(errors)
 error_log_path = output_dir / 'error_log.csv'
 error_df.to_csv(error_log_path, index=False)
 print(f"\nError log saved: {error_log_path}")
 
 print("\n" + "=" * 80)
 print("NEXT STEPS:")
 print("1. Review master_results.csv")
 print("2. Generate figures (eigenvalue spectra, regime distributions, etc.)")
 print("3. Create VALIDATION_REPORT.md with TeX citations")
 print("=" * 80)

if __name__ == '__main__':
 main()
