#!/usr/bin/env python3
"""
Complete EntPTC Feature Extraction with ALL Modules
Includes: Absurdity Gap, Geodesics, Clifford, Quaternion, THz, Entropy, etc.
"""

import numpy as np
import h5py
import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add entptc to path
sys.path.insert(0, '/home/ubuntu/entptc-archive')

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
 """Aggregate 64 channels to 16 ROIs by averaging."""
 n_rois = len(roi_map)
 n_timepoints = eeg_data.shape[1]
 roi_data = np.zeros((n_rois, n_timepoints))
 
 for roi_idx, channel_indices in roi_map.items():
 roi_data[roi_idx, :] = np.mean(eeg_data[channel_indices, :], axis=0)
 
 return roi_data

def extract_complete_features(eeg_data):
 """Extract ALL EntPTC features from EEG data."""
 features = {}
 
 # Aggregate to ROIs
 roi_data = aggregate_to_rois(eeg_data, ROI_MAP)
 n_rois = roi_data.shape[0]
 
 # 1. Shannon Entropy
 entropy_field = EntropyField(n_rois)
 shannon_entropies = []
 for i in range(n_rois):
 signal = roi_data[i, :]
 # Compute probability distribution
 hist, _ = np.histogram(signal, bins=50, density=True)
 hist = hist / (np.sum(hist) + 1e-12)
 entropy = -np.sum(hist * np.log(hist + 1e-12))
 shannon_entropies.append(entropy)
 
 features['shannon_entropy_mean'] = np.mean(shannon_entropies)
 features['shannon_entropy_std'] = np.std(shannon_entropies)
 
 # 2. Spectral Entropy
 spectral_entropies = []
 for i in range(n_rois):
 signal = roi_data[i, :]
 fft = np.fft.fft(signal)
 power = np.abs(fft)**2
 power_norm = power / (np.sum(power) + 1e-12)
 spec_entropy = -np.sum(power_norm * np.log(power_norm + 1e-12))
 spectral_entropies.append(spec_entropy)
 
 features['spectral_entropy_mean'] = np.mean(spectral_entropies)
 features['spectral_entropy_std'] = np.std(spectral_entropies)
 
 # 3. Progenitor Matrix
 progenitor = ProgenitorMatrix(n_rois)
 P = progenitor.construct_from_eeg_data(roi_data)
 
 # Store matrix elements
 P_flat = P.flatten()
 for i, val in enumerate(P_flat):
 features[f'progenitor_elem_{i}'] = val
 
 # 4. Perron-Frobenius Eigendecomposition
 pf_operator = PerronFrobeniusOperator(P)
 eigenvalues, eigenvectors = pf_operator.compute_eigendecomposition()
 
 features['dominant_eigenvalue'] = eigenvalues[0]
 features['eigenvalue_decay'] = np.mean(np.diff(np.sort(eigenvalues)[::-1]))
 
 # Store all eigenvalues
 for i, eig in enumerate(eigenvalues):
 features[f'eigenvalue_{i}'] = eig
 
 # 5. THz Structural Invariants
 thz_extractor = THzStructuralInvariants()
 thz_invariants = thz_extractor.extract_all_invariants(eigenvalues)
 
 features['thz_eigenvalue_ratios_mean'] = np.mean(thz_invariants['eigenvalue_ratios'])
 features['thz_spectral_gaps_mean'] = np.mean(thz_invariants['spectral_gaps'])
 features['thz_symmetry_breaking'] = thz_invariants['symmetry_breaking']
 features['thz_dominant_eigenvalue'] = thz_invariants['dominant_eigenvalue']
 features['thz_spectral_radius'] = thz_invariants['spectral_radius']
 
 # 6. THz Pattern Matching
 thz_matcher = THzPatternMatcher()
 thz_report = thz_matcher.compute_thz_inference_report(eigenvalues)
 
 features['thz_dominant_pattern'] = thz_report['dominant_pattern']
 features['thz_dominant_score'] = thz_report['dominant_score']
 features['thz_confidence'] = thz_report['confidence']
 
 # Store all pattern match scores
 for pattern, score in thz_report['thz_pattern_matches'].items():
 features[f'thz_match_{pattern}'] = score
 
 # 7. Absurdity Gap (requires pre and post states)
 # For single recording, compute gap between initial and final states
 n_timepoints = roi_data.shape[1]
 psi_pre = roi_data[:, 0] # Initial state
 psi_post = eigenvectors[:, 0] # Dominant eigenvector (collapsed state)
 
 absurdity_gap = AbsurdityGap()
 gap_components = absurdity_gap.compute_gap_components(psi_pre, psi_post)
 
 features['absurdity_gap_L1'] = gap_components['gap_L1']
 features['absurdity_gap_L2'] = gap_components['gap_L2']
 features['absurdity_gap_Linf'] = gap_components['gap_Linf']
 features['absurdity_gap_overlap'] = gap_components['overlap']
 features['absurdity_gap_info_loss'] = gap_components['info_loss']
 features['absurdity_gap_entropy_change'] = gap_components['entropy_change']
 features['absurdity_gap_regime'] = gap_components['regime']
 
 # 8. Clifford Algebra Operations
 # Create Clifford element from first 3 eigenvalues as vector
 if len(eigenvalues) >= 3:
 mv = CliffordElement(e1=eigenvalues[0], e2=eigenvalues[1], e3=eigenvalues[2])
 mv_norm = np.linalg.norm(mv.to_array())
 features['clifford_multivector_norm'] = mv_norm
 
 # Geometric product of two Clifford vectors
 if len(eigenvectors) >= 2:
 v1 = eigenvectors[:3, 0]
 v2 = eigenvectors[:3, 1]
 mv1 = CliffordElement(e1=v1[0], e2=v1[1], e3=v1[2])
 mv2 = CliffordElement(e1=v2[0], e2=v2[1], e3=v2[2])
 prod = mv1 * mv2
 features['clifford_geometric_product_norm'] = np.linalg.norm(prod.to_array())
 else:
 features['clifford_multivector_norm'] = np.nan
 features['clifford_geometric_product_norm'] = np.nan
 
 # 9. Quaternion Operations
 # Create quaternion from first 4 eigenvalues
 if len(eigenvalues) >= 4:
 q = Quaternion(w=eigenvalues[0], x=eigenvalues[1], y=eigenvalues[2], z=eigenvalues[3])
 q_norm = q.norm()
 q_conj = q.conjugate()
 
 features['quaternion_norm'] = q_norm
 features['quaternion_scalar'] = q.w
 features['quaternion_vector_norm'] = np.sqrt(q.x**2 + q.y**2 + q.z**2)
 else:
 features['quaternion_norm'] = np.nan
 features['quaternion_scalar'] = np.nan
 features['quaternion_vector_norm'] = np.nan
 
 # 10. Geodesics (compute representative geodesic distance)
 # Use entropy field for geodesic computation
 geodesic_solver = GeodesicSolver(entropy_field, alpha=0.1)
 
 # Compute geodesic distance between two representative points on T³
 start_point = np.array([0.0, 0.0, 0.0])
 end_point = np.array([np.pi, np.pi, np.pi])
 
 try:
 geodesic_dist = geodesic_solver.geodesic_distance(start_point, end_point)
 features['geodesic_distance_representative'] = geodesic_dist
 except:
 features['geodesic_distance_representative'] = np.nan
 
 return features

def main():
 """Main execution."""
 print("=" * 80)
 print("COMPLETE EntPTC Feature Extraction - ALL MODULES")
 print("=" * 80)
 
 # Data directory
 data_dir = Path('/home/ubuntu/entptc-archive/data')
 
 if not data_dir.exists():
 print(f"ERROR: Data directory not found: {data_dir}")
 return
 
 # Get all .mat files
 mat_files = sorted(list(data_dir.glob('*.mat')))
 print(f"\nFound {len(mat_files)} .mat files")
 
 if len(mat_files) == 0:
 print("ERROR: No .mat files found!")
 return
 
 # Process all files
 results = []
 
 for mat_file in tqdm(mat_files, desc="Processing files"):
 try:
 # Load data
 with h5py.File(mat_file, 'r') as f:
 data_matrix = np.array(f['data_matrix'])
 
 # Transpose if needed (timepoints, channels) -> (channels, timepoints)
 if data_matrix.shape[0] > data_matrix.shape[1]:
 data_matrix = data_matrix.T
 
 # Validate 64 channels
 assert data_matrix.shape[0] == 64, f"Expected 64 channels, got {data_matrix.shape[0]}"
 
 # Extract features
 features = extract_complete_features(data_matrix)
 
 # Add metadata
 filename = mat_file.stem
 features['filename'] = filename
 
 # Parse subject and condition
 parts = filename.split('_')
 features['subject_id'] = parts[0]
 
 # Determine condition (pre/post)
 if 'pre' in filename.lower():
 features['condition'] = 'pre'
 elif 'post' in filename.lower():
 features['condition'] = 'post'
 else:
 features['condition'] = 'unknown'
 
 results.append(features)
 
 except Exception as e:
 print(f"\nERROR processing {mat_file.name}: {e}")
 continue
 
 # Convert to DataFrame
 df = pd.DataFrame(results)
 
 # Save to CSV
 output_file = '/home/ubuntu/entptc-archive/entptc_features_COMPLETE_ALL_MODULES.csv'
 df.to_csv(output_file, index=False)
 
 print(f"\n{'=' * 80}")
 print(f"COMPLETE! Extracted {len(df)} feature sets")
 print(f"Output: {output_file}")
 print(f"{'=' * 80}")
 
 # Print summary statistics
 print("\nSummary Statistics:")
 print(f" Total files processed: {len(df)}")
 print(f" Pre-treatment: {len(df[df['condition'] == 'pre'])}")
 print(f" Post-treatment: {len(df[df['condition'] == 'post'])}")
 print(f" Total features per file: {len(df.columns)}")
 
 # Print key metrics
 print("\nKey Metrics (Mean ± Std):")
 print(f" Shannon Entropy: {df['shannon_entropy_mean'].mean():.4f} ± {df['shannon_entropy_mean'].std():.4f}")
 print(f" Dominant Eigenvalue: {df['dominant_eigenvalue'].mean():.4f} ± {df['dominant_eigenvalue'].std():.4f}")
 print(f" Absurdity Gap (L2): {df['absurdity_gap_L2'].mean():.4f} ± {df['absurdity_gap_L2'].std():.4f}")
 print(f" THz Symmetry Breaking: {df['thz_symmetry_breaking'].mean():.4f} ± {df['thz_symmetry_breaking'].std():.4f}")
 print(f" Clifford MV Norm: {df['clifford_multivector_norm'].mean():.4f} ± {df['clifford_multivector_norm'].std():.4f}")
 print(f" Quaternion Norm: {df['quaternion_norm'].mean():.4f} ± {df['quaternion_norm'].std():.4f}")

if __name__ == '__main__':
 main()
