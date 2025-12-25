#!/usr/bin/env python3.11
"""
EntPTC Feature Extraction: COMPLETE IMPLEMENTATION
Uses ALL components of the 9,100+ line EntPTC model.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Import ALL EntPTC model components
from entptc.core.progenitor import ProgenitorMatrix
from entptc.core.perron_frobenius import PerronFrobeniusOperator
from entptc.core.absurdity_gap import AbsurdityGap, AbsurdityGapAnalyzer
from entptc.core.entropy import ToroidalManifold, EntropyField
from entptc.core.quaternion import QuaternionicHilbertSpace, eeg_to_quaternionic_vector
from entptc.core.geodesics import GeodesicSolver, GeodesicAnalyzer
from entptc.analysis.thz_inference import THzStructuralInvariants

def load_mat_file(filepath):
 """Load .mat file (v7.3 HDF5 format)"""
 with h5py.File(filepath, 'r') as f:
 data_matrix = np.array(f['data_matrix'])
 srate = float(f['srate'][0, 0])
 return data_matrix, srate

def aggregate_to_rois(data_64ch):
 """Aggregate 64 channels to 16 ROIs"""
 channels_per_roi = 4
 n_rois = 16
 n_timepoints = data_64ch.shape[1]
 roi_data = np.zeros((n_rois, n_timepoints))
 
 for i in range(n_rois):
 start_ch = i * channels_per_roi
 end_ch = start_ch + channels_per_roi
 roi_data[i, :] = np.mean(data_64ch[start_ch:end_ch, :], axis=0)
 
 return roi_data

def extract_complete_features(filepath, subject_pairs):
 """Extract ALL EntPTC features using the complete model"""
 
 # Load data
 data_matrix, srate = load_mat_file(filepath)
 
 # Validate 64 channels
 assert data_matrix.shape[1] == 64, f"Expected 64 channels, got {data_matrix.shape[1]}"
 
 # Transpose to (channels, time)
 data_matrix = data_matrix.T
 
 # Aggregate to 16 ROIs
 roi_data = aggregate_to_rois(data_matrix)
 
 # Parse filename
 filename = Path(filepath).stem
 parts = filename.split('_')
 subject_id = parts[0].replace('sub-', '')
 condition = '_'.join(parts[3:])
 
 # 1. PROGENITOR MATRIX
 progenitor = ProgenitorMatrix()
 progenitor_matrix = progenitor.construct_from_eeg_data(roi_data, srate)
 
 # 2. PERRON-FROBENIUS OPERATOR
 pf_operator = PerronFrobeniusOperator(progenitor_matrix)
 eigendecomp = pf_operator.compute_eigendecomposition()
 dominant_eigenvalue = eigendecomp['dominant_eigenvalue']
 dominant_eigenvector = eigendecomp['dominant_eigenvector']
 eigenvalue_decay = eigendecomp['decay_rate']
 
 # 3. THZ STRUCTURAL INVARIANTS (NO GHz conversion)
 thz_invariants = THzStructuralInvariants(progenitor_matrix)
 invariants = thz_invariants.extract_all_invariants()
 
 # 4. ENTROPY FIELD ON T³
 toroidal_manifold = ToroidalManifold()
 entropy_field = EntropyField(toroidal_manifold)
 entropy_field.construct_from_progenitor(progenitor_matrix)
 shannon_entropy_mean = entropy_field.compute_shannon_entropy_mean()
 spectral_entropy_mean = entropy_field.compute_spectral_entropy_mean()
 
 # 5. QUATERNIONIC REPRESENTATION
 quat_space = QuaternionicHilbertSpace(roi_data)
 quaternionic_vector = eeg_to_quaternionic_vector(roi_data)
 
 # 6. GEODESICS ON T³
 geodesic_solver = GeodesicSolver(toroidal_manifold)
 geodesic_analyzer = GeodesicAnalyzer(geodesic_solver)
 geodesic_length = geodesic_analyzer.compute_total_geodesic_length(entropy_field)
 
 # 7. ABSURDITY GAP (POST-OPERATOR)
 # This requires pre/post pairs, so computing it separately
 absurdity_gap_value = None
 if 'pre' in condition:
 # Find corresponding post file
 post_filename = filename.replace('pre', 'post')
 post_filepath = Path(filepath).parent / f"{post_filename}.mat"
 if post_filepath.exists():
 # Load post data
 post_data_matrix, _ = load_mat_file(str(post_filepath))
 post_data_matrix = post_data_matrix.T
 post_roi_data = aggregate_to_rois(post_data_matrix)
 
 # Compute post progenitor and eigenvector
 post_progenitor = ProgenitorMatrix()
 post_progenitor_matrix = post_progenitor.construct_from_eeg_data(post_roi_data, srate)
 post_pf_operator = PerronFrobeniusOperator(post_progenitor_matrix)
 post_eigendecomp = post_pf_operator.compute_eigendecomposition()
 post_eigenvector = post_eigendecomp['dominant_eigenvector']
 
 # Compute Absurdity Gap
 absurdity_gap = AbsurdityGap()
 absurdity_gap_value = absurdity_gap.compute(dominant_eigenvector, post_eigenvector)
 
 # Flatten progenitor matrix
 matrix_flat = progenitor_matrix.flatten()
 
 # Build feature dictionary
 features = {
 'subject_id': subject_id,
 'condition': condition,
 'shannon_entropy_mean': shannon_entropy_mean,
 'spectral_entropy_mean': spectral_entropy_mean,
 'coherence_eigen_decay': eigenvalue_decay,
 'dominant_eigenvalue': dominant_eigenvalue,
 'thz_eigenvalue_ratios_mean': invariants['eigenvalue_ratios_mean'],
 'thz_spectral_gaps_mean': invariants['spectral_gaps_mean'],
 'thz_symmetry_breaking': invariants['symmetry_breaking'],
 'geodesic_length': geodesic_length,
 'absurdity_gap': absurdity_gap_value if absurdity_gap_value is not None else np.nan,
 }
 
 # Add matrix elements
 for i in range(256):
 features[f'matrix_elem_{i}'] = matrix_flat[i]
 
 return features

def main():
 data_dir = Path('data')
 mat_files = sorted(data_dir.glob('*.mat'))
 
 print(f"Found {len(mat_files)} .mat files")
 print("Starting COMPLETE feature extraction with ALL EntPTC components...")
 print()
 
 # Build subject pairs for absurdity gap
 subject_pairs = {}
 for f in mat_files:
 parts = f.stem.split('_')
 subject_id = parts[0].replace('sub-', '')
 if subject_id not in subject_pairs:
 subject_pairs[subject_id] = {'pre': None, 'post': None}
 if 'pre' in f.stem:
 subject_pairs[subject_id]['pre'] = str(f)
 elif 'post' in f.stem:
 subject_pairs[subject_id]['post'] = str(f)
 
 results = []
 for idx, mat_file in enumerate(mat_files, 1):
 try:
 print(f"[{idx}/{len(mat_files)}] Processing: {mat_file.name}...", end=' ')
 features = extract_complete_features(str(mat_file), subject_pairs)
 results.append(features)
 print("✓")
 except Exception as e:
 print(f"✗ Error: {e}")
 continue
 
 # Save to CSV
 df = pd.DataFrame(results)
 output_file = 'entptc_features_COMPLETE.csv'
 df.to_csv(output_file, index=False)
 
 print()
 print(f"COMPLETE feature extraction finished!")
 print(f"Output: {output_file}")
 print(f"Rows: {len(df)}")
 print(f"Columns: {len(df.columns)}")
 print()
 print("Features extracted:")
 print(" - Shannon Entropy (T³)")
 print(" - Spectral Entropy (T³)")
 print(" - Dominant Eigenvalue (Perron-Frobenius)")
 print(" - Eigenvalue Decay Rate")
 print(" - THz Structural Invariants (NO GHz conversion)")
 print(" - Geodesic Length (T³)")
 print(" - Absurdity Gap (POST-OPERATOR)")
 print(" - 256 Progenitor Matrix Elements")

if __name__ == '__main__':
 main()
