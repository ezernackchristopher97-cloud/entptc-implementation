#!/usr/bin/env python3.11
"""
EntPTC Feature Extraction: WITH ABSURDITY GAP
Adds the missing Absurdity Gap component to the working extraction.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# Import working components
from entptc.core.progenitor import ProgenitorMatrix
from entptc.core.perron_frobenius import PerronFrobeniusOperator
from entptc.analysis.thz_inference import THzStructuralInvariants
from entptc.core.absurdity_gap import AbsurdityGap

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

def compute_shannon_entropy(roi_data):
 """Compute Shannon entropy for each ROI"""
 entropies = []
 for i in range(roi_data.shape[0]):
 signal = roi_data[i, :]
 hist, _ = np.histogram(signal, bins=50, density=True)
 hist = hist[hist > 0]
 entropy = -np.sum(hist * np.log2(hist + 1e-12))
 entropies.append(entropy)
 return np.mean(entropies)

def compute_spectral_entropy(roi_data, srate):
 """Compute spectral entropy for each ROI"""
 entropies = []
 for i in range(roi_data.shape[0]):
 signal = roi_data[i, :]
 fft = np.fft.rfft(signal)
 psd = np.abs(fft) ** 2
 psd_norm = psd / (np.sum(psd) + 1e-12)
 psd_norm = psd_norm[psd_norm > 0]
 entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
 entropies.append(entropy)
 return np.mean(entropies)

def extract_features(filepath):
 """Extract features using working components + Absurdity Gap"""
 
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
 
 # 4. ENTROPY
 shannon_entropy_mean = compute_shannon_entropy(roi_data)
 spectral_entropy_mean = compute_spectral_entropy(roi_data, srate)
 
 # 5. ABSURDITY GAP (POST-OPERATOR)
 # Store eigenvector for later pairing
 absurdity_gap_value = None
 
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
 'dominant_eigenvector': dominant_eigenvector, # Store for absurdity gap
 }
 
 # Add matrix elements
 for i in range(256):
 features[f'matrix_elem_{i}'] = matrix_flat[i]
 
 return features

def compute_absurdity_gaps(results):
 """Compute Absurdity Gap for all pre/post pairs"""
 absurdity_gap_calc = AbsurdityGap()
 
 # Group by subject
 by_subject = {}
 for r in results:
 sid = r['subject_id']
 if sid not in by_subject:
 by_subject[sid] = {}
 
 if 'pre' in r['condition']:
 by_subject[sid]['pre'] = r
 elif 'post' in r['condition']:
 by_subject[sid]['post'] = r
 
 # Compute gaps
 for sid, pair in by_subject.items():
 if 'pre' in pair and 'post' in pair:
 psi_pre = pair['pre']['dominant_eigenvector']
 psi_post = pair['post']['dominant_eigenvector']
 gap = absurdity_gap_calc.compute_gap(psi_pre, psi_post)
 
 # Add to both pre and post
 pair['pre']['absurdity_gap'] = gap
 pair['post']['absurdity_gap'] = gap
 
 # Remove eigenvector from results (don't save to CSV)
 for r in results:
 if 'absurdity_gap' not in r:
 r['absurdity_gap'] = np.nan
 del r['dominant_eigenvector']
 
 return results

def main():
 data_dir = Path('data')
 mat_files = sorted(data_dir.glob('*.mat'))
 
 print(f"Found {len(mat_files)} .mat files")
 print("Starting feature extraction WITH ABSURDITY GAP...")
 print()
 
 results = []
 for idx, mat_file in enumerate(mat_files, 1):
 try:
 print(f"[{idx}/{len(mat_files)}] Processing: {mat_file.name}...", end=' ')
 features = extract_features(str(mat_file))
 results.append(features)
 print("✓")
 except Exception as e:
 print(f"✗ Error: {e}")
 continue
 
 # Compute Absurdity Gaps for all pre/post pairs
 print()
 print("Computing Absurdity Gaps for pre/post pairs...")
 results = compute_absurdity_gaps(results)
 
 # Save to CSV
 df = pd.DataFrame(results)
 output_file = 'entptc_features_WITH_ABSURDITY_GAP.csv'
 df.to_csv(output_file, index=False)
 
 print()
 print(f"Feature extraction WITH ABSURDITY GAP finished!")
 print(f"Output: {output_file}")
 print(f"Rows: {len(df)}")
 print(f"Columns: {len(df.columns)}")
 print()
 print("Features extracted:")
 print(" - Shannon Entropy")
 print(" - Spectral Entropy")
 print(" - Dominant Eigenvalue (Perron-Frobenius)")
 print(" - Eigenvalue Decay Rate")
 print(" - THz Structural Invariants (NO GHz conversion)")
 print(" - Absurdity Gap (POST-OPERATOR) ← NEW!")
 print(" - 256 Progenitor Matrix Elements")

if __name__ == '__main__':
 main()
