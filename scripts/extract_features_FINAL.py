#!/usr/bin/env python3.11
"""
EntPTC Feature Extraction - ABSOLUTE FINAL VERSION
Uses REAL existing model with VERIFIED method names
"""
import numpy as np
import h5py
import os
import csv
import sys
from glob import glob

sys.path.insert(0, '/home/ubuntu/entptc-archive')

from entptc.core.progenitor import ProgenitorMatrix
from entptc.core.perron_frobenius import PerronFrobeniusOperator 
from entptc.analysis.thz_inference import THzStructuralInvariants

print("=" * 80)
print("EntPTC Feature Extraction - FINAL - REAL Model")
print("=" * 80)

data_dir = '/home/ubuntu/entptc-archive/data'
mat_files = sorted(glob(os.path.join(data_dir, 'sub-*.mat')))
print(f"\nFound {len(mat_files)} .mat files\n")

progenitor_builder = ProgenitorMatrix()
pf_operator = PerronFrobeniusOperator()
thz_analyzer = THzStructuralInvariants()

results = []

for idx, mat_file in enumerate(mat_files):
 filename = os.path.basename(mat_file)
 print(f"[{idx+1}/{len(mat_files)}] {filename}", end=" ... ")
 
 parts = filename.replace('.mat', '').split('_')
 subject_id = parts[0].replace('sub-', '')
 condition = f"{parts[3]}_{parts[4]}"
 
 try:
 with h5py.File(mat_file, 'r') as f:
 data_matrix = np.array(f['data_matrix']).T # (64, time)
 
 nChan = data_matrix.shape[0]
 assert nChan == 64, f"Expected 64 channels, got {nChan}"
 
 # Aggregate to 16 ROIs
 roi_data = np.zeros((16, data_matrix.shape[1]))
 for i in range(16):
 roi_data[i] = np.mean(data_matrix[i*4:(i+1)*4, :], axis=0)
 
 # Build Progenitor Matrix - VERIFIED METHOD
 progenitor_matrix = progenitor_builder.construct_from_eeg_data(roi_data)
 
 # Perron-Frobenius - VERIFIED METHOD
 eigenvalues, eigenvectors = pf_operator.compute_eigendecomposition(progenitor_matrix)
 
 # THz structural invariants - VERIFIED METHOD
 thz_results = thz_analyzer.extract_all_invariants(eigenvalues)
 
 # Compute entropy metrics
 shannon_entropy_mean = -np.mean([
 np.sum(p * np.log(p + 1e-10)) 
 for p in [np.abs(roi_data[i]) / (np.sum(np.abs(roi_data[i])) + 1e-10) 
 for i in range(16)]
 ])
 
 spectral_entropy_mean = np.mean([
 -np.sum((np.abs(np.fft.fft(roi_data[i]))**2) * 
 np.log(np.abs(np.fft.fft(roi_data[i]))**2 + 1e-10))
 for i in range(16)
 ])
 
 log_eigenvalues = np.log(np.abs(eigenvalues) + 1e-10)
 coherence_eigen_decay = np.polyfit(range(len(log_eigenvalues)), log_eigenvalues, 1)[0]
 dominant_eigenvalue = np.abs(eigenvalues[0])
 
 matrix_16x16_flat = progenitor_matrix.flatten().tolist()
 
 result = {
 'subject_id': subject_id,
 'condition': condition,
 'shannon_entropy_mean': float(shannon_entropy_mean),
 'spectral_entropy_mean': float(spectral_entropy_mean),
 'coherence_eigen_decay': float(coherence_eigen_decay),
 'dominant_eigenvalue': float(dominant_eigenvalue),
 'thz_eigenvalue_ratios_mean': float(np.mean(thz_results['eigenvalue_ratios'])),
 'thz_spectral_gaps_mean': float(np.mean(thz_results['spectral_gaps'])),
 'thz_symmetry_breaking': float(thz_results['symmetry_breaking']),
 }
 
 for i, val in enumerate(matrix_16x16_flat):
 result[f'matrix_elem_{i}'] = float(val)
 
 results.append(result)
 print("✓")
 
 except Exception as e:
 print(f"ERROR: {e}")
 continue

output_file = '/home/ubuntu/entptc-archive/entptc_features.csv'
if results:
 with open(output_file, 'w', newline='') as f:
 writer = csv.DictWriter(f, fieldnames=results[0].keys())
 writer.writeheader()
 writer.writerows(results)
 
 print(f"\n{'=' * 80}")
 print(f"✓ SUCCESS! Processed {len(results)}/{len(mat_files)} files")
 print(f" Output: {output_file}")
 print(f" Rows: {len(results)+1} (including header)")
 print(f" Columns: {len(results[0].keys())}")
 print(f"{'=' * 80}")
else:
 print("\nERROR: No results generated")
