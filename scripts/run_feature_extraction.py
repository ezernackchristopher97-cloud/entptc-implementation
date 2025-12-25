#!/usr/bin/env python3.11
"""
EntPTC Feature Extraction Script
Uses existing EntPTC model modules to process .mat files and generate entptc_features.csv
"""

import numpy as np
import scipy.io
import h5py
import os
import csv
import sys
from glob import glob

# Add repo to path
sys.path.insert(0, '/home/ubuntu/entptc-archive')

# Import existing EntPTC modules
from entptc.core.progenitor import ProgenitorMatrix
from entptc.core.perron_frobenius import PerronFrobeniusOperator
from entptc.core.quaternion import Quaternion, QuaternionicHilbertSpace
from entptc.core.entropy import EntropyField, ToroidalManifold
from entptc.core.thz_inference import THzStructuralInvariants

print("=" * 80)
print("EntPTC Feature Extraction")
print("Using existing 9,100+ line EntPTC model")
print("=" * 80)

# Find all .mat files
data_dir = '/home/ubuntu/entptc-archive/data'
mat_files = sorted(glob(os.path.join(data_dir, '*.mat')))
print(f"\nFound {len(mat_files)} .mat files")

# Initialize EntPTC components
progenitor_builder = ProgenitorMatrix()
pf_operator = PerronFrobeniusOperator()
quat_space = QuaternionicHilbertSpace(dimension=16)
thz_analyzer = THzStructuralInvariants()

# Results list
results = []

# Process each file
for idx, mat_file in enumerate(mat_files):
 filename = os.path.basename(mat_file)
 print(f"\n[{idx+1}/{len(mat_files)}] Processing: {filename}")
 
 # Parse filename
 parts = filename.replace('.mat', '').split('_')
 subject_id = parts[0].replace('sub-', '')
 condition = f"{parts[3]}_{parts[4]}" # task_acq
 
 try:
 # Load .mat file (try scipy first, fall back to h5py for v7.3)
 try:
 mat_data = scipy.io.loadmat(mat_file)
 # Extract data_matrix
 if 'data_matrix' in mat_data:
 data_matrix = mat_data['data_matrix']
 else:
 # Try other common variable names
 keys = [k for k in mat_data.keys() if not k.startswith('__')]
 data_matrix = mat_data[keys[0]]
 except NotImplementedError:
 # MATLAB v7.3 file, use h5py
 with h5py.File(mat_file, 'r') as f:
 if 'data_matrix' in f:
 data_matrix = np.array(f["data_matrix"]).T
 else:
 # Try other common variable names
 keys = [k for k in f.keys()]
 data_matrix = np.array(f[keys[0]])
 
 # CRITICAL: Verify 64 channels
 nChan = data_matrix.shape[0]
 print(f" Channels: {nChan}")
 
 if nChan != 64:
 print(f" WARNING: Expected 64 channels, got {nChan}. Skipping.")
 continue
 
 # Assert 0-indexed (Python default)
 assert data_matrix.shape[0] == 64, f"Channel count assertion failed: {nChan} != 64"
 
 # Aggregate to 16 ROIs (4 channels per ROI)
 roi_data = np.zeros((16, data_matrix.shape[1]))
 for i in range(16):
 roi_data[i] = np.mean(data_matrix[i*4:(i+1)*4, :], axis=0)
 
 print(f" Aggregated to 16 ROIs")
 
 # Construct 16 quaternions from 16 ROIs
 quaternions = np.zeros((16, 4))
 for i in range(16):
 signal = roi_data[i, :]
 # Simple quaternion construction
 quaternions[i] = [
 np.mean(signal),
 np.std(signal),
 np.percentile(signal, 75) - np.percentile(signal, 25),
 np.max(np.abs(signal))
 ]
 
 # Normalize quaternions
 for i in range(16):
 norm = np.linalg.norm(quaternions[i])
 if norm > 0:
 quaternions[i] /= norm
 
 print(f" Constructed 16 quaternions")
 
 # Compute coherence matrix (using correlation as proxy)
 coherence = np.corrcoef(roi_data)
 
 # Build 16x16 Progenitor Matrix using existing module
 progenitor_matrix = progenitor_builder.construct_progenitor_matrix(
 quaternions, coherence
 )
 
 print(f" Built 16x16 Progenitor Matrix")
 
 # Perron-Frobenius collapse
 dominant_eigenvector, eigenvalues = pf_operator.compute_dominant_eigenvector(
 progenitor_matrix
 )
 
 print(f" Perron-Frobenius collapse complete")
 
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
 
 # THz structural invariants using existing module
 thz_results = thz_analyzer.compute_structural_invariants(eigenvalues)
 
 # Eigenvalue decay slope
 log_eigenvalues = np.log(np.abs(eigenvalues) + 1e-10)
 coherence_eigen_decay = np.polyfit(range(len(log_eigenvalues)), log_eigenvalues, 1)[0]
 
 # Dominant eigenvalue
 dominant_eigenvalue = np.abs(eigenvalues[0])
 
 # Flatten 16x16 matrix to 256 values
 matrix_16x16_flat = progenitor_matrix.flatten().tolist()
 
 # Store results
 result = {
 'subject_id': subject_id,
 'condition': condition,
 'shannon_entropy_mean': float(shannon_entropy_mean),
 'spectral_entropy_mean': float(spectral_entropy_mean),
 'coherence_eigen_decay': float(coherence_eigen_decay),
 'dominant_eigenvalue': float(dominant_eigenvalue),
 'thz_eigenvalue_ratio': float(thz_results['eigenvalue_ratio']),
 'thz_spectral_gap': float(thz_results['spectral_gap']),
 'thz_decay_exponent': float(thz_results['decay_exponent']),
 }
 
 # Add flattened matrix
 for i, val in enumerate(matrix_16x16_flat):
 result[f'matrix_elem_{i}'] = float(val)
 
 results.append(result)
 
 print(f" ✓ Features extracted successfully")
 
 except Exception as e:
 print(f" ERROR: {e}")
 continue

# Save to CSV
output_file = '/home/ubuntu/entptc-archive/entptc_features.csv'
if results:
 with open(output_file, 'w', newline='') as f:
 writer = csv.DictWriter(f, fieldnames=results[0].keys())
 writer.writeheader()
 writer.writerows(results)
 
 print(f"\n{'=' * 80}")
 print(f"✓ Feature extraction complete!")
 print(f" Processed: {len(results)} files")
 print(f" Output: {output_file}")
 print(f"{'=' * 80}")
else:
 print("\nERROR: No results generated")
