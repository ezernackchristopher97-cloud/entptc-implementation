"""
Self-Contained EntPTC Analysis with Toroidal Constraints

Implements the complete EntPTC model with proper toroidal topology.
All core functions are included to avoid import issues.

"""

import os
import sys
import numpy as np
from scipy.io import loadmat
from scipy import signal
from pathlib import Path
import csv
import h5py
import warnings
warnings.filterwarnings('ignore')

# Add path
sys.path.insert(0, '/home/ubuntu/entptc-implementation')
from entptc.refinements.toroidal_grid_topology import (
 ToroidalGrid,
 apply_toroidal_constraint_to_progenitor
)

# Directories
DATA_DIR_SET_1 = "/home/ubuntu/entptc-implementation/data"
DATA_DIR_SET_2 = "/home/ubuntu/entptc-implementation/data/dataset_set_2"
OUTPUT_DIR = "/home/ubuntu/entptc-implementation/outputs/toroidal_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize toroidal grid
TOROIDAL_GRID = ToroidalGrid(grid_size=4, connectivity='von_neumann')
CONSTRAINT_STRENGTH = 0.8

def load_mat_file(file_path):
 """Load MAT file and extract EEG data (supports v7.3 HDF5 format)."""
 try:
 # Try standard loadmat first
 try:
 data = loadmat(file_path)
 is_hdf5 = False
 except NotImplementedError:
 # MATLAB v7.3 format (HDF5)
 data = h5py.File(file_path, 'r')
 is_hdf5 = True
 
 # Try different possible keys
 for key in ['eeg_data', 'data', 'EEG']:
 if key in data:
 eeg = data[key]
 if is_hdf5:
 eeg = np.array(eeg)
 break
 else:
 # Find first non-metadata key
 if is_hdf5:
 keys = [k for k in data.keys()]
 else:
 keys = [k for k in data.keys() if not k.startswith('__')]
 
 if keys:
 eeg = data[keys[0]]
 if is_hdf5:
 eeg = np.array(eeg)
 else:
 if is_hdf5:
 data.close()
 return None
 
 if is_hdf5:
 data.close()
 
 # Ensure shape is (16, n_samples)
 if eeg.shape[0] == 16:
 return eeg
 elif eeg.shape[1] == 16:
 return eeg.T
 else:
 return None
 
 except Exception as e:
 print(f" Error: {e}")
 return None

def compute_plv_coherence(eeg_data):
 """
 Compute Phase Locking Value (PLV) coherence matrix.
 
 Args:
 eeg_data: (16, n_samples) array
 
 Returns:
 coherence: (16, 16) matrix
 """
 n_rois = eeg_data.shape[0]
 coherence = np.zeros((n_rois, n_rois))
 
 # Compute analytic signal using Hilbert transform
 analytic_signals = np.zeros((n_rois, eeg_data.shape[1]), dtype=complex)
 for i in range(n_rois):
 analytic_signals[i, :] = signal.hilbert(eeg_data[i, :])
 
 # Compute PLV between all pairs
 for i in range(n_rois):
 for j in range(i, n_rois):
 phase_i = np.angle(analytic_signals[i, :])
 phase_j = np.angle(analytic_signals[j, :])
 
 phase_diff = phase_i - phase_j
 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
 
 coherence[i, j] = plv
 coherence[j, i] = plv
 
 return coherence

def compute_progenitor_matrix(eeg_data):
 """
 Compute Progenitor Matrix from EEG data.
 
 Per ENTPC.tex: c_ij = λ_ij * exp(-∇S_ij) * |Q(θ_ij)|
 
 Args:
 eeg_data: (16, n_samples) array
 
 Returns:
 progenitor: (16, 16) matrix
 """
 # 1. Coherence matrix (λ_ij)
 coherence = compute_plv_coherence(eeg_data)
 
 # 2. Entropy gradient matrix (∇S_ij)
 # Compute local entropy for each ROI
 entropies = np.zeros(16)
 for i in range(16):
 # Histogram-based entropy
 hist, _ = np.histogram(eeg_data[i, :], bins=50, density=True)
 hist = hist[hist > 0] # Remove zeros
 entropies[i] = -np.sum(hist * np.log(hist + 1e-10))
 
 # Entropy gradient between pairs
 entropy_gradient = np.zeros((16, 16))
 for i in range(16):
 for j in range(16):
 entropy_gradient[i, j] = abs(entropies[i] - entropies[j])
 
 # 3. Quaternion norm matrix (|Q(θ_ij)|)
 # Simplified: use coherence-based quaternion norms
 quaternion_norms = np.ones((16, 16))
 for i in range(16):
 for j in range(16):
 # Quaternion norm based on phase relationship
 quaternion_norms[i, j] = np.sqrt(1 + coherence[i, j]**2)
 
 # Construct Progenitor Matrix
 progenitor = coherence * np.exp(-entropy_gradient) * quaternion_norms
 
 # Ensure non-negative
 progenitor = np.abs(progenitor)
 
 return progenitor

def compute_eigendecomposition(matrix):
 """
 Compute eigendecomposition for Perron-Frobenius operator.
 
 Args:
 matrix: (16, 16) matrix
 
 Returns:
 eigenvalues: sorted eigenvalues (descending)
 eigenvectors: corresponding eigenvectors
 """
 eigenvalues, eigenvectors = np.linalg.eig(matrix)
 
 # Sort by magnitude (descending)
 idx = np.argsort(np.abs(eigenvalues))[::-1]
 eigenvalues = eigenvalues[idx]
 eigenvectors = eigenvectors[:, idx]
 
 # Take real parts
 eigenvalues = np.real(eigenvalues)
 eigenvectors = np.real(eigenvectors)
 
 return eigenvalues, eigenvectors

def compute_entropy(matrix):
 """
 Compute entropy of matrix.
 
 Args:
 matrix: (16, 16) matrix
 
 Returns:
 entropy: scalar value
 """
 # Normalize to probability distribution
 matrix_flat = matrix.flatten()
 matrix_flat = matrix_flat[matrix_flat > 0] # Remove zeros
 matrix_flat = matrix_flat / np.sum(matrix_flat) # Normalize
 
 # Shannon entropy
 entropy = -np.sum(matrix_flat * np.log(matrix_flat + 1e-10))
 
 return entropy

def compute_absurdity_gap(progenitor, eigenvalues, eigenvectors):
 """
 Compute Absurdity Gap (POST-OPERATOR ONLY).
 
 Per ENTPC.tex lines 733-734: computed after operator application.
 
 Args:
 progenitor: (16, 16) Progenitor Matrix
 eigenvalues: eigenvalues from decomposition
 eigenvectors: eigenvectors from decomposition
 
 Returns:
 dict with L1, L2, Linf norms
 """
 # Reconstruct matrix from dominant mode
 dominant_mode = eigenvalues[0] * np.outer(eigenvectors[:, 0], eigenvectors[:, 0])
 
 # Absurdity Gap = difference between full matrix and dominant mode
 gap = progenitor - dominant_mode
 
 # Compute norms
 l1 = np.sum(np.abs(gap))
 l2 = np.linalg.norm(gap, 'fro') # Frobenius norm
 linf = np.max(np.abs(gap))
 
 return {
 'L1': l1,
 'L2': l2,
 'Linf': linf
 }

def classify_regime(spectral_gap):
 """
 Classify regime based on spectral gap.
 
 Per ENTPC.tex lines 669-676:
 - Regime I (Local Stabilized): spectral_gap > 2.0
 - Regime II (Transitional): 1.2 ≤ spectral_gap ≤ 2.0
 - Regime III (Global Experience): spectral_gap < 1.5
 
 Args:
 spectral_gap: λ_max / λ_2
 
 Returns:
 regime name
 """
 if spectral_gap > 2.0:
 return "Regime_I"
 elif 1.2 <= spectral_gap <= 2.0:
 return "Regime_II"
 else:
 return "Regime_III"

def analyze_file(file_path, subject_id, condition, dataset_name):
 """Analyze single MAT file with toroidal constraints."""
 print(f" {os.path.basename(file_path)}", end=" ... ")
 
 # Load data
 eeg_data = load_mat_file(file_path)
 if eeg_data is None:
 print("FAILED (load error)")
 return None
 
 if eeg_data.shape[0] != 16:
 print(f"FAILED (wrong shape: {eeg_data.shape})")
 return None
 
 try:
 # Compute Progenitor Matrix (UNCONSTRAINED)
 progenitor_unconstrained = compute_progenitor_matrix(eeg_data)
 
 # Apply TOROIDAL CONSTRAINT
 progenitor_constrained = apply_toroidal_constraint_to_progenitor(
 progenitor_unconstrained,
 TOROIDAL_GRID,
 strength=CONSTRAINT_STRENGTH
 )
 
 # Eigendecomposition
 eigenvalues, eigenvectors = compute_eigendecomposition(progenitor_constrained)
 
 # Metrics
 lambda_max = eigenvalues[0]
 spectral_gap = eigenvalues[0] / eigenvalues[1] if len(eigenvalues) > 1 else 0.0
 entropy = compute_entropy(progenitor_constrained)
 absurdity_gap = compute_absurdity_gap(progenitor_constrained, eigenvalues, eigenvectors)
 regime = classify_regime(spectral_gap)
 
 print("OK")
 
 return {
 'subject_id': subject_id,
 'condition': condition,
 'dataset': dataset_name,
 'file_path': file_path,
 'lambda_max': lambda_max,
 'spectral_gap': spectral_gap,
 'entropy': entropy,
 'absurdity_gap_l1': absurdity_gap['L1'],
 'absurdity_gap_l2': absurdity_gap['L2'],
 'absurdity_gap_linf': absurdity_gap['Linf'],
 'regime': regime,
 'toroidal_constrained': True,
 'constraint_strength': CONSTRAINT_STRENGTH
 }
 
 except Exception as e:
 print(f"FAILED ({e})")
 return None

def process_dataset(data_dir, dataset_name, file_pattern="*.mat"):
 """Process all files in a dataset."""
 print(f"\n{'='*80}")
 print(f"DATASET: {dataset_name}")
 print(f"{'='*80}")
 
 mat_files = sorted(Path(data_dir).glob(file_pattern))
 print(f"Files found: {len(mat_files)}\n")
 
 results = []
 
 for i, mat_file in enumerate(mat_files, 1):
 print(f"[{i}/{len(mat_files)}]", end=" ")
 
 # Parse filename
 filename = mat_file.stem
 
 if 'sub-' in filename:
 parts = filename.split('_')
 subject_id = parts[0]
 task = [p for p in parts if 'task-' in p][0] if any('task-' in p for p in parts) else 'unknown'
 condition = task.replace('task-', '')
 elif filename.startswith('S'):
 parts = filename.split('_')
 subject_id = parts[0]
 condition = parts[1].replace('task-', '') if len(parts) > 1 else 'unknown'
 else:
 subject_id = filename
 condition = 'unknown'
 
 result = analyze_file(str(mat_file), subject_id, condition, dataset_name)
 
 if result:
 results.append(result)
 
 print(f"\n{'='*80}")
 print(f"Processed: {len(results)}/{len(mat_files)}")
 print(f"{'='*80}")
 
 return results

def save_results(results, output_path):
 """Save results to CSV."""
 if not results:
 return
 
 keys = results[0].keys()
 
 with open(output_path, 'w', newline='') as f:
 writer = csv.DictWriter(f, fieldnames=keys)
 writer.writeheader()
 writer.writerows(results)
 
 print(f"\nSaved: {output_path} ({len(results)} rows)")

def main():
 print("=" * 80)
 print("ENTPTC ANALYSIS WITH TOROIDAL CONSTRAINTS")
 print("=" * 80)
 print(f"Toroidal grid: 4×4 on T²")
 print(f"Connectivity: von Neumann (4-neighbors)")
 print(f"Constraint strength: {CONSTRAINT_STRENGTH}")
 print("=" * 80)
 
 all_results = []
 
 # Dataset Set 1
 if os.path.exists(DATA_DIR_SET_1):
 results_set1 = process_dataset(DATA_DIR_SET_1, "Dataset_Set_1", "sub-*.mat")
 all_results.extend(results_set1)
 save_results(results_set1, os.path.join(OUTPUT_DIR, "dataset_set_1_toroidal.csv"))
 
 # Dataset Set 2
 if os.path.exists(DATA_DIR_SET_2):
 results_set2 = process_dataset(DATA_DIR_SET_2, "Dataset_Set_2", "*.mat")
 all_results.extend(results_set2)
 save_results(results_set2, os.path.join(OUTPUT_DIR, "dataset_set_2_toroidal.csv"))
 
 # Combined
 save_results(all_results, os.path.join(OUTPUT_DIR, "all_datasets_toroidal.csv"))
 
 print("\n" + "=" * 80)
 print("ANALYSIS COMPLETE")
 print("=" * 80)
 print(f"Total files: {len(all_results)}")
 print(f"Output: {OUTPUT_DIR}")
 print("=" * 80)

if __name__ == '__main__':
 main()
