"""
EntPTC Analysis with Toroidal Grid-Cell Constraints

Runs the complete EntPTC pipeline with PROPER toroidal topology enforcement.

Key Features:
- 16 ROIs arranged as 4×4 grid on T²
- Periodic boundary conditions (wraparound)
- Von Neumann connectivity (4-neighbors)
- Toroidal constraint applied to Progenitor Matrix
- Geodesics computed on torus
- All metrics computed with toroidal structure

"""

import os
import sys
import numpy as np
from scipy.io import loadmat, savemat
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add entptc to path
sys.path.insert(0, '/home/ubuntu/entptc-implementation')

# Import toroidal grid
from entptc.refinements.toroidal_grid_topology import (
 ToroidalGrid,
 apply_toroidal_constraint_to_progenitor
)

# Import EntPTC modules
from entptc.core.progenitor import ProgenitorMatrix
from entptc.core.perron_frobenius import PerronFrobeniusOperator
from entptc.core.entropy import compute_entropy
from entptc.core.absurdity_gap import compute_absurdity_gap_post_operator
from entptc.core.quaternion import Quaternion

# Directories
DATA_DIR_SET_1 = "/home/ubuntu/entptc-implementation/data"
DATA_DIR_SET_2 = "/home/ubuntu/entptc-implementation/data/dataset_set_2"
OUTPUT_DIR = "/home/ubuntu/entptc-implementation/outputs/toroidal_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize toroidal grid
TOROIDAL_GRID = ToroidalGrid(grid_size=4, connectivity='von_neumann')
TOROIDAL_CONSTRAINT_STRENGTH = 0.8 # 80% constraint strength

def load_mat_file(file_path):
 """Load MAT file and extract EEG data."""
 try:
 data = loadmat(file_path)
 
 # Try different possible keys
 if 'eeg_data' in data:
 eeg = data['eeg_data']
 elif 'data' in data:
 eeg = data['data']
 elif 'EEG' in data:
 eeg = data['EEG']
 else:
 # Find first non-metadata key
 keys = [k for k in data.keys() if not k.startswith('__')]
 if keys:
 eeg = data[keys[0]]
 else:
 return None
 
 # Ensure shape is (16, n_samples) or (n_samples, 16)
 if eeg.shape[0] == 16:
 return eeg # (16, n_samples)
 elif eeg.shape[1] == 16:
 return eeg.T # Transpose to (16, n_samples)
 else:
 print(f" WARNING: Unexpected shape {eeg.shape}")
 return None
 
 except Exception as e:
 print(f" Error loading {file_path}: {e}")
 return None

def compute_coherence_matrix(eeg_data):
 """
 Compute coherence matrix from EEG data.
 
 Args:
 eeg_data: (16, n_samples) array
 
 Returns:
 coherence: (16, 16) matrix
 """
 n_rois = eeg_data.shape[0]
 coherence = np.zeros((n_rois, n_rois))
 
 for i in range(n_rois):
 for j in range(i, n_rois):
 # Compute phase locking value (PLV)
 phase_i = np.angle(np.fft.fft(eeg_data[i, :]))
 phase_j = np.angle(np.fft.fft(eeg_data[j, :]))
 
 phase_diff = phase_i - phase_j
 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
 
 coherence[i, j] = plv
 coherence[j, i] = plv
 
 return coherence

def analyze_single_file(file_path, subject_id, condition, dataset_name):
 """
 Analyze single MAT file with toroidal constraints.
 
 Returns:
 dict with all metrics
 """
 print(f"\n Processing: {os.path.basename(file_path)}")
 
 # Load data
 eeg_data = load_mat_file(file_path)
 if eeg_data is None:
 return None
 
 assert eeg_data.shape[0] == 16, f"Expected 16 ROIs, got {eeg_data.shape[0]}"
 
 # Compute coherence matrix
 coherence = compute_coherence_matrix(eeg_data)
 
 # Compute Progenitor Matrix (UNCONSTRAINED)
 pm = ProgenitorMatrix()
 progenitor_unconstrained = pm.construct_from_eeg_data(eeg_data)
 
 # Apply TOROIDAL CONSTRAINT
 progenitor_constrained = apply_toroidal_constraint_to_progenitor(
 progenitor_unconstrained,
 TOROIDAL_GRID,
 strength=TOROIDAL_CONSTRAINT_STRENGTH
 )
 
 # Perron-Frobenius operator
 pf_operator = PerronFrobeniusOperator(progenitor_constrained)
 eigenvalues, eigenvectors = pf_operator.compute_eigendecomposition()
 
 # Extract metrics
 lambda_max = eigenvalues[0]
 spectral_gap = eigenvalues[0] / eigenvalues[1] if len(eigenvalues) > 1 else 0.0
 
 # Entropy
 entropy = compute_entropy(progenitor_constrained)
 
 # Absurdity Gap (POST-OPERATOR ONLY)
 absurdity_gap = compute_absurdity_gap_post_operator(
 progenitor_constrained,
 eigenvalues,
 eigenvectors
 )
 
 # Regime classification
 if spectral_gap > 2.0:
 regime = "Regime_I"
 elif 1.2 <= spectral_gap <= 2.0:
 regime = "Regime_II"
 else:
 regime = "Regime_III"
 
 # Return metrics
 return {
 'subject_id': subject_id,
 'condition': condition,
 'dataset': dataset_name,
 'file_path': file_path,
 'lambda_max': lambda_max,
 'spectral_gap': spectral_gap,
 'entropy': entropy,
 'absurdity_gap_l2': absurdity_gap.get('L2', np.nan),
 'absurdity_gap_l1': absurdity_gap.get('L1', np.nan),
 'absurdity_gap_linf': absurdity_gap.get('Linf', np.nan),
 'regime': regime,
 'toroidal_constrained': True,
 'constraint_strength': TOROIDAL_CONSTRAINT_STRENGTH
 }

def process_dataset(data_dir, dataset_name, file_pattern="*.mat"):
 """
 Process all files in a dataset directory.
 
 Args:
 data_dir: directory containing MAT files
 dataset_name: name of dataset (for labeling)
 file_pattern: glob pattern for files
 
 Returns:
 list of result dictionaries
 """
 print(f"\n{'='*80}")
 print(f"PROCESSING DATASET: {dataset_name}")
 print(f"{'='*80}")
 print(f"Directory: {data_dir}")
 
 # Find all MAT files
 mat_files = sorted(Path(data_dir).glob(file_pattern))
 print(f"Found {len(mat_files)} MAT files")
 
 results = []
 
 for i, mat_file in enumerate(mat_files):
 print(f"\n[{i+1}/{len(mat_files)}] {mat_file.name}")
 
 # Parse filename to extract metadata
 filename = mat_file.stem
 
 # Try to extract subject_id and condition
 if 'sub-' in filename:
 # Dataset Set 1 format: sub-XXX_ses-XX_task-EyesClosed_acq-pre_eeg
 parts = filename.split('_')
 subject_id = parts[0]
 task = [p for p in parts if 'task-' in p][0] if any('task-' in p for p in parts) else 'unknown'
 condition = task.replace('task-', '')
 elif filename.startswith('S'):
 # Dataset Set 2 format: S001_task-EyesOpen_eeg
 parts = filename.split('_')
 subject_id = parts[0]
 condition = parts[1].replace('task-', '') if len(parts) > 1 else 'unknown'
 else:
 subject_id = filename
 condition = 'unknown'
 
 # Analyze file
 result = analyze_single_file(str(mat_file), subject_id, condition, dataset_name)
 
 if result:
 results.append(result)
 
 print(f"\n{'='*80}")
 print(f"DATASET COMPLETE: {dataset_name}")
 print(f"Successfully processed: {len(results)}/{len(mat_files)} files")
 print(f"{'='*80}")
 
 return results

def save_results(results, output_path):
 """Save results to CSV."""
 import csv
 
 if not results:
 print("No results to save")
 return
 
 # Get all keys
 keys = results[0].keys()
 
 with open(output_path, 'w', newline='') as f:
 writer = csv.DictWriter(f, fieldnames=keys)
 writer.writeheader()
 writer.writerows(results)
 
 print(f"\nResults saved: {output_path}")
 print(f"Total rows: {len(results)}")

def main():
 print("=" * 80)
 print("ENTPTC ANALYSIS WITH TOROIDAL CONSTRAINTS")
 print("=" * 80)
 print(f"Toroidal grid: 4×4 on T²")
 print(f"Connectivity: von Neumann (4-neighbors)")
 print(f"Constraint strength: {TOROIDAL_CONSTRAINT_STRENGTH}")
 print("=" * 80)
 
 all_results = []
 
 # Process Dataset Set 1
 if os.path.exists(DATA_DIR_SET_1):
 results_set1 = process_dataset(DATA_DIR_SET_1, "Dataset_Set_1", "sub-*.mat")
 all_results.extend(results_set1)
 
 # Save Set 1 results
 save_results(results_set1, os.path.join(OUTPUT_DIR, "dataset_set_1_toroidal_results.csv"))
 
 # Process Dataset Set 2
 if os.path.exists(DATA_DIR_SET_2):
 results_set2 = process_dataset(DATA_DIR_SET_2, "Dataset_Set_2", "*.mat")
 all_results.extend(results_set2)
 
 # Save Set 2 results
 save_results(results_set2, os.path.join(OUTPUT_DIR, "dataset_set_2_toroidal_results.csv"))
 
 # Save combined results
 save_results(all_results, os.path.join(OUTPUT_DIR, "all_datasets_toroidal_results.csv"))
 
 print("\n" + "=" * 80)
 print("ANALYSIS COMPLETE")
 print("=" * 80)
 print(f"Total files processed: {len(all_results)}")
 print(f"Output directory: {OUTPUT_DIR}")
 print("=" * 80)

if __name__ == '__main__':
 main()
