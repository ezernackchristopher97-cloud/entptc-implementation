"""
Matrix-First Pipeline
=====================

Per MATRIX_FIRST_PROTOCOL:
- Map ALL conditions through Progenitor Matrix
- Report collapse objects (eigenstructure, spectral gap, entropy)
- Compare via collapse drift (NOT projection metrics)

Conditions:
- Intact (baseline)
- Ablations (boundary removal, adjacency scramble, phase destruction, channel randomization)
- EO (eyes open)
- EC (eyes closed)
- Task states (if applicable)

"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import scipy.io as sio

# Set random seed
np.random.seed(42)

# ============================================================================
# PROGENITOR MATRIX CONSTRUCTION
# ============================================================================

def construct_progenitor_matrix(data: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
 """
 Construct Progenitor Matrix from data and geometry.
 
 Per paper: Constraint-weighted covariance matrix.
 
 Args:
 data: (n_rois, n_samples) array
 adjacency: (n_rois, n_rois) adjacency matrix (toroidal grid)
 
 Returns:
 progenitor: (n_rois, n_rois) Progenitor Matrix
 """
 # Covariance matrix (spatial)
 cov = np.cov(data)
 
 # Normalize by trace
 cov_norm = cov / (np.trace(cov) + 1e-10)
 
 # Apply toroidal constraint (adjacency weighting)
 progenitor = cov_norm * adjacency
 
 # Symmetrize
 progenitor = (progenitor + progenitor.T) / 2
 
 return progenitor

# ============================================================================
# COLLAPSE OBJECT EXTRACTION
# ============================================================================

def extract_collapse_object(progenitor: np.ndarray) -> Dict:
 """
 Extract collapse object from Progenitor Matrix.
 
 Collapse object contains:
 - Eigenmode dominance profile (λ₁, λ₂, ..., λₙ)
 - Spectral gap (λ₁ - λ₂)
 - Participation ratio (effective dimensionality)
 - Von Neumann entropy
 - Entropy gradient (if multiple segments)
 - Stability (CV across segments)
 
 Args:
 progenitor: (n_rois, n_rois) Progenitor Matrix
 
 Returns:
 collapse_object: dict with all collapse quantities
 """
 # Eigendecomposition
 eigenvalues, eigenvectors = np.linalg.eigh(progenitor)
 
 # Sort descending
 idx = np.argsort(eigenvalues)[::-1]
 eigenvalues = eigenvalues[idx]
 eigenvectors = eigenvectors[:, idx]
 
 # Eigenmode dominance profile
 eigenmode_profile = eigenvalues.copy()
 
 # Spectral gap
 spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0
 
 # Participation ratio (effective dimensionality)
 participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2) if np.sum(eigenvalues**2) > 0 else 0
 
 # Von Neumann entropy
 eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
 eigenvalues_norm = eigenvalues_pos / np.sum(eigenvalues_pos)
 von_neumann_entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm))
 
 # Maximum entropy (uniform distribution)
 max_entropy = np.log(len(eigenvalues_pos))
 
 # Normalized entropy
 entropy_normalized = von_neumann_entropy / max_entropy if max_entropy > 0 else 0
 
 # Eigenvalue decay rate (exponential fit)
 # log(λ_i) ~ -α * i
 if len(eigenvalues_pos) > 2:
 indices = np.arange(len(eigenvalues_pos))
 log_eigenvalues = np.log(eigenvalues_pos + 1e-10)
 decay_rate = -np.polyfit(indices, log_eigenvalues, 1)[0]
 else:
 decay_rate = 0
 
 collapse_object = {
 'eigenmode_profile': eigenmode_profile.tolist(),
 'dominant_eigenvalue': float(eigenvalues[0]),
 'spectral_gap': float(spectral_gap),
 'participation_ratio': float(participation_ratio),
 'von_neumann_entropy': float(von_neumann_entropy),
 'entropy_normalized': float(entropy_normalized),
 'eigenvalue_decay_rate': float(decay_rate),
 'n_modes': len(eigenvalues),
 'progenitor_trace': float(np.trace(progenitor))
 }
 
 return collapse_object

# ============================================================================
# COLLAPSE DRIFT COMPUTATION
# ============================================================================

def compute_collapse_drift(collapse_baseline: Dict, collapse_condition: Dict) -> Dict:
 """
 Compute drift between two collapse objects.
 
 Per MATRIX_FIRST_PROTOCOL:
 - Compare via collapse drift, NOT projection metrics
 - Systematic deformation = uniqueness
 
 Args:
 collapse_baseline: Collapse object for baseline condition
 collapse_condition: Collapse object for test condition
 
 Returns:
 drift_metrics: dict with drift quantities
 """
 # Eigenvalue profile drift (Euclidean distance)
 profile_baseline = np.array(collapse_baseline['eigenmode_profile'])
 profile_condition = np.array(collapse_condition['eigenmode_profile'])
 
 # Pad to same length if needed
 max_len = max(len(profile_baseline), len(profile_condition))
 profile_baseline_padded = np.pad(profile_baseline, (0, max_len - len(profile_baseline)))
 profile_condition_padded = np.pad(profile_condition, (0, max_len - len(profile_condition)))
 
 eigenmode_drift = np.linalg.norm(profile_baseline_padded - profile_condition_padded)
 
 # Spectral gap drift (absolute change)
 spectral_gap_drift = abs(collapse_baseline['spectral_gap'] - collapse_condition['spectral_gap'])
 
 # Entropy drift (absolute change)
 entropy_drift = abs(collapse_baseline['von_neumann_entropy'] - collapse_condition['von_neumann_entropy'])
 
 # Participation ratio drift (absolute change)
 participation_drift = abs(collapse_baseline['participation_ratio'] - collapse_condition['participation_ratio'])
 
 # Decay rate drift (absolute change)
 decay_rate_drift = abs(collapse_baseline['eigenvalue_decay_rate'] - collapse_condition['eigenvalue_decay_rate'])
 
 # Dominant eigenvalue drift (relative change)
 dominant_eigenvalue_drift = abs(collapse_baseline['dominant_eigenvalue'] - collapse_condition['dominant_eigenvalue']) / (collapse_baseline['dominant_eigenvalue'] + 1e-10)
 
 drift_metrics = {
 'eigenmode_drift': float(eigenmode_drift),
 'spectral_gap_drift': float(spectral_gap_drift),
 'entropy_drift': float(entropy_drift),
 'participation_drift': float(participation_drift),
 'decay_rate_drift': float(decay_rate_drift),
 'dominant_eigenvalue_drift_relative': float(dominant_eigenvalue_drift)
 }
 
 return drift_metrics

# ============================================================================
# MATRIX-FIRST PIPELINE
# ============================================================================

def matrix_first_pipeline(data_dict: Dict[str, np.ndarray], adjacency: np.ndarray) -> Dict:
 """
 Run matrix-first pipeline on all conditions.
 
 Per MATRIX_FIRST_PROTOCOL:
 1. Map each condition through Progenitor Matrix
 2. Extract collapse objects
 3. Compare via collapse drift
 
 Args:
 data_dict: Dict mapping condition names to (n_rois, n_samples) arrays
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 results: Dict with collapse objects and drift metrics for all conditions
 """
 results = {
 'collapse_objects': {},
 'collapse_drifts': {}
 }
 
 # Extract collapse objects for all conditions
 print("\n" + "="*80)
 print("MATRIX-FIRST PIPELINE: EXTRACTING COLLAPSE OBJECTS")
 print("="*80)
 
 for condition_name, data in data_dict.items():
 print(f"\nCondition: {condition_name}")
 print(f" Data shape: {data.shape}")
 
 # Construct Progenitor Matrix
 progenitor = construct_progenitor_matrix(data, adjacency)
 print(f" Progenitor trace: {np.trace(progenitor):.6f}")
 
 # Extract collapse object
 collapse_object = extract_collapse_object(progenitor)
 print(f" Dominant eigenvalue: {collapse_object['dominant_eigenvalue']:.6f}")
 print(f" Spectral gap: {collapse_object['spectral_gap']:.6f}")
 print(f" Von Neumann entropy: {collapse_object['von_neumann_entropy']:.4f}")
 print(f" Participation ratio: {collapse_object['participation_ratio']:.4f}")
 
 results['collapse_objects'][condition_name] = collapse_object
 
 # Compute collapse drifts (all conditions vs baseline)
 print("\n" + "="*80)
 print("MATRIX-FIRST PIPELINE: COMPUTING COLLAPSE DRIFTS")
 print("="*80)
 
 baseline_name = list(data_dict.keys())[0] # First condition is baseline
 collapse_baseline = results['collapse_objects'][baseline_name]
 
 for condition_name in data_dict.keys():
 if condition_name == baseline_name:
 continue
 
 print(f"\nDrift: {baseline_name} → {condition_name}")
 
 collapse_condition = results['collapse_objects'][condition_name]
 drift_metrics = compute_collapse_drift(collapse_baseline, collapse_condition)
 
 print(f" Eigenmode drift: {drift_metrics['eigenmode_drift']:.6f}")
 print(f" Spectral gap drift: {drift_metrics['spectral_gap_drift']:.6f}")
 print(f" Entropy drift: {drift_metrics['entropy_drift']:.6f}")
 print(f" Participation drift: {drift_metrics['participation_drift']:.6f}")
 
 results['collapse_drifts'][f"{baseline_name}_to_{condition_name}"] = drift_metrics
 
 print("\n" + "="*80)
 print("MATRIX-FIRST PIPELINE COMPLETE")
 print("="*80)
 
 return results

# ============================================================================
# MAIN RUNNER (EXAMPLE)
# ============================================================================

if __name__ == '__main__':
 from entptc.utils.grid_utils import create_toroidal_grid
 
 # Load data (example: ds004706)
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/matrix_first')
 output_dir.mkdir(parents=True, exist_ok=True)
 
 if not data_path.exists():
 print(f"Data file not found: {data_path}")
 print("This is an example runner. Modify paths as needed.")
 exit(1)
 
 print("Loading data...")
 mat = sio.loadmat(data_path)
 data_intact = mat['eeg_data']
 fs = float(mat['fs'][0, 0])
 
 grid_size = int(np.sqrt(data_intact.shape[0]))
 adjacency = create_toroidal_grid(grid_size)
 
 print(f"Data shape: {data_intact.shape}")
 print(f"Sampling rate: {fs} Hz")
 print(f"Grid size: {grid_size}×{grid_size}")
 
 # Create example conditions (replace with actual ablations/EO/EC)
 data_dict = {
 'intact': data_intact,
 # Add more conditions here:
 # 'boundary_removal': data_boundary_removal,
 # 'adjacency_scramble': data_adjacency_scramble,
 # 'EO': data_eo,
 # 'EC': data_ec,
 }
 
 # Run matrix-first pipeline
 results = matrix_first_pipeline(data_dict, adjacency)
 
 # Save results
 output_path = output_dir / 'matrix_first_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")
