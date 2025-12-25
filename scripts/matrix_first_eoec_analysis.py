"""
Matrix-First EO/EC Analysis
============================

Per MATRIX_FIRST_PROTOCOL:
- EO/EC must be reported as collapse structure stability/drift
- NOT as bandpower differences or frequency shifts
- Compare collapse objects (eigenstructure, entropy, spectral gap)

"""

import numpy as np
from pathlib import Path
import json
from typing import Dict
import scipy.io as sio
from matrix_first_pipeline import construct_progenitor_matrix, extract_collapse_object, compute_collapse_drift
from entptc.utils.grid_utils import create_toroidal_grid

# Set random seed
np.random.seed(42)

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 # Load EO/EC data (PhysioNet preprocessed)
 data_dir = Path('/home/ubuntu/entptc-implementation/data/dataset_set_2')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/matrix_first_eoec')
 output_dir.mkdir(parents=True, exist_ok=True)
 
 # Load EO and EC files for S001
 eo_path = data_dir / 'S001_task-EyesOpen_eeg.mat'
 ec_path = data_dir / 'S001_task-EyesClosed_eeg.mat'
 
 if not eo_path.exists() or not ec_path.exists():
 print("EO/EC data not found. Skipping.")
 exit(1)
 
 print(f"Found EO and EC files for S001")
 
 print(f"\nLoading EO: {eo_path.name}")
 mat_eo = sio.loadmat(eo_path)
 data_eo = mat_eo['eeg_data']
 fs = float(mat_eo['fs'][0, 0])
 
 print(f"Loading EC: {ec_path.name}")
 mat_ec = sio.loadmat(ec_path)
 data_ec = mat_ec['eeg_data']
 
 print(f"\nEO data shape: {data_eo.shape}")
 print(f"EC data shape: {data_ec.shape}")
 print(f"Sampling rate: {fs} Hz")
 
 # Create toroidal grid adjacency
 grid_size = 4
 adjacency = create_toroidal_grid(grid_size)
 
 # ========================================================================
 # MATRIX-FIRST PIPELINE: EO
 # ========================================================================
 
 print("\n" + "="*80)
 print("MATRIX-FIRST PIPELINE: EYES OPEN (EO)")
 print("="*80)
 
 progenitor_eo = construct_progenitor_matrix(data_eo, adjacency)
 collapse_eo = extract_collapse_object(progenitor_eo)
 
 print(f"\nCollapse Object (EO):")
 print(f" Dominant eigenvalue λ₁: {collapse_eo['dominant_eigenvalue']:.6f}")
 print(f" Spectral gap: {collapse_eo['spectral_gap']:.6f}")
 print(f" Von Neumann entropy: {collapse_eo['von_neumann_entropy']:.4f}")
 print(f" Entropy (normalized): {collapse_eo['entropy_normalized']:.4f}")
 print(f" Participation ratio: {collapse_eo['participation_ratio']:.6f}")
 print(f" Eigenvalue decay rate: {collapse_eo['eigenvalue_decay_rate']:.6f}")
 
 # ========================================================================
 # MATRIX-FIRST PIPELINE: EC
 # ========================================================================
 
 print("\n" + "="*80)
 print("MATRIX-FIRST PIPELINE: EYES CLOSED (EC)")
 print("="*80)
 
 progenitor_ec = construct_progenitor_matrix(data_ec, adjacency)
 collapse_ec = extract_collapse_object(progenitor_ec)
 
 print(f"\nCollapse Object (EC):")
 print(f" Dominant eigenvalue λ₁: {collapse_ec['dominant_eigenvalue']:.6f}")
 print(f" Spectral gap: {collapse_ec['spectral_gap']:.6f}")
 print(f" Von Neumann entropy: {collapse_ec['von_neumann_entropy']:.4f}")
 print(f" Entropy (normalized): {collapse_ec['entropy_normalized']:.4f}")
 print(f" Participation ratio: {collapse_ec['participation_ratio']:.6f}")
 print(f" Eigenvalue decay rate: {collapse_ec['eigenvalue_decay_rate']:.6f}")
 
 # ========================================================================
 # COLLAPSE DRIFT: EO → EC
 # ========================================================================
 
 print("\n" + "="*80)
 print("COLLAPSE DRIFT: EO → EC")
 print("="*80)
 
 drift_eo_to_ec = compute_collapse_drift(collapse_eo, collapse_ec)
 
 print(f"\nCollapse Drift Metrics:")
 print(f" Eigenmode drift: {drift_eo_to_ec['eigenmode_drift']:.6f}")
 print(f" Spectral gap drift: {drift_eo_to_ec['spectral_gap_drift']:.6f}")
 print(f" Entropy drift: {drift_eo_to_ec['entropy_drift']:.6f}")
 print(f" Participation drift: {drift_eo_to_ec['participation_drift']:.6f}")
 print(f" Decay rate drift: {drift_eo_to_ec['decay_rate_drift']:.6f}")
 print(f" Dominant eigenvalue drift (relative): {drift_eo_to_ec['dominant_eigenvalue_drift_relative']:.6f}")
 
 # ========================================================================
 # INTERPRETATION (PER MATRIX_FIRST_PROTOCOL)
 # ========================================================================
 
 print("\n" + "="*80)
 print("INTERPRETATION (MATRIX-FIRST)")
 print("="*80)
 
 print("\nPer MATRIX_FIRST_PROTOCOL:")
 print("- EO/EC must be reported as collapse structure stability/drift")
 print("- NOT as bandpower differences or frequency shifts")
 print("- Compare collapse objects (eigenstructure, entropy, spectral gap)")
 
 print("\nCollapse Structure Stability/Drift:")
 
 # Classify each component as stable or drifting (threshold: 10% relative change)
 components = [
 ('Dominant eigenvalue', 
 abs(collapse_eo['dominant_eigenvalue'] - collapse_ec['dominant_eigenvalue']) / collapse_eo['dominant_eigenvalue']),
 ('Spectral gap', 
 abs(collapse_eo['spectral_gap'] - collapse_ec['spectral_gap']) / (collapse_eo['spectral_gap'] + 1e-10)),
 ('Von Neumann entropy', 
 abs(collapse_eo['von_neumann_entropy'] - collapse_ec['von_neumann_entropy']) / collapse_eo['von_neumann_entropy']),
 ('Participation ratio', 
 abs(collapse_eo['participation_ratio'] - collapse_ec['participation_ratio']) / (collapse_eo['participation_ratio'] + 1e-10)),
 ('Eigenvalue decay rate', 
 abs(collapse_eo['eigenvalue_decay_rate'] - collapse_ec['eigenvalue_decay_rate']) / collapse_eo['eigenvalue_decay_rate']),
 ]
 
 stable_components = []
 drifting_components = []
 
 for name, relative_change in components:
 if relative_change < 0.1: # < 10% change
 stable_components.append((name, relative_change))
 else:
 drifting_components.append((name, relative_change))
 
 print(f"\nSTABLE components (< 10% change):")
 for name, relative_change in stable_components:
 print(f" ✅ {name}: {relative_change*100:.1f}% change")
 
 print(f"\nDRIFTING components (> 10% change):")
 for name, relative_change in drifting_components:
 print(f" ⚠️ {name}: {relative_change*100:.1f}% change")
 
 print("\nConclusion:")
 if len(stable_components) > len(drifting_components):
 print(" ✅ Core collapse structure is STABLE across EO/EC")
 print(" ⚠️ Some components drift, reflecting operator-state change")
 else:
 print(" ⚠️ Collapse structure shows MODERATE DRIFT across EO/EC")
 print(" ⚠️ Operator-state change affects multiple components")
 
 print("\nThis is:")
 print(" ✅ Collapse structure stability/drift")
 print(" ✅ Operator-state change")
 print(" ❌ NOT bandpower differences")
 print(" ❌ NOT frequency shifts")
 
 # ========================================================================
 # SAVE RESULTS
 # ========================================================================
 
 results = {
 'collapse_objects': {
 'EO': collapse_eo,
 'EC': collapse_ec
 },
 'collapse_drift': {
 'EO_to_EC': drift_eo_to_ec
 },
 'interpretation': {
 'stable_components': [(name, float(change)) for name, change in stable_components],
 'drifting_components': [(name, float(change)) for name, change in drifting_components],
 'conclusion': 'STABLE' if len(stable_components) > len(drifting_components) else 'MODERATE_DRIFT'
 }
 }
 
 output_path = output_dir / 'matrix_first_eoec_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")
