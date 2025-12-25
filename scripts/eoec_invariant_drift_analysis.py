"""
EO/EC Invariant Drift Analysis (Class C)
=========================================

Per locked protocol: Compare invariant signatures EO vs EC, analyze as operator-level control drift.

NOT band changes, NOT spectral power.

"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import json
from composite_invariant_signature import compute_composite_signature_U
from entptc.utils.grid_utils import create_toroidal_grid

# Set random seed
np.random.seed(42)

# ============================================================================
# EO/EC COMPARISON
# ============================================================================

def compare_eoec_signatures(data_eo: np.ndarray, data_ec: np.ndarray, fs: float, adjacency: np.ndarray) -> dict:
 """
 Compare invariant signatures between EO and EC conditions.
 
 Args:
 data_eo: (n_rois, n_samples) EO data
 data_ec: (n_rois, n_samples) EC data
 fs: sampling rate
 adjacency: adjacency matrix
 
 Returns:
 comparison: dict with EO/EC signatures and differences
 """
 print("\n" + "="*80)
 print("EO/EC INVARIANT DRIFT ANALYSIS")
 print("="*80)
 
 # Compute signatures
 print("\nComputing EO signature...")
 U_eo = compute_composite_signature_U(data_eo, fs, adjacency)
 
 print("\nComputing EC signature...")
 U_ec = compute_composite_signature_U(data_ec, fs, adjacency)
 
 # Compute differences
 print("\n" + "-"*80)
 print("COMPUTING SIGNATURE DIFFERENCES")
 print("-"*80)
 
 differences = {}
 
 for component_key in sorted(U_eo.keys()):
 component_eo = U_eo[component_key]
 component_ec = U_ec[component_key]
 
 component_diff = {}
 
 for metric_key in sorted(component_eo.keys()):
 val_eo = component_eo[metric_key]
 val_ec = component_ec[metric_key]
 
 # Absolute difference
 abs_diff = val_ec - val_eo
 
 # Relative difference (percent change)
 rel_diff = (abs_diff / (abs(val_eo) + 1e-10)) * 100
 
 component_diff[metric_key] = {
 'eo': float(val_eo),
 'ec': float(val_ec),
 'abs_diff': float(abs_diff),
 'rel_diff_percent': float(rel_diff)
 }
 
 differences[component_key] = component_diff
 
 # Print summary
 print("\nSIGNATURE DIFFERENCES (EO → EC):")
 print("-"*80)
 
 for component_key in sorted(differences.keys()):
 print(f"\n{component_key}:")
 for metric_key in sorted(differences[component_key].keys()):
 diff_data = differences[component_key][metric_key]
 print(f" {metric_key}:")
 print(f" EO: {diff_data['eo']:.6f}")
 print(f" EC: {diff_data['ec']:.6f}")
 print(f" Δ: {diff_data['abs_diff']:.6f} ({diff_data['rel_diff_percent']:.2f}%)")
 
 return {
 'signature_eo': U_eo,
 'signature_ec': U_ec,
 'differences': differences
 }

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 # Load EO/EC data
 data_dir = Path('/home/ubuntu/entptc-implementation/data/dataset_set_2')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/eoec_analysis')
 output_dir.mkdir(parents=True, exist_ok=True)
 
 # Use first subject as example
 eo_path = data_dir / 'S001_task-EyesOpen_eeg.mat'
 ec_path = data_dir / 'S001_task-EyesClosed_eeg.mat'
 
 if not eo_path.exists() or not ec_path.exists():
 print(f"Data files not found")
 exit(1)
 
 print("Loading EO/EC data...")
 mat_eo = sio.loadmat(eo_path)
 mat_ec = sio.loadmat(ec_path)
 
 data_eo = mat_eo['eeg_data']
 data_ec = mat_ec['eeg_data']
 fs = float(mat_eo['fs'][0, 0])
 
 grid_size = int(np.sqrt(data_eo.shape[0]))
 adjacency = create_toroidal_grid(grid_size)
 
 print(f"EO data shape: {data_eo.shape}")
 print(f"EC data shape: {data_ec.shape}")
 print(f"Sampling rate: {fs} Hz")
 print(f"Grid size: {grid_size}×{grid_size}")
 
 # Compare signatures
 comparison = compare_eoec_signatures(data_eo, data_ec, fs, adjacency)
 
 # Save results
 output_path = output_dir / 'eoec_comparison_S001.json'
 with open(output_path, 'w') as f:
 json.dump(comparison, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")
 
 # Interpretation
 print("\n" + "="*80)
 print("INTERPRETATION (PER ABSOLUTE GUARDRAILS)")
 print("="*80)
 
 print("""
Per locked protocol (ABSOLUTE_INTERPRETATION_GUARDRAILS.md):

EO/EC are LOW-EXCITATION PROJECTIONS that test whether control structure is CONSTITUTIVE.

Correct interpretation:
- Persistence or drift of invariant structure explains EO/EC changes
- Changes are caused by DRIFT IN OPERATOR COLLAPSE, not bandpower shifts

This analysis shows:
- How invariant signature U changes between EO and EC
- Which components are stable vs drifting
- Whether control structure persists or deforms

NOT:
- Frequency discovery
- Band power changes
- Spectral features
 """)
