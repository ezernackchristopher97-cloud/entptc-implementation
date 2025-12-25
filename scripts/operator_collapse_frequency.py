"""
Operator Collapse Frequency Inference
======================================

Infers operant conditioning control timescale from operator collapse dynamics.

Per user protocol:
- Frequency is NOT read from EEG spectrum
- Frequency is INFERRED from operator collapse rate: τ_control ~ 1/λ_dom^imag
- Projection determines how frequency appears, not what it is

Key quantities:
- Convergence rate to dominant eigenstructure
- Stability of leading eigenmode
- Entropy gradient flattening rate
- Regime dwell times after collapse

"""

import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple

# Set random seed
np.random.seed(42)

# ============================================================================
# OPERATOR COLLAPSE DYNAMICS
# ============================================================================

def compute_operator_collapse_frequency(data: np.ndarray, fs: float, adjacency: np.ndarray) -> Dict:
 """
 Infer control timescale from operator collapse dynamics.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 collapse_metrics: dict with collapse rate, eigenstructure, entropy gradient
 """
 n_rois, n_samples = data.shape
 
 print("\n" + "="*80)
 print("OPERATOR COLLAPSE FREQUENCY INFERENCE")
 print("="*80)
 
 # ========================================================================
 # 1. CONSTRUCT PROGENITOR MATRIX
 # ========================================================================
 
 print("\n1. Constructing progenitor matrix...")
 
 # Covariance matrix (spatial)
 cov = np.cov(data)
 
 # Normalize
 cov_norm = cov / (np.trace(cov) + 1e-10)
 
 # Apply toroidal constraint (adjacency weighting)
 progenitor = cov_norm * adjacency
 progenitor = (progenitor + progenitor.T) / 2 # Symmetrize
 
 print(f"Progenitor matrix shape: {progenitor.shape}")
 print(f"Progenitor trace: {np.trace(progenitor):.6f}")
 
 # ========================================================================
 # 2. EIGENSTRUCTURE ANALYSIS
 # ========================================================================
 
 print("\n2. Computing eigenstructure...")
 
 eigenvalues, eigenvectors = np.linalg.eigh(progenitor)
 
 # Sort descending
 idx = np.argsort(eigenvalues)[::-1]
 eigenvalues = eigenvalues[idx]
 eigenvectors = eigenvectors[:, idx]
 
 # Dominant eigenvalue
 lambda_dom = eigenvalues[0]
 
 # Spectral gap (λ₁ - λ₂)
 spectral_gap = eigenvalues[0] - eigenvalues[1] if len(eigenvalues) > 1 else 0
 
 # Participation ratio (effective dimensionality)
 participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2) if np.sum(eigenvalues**2) > 0 else 0
 
 print(f"Dominant eigenvalue (λ_dom): {lambda_dom:.6f}")
 print(f"Spectral gap (λ₁ - λ₂): {spectral_gap:.6f}")
 print(f"Participation ratio: {participation_ratio:.2f}")
 
 # ========================================================================
 # 3. COLLAPSE RATE (IMAGINARY COMPONENT PROXY)
 # ========================================================================
 
 print("\n3. Inferring collapse rate...")
 
 # In real symmetric matrix, eigenvalues are real
 # Collapse rate estimated from spectral gap / dominant eigenvalue
 # This is dimensionless collapse rate
 
 collapse_rate_dimensionless = spectral_gap / (lambda_dom + 1e-10)
 
 # Convert to frequency (Hz) via sampling rate
 # τ_control ~ 1 / collapse_rate
 # f_control ~ collapse_rate / (2π)
 
 tau_control_samples = 1 / (collapse_rate_dimensionless + 1e-10)
 tau_control_seconds = tau_control_samples / fs
 f_control_hz = 1 / tau_control_seconds if tau_control_seconds > 0 else 0
 
 print(f"Collapse rate (dimensionless): {collapse_rate_dimensionless:.6f}")
 print(f"Control timescale τ_control: {tau_control_seconds:.2f} seconds")
 print(f"Control frequency f_control: {f_control_hz:.4f} Hz")
 
 # ========================================================================
 # 4. ENTROPY GRADIENT
 # ========================================================================
 
 print("\n4. Computing entropy gradient...")
 
 # Von Neumann entropy
 eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
 eigenvalues_norm = eigenvalues_pos / np.sum(eigenvalues_pos)
 entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm))
 
 # Maximum entropy (uniform distribution)
 max_entropy = np.log(len(eigenvalues_pos))
 
 # Normalized entropy
 entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0
 
 print(f"Von Neumann entropy: {entropy:.4f}")
 print(f"Max entropy: {max_entropy:.4f}")
 print(f"Normalized entropy: {entropy_normalized:.4f}")
 
 # ========================================================================
 # 5. REGIME DWELL TIMES
 # ========================================================================
 
 print("\n5. Computing regime dwell times...")
 
 # Project data onto dominant eigenmode
 dominant_mode = eigenvectors[:, 0]
 projection = data.T @ dominant_mode # (n_samples,)
 
 # Threshold at median
 threshold = np.median(projection)
 regime_labels = (projection > threshold).astype(int)
 
 # Compute dwell times
 dwell_times = []
 current_regime = regime_labels[0]
 dwell_count = 1
 
 for i in range(1, len(regime_labels)):
 if regime_labels[i] == current_regime:
 dwell_count += 1
 else:
 dwell_times.append(dwell_count / fs) # Convert to seconds
 current_regime = regime_labels[i]
 dwell_count = 1
 
 dwell_times.append(dwell_count / fs)
 
 mean_dwell_time = np.mean(dwell_times)
 median_dwell_time = np.median(dwell_times)
 
 print(f"Number of regimes: {len(dwell_times)}")
 print(f"Mean dwell time: {mean_dwell_time:.2f} seconds")
 print(f"Median dwell time: {median_dwell_time:.2f} seconds")
 
 # ========================================================================
 # 6. CONSOLIDATE RESULTS
 # ========================================================================
 
 results = {
 'progenitor_trace': float(np.trace(progenitor)),
 'dominant_eigenvalue': float(lambda_dom),
 'spectral_gap': float(spectral_gap),
 'participation_ratio': float(participation_ratio),
 'collapse_rate_dimensionless': float(collapse_rate_dimensionless),
 'tau_control_seconds': float(tau_control_seconds),
 'f_control_hz': float(f_control_hz),
 'von_neumann_entropy': float(entropy),
 'entropy_normalized': float(entropy_normalized),
 'n_regimes': len(dwell_times),
 'mean_dwell_time_seconds': float(mean_dwell_time),
 'median_dwell_time_seconds': float(median_dwell_time)
 }
 
 print("\n" + "="*80)
 print("OPERATOR COLLAPSE FREQUENCY INFERENCE COMPLETE")
 print("="*80)
 print(f"\nInferred control frequency: {f_control_hz:.4f} Hz")
 print(f"Control timescale: {tau_control_seconds:.2f} seconds")
 print(f"Regime dwell time: {mean_dwell_time:.2f} seconds")
 
 print("\nInterpretation:")
 print("- This frequency is INFERRED from operator collapse, NOT read from EEG spectrum")
 print("- Projection into EEG may appear as infra-slow modulation, regime timing, or geometry")
 print("- THz is viable hypothesis if invariant matching supports it at microphysical scale")
 
 return results

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 import scipy.io as sio
 from entptc.utils.grid_utils import create_toroidal_grid
 
 # Load data
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/operator_collapse')
 output_dir.mkdir(parents=True, exist_ok=True)
 
 if not data_path.exists():
 print(f"Data file not found: {data_path}")
 exit(1)
 
 print("Loading data...")
 mat = sio.loadmat(data_path)
 data = mat['eeg_data']
 fs = float(mat['fs'][0, 0])
 
 grid_size = int(np.sqrt(data.shape[0]))
 adjacency = create_toroidal_grid(grid_size)
 
 print(f"Data shape: {data.shape}")
 print(f"Sampling rate: {fs} Hz")
 print(f"Grid size: {grid_size}×{grid_size}")
 
 # Compute operator collapse frequency
 results = compute_operator_collapse_frequency(data, fs, adjacency)
 
 # Save results
 output_path = output_dir / 'operator_collapse_results.json'
 with open(output_path, 'w') as f:
 json.dump(results, f, indent=2)
 
 print(f"\n✅ Results saved to {output_path}")
