"""
Composite Invariant Signature U
================================

Per locked protocol: U is a VECTOR, not a scalar.

Components:
1. Eigenvalue collapse profile (λ₁, λ₂, ..., participation ratio)
2. Spectral gap stability (λ₁ - λ₂ over time windows)
3. Entropy flow signature (Von Neumann entropy, gradient, rate)
4. Winding/coverage stats (phase winding per dimension, circular variance)
5. Regime dwell distribution (mean, std, CV, histogram)
6. Path curvature/torsion stats (trajectory geometry)
7. Graph locality vs long-range suppression (adjacency structure)

Uniqueness assessed by signature separation under ablations, not single-metric monotonicity.

"""

import numpy as np
import scipy.signal as signal
from typing import Dict, Tuple
import json

# Set random seed
np.random.seed(42)

# ============================================================================
# COMPOSITE INVARIANT SIGNATURE U
# ============================================================================

def compute_composite_signature_U(data: np.ndarray, fs: float, adjacency: np.ndarray) -> Dict:
 """
 Compute composite invariant signature U (vector).
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 adjacency: (n_rois, n_rois) adjacency matrix
 
 Returns:
 U: dict with all signature components
 """
 from entptc.t3_to_r3_mapping import compute_t3_coordinates
 
 print("\n" + "="*80)
 print("COMPUTING COMPOSITE INVARIANT SIGNATURE U")
 print("="*80)
 
 n_rois, n_samples = data.shape
 
 # ========================================================================
 # 1. EIGENVALUE COLLAPSE PROFILE
 # ========================================================================
 
 print("\n1. Eigenvalue collapse profile...")
 
 # Covariance matrix
 cov = np.cov(data)
 
 # Normalize
 cov_norm = cov / (np.trace(cov) + 1e-10)
 
 # Apply toroidal constraint
 progenitor = cov_norm * adjacency
 progenitor = (progenitor + progenitor.T) / 2
 
 # Eigenvalues
 eigenvalues = np.linalg.eigvalsh(progenitor)
 eigenvalues = np.sort(eigenvalues)[::-1]
 
 # Dominant eigenvalue
 lambda_1 = eigenvalues[0]
 lambda_2 = eigenvalues[1] if len(eigenvalues) > 1 else 0
 
 # Spectral gap
 spectral_gap = lambda_1 - lambda_2
 
 # Participation ratio
 participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2) if np.sum(eigenvalues**2) > 0 else 0
 
 # Eigenvalue decay rate (exponential fit)
 eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
 if len(eigenvalues_pos) > 2:
 log_eigenvalues = np.log(eigenvalues_pos + 1e-10)
 decay_rate = -np.polyfit(np.arange(len(log_eigenvalues)), log_eigenvalues, 1)[0]
 else:
 decay_rate = 0
 
 print(f" λ₁: {lambda_1:.6f}")
 print(f" λ₂: {lambda_2:.6f}")
 print(f" Spectral gap: {spectral_gap:.6f}")
 print(f" Participation ratio: {participation_ratio:.2f}")
 print(f" Eigenvalue decay rate: {decay_rate:.4f}")
 
 # ========================================================================
 # 2. SPECTRAL GAP STABILITY (TIME-RESOLVED)
 # ========================================================================
 
 print("\n2. Spectral gap stability...")
 
 window_length_sec = 10.0
 window_length_samples = int(window_length_sec * fs)
 overlap = 0.5
 step = int(window_length_samples * (1 - overlap))
 
 spectral_gaps = []
 
 for start in range(0, n_samples - window_length_samples, step):
 end = start + window_length_samples
 window_data = data[:, start:end]
 
 # Covariance
 cov_window = np.cov(window_data)
 cov_window_norm = cov_window / (np.trace(cov_window) + 1e-10)
 progenitor_window = cov_window_norm * adjacency
 progenitor_window = (progenitor_window + progenitor_window.T) / 2
 
 # Eigenvalues
 eigenvalues_window = np.linalg.eigvalsh(progenitor_window)
 eigenvalues_window = np.sort(eigenvalues_window)[::-1]
 
 if len(eigenvalues_window) >= 2:
 gap = eigenvalues_window[0] - eigenvalues_window[1]
 spectral_gaps.append(gap)
 
 spectral_gaps = np.array(spectral_gaps)
 
 spectral_gap_mean = np.mean(spectral_gaps)
 spectral_gap_std = np.std(spectral_gaps)
 spectral_gap_cv = spectral_gap_std / spectral_gap_mean if spectral_gap_mean > 0 else 0
 
 print(f" Mean spectral gap: {spectral_gap_mean:.6f}")
 print(f" Std spectral gap: {spectral_gap_std:.6f}")
 print(f" CV: {spectral_gap_cv:.4f}")
 
 # ========================================================================
 # 3. ENTROPY FLOW SIGNATURE
 # ========================================================================
 
 print("\n3. Entropy flow signature...")
 
 # Von Neumann entropy
 eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
 eigenvalues_norm = eigenvalues_pos / np.sum(eigenvalues_pos)
 entropy = -np.sum(eigenvalues_norm * np.log(eigenvalues_norm))
 
 # Maximum entropy
 max_entropy = np.log(len(eigenvalues_pos))
 
 # Normalized entropy
 entropy_normalized = entropy / max_entropy if max_entropy > 0 else 0
 
 # Entropy gradient (time-resolved)
 entropies = []
 
 for start in range(0, n_samples - window_length_samples, step):
 end = start + window_length_samples
 window_data = data[:, start:end]
 
 cov_window = np.cov(window_data)
 eigenvalues_window = np.linalg.eigvalsh(cov_window)
 eigenvalues_window_pos = eigenvalues_window[eigenvalues_window > 1e-10]
 eigenvalues_window_norm = eigenvalues_window_pos / np.sum(eigenvalues_window_pos)
 entropy_window = -np.sum(eigenvalues_window_norm * np.log(eigenvalues_window_norm))
 entropies.append(entropy_window)
 
 entropies = np.array(entropies)
 
 # Entropy gradient (linear fit)
 if len(entropies) > 2:
 entropy_gradient = np.polyfit(np.arange(len(entropies)), entropies, 1)[0]
 else:
 entropy_gradient = 0
 
 print(f" Von Neumann entropy: {entropy:.4f}")
 print(f" Normalized entropy: {entropy_normalized:.4f}")
 print(f" Entropy gradient: {entropy_gradient:.6f}")
 
 # ========================================================================
 # 4. WINDING/COVERAGE STATS
 # ========================================================================
 
 print("\n4. Winding/coverage stats...")
 
 t3_coords = compute_t3_coordinates(data, fs)
 
 # Phase winding per dimension
 phase_winding = []
 circular_variance = []
 
 for i, phases in enumerate(t3_coords):
 # Phase winding (PLV-like)
 winding = np.abs(np.mean(np.exp(1j * phases), axis=1))
 phase_winding.append(np.mean(winding))
 
 # Circular variance
 cv = 1 - np.abs(np.mean(np.exp(1j * phases), axis=1))
 circular_variance.append(np.mean(cv))
 
 print(f" Phase winding (θ₁, θ₂, θ₃): {phase_winding}")
 print(f" Circular variance (θ₁, θ₂, θ₃): {circular_variance}")
 
 # ========================================================================
 # 5. REGIME DWELL DISTRIBUTION
 # ========================================================================
 
 print("\n5. Regime dwell distribution...")
 
 # Threshold at median
 threshold = np.median(spectral_gaps)
 regime_labels_raw = (spectral_gaps > threshold).astype(int)
 
 # Apply minimum dwell time filter
 min_dwell_sec = 15
 min_dwell_windows = int(min_dwell_sec / (window_length_sec * (1 - overlap)))
 
 regime_labels = regime_labels_raw.copy()
 current_regime = regime_labels[0]
 dwell_count = 1
 
 for i in range(1, len(regime_labels)):
 if regime_labels_raw[i] == current_regime:
 dwell_count += 1
 else:
 if dwell_count >= min_dwell_windows:
 current_regime = regime_labels_raw[i]
 dwell_count = 1
 else:
 regime_labels[i] = current_regime
 dwell_count += 1
 
 # Compute dwell times
 dwell_times = []
 current_regime = regime_labels[0]
 dwell_count = 1
 
 for i in range(1, len(regime_labels)):
 if regime_labels[i] == current_regime:
 dwell_count += 1
 else:
 dwell_times.append(dwell_count * window_length_sec * (1 - overlap))
 current_regime = regime_labels[i]
 dwell_count = 1
 
 dwell_times.append(dwell_count * window_length_sec * (1 - overlap))
 dwell_times = np.array(dwell_times)
 
 dwell_mean = np.mean(dwell_times)
 dwell_std = np.std(dwell_times)
 dwell_cv = dwell_std / dwell_mean if dwell_mean > 0 else 0
 
 print(f" Mean dwell time: {dwell_mean:.2f} seconds")
 print(f" Std dwell time: {dwell_std:.2f} seconds")
 print(f" CV: {dwell_cv:.4f}")
 
 # ========================================================================
 # 6. PATH CURVATURE/TORSION STATS
 # ========================================================================
 
 print("\n6. Path curvature/torsion stats...")
 
 # Trajectory curvature (discrete approximation)
 curvatures = []
 
 for i, phases in enumerate(t3_coords):
 # Average across ROIs
 trajectory = np.mean(phases, axis=0)
 
 # First and second derivatives
 velocity = np.gradient(trajectory)
 acceleration = np.gradient(velocity)
 
 # Curvature = |v × a| / |v|^3
 curvature = np.abs(acceleration) / (np.abs(velocity)**3 + 1e-10)
 curvatures.append(np.mean(curvature))
 
 print(f" Mean curvature (θ₁, θ₂, θ₃): {curvatures}")
 
 # ========================================================================
 # 7. GRAPH LOCALITY VS LONG-RANGE SUPPRESSION
 # ========================================================================
 
 print("\n7. Graph locality vs long-range suppression...")
 
 # Compute distance matrix (toroidal grid)
 n_rois = adjacency.shape[0]
 grid_size = int(np.sqrt(n_rois))
 
 distance_matrix = np.zeros((n_rois, n_rois))
 
 for i in range(n_rois):
 for j in range(n_rois):
 ix, iy = i // grid_size, i % grid_size
 jx, jy = j // grid_size, j % grid_size
 
 # Toroidal distance
 dx = min(abs(ix - jx), grid_size - abs(ix - jx))
 dy = min(abs(iy - jy), grid_size - abs(iy - jy))
 distance_matrix[i, j] = np.sqrt(dx**2 + dy**2)
 
 # Compute PLV matrix
 plv_matrix = np.zeros((n_rois, n_rois))
 
 for i in range(n_rois):
 for j in range(n_rois):
 phase_diff = t3_coords[0][i] - t3_coords[0][j] # θ₁
 plv_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff)))
 
 # Locality: correlation between distance and PLV
 locality_correlation = np.corrcoef(distance_matrix.flatten(), plv_matrix.flatten())[0, 1]
 
 # Long-range suppression: mean PLV for distant pairs (distance > grid_size/2)
 distant_mask = distance_matrix > grid_size / 2
 long_range_plv = np.mean(plv_matrix[distant_mask]) if np.sum(distant_mask) > 0 else 0
 
 print(f" Locality correlation (distance vs PLV): {locality_correlation:.4f}")
 print(f" Long-range PLV (distant pairs): {long_range_plv:.4f}")
 
 # ========================================================================
 # CONSOLIDATE SIGNATURE U
 # ========================================================================
 
 U = {
 '1_eigenvalue_profile': {
 'lambda_1': float(lambda_1),
 'lambda_2': float(lambda_2),
 'spectral_gap': float(spectral_gap),
 'participation_ratio': float(participation_ratio),
 'eigenvalue_decay_rate': float(decay_rate)
 },
 '2_spectral_gap_stability': {
 'mean': float(spectral_gap_mean),
 'std': float(spectral_gap_std),
 'cv': float(spectral_gap_cv)
 },
 '3_entropy_flow': {
 'von_neumann_entropy': float(entropy),
 'normalized_entropy': float(entropy_normalized),
 'entropy_gradient': float(entropy_gradient)
 },
 '4_winding_coverage': {
 'phase_winding_theta1': float(phase_winding[0]),
 'phase_winding_theta2': float(phase_winding[1]),
 'phase_winding_theta3': float(phase_winding[2]),
 'circular_variance_theta1': float(circular_variance[0]),
 'circular_variance_theta2': float(circular_variance[1]),
 'circular_variance_theta3': float(circular_variance[2])
 },
 '5_regime_dwell': {
 'mean_dwell_time_sec': float(dwell_mean),
 'std_dwell_time_sec': float(dwell_std),
 'cv': float(dwell_cv)
 },
 '6_path_curvature': {
 'mean_curvature_theta1': float(curvatures[0]),
 'mean_curvature_theta2': float(curvatures[1]),
 'mean_curvature_theta3': float(curvatures[2])
 },
 '7_graph_locality': {
 'locality_correlation': float(locality_correlation),
 'long_range_plv': float(long_range_plv)
 }
 }
 
 print("\n" + "="*80)
 print("COMPOSITE INVARIANT SIGNATURE U COMPLETE")
 print("="*80)
 
 return U

# ============================================================================
# MAIN RUNNER
# ============================================================================

if __name__ == '__main__':
 import scipy.io as sio
 from pathlib import Path
 from entptc.utils.grid_utils import create_toroidal_grid
 
 # Load data
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/composite_signature')
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
 
 # Compute composite signature U
 U = compute_composite_signature_U(data, fs, adjacency)
 
 # Save results
 output_path = output_dir / 'composite_signature_U.json'
 with open(output_path, 'w') as f:
 json.dump(U, f, indent=2)
 
 print(f"\n✅ Signature U saved to {output_path}")
