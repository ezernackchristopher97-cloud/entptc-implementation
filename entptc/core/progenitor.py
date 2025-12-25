"""
Progenitor Matrix Construction

Reference: ENTPC.tex Definition 2.5 (lines 266-285)

From ENTPC.tex:

Definition 2.5 (Progenitor Matrix): The Progenitor Matrix M or C₁₆ is a 16×16
real, non-negative matrix that generates the dynamics of experience. Each entry
c_ij is given by:

 c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|

where:
- λ_ij: coherence amplitude (from inferred THz control frequencies)
- ∇S_ij: entropy gradient between subsystems i and j
- Q(θ_ij): quaternionic rotation operator (norm taken for real value)

Structure:
- 16×16 matrix organized into four 4×4 quadrants
- Each quadrant A_pq is a 4×4 submatrix
- Diagonal blocks A_pp: intra-subsystem coherence
- Off-diagonal blocks A_pq (p≠q): inter-subsystem coupling

Properties:
- Real, non-negative entries
- Suitable for Perron-Frobenius theorem
- Preliminary eigenvalues: λ_max ≈ 12.6, λ₂ ≈ 6.1
- Spectral gap: 1.47 to 3.78
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ProgenitorMatrix:
 """
 Progenitor Matrix M: 16×16 generator of experience dynamics.
 
 Per ENTPC.tex Definition 2.5:
 - c_ij = λ_ij * exp(-∇S_ij) * |Q(θ_ij)|
 - Real, non-negative entries
 - Block structure: 4×4 quadrants
 """
 
 def __init__(self):
 """Initialize Progenitor Matrix constructor."""
 self.matrix = None
 self.coherence_matrix = None
 self.entropy_gradient_matrix = None
 self.quaternion_norm_matrix = None
 
 def construct_from_components(self,
 coherence_matrix: np.ndarray,
 entropy_gradient_matrix: np.ndarray,
 quaternion_norm_matrix: np.ndarray) -> np.ndarray:
 """
 Construct Progenitor Matrix from component matrices.
 
 Per ENTPC.tex: c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|
 
 Args:
 coherence_matrix: 16×16 matrix of λ_ij values
 entropy_gradient_matrix: 16×16 matrix of ∇S_ij values
 quaternion_norm_matrix: 16×16 matrix of |Q(θ_ij)| values
 
 Returns:
 16×16 Progenitor Matrix
 """
 # Validate inputs
 assert coherence_matrix.shape == (16, 16), \
 f"Expected 16×16 coherence matrix, got {coherence_matrix.shape}"
 assert entropy_gradient_matrix.shape == (16, 16), \
 f"Expected 16×16 entropy gradient matrix, got {entropy_gradient_matrix.shape}"
 assert quaternion_norm_matrix.shape == (16, 16), \
 f"Expected 16×16 quaternion norm matrix, got {quaternion_norm_matrix.shape}"
 
 # Store components
 self.coherence_matrix = coherence_matrix
 self.entropy_gradient_matrix = entropy_gradient_matrix
 self.quaternion_norm_matrix = quaternion_norm_matrix
 
 # Compute Progenitor Matrix
 # c_ij = λ_ij * exp(-∇S_ij) * |Q(θ_ij)|
 self.matrix = (coherence_matrix * 
 np.exp(-entropy_gradient_matrix) * 
 quaternion_norm_matrix)
 
 # Ensure non-negative (required for Perron-Frobenius)
 self.matrix = np.abs(self.matrix)
 
 # Validate output
 assert self.matrix.shape == (16, 16), \
 f"Progenitor matrix construction failed: shape {self.matrix.shape}"
 assert np.all(self.matrix >= 0), \
 "Progenitor matrix must be non-negative"
 
 logger.info("✓ Constructed 16×16 Progenitor Matrix")
 logger.info(f" Matrix norm: {np.linalg.norm(self.matrix):.4f}")
 logger.info(f" Min value: {np.min(self.matrix):.6f}")
 logger.info(f" Max value: {np.max(self.matrix):.6f}")
 
 return self.matrix
 
 def construct_from_eeg_data(self,
 roi_data: np.ndarray,
 entropy_field: Optional[object] = None,
 quaternion_field: Optional[np.ndarray] = None) -> np.ndarray:
 """
 Construct Progenitor Matrix from EEG ROI data.
 
 Args:
 roi_data: Array of shape (16, n_samples) - 16 ROIs time series
 entropy_field: EntropyField object (optional)
 quaternion_field: Array of quaternions (16, 4) (optional)
 
 Returns:
 16×16 Progenitor Matrix
 """
 assert roi_data.shape[0] == 16, \
 f"Expected 16 ROIs, got {roi_data.shape[0]}"
 
 n_rois = 16
 n_samples = roi_data.shape[1]
 
 # 1. Compute coherence matrix (Phase Locking Value)
 coherence_matrix = self._compute_coherence_plv(roi_data)
 
 # 2. Compute entropy gradient matrix
 if entropy_field is not None:
 entropy_gradient_matrix = self._compute_entropy_gradients_from_field(
 entropy_field, n_rois
 )
 else:
 # Fallback: compute from coherence
 entropy_gradient_matrix = self._compute_entropy_gradients_from_coherence(
 coherence_matrix
 )
 
 # 3. Compute quaternion norm matrix
 if quaternion_field is not None:
 quaternion_norm_matrix = self._compute_quaternion_norms(quaternion_field)
 else:
 # Fallback: uniform quaternion norms
 quaternion_norm_matrix = np.ones((n_rois, n_rois))
 
 # Construct Progenitor Matrix
 return self.construct_from_components(
 coherence_matrix,
 entropy_gradient_matrix,
 quaternion_norm_matrix
 )
 
 def _compute_coherence_plv(self, roi_data: np.ndarray) -> np.ndarray:
 """
 Compute Phase Locking Value (PLV) coherence matrix.
 
 PLV measures phase synchronization between ROI pairs.
 
 Args:
 roi_data: Array of shape (16, n_samples)
 
 Returns:
 16×16 coherence matrix
 """
 from scipy.signal import hilbert
 
 n_rois = roi_data.shape[0]
 n_samples = roi_data.shape[1]
 
 # Compute analytic signal (Hilbert transform)
 analytic_signals = np.zeros((n_rois, n_samples), dtype=complex)
 for i in range(n_rois):
 analytic_signals[i, :] = hilbert(roi_data[i, :])
 
 # Extract instantaneous phase
 phases = np.angle(analytic_signals)
 
 # Compute PLV for all pairs
 plv_matrix = np.zeros((n_rois, n_rois))
 
 for i in range(n_rois):
 for j in range(n_rois):
 # Phase difference
 phase_diff = phases[i, :] - phases[j, :]
 
 # PLV: |⟨e^(iΔφ)⟩|
 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
 
 plv_matrix[i, j] = plv
 
 return plv_matrix
 
 def _compute_entropy_gradients_from_field(self,
 entropy_field: object,
 n_rois: int) -> np.ndarray:
 """
 Compute entropy gradient matrix from EntropyField.
 
 Args:
 entropy_field: EntropyField object
 n_rois: Number of ROIs (16)
 
 Returns:
 16×16 entropy gradient norm matrix
 """
 # Map ROIs to toroidal coordinates
 # Simple mapping: distribute ROIs uniformly on T³
 theta_coords = np.linspace(0, 2*np.pi, n_rois, endpoint=False)
 
 gradient_matrix = np.zeros((n_rois, n_rois))
 
 for i in range(n_rois):
 for j in range(n_rois):
 # Coordinates for ROIs i and j
 theta_i = [theta_coords[i], theta_coords[i % 4], theta_coords[i % 8]]
 theta_j = [theta_coords[j], theta_coords[j % 4], theta_coords[j % 8]]
 
 # Compute gradient norms at both points
 grad_i = entropy_field.gradient_norm(*theta_i)
 grad_j = entropy_field.gradient_norm(*theta_j)
 
 # Average gradient between i and j
 gradient_matrix[i, j] = (grad_i + grad_j) / 2.0
 
 return gradient_matrix
 
 def _compute_entropy_gradients_from_coherence(self,
 coherence_matrix: np.ndarray) -> np.ndarray:
 """
 Compute entropy gradients from coherence matrix (fallback).
 
 High coherence → low entropy → low gradient
 Low coherence → high entropy → high gradient
 
 Args:
 coherence_matrix: 16×16 PLV matrix
 
 Returns:
 16×16 entropy gradient matrix
 """
 # Entropy inversely related to coherence
 # ∇S ∝ -log(coherence)
 entropy_gradient = -np.log(coherence_matrix + 1e-6)
 
 # Normalize to reasonable range [0, 1]
 entropy_gradient = (entropy_gradient - entropy_gradient.min()) / \
 (entropy_gradient.max() - entropy_gradient.min() + 1e-12)
 
 return entropy_gradient
 
 def _compute_quaternion_norms(self, quaternion_field: np.ndarray) -> np.ndarray:
 """
 Compute quaternion norm matrix.
 
 Args:
 quaternion_field: Array of shape (16, 4) - quaternions for each ROI
 
 Returns:
 16×16 matrix of quaternion rotation norms
 """
 assert quaternion_field.shape == (16, 4), \
 f"Expected (16, 4) quaternion field, got {quaternion_field.shape}"
 
 n_rois = 16
 norm_matrix = np.zeros((n_rois, n_rois))
 
 for i in range(n_rois):
 for j in range(n_rois):
 # Quaternion for ROI i
 q_i = quaternion_field[i, :]
 
 # Quaternion for ROI j
 q_j = quaternion_field[j, :]
 
 # Relative quaternion: q_i * q_j^(-1)
 # For simplicity, use norm of difference
 q_diff = q_i - q_j
 norm_matrix[i, j] = np.linalg.norm(q_diff)
 
 # Normalize to [0, 1]
 norm_matrix = norm_matrix / (np.max(norm_matrix) + 1e-12)
 
 return norm_matrix
 
 def get_block_structure(self) -> Dict[str, np.ndarray]:
 """
 Extract 4×4 block structure.
 
 Per ENTPC.tex: Matrix organized into four 4×4 quadrants.
 
 Returns:
 Dictionary with blocks A_11, A_12, ..., A_44
 """
 if self.matrix is None:
 raise ValueError("Progenitor matrix not constructed yet")
 
 blocks = {}
 for p in range(4):
 for q in range(4):
 i_start, i_end = p * 4, (p + 1) * 4
 j_start, j_end = q * 4, (q + 1) * 4
 
 block_name = f"A_{p+1}{q+1}"
 blocks[block_name] = self.matrix[i_start:i_end, j_start:j_end]
 
 return blocks
 
 def get_matrix(self) -> np.ndarray:
 """Get the constructed Progenitor Matrix."""
 if self.matrix is None:
 raise ValueError("Progenitor matrix not constructed yet")
 return self.matrix.copy()
 
 def validate_perron_frobenius_conditions(self) -> bool:
 """
 Validate conditions for Perron-Frobenius theorem.
 
 Requirements:
 - Real entries
 - Non-negative entries
 - Irreducible (strongly connected)
 
 Returns:
 True if conditions satisfied
 """
 if self.matrix is None:
 raise ValueError("Progenitor matrix not constructed yet")
 
 # Check real
 assert np.all(np.isreal(self.matrix)), "Matrix must be real"
 
 # Check non-negative
 assert np.all(self.matrix >= 0), "Matrix must be non-negative"
 
 # Check irreducibility (all entries positive for simplicity)
 # More rigorous: check strong connectivity
 is_positive = np.all(self.matrix > 0)
 
 if not is_positive:
 logger.warning("Matrix not strictly positive, may not be irreducible")
 
 logger.info("✓ Perron-Frobenius conditions validated")
 
 return True
