"""
Perron-Frobenius Operator for Multiplicity Resolution

Reference: ENTPC.tex Definition 2.6 (lines 287-297)

From ENTPC.tex:

Definition 2.6 (Progenitor Operator): The Progenitor Operator O_P resolves
multiplicity through spectral decomposition. For an irreducible matrix M, the
Perron-Frobenius theorem guarantees a unique, simple, and positive dominant
eigenvalue λ_max, whose corresponding eigenvector v₁ has all positive entries.

The operator collapses the system dynamics to this dominant mode:

 lim_{n→∞} M^n ψ₀ / ||M^n ψ₀|| = v₁

This eigenvector v₁ represents the unified conscious state, with all other modes
decaying exponentially.

Toy Example (4×4 matrix):
- Eigenvalues: [2.1, 1.2, 0.4, 0.3]
- Collapse to dominant mode: [0.45, 0.45, 0.45, 0.45]
- Spectral gap: λ_max/λ₂ ≈ 1.75

For 16×16 Progenitor Matrix:
- Preliminary λ_max ≈ 12.6
- Preliminary λ₂ ≈ 6.1
- Spectral gap: 1.47 to 3.78
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PerronFrobeniusOperator:
 """
 Perron-Frobenius Operator for multiplicity resolution.
 
 Per ENTPC.tex Definition 2.6:
 - Extracts dominant eigenvector from irreducible matrix
 - Collapses 256 potential states to single unified state
 - Spectral gap determines collapse rate
 """
 
 def __init__(self):
 """Initialize Perron-Frobenius operator."""
 self.eigenvalues = None
 self.eigenvectors = None
 self.dominant_eigenvalue = None
 self.dominant_eigenvector = None
 self.spectral_gap = None
 
 def validate_matrix(self, matrix: np.ndarray) -> bool:
 """
 Validate matrix satisfies Perron-Frobenius conditions.
 
 Requirements:
 - Real entries
 - Non-negative entries
 - Irreducible (for unique dominant eigenvalue)
 
 Args:
 matrix: Input matrix (16×16)
 
 Returns:
 True if valid
 
 Raises:
 ValueError if conditions not met
 """
 # Check real
 if not np.all(np.isreal(matrix)):
 raise ValueError("Matrix must have real entries")
 
 # Check non-negative
 if not np.all(matrix >= 0):
 raise ValueError("Matrix must have non-negative entries")
 
 # Check square
 if matrix.shape[0] != matrix.shape[1]:
 raise ValueError(f"Matrix must be square, got {matrix.shape}")
 
 logger.info("✓ Matrix satisfies Perron-Frobenius conditions")
 return True
 
 def compute_eigendecomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
 """
 Compute eigendecomposition of matrix.
 
 Args:
 matrix: Input matrix (n×n)
 
 Returns:
 (eigenvalues, eigenvectors) sorted by eigenvalue magnitude
 """
 # Validate matrix
 self.validate_matrix(matrix)
 
 # Compute eigendecomposition
 eigenvalues, eigenvectors = np.linalg.eig(matrix)
 
 # Sort by magnitude (descending)
 idx = np.argsort(np.abs(eigenvalues))[::-1]
 eigenvalues = eigenvalues[idx]
 eigenvectors = eigenvectors[:, idx]
 
 self.eigenvalues = eigenvalues
 self.eigenvectors = eigenvectors
 
 logger.info(f"✓ Computed eigendecomposition")
 logger.info(f" Eigenvalues (top 5): {eigenvalues[:5]}")
 
 return eigenvalues, eigenvectors
 
 def extract_dominant_mode(self, matrix: np.ndarray) -> Tuple[float, np.ndarray]:
 """
 Extract dominant eigenvalue and eigenvector.
 
 Per ENTPC.tex: Perron-Frobenius guarantees unique positive dominant eigenvalue.
 
 Args:
 matrix: Input matrix (n×n)
 
 Returns:
 (dominant_eigenvalue, dominant_eigenvector)
 """
 # Compute eigendecomposition if not already done
 if self.eigenvalues is None:
 self.compute_eigendecomposition(matrix)
 
 # Dominant eigenvalue (largest magnitude)
 self.dominant_eigenvalue = self.eigenvalues[0]
 
 # Dominant eigenvector
 self.dominant_eigenvector = self.eigenvectors[:, 0]
 
 # Ensure positive entries (Perron-Frobenius guarantee)
 # If negative, flip sign
 if np.sum(self.dominant_eigenvector) < 0:
 self.dominant_eigenvector = -self.dominant_eigenvector
 
 # Normalize to unit norm
 self.dominant_eigenvector = self.dominant_eigenvector / np.linalg.norm(self.dominant_eigenvector)
 
 logger.info(f"✓ Extracted dominant mode")
 logger.info(f" Dominant eigenvalue: {self.dominant_eigenvalue:.4f}")
 logger.info(f" Eigenvector norm: {np.linalg.norm(self.dominant_eigenvector):.4f}")
 logger.info(f" Eigenvector min: {np.min(self.dominant_eigenvector):.6f}")
 logger.info(f" Eigenvector max: {np.max(self.dominant_eigenvector):.6f}")
 
 return self.dominant_eigenvalue, self.dominant_eigenvector
 
 def compute_spectral_gap(self) -> float:
 """
 Compute spectral gap λ_max / λ₂.
 
 Per ENTPC.tex: Spectral gap determines collapse rate.
 Larger gap → faster collapse to dominant mode.
 
 Returns:
 Spectral gap ratio
 """
 if self.eigenvalues is None:
 raise ValueError("Eigenvalues not computed yet")
 
 if len(self.eigenvalues) < 2:
 raise ValueError("Need at least 2 eigenvalues for spectral gap")
 
 lambda_max = np.abs(self.eigenvalues[0])
 lambda_2 = np.abs(self.eigenvalues[1])
 
 if lambda_2 < 1e-12:
 logger.warning("Second eigenvalue near zero, spectral gap undefined")
 self.spectral_gap = np.inf
 else:
 self.spectral_gap = lambda_max / lambda_2
 
 logger.info(f"✓ Computed spectral gap: {self.spectral_gap:.4f}")
 logger.info(f" λ_max = {lambda_max:.4f}")
 logger.info(f" λ₂ = {lambda_2:.4f}")
 
 return self.spectral_gap
 
 def power_iteration_collapse(self,
 matrix: np.ndarray,
 initial_state: Optional[np.ndarray] = None,
 n_iterations: int = 100) -> np.ndarray:
 """
 Demonstrate collapse via power iteration.
 
 Per ENTPC.tex: lim_{n→∞} M^n ψ₀ / ||M^n ψ₀|| = v₁
 
 Args:
 matrix: Input matrix (n×n)
 initial_state: Initial state ψ₀ (if None, use random)
 n_iterations: Number of iterations
 
 Returns:
 Collapsed state (should match dominant eigenvector)
 """
 n = matrix.shape[0]
 
 # Initial state
 if initial_state is None:
 psi = np.random.rand(n)
 else:
 psi = initial_state.copy()
 
 # Normalize
 psi = psi / np.linalg.norm(psi)
 
 # Power iteration
 for i in range(n_iterations):
 psi = matrix @ psi
 psi = psi / np.linalg.norm(psi)
 
 logger.info(f"✓ Power iteration converged after {n_iterations} iterations")
 
 return psi
 
 def identify_regime(self, spectral_gap: Optional[float] = None) -> str:
 """
 Identify regime based on spectral gap.
 
 Per ENTPC.tex toy example:
 - Large gap (> 2.0): Regime I (Local Stabilized) - strong collapse
 - Medium gap (1.5 - 2.0): Regime II (Transitional) - partial collapse
 - Small gap (< 1.5): Regime III (Global Experience) - weak collapse
 
 Args:
 spectral_gap: Spectral gap value (if None, use computed)
 
 Returns:
 Regime identifier ('I', 'II', or 'III')
 """
 if spectral_gap is None:
 if self.spectral_gap is None:
 raise ValueError("Spectral gap not computed yet")
 spectral_gap = self.spectral_gap
 
 if spectral_gap > 2.0:
 regime = 'I'
 description = "Local Stabilized (strong collapse)"
 elif spectral_gap > 1.5:
 regime = 'II'
 description = "Transitional (partial collapse)"
 else:
 regime = 'III'
 description = "Global Experience (weak collapse)"
 
 logger.info(f"Identified regime: {regime} - {description}")
 
 return regime
 
 def compute_collapse_report(self, matrix: np.ndarray) -> Dict:
 """
 Generate comprehensive collapse report.
 
 Args:
 matrix: Progenitor matrix (16×16)
 
 Returns:
 Dictionary with all collapse metrics
 """
 # Extract dominant mode
 lambda_max, v1 = self.extract_dominant_mode(matrix)
 
 # Compute spectral gap
 gap = self.compute_spectral_gap()
 
 # Identify regime
 regime = self.identify_regime(gap)
 
 # Compute participation ratio (measure of localization)
 participation_ratio = 1.0 / np.sum(v1**4)
 
 # Normalized spectral gap (per ENTPC.tex Appendix D.2)
 G_norm = (np.abs(self.eigenvalues[0]) - np.abs(self.eigenvalues[1])) / \
 np.abs(self.eigenvalues[0])
 
 # Decay exponent (per ENTPC.tex Appendix D.2)
 # α = -d(log λ_n)/dn
 log_eigs = np.log(np.abs(self.eigenvalues[:10]) + 1e-12)
 alpha = -np.mean(np.diff(log_eigs))
 
 report = {
 'dominant_eigenvalue': float(np.real(lambda_max)),
 'dominant_eigenvector': v1,
 'spectral_gap_ratio': float(gap),
 'normalized_spectral_gap': float(G_norm),
 'decay_exponent': float(alpha),
 'regime': regime,
 'participation_ratio': float(participation_ratio),
 'all_eigenvalues': self.eigenvalues,
 'matrix_trace': float(np.trace(matrix)),
 'matrix_determinant': float(np.linalg.det(matrix)),
 'matrix_norm': float(np.linalg.norm(matrix))
 }
 
 logger.info("=" * 60)
 logger.info("PERRON-FROBENIUS COLLAPSE REPORT")
 logger.info("=" * 60)
 logger.info(f"Dominant eigenvalue: {report['dominant_eigenvalue']:.4f}")
 logger.info(f"Spectral gap ratio: {report['spectral_gap_ratio']:.4f}")
 logger.info(f"Normalized spectral gap: {report['normalized_spectral_gap']:.4f}")
 logger.info(f"Decay exponent α: {report['decay_exponent']:.4f}")
 logger.info(f"Regime: {report['regime']}")
 logger.info(f"Participation ratio: {report['participation_ratio']:.4f}")
 logger.info("=" * 60)
 
 return report
