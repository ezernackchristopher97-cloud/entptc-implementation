"""
Absurdity Gap Calculation

Reference: ENTPC.tex Section 5.2 (lines 649-659, 728-733)

From ENTPC.tex Section 5.2:

"The Absurdity Gap quantifies the discrepancy between the collapsed state (post-
Perron-Frobenius) and the pre-collapse distribution. It is defined as:

Δ_absurd = ||ψ_pre - ψ_post||

where ψ_pre is the pre-collapse state vector and ψ_post is the dominant eigenvector
from the Perron-Frobenius collapse. This gap measures the 'surprise' or 'absurdity'
of the collapse: how much information is lost or reorganized during the transition
from distributed to localized state."

Lines 728-733:
"The Absurdity Gap serves as a diagnostic for regime identification:
- Small gap (Δ < 0.3): Regime I (Local Stabilized) - collapse is expected
- Medium gap (0.3 ≤ Δ < 0.7): Regime II (Transitional) - partial surprise
- Large gap (Δ ≥ 0.7): Regime III (Global Experience) - maximal surprise

CRITICAL: The Absurdity Gap is a POST-OPERATOR ONLY. It is computed AFTER the
Perron-Frobenius collapse has occurred, not before. It measures the consequence
of collapse, not a property of the pre-collapse state alone."
"""

import numpy as np
from typing import Tuple, Dict, Optional

class AbsurdityGap:
 """
 Absurdity Gap Computation
 
 Per ENTPC.tex Section 5.2:
 - POST-OPERATOR ONLY: Applied AFTER Perron-Frobenius collapse
 - Measures discrepancy between pre and post collapse states
 - Diagnostic for regime identification (I, II, III)
 """
 
 # Regime thresholds per ENTPC.tex lines 728-733
 REGIME_I_THRESHOLD = 0.3
 REGIME_II_THRESHOLD = 0.7
 
 def __init__(self):
 """Initialize Absurdity Gap calculator."""
 pass
 
 def compute_gap(self, psi_pre: np.ndarray, psi_post: np.ndarray,
 norm_type: str = 'L2') -> float:
 """
 Compute Absurdity Gap Δ_absurd = ||ψ_pre - ψ_post||.
 
 Per ENTPC.tex: POST-OPERATOR ONLY.
 
 Args:
 psi_pre: Pre-collapse state vector (length n)
 psi_post: Post-collapse state vector (dominant eigenvector, length n)
 norm_type: Norm to use ('L1', 'L2', 'Linf')
 
 Returns:
 Absurdity gap value
 """
 assert len(psi_pre) == len(psi_post), \
 f"State vectors must have same length: {len(psi_pre)} vs {len(psi_post)}"
 
 # Normalize both vectors to unit norm for fair comparison
 psi_pre_norm = psi_pre / (np.linalg.norm(psi_pre) + 1e-12)
 psi_post_norm = psi_post / (np.linalg.norm(psi_post) + 1e-12)
 
 # Compute difference
 diff = psi_pre_norm - psi_post_norm
 
 # Compute norm
 if norm_type == 'L1':
 gap = np.sum(np.abs(diff))
 elif norm_type == 'L2':
 gap = np.linalg.norm(diff)
 elif norm_type == 'Linf':
 gap = np.max(np.abs(diff))
 else:
 raise ValueError(f"Unknown norm type: {norm_type}")
 
 return float(gap)
 
 def identify_regime(self, gap: float) -> str:
 """
 Identify regime based on Absurdity Gap value.
 
 Per ENTPC.tex lines 728-733:
 - Δ < 0.3: Regime I (Local Stabilized)
 - 0.3 ≤ Δ < 0.7: Regime II (Transitional)
 - Δ ≥ 0.7: Regime III (Global Experience)
 
 Args:
 gap: Absurdity gap value
 
 Returns:
 Regime identifier ('I', 'II', or 'III')
 """
 if gap < self.REGIME_I_THRESHOLD:
 return 'I'
 elif gap < self.REGIME_II_THRESHOLD:
 return 'II'
 else:
 return 'III'
 
 def compute_gap_components(self, psi_pre: np.ndarray, psi_post: np.ndarray) -> Dict[str, float]:
 """
 Compute detailed gap components for analysis.
 
 Returns multiple measures of discrepancy.
 
 Args:
 psi_pre: Pre-collapse state vector
 psi_post: Post-collapse state vector
 
 Returns:
 Dictionary with gap measures and diagnostics
 """
 # Normalize vectors
 psi_pre_norm = psi_pre / (np.linalg.norm(psi_pre) + 1e-12)
 psi_post_norm = psi_post / (np.linalg.norm(psi_post) + 1e-12)
 
 # Compute various gap measures
 gap_L1 = self.compute_gap(psi_pre, psi_post, norm_type='L1')
 gap_L2 = self.compute_gap(psi_pre, psi_post, norm_type='L2')
 gap_Linf = self.compute_gap(psi_pre, psi_post, norm_type='Linf')
 
 # Overlap (fidelity): |⟨ψ_pre|ψ_post⟩|
 overlap = abs(np.dot(psi_pre_norm, psi_post_norm))
 
 # Information loss: 1 - overlap
 info_loss = 1.0 - overlap
 
 # Entropy change (if states are probability distributions)
 # Ensure non-negative for entropy calculation
 p_pre = np.abs(psi_pre_norm)**2
 p_post = np.abs(psi_post_norm)**2
 
 # Shannon entropy
 H_pre = -np.sum(p_pre * np.log(p_pre + 1e-12))
 H_post = -np.sum(p_post * np.log(p_post + 1e-12))
 entropy_change = H_post - H_pre
 
 # Identify regime
 regime = self.identify_regime(gap_L2)
 
 return {
 'gap_L1': gap_L1,
 'gap_L2': gap_L2,
 'gap_Linf': gap_Linf,
 'overlap': overlap,
 'info_loss': info_loss,
 'entropy_pre': H_pre,
 'entropy_post': H_post,
 'entropy_change': entropy_change,
 'regime': regime
 }
 
 def compute_gap_matrix(self, psi_pre_matrix: np.ndarray,
 psi_post_matrix: np.ndarray) -> np.ndarray:
 """
 Compute Absurdity Gap for matrix of state vectors.
 
 Used for batch processing of multiple subjects or time points.
 
 Args:
 psi_pre_matrix: Matrix of shape (n_samples, n_features)
 Each row is a pre-collapse state vector
 psi_post_matrix: Matrix of shape (n_samples, n_features)
 Each row is a post-collapse state vector
 
 Returns:
 Array of shape (n_samples,) with gap values
 """
 assert psi_pre_matrix.shape == psi_post_matrix.shape, \
 "Pre and post matrices must have same shape"
 
 n_samples = psi_pre_matrix.shape[0]
 gaps = np.zeros(n_samples)
 
 for i in range(n_samples):
 gaps[i] = self.compute_gap(psi_pre_matrix[i], psi_post_matrix[i])
 
 return gaps
 
 def compute_temporal_gap(self, psi_pre_sequence: np.ndarray,
 psi_post_sequence: np.ndarray) -> Tuple[np.ndarray, Dict]:
 """
 Compute Absurdity Gap over temporal sequence.
 
 Used for EEG time series analysis.
 
 Args:
 psi_pre_sequence: Array of shape (T, n) with pre-collapse states
 psi_post_sequence: Array of shape (T, n) with post-collapse states
 
 Returns:
 (gaps, statistics) where gaps is array of length T
 """
 T = len(psi_pre_sequence)
 gaps = self.compute_gap_matrix(psi_pre_sequence, psi_post_sequence)
 
 # Compute temporal statistics
 statistics = {
 'mean_gap': float(np.mean(gaps)),
 'std_gap': float(np.std(gaps)),
 'min_gap': float(np.min(gaps)),
 'max_gap': float(np.max(gaps)),
 'median_gap': float(np.median(gaps)),
 'regime_distribution': self._compute_regime_distribution(gaps)
 }
 
 return gaps, statistics
 
 def _compute_regime_distribution(self, gaps: np.ndarray) -> Dict[str, float]:
 """
 Compute distribution of regimes from gap values.
 
 Args:
 gaps: Array of gap values
 
 Returns:
 Dictionary with fraction of time in each regime
 """
 regime_I = np.sum(gaps < self.REGIME_I_THRESHOLD)
 regime_II = np.sum((gaps >= self.REGIME_I_THRESHOLD) & (gaps < self.REGIME_II_THRESHOLD))
 regime_III = np.sum(gaps >= self.REGIME_II_THRESHOLD)
 
 total = len(gaps)
 
 return {
 'regime_I': float(regime_I / total),
 'regime_II': float(regime_II / total),
 'regime_III': float(regime_III / total)
 }

class AbsurdityGapAnalyzer:
 """
 Analyze Absurdity Gap patterns for subject comparison.
 
 Per ENTPC.tex: Gap patterns distinguish treatment effects.
 """
 
 def __init__(self):
 """Initialize analyzer."""
 self.gap_calculator = AbsurdityGap()
 
 def compare_pre_post_treatment(self, pre_treatment_gaps: np.ndarray,
 post_treatment_gaps: np.ndarray) -> Dict:
 """
 Compare Absurdity Gap distributions pre vs post treatment.
 
 Args:
 pre_treatment_gaps: Array of gap values before treatment
 post_treatment_gaps: Array of gap values after treatment
 
 Returns:
 Dictionary with comparison statistics
 """
 # Mean gap change
 mean_change = np.mean(post_treatment_gaps) - np.mean(pre_treatment_gaps)
 
 # Statistical test (paired t-test approximation)
 diff = post_treatment_gaps - pre_treatment_gaps
 t_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)) + 1e-12)
 
 # Regime distribution changes
 pre_regimes = self.gap_calculator._compute_regime_distribution(pre_treatment_gaps)
 post_regimes = self.gap_calculator._compute_regime_distribution(post_treatment_gaps)
 
 regime_changes = {
 f'regime_{r}_change': post_regimes[f'regime_{r}'] - pre_regimes[f'regime_{r}']
 for r in ['I', 'II', 'III']
 }
 
 return {
 'mean_gap_pre': float(np.mean(pre_treatment_gaps)),
 'mean_gap_post': float(np.mean(post_treatment_gaps)),
 'mean_gap_change': float(mean_change),
 'std_gap_pre': float(np.std(pre_treatment_gaps)),
 'std_gap_post': float(np.std(post_treatment_gaps)),
 't_statistic': float(t_stat),
 'pre_regime_distribution': pre_regimes,
 'post_regime_distribution': post_regimes,
 'regime_changes': regime_changes
 }
 
 def compute_subject_gap_profile(self, psi_pre_sequence: np.ndarray,
 psi_post_sequence: np.ndarray,
 subject_id: str) -> Dict:
 """
 Compute comprehensive Absurdity Gap profile for single subject.
 
 Args:
 psi_pre_sequence: Pre-collapse state sequence (T, n)
 psi_post_sequence: Post-collapse state sequence (T, n)
 subject_id: Subject identifier
 
 Returns:
 Dictionary with complete gap profile
 """
 # Compute temporal gaps
 gaps, statistics = self.gap_calculator.compute_temporal_gap(
 psi_pre_sequence, psi_post_sequence
 )
 
 # Compute detailed components for representative time points
 T = len(gaps)
 representative_indices = [0, T//4, T//2, 3*T//4, T-1]
 
 components = []
 for idx in representative_indices:
 comp = self.gap_calculator.compute_gap_components(
 psi_pre_sequence[idx], psi_post_sequence[idx]
 )
 comp['time_index'] = idx
 components.append(comp)
 
 return {
 'subject_id': subject_id,
 'temporal_gaps': gaps,
 'statistics': statistics,
 'representative_components': components,
 'dominant_regime': max(statistics['regime_distribution'].items(),
 key=lambda x: x[1])[0]
 }
 
 def compute_cohort_gap_summary(self, subject_profiles: list) -> Dict:
 """
 Compute summary statistics across cohort.
 
 Args:
 subject_profiles: List of subject gap profiles
 
 Returns:
 Dictionary with cohort-level statistics
 """
 # Extract mean gaps for each subject
 subject_mean_gaps = [
 profile['statistics']['mean_gap']
 for profile in subject_profiles
 ]
 
 # Extract regime distributions
 regime_distributions = [
 profile['statistics']['regime_distribution']
 for profile in subject_profiles
 ]
 
 # Aggregate regime distributions
 cohort_regime_dist = {
 'regime_I': np.mean([rd['regime_I'] for rd in regime_distributions]),
 'regime_II': np.mean([rd['regime_II'] for rd in regime_distributions]),
 'regime_III': np.mean([rd['regime_III'] for rd in regime_distributions])
 }
 
 # Identify dominant regime for each subject
 dominant_regimes = [profile['dominant_regime'] for profile in subject_profiles]
 regime_counts = {
 'regime_I': dominant_regimes.count('regime_I'),
 'regime_II': dominant_regimes.count('regime_II'),
 'regime_III': dominant_regimes.count('regime_III')
 }
 
 return {
 'n_subjects': len(subject_profiles),
 'mean_gap_across_subjects': float(np.mean(subject_mean_gaps)),
 'std_gap_across_subjects': float(np.std(subject_mean_gaps)),
 'median_gap_across_subjects': float(np.median(subject_mean_gaps)),
 'cohort_regime_distribution': cohort_regime_dist,
 'dominant_regime_counts': regime_counts,
 'subject_mean_gaps': subject_mean_gaps
 }

def validate_absurdity_gap_computation(psi_pre: np.ndarray, psi_post: np.ndarray) -> bool:
 """
 Validate that Absurdity Gap computation is being used correctly.
 
 Per ENTPC.tex: POST-OPERATOR ONLY.
 
 Args:
 psi_pre: Pre-collapse state vector
 psi_post: Post-collapse state vector (must be from Perron-Frobenius)
 
 Returns:
 True if validation passes
 
 Raises:
 AssertionError if misused
 """
 # Check that psi_post is normalized (characteristic of eigenvector)
 post_norm = np.linalg.norm(psi_post)
 assert abs(post_norm - 1.0) < 0.1 or post_norm > 0.9, \
 "psi_post should be normalized eigenvector from Perron-Frobenius collapse"
 
 # Check dimensions match
 assert len(psi_pre) == len(psi_post), \
 f"State vectors must have same dimension: {len(psi_pre)} vs {len(psi_post)}"
 
 # Check that vectors are real (no complex components from incorrect usage)
 assert np.all(np.isreal(psi_pre)), "psi_pre must be real-valued"
 assert np.all(np.isreal(psi_post)), "psi_post must be real-valued"
 
 return True
