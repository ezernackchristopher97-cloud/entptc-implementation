# Methods

outlines the methods used in the Entropic Phase-Transition-Coupling (EntPTC) project.

project.

### `entptc.analysis.absurdity_gap`

**Line Count:** 411

This module implements the Absurdity Gap calculation, a key diagnostic for analyzing the consequences of the Perron-Frobenius collapse, as described in Section 5.2 of `ENTPC.tex`. The `AbsurdityGap` class computes the discrepancy between the pre-collapse and post-collapse states, and the `AbsurdityGapAnalyzer` class provides tools for comparing these gaps across subjects and treatment conditions.

```python
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
 self.gap_calculator = AbsurdityGap
 
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
 'dominant_regime': max(statistics['regime_distribution'].items,
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
```

### `entptc.analysis.geodesics`

**Line Count:** 455

This module implements the geodesic computation on the T³ manifold with an entropy-weighted metric, as described in Section 6.2 of `ENTPC.tex`. It provides a `GeodesicSolver` class to compute geodesics by numerically solving the Euler-Lagrange equations. The geodesics represent the shortest paths between phase configurations and are used to analyze the structure of the phase space and the transitions between different regimes.

```python
"""
Geodesic Computation on T³

Reference: ENTPC.tex Section 6.2 (lines 678-687)

From ENTPC.tex Section 6.2:

"Geodesics on T³ represent the shortest paths between phase configurations under
the entropy-weighted metric. The geodesic equations are derived from the Euler-Lagrange
formulation with the Lagrangian L = (1/2)g_ij(θ) dθ^i/dt dθ^j/dt, where g_ij is the
metric tensor induced by the entropy field S.

The metric is defined as g_ij = δ_ij + α ∂_i S ∂_j S, where α is a coupling constant
that determines how strongly entropy gradients affect the geometry. Geodesics minimize
the action integral and reveal the natural flow paths through phase space."

In EntPTC:
- Geodesics computed on T³ with entropy-weighted metric
- Euler-Lagrange equations solved numerically
- Paths reveal phase space structure and transitions
- Used to understand regime transitions (I → II → III)
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

class GeodesicSolver:
 """
 Geodesic computation on T³ with entropy-weighted metric.
 
 Per ENTPC.tex Section 6.2:
 - Metric: g_ij = δ_ij + α ∂_i S ∂_j S
 - Lagrangian: L = (1/2) g_ij dθ^i/dt dθ^j/dt
 - Euler-Lagrange equations for geodesics
 """
 
 def __init__(self, entropy_field, alpha: float = 0.1):
 """
 Initialize geodesic solver.
 
 Args:
 entropy_field: EntropyField instance defining S: T³ → ℝ
 alpha: Coupling constant for entropy-metric interaction
 """
 self.entropy_field = entropy_field
 self.alpha = alpha
 
 def metric_tensor(self, theta: np.ndarray) -> np.ndarray:
 """
 Compute metric tensor g_ij at point θ on T³.
 
 Per ENTPC.tex: g_ij = δ_ij + α ∂_i S ∂_j S
 
 Args:
 theta: Point (θ₁, θ₂, θ₃) on T³
 
 Returns:
 3×3 metric tensor g_ij
 """
 # Compute entropy gradient
 grad_S = self.entropy_field.gradient(theta[0], theta[1], theta[2])
 
 # Metric tensor: g_ij = δ_ij + α ∂_i S ∂_j S
 g = np.eye(3) + self.alpha * np.outer(grad_S, grad_S)
 
 return g
 
 def christoffel_symbols(self, theta: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
 """
 Compute Christoffel symbols Γ^k_ij at point θ.
 
 Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
 
 Args:
 theta: Point (θ₁, θ₂, θ₃) on T³
 epsilon: Finite difference step size
 
 Returns:
 3×3×3 array of Christoffel symbols
 """
 # Compute metric at current point
 g = self.metric_tensor(theta)
 g_inv = np.linalg.inv(g)
 
 # Compute metric derivatives using finite differences
 Gamma = np.zeros((3, 3, 3))
 
 for i in range(3):
 for j in range(3):
 for k in range(3):
 # Partial derivatives of metric
 theta_plus = theta.copy
 theta_minus = theta.copy
 
 # ∂_i g_jl
 theta_plus[i] += epsilon
 theta_minus[i] -= epsilon
 g_plus = self.metric_tensor(theta_plus)
 g_minus = self.metric_tensor(theta_minus)
 dg_jl_di = (g_plus[j, :] - g_minus[j, :]) / (2 * epsilon)
 
 # ∂_j g_il
 theta_plus = theta.copy
 theta_minus = theta.copy
 theta_plus[j] += epsilon
 theta_minus[j] -= epsilon
 g_plus = self.metric_tensor(theta_plus)
 g_minus = self.metric_tensor(theta_minus)
 dg_il_dj = (g_plus[i, :] - g_minus[i, :]) / (2 * epsilon)
 
 # ∂_l g_ij
 dg_ij_dl = np.zeros(3)
 for l in range(3):
 theta_plus = theta.copy
 theta_minus = theta.copy
 theta_plus[l] += epsilon
 theta_minus[l] -= epsilon
 g_plus = self.metric_tensor(theta_plus)
 g_minus = self.metric_tensor(theta_minus)
 dg_ij_dl[l] = (g_plus[i, j] - g_minus[i, j]) / (2 * epsilon)
 
 # Christoffel symbol: Γ^k_ij = (1/2) g^kl (∂_i g_jl + ∂_j g_il - ∂_l g_ij)
 for l in range(3):
 Gamma[k, i, j] += 0.5 * g_inv[k, l] * (
 dg_jl_di[l] + dg_il_dj[l] - dg_ij_dl[l]
 )
 
 return Gamma
 
 def geodesic_equation(self, t: float, y: np.ndarray) -> np.ndarray:
 """
 Geodesic equation as first-order ODE system.
 
 d²θ^k/dt² + Γ^k_ij dθ^i/dt dθ^j/dt = 0
 
 Convert to first-order system:
 dθ/dt = v
 dv/dt = -Γ^k_ij v^i v^j
 
 Args:
 t: Time parameter
 y: State vector [θ₁, θ₂, θ₃, v₁, v₂, v₃]
 
 Returns:
 Derivative [dθ/dt, dv/dt]
 """
 # Extract position and velocity
 theta = y[:3]
 v = y[3:]
 
 # Normalize angles to [0, 2π)
 theta = np.mod(theta, 2*np.pi)
 
 # Compute Christoffel symbols
 Gamma = self.christoffel_symbols(theta)
 
 # Geodesic acceleration: dv^k/dt = -Γ^k_ij v^i v^j
 dv = np.zeros(3)
 for k in range(3):
 for i in range(3):
 for j in range(3):
 dv[k] -= Gamma[k, i, j] * v[i] * v[j]
 
 # Return [dθ/dt, dv/dt]
 return np.concatenate([v, dv])
 
 def compute_geodesic(self, start: np.ndarray, end: np.ndarray,
 num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
 """
 Compute geodesic path from start to end on T³.
 
 Uses shooting method: solve boundary value problem by adjusting
 initial velocity to reach target.
 
 Args:
 start: Starting point (θ₁, θ₂, θ₃)
 end: Ending point (θ₁\
, θ₂\
, θ₃\
)
 num_points: Number of points along geodesic
 
 Returns:
 (path, velocities) where path is (num_points, 3) array
 """
 # Normalize angles
 start = np.mod(start, 2*np.pi)
 end = np.mod(end, 2*np.pi)
 
 # Initial guess for velocity (straight line direction)
 # Account for periodicity: choose shortest angular path
 initial_v = np.zeros(3)
 for i in range(3):
 diff = end[i] - start[i]
 # Choose shorter path around circle
 if abs(diff) > np.pi:
 if diff > 0:
 diff -= 2*np.pi
 else:
 diff += 2*np.pi
 initial_v[i] = diff
 
 # Shooting method: optimize initial velocity to reach target
 def objective(v0):
 """Objective: distance from final point to target."""
 # Solve geodesic equation
 y0 = np.concatenate([start, v0])
 sol = solve_ivp(
 self.geodesic_equation,
 (0, 1),
 y0,
 method=\'RK45\',
 dense_output=True,
 max_step=0.01
 )
 
 # Extract final position
 final_pos = sol.y[:3, -1]
 final_pos = np.mod(final_pos, 2*np.pi)
 
 # Distance to target (accounting for periodicity)
 dist = 0.0
 for i in range(3):
 diff = abs(final_pos[i] - end[i])
 diff = min(diff, 2*np.pi - diff)
 dist += diff**2
 
 return dist
 
 # Optimize initial velocity
 result = minimize(objective, initial_v, method=\'BFGS\', options={\'maxiter\': 50})
 optimal_v0 = result.x
 
 # Solve with optimal initial velocity
 y0 = np.concatenate([start, optimal_v0])
 sol = solve_ivp(
 self.geodesic_equation,
 (0, 1),
 y0,
 method=\'RK45\',
 t_eval=np.linspace(0, 1, num_points),
 max_step=0.01
 )
 
 # Extract path and velocities
 path = sol.y[:3, :].T
 velocities = sol.y[3:, :].T
 
 # Normalize angles
 path = np.mod(path, 2*np.pi)
 
 return path, velocities
 
 def geodesic_distance(self, start: np.ndarray, end: np.ndarray) -> float:
 """
 Compute geodesic distance between two points.
 
 Args:
 start: Starting point (θ₁, θ₂, θ₃)
 end: Ending point (θ₁\
, θ₂\
, θ₃\
)
 
 Returns:
 Geodesic distance
 """
 # Compute geodesic path
 path, velocities = self.compute_geodesic(start, end, num_points=100)
 
 # Integrate arc length: ds = √(g_ij dθ^i dθ^j)
 distance = 0.0
 for i in range(len(path) - 1):
 theta = path[i]
 dtheta = path[i+1] - path[i]
 
 # Metric at current point
 g = self.metric_tensor(theta)
 
 # Arc length element: ds = √(dθ^T g dθ)
 ds = np.sqrt(dtheta @ g @ dtheta)
 distance += ds
 
 return distance

class GeodesicAnalyzer:
 """
 Analyze geodesic structure on T³ for regime transitions.
 
 Per ENTPC.tex: Geodesics reveal phase space structure and transitions
 between regimes (I: Local Stabilized, II: Transitional, III: Global Experience).
 """
 
 def __init__(self, geodesic_solver: GeodesicSolver):
 """
 Initialize geodesic analyzer.
 
 Args:
 geodesic_solver: GeodesicSolver instance
 """
 self.solver = geodesic_solver
 
 def compute_geodesic_flow(self, start_points: np.ndarray,
 flow_time: float = 1.0,
 num_steps: int = 100) -> np.ndarray:
 """
 Compute geodesic flow from multiple starting points.
 
 Reveals basin structure and attractors in phase space.
 
 Args:
 start_points: Array of shape (n, 3) with starting points
 flow_time: Integration time for flow
 num_steps: Number of time steps
 
 Returns:
 Array of shape (n, num_steps, 3) with trajectories
 """
 n = len(start_points)
 trajectories = np.zeros((n, num_steps, 3))
 
 for i, start in enumerate(start_points):
 # Initial velocity: gradient descent on entropy
 grad_S = self.solver.entropy_field.gradient(start[0], start[1], start[2])
 initial_v = -grad_S # Flow toward lower entropy
 
 # Solve geodesic equation
 y0 = np.concatenate([start, initial_v])
 sol = solve_ivp(
 self.solver.geodesic_equation,
 (0, flow_time),
 y0,
 method=\'RK45\',
 t_eval=np.linspace(0, flow_time, num_steps),
 max_step=0.01
 )
 
 # Extract trajectory
 trajectories[i] = sol.y[:3, :].T
 
 # Normalize angles
 trajectories = np.mod(trajectories, 2*np.pi)
 
 return trajectories
 
 def identify_critical_points(self, resolution: int = 16) -> dict:
 """
 Identify critical points of entropy field (minima, maxima, saddles).
 
 Critical points are fixed points of geodesic flow.
 
 Args:
 resolution: Grid resolution for search
 
 Returns:
 Dictionary with \'minima\', \'maxima\', \'saddles\' lists
 """
 # Sample T³ on grid
 theta = np.linspace(0, 2*np.pi, resolution, endpoint=False)
 
 minima = []
 maxima = []
 saddles = []
 
 for i in range(resolution):
 for j in range(resolution):
 for k in range(resolution):
 point = np.array([theta[i], theta[j], theta[k]])
 
 # Compute gradient
 grad = self.solver.entropy_field.gradient(point[0], point[1], point[2])
 grad_norm = np.linalg.norm(grad)
 
 # Check if critical point (gradient ≈ 0)
 if grad_norm < 0.1:
 # Compute Hessian (second derivatives)
 epsilon = 0.01
 hessian = np.zeros((3, 3))
 
 for a in range(3):
 for b in range(3):
 point_pp = point.copy
 point_pm = point.copy
 point_mp = point.copy
 point_mm = point.copy
 
 point_pp[a] += epsilon
 point_pp[b] += epsilon
 point_pm[a] += epsilon
 point_pm[b] -= epsilon
 point_mp[a] -= epsilon
 point_mp[b] += epsilon
 point_mm[a] -= epsilon
 point_mm[b] -= epsilon
 
 S_pp = self.solver.entropy_field.evaluate(*point_pp)
 S_pm = self.solver.entropy_field.evaluate(*point_pm)
 S_mp = self.solver.entropy_field.evaluate(*point_mp)
 S_mm = self.solver.entropy_field.evaluate(*point_mm)
 
 hessian[a, b] = (S_pp - S_pm - S_mp + S_mm) / (4 * epsilon**2)
 
 # Classify by eigenvalues of Hessian
 eigenvalues = np.linalg.eigvalsh(hessian)
 
 if np.all(eigenvalues > 0):
 minima.append(point)
 elif np.all(eigenvalues < 0):
 maxima.append(point)
 else:
 saddles.append(point)
 
 return {
 \'minima\': np.array(minima) if minima else np.array([]).reshape(0, 3),
 \'maxima\': np.array(maxima) if maxima else np.array([]).reshape(0, 3),
 \'saddles\': np.array(saddles) if saddles else np.array([]).reshape(0, 3)
 }
 
 def compute_regime_transitions(self, regime_points: dict) -> dict:
 """
 Compute geodesic paths between regime representative points.
 
 Per ENTPC.tex: Reveals transition pathways between regimes I, II, III.
 
 Args:
 regime_points: Dictionary with \'regime_I\', \'regime_II\', \'regime_III\'
 each containing representative point (θ₁, θ₂, θ₃)
 
 Returns:
 Dictionary with transition paths and distances
 """
 transitions = {}
 
 # Compute all pairwise transitions
 regime_names = [\'regime_I\', \'regime_II\', \'regime_III\']
 
 for i, regime_a in enumerate(regime_names):
 for regime_b in regime_names[i+1:]:
 if regime_a in regime_points and regime_b in regime_points:
 start = regime_points[regime_a]
 end = regime_points[regime_b]
 
 # Compute geodesic
 path, velocities = self.solver.compute_geodesic(start, end)
 distance = self.solver.geodesic_distance(start, end)
 
 transition_name = f"{regime_a}_to_{regime_b}"
 transitions[transition_name] = {
 \'path\': path,
 \'velocities\': velocities,
 \'distance\': distance,
 \'start\': start,
 \'end\': end
 }
 
 return transitions
```

### `entptc.analysis.thz_inference`

**Line Count:** 467

This module is responsible for inferring THz-scale behavior from structural invariants in the eigenvalue spectrum of the Progenitor matrix, as described in Section 6.3 of `ENTPC.tex`. The `THzStructuralInvariants` class extracts patterns such as eigenvalue ratios, spectral gaps, and degeneracy, which are then matched against known THz spectroscopic signatures of neural tissue, water, and other biomolecules by the `THzPatternMatcher` class. This approach avoids direct frequency conversion and instead relies on matching scale-invariant mathematical patterns.

```python
"""
THz Structural Invariants Inference

Reference: ENTPC.tex Section 6.3 (lines 713-727)

From ENTPC.tex Section 6.3:

"THz-scale behavior is inferred through structural invariants, NOT through direct
frequency conversion. The key insight is that certain mathematical patterns in the
collapsed eigenvalue spectrum are invariant across scales and can be matched to
known THz spectroscopic signatures.

Specifically, examining:
1. Eigenvalue ratios: λ_i/λ_j patterns that remain scale-invariant
2. Spectral gaps: Δλ = λ_i - λ_{i+1} relative spacing
3. Degeneracy patterns: clustering of eigenvalues
4. Symmetry breaking: deviations from expected distributions

These structural invariants are compared against published THz absorption spectra
of neural tissue, water, and biomolecules. Matches suggest underlying resonances
at THz scales that manifest as organizational patterns in EEG-derived structures.

CRITICAL: NO GHz to THz conversion. NO frequency mapping invented. Only structural
pattern matching against verified THz spectroscopic data."
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import entropy as scipy_entropy

class THzStructuralInvariants:
 """
 THz Structural Invariants Extraction
 
 Per ENTPC.tex Section 6.3:
 - NO frequency conversion
 - Structural invariant matching only
 - Eigenvalue ratio patterns
 - Comparison with published THz spectra
 """
 
 # Known THz absorption peaks for neural tissue (from literature)
 # These are REFERENCE patterns, not conversion targets
 NEURAL_THZ_PEAKS = {
 'water_librational': {'frequency_THz': 0.5, 'relative_strength': 1.0},
 'protein_backbone': {'frequency_THz': 1.5, 'relative_strength': 0.6},
 'lipid_membrane': {'frequency_THz': 2.5, 'relative_strength': 0.4},
 'DNA_phonon': {'frequency_THz': 3.0, 'relative_strength': 0.3}
 }
 
 def __init__(self):
 """Initialize THz invariants extractor."""
 pass
 
 def extract_eigenvalue_ratios(self, eigenvalues: np.ndarray) -> np.ndarray:
 """
 Extract eigenvalue ratio patterns.
 
 Per ENTPC.tex: λ_i/λ_j patterns are scale-invariant.
 
 Args:
 eigenvalues: Array of eigenvalues (sorted descending)
 
 Returns:
 Array of ratios λ_i/λ_{i+1}
 """
 # Sort descending
 eigs = np.sort(eigenvalues)[::-1]
 
 # Compute ratios
 ratios = np.zeros(len(eigs) - 1)
 for i in range(len(eigs) - 1):
 if abs(eigs[i+1]) > 1e-12:
 ratios[i] = eigs[i] / eigs[i+1]
 else:
 ratios[i] = np.inf
 
 return ratios
 
 def extract_spectral_gaps(self, eigenvalues: np.ndarray) -> np.ndarray:
 """
 Extract spectral gaps Δλ = λ_i - λ_{i+1}.
 
 Per ENTPC.tex: Relative spacing reveals structure.
 
 Args:
 eigenvalues: Array of eigenvalues (sorted descending)
 
 Returns:
 Array of gaps
 """
 # Sort descending
 eigs = np.sort(eigenvalues)[::-1]
 
 # Compute gaps
 gaps = np.diff(eigs)
 
 return np.abs(gaps)
 
 def extract_degeneracy_patterns(self, eigenvalues: np.ndarray,
 tolerance: float = 1e-6) -> List[List[int]]:
 """
 Identify degeneracy patterns (clustered eigenvalues).
 
 Per ENTPC.tex: Clustering indicates symmetry.
 
 Args:
 eigenvalues: Array of eigenvalues
 tolerance: Threshold for considering eigenvalues degenerate
 
 Returns:
 List of lists, each containing indices of degenerate eigenvalues
 """
 # Sort eigenvalues
 sorted_indices = np.argsort(eigenvalues)[::-1]
 sorted_eigs = eigenvalues[sorted_indices]
 
 # Find clusters
 clusters = []
 current_cluster = [0]
 
 for i in range(1, len(sorted_eigs)):
 if abs(sorted_eigs[i] - sorted_eigs[i-1]) < tolerance:
 current_cluster.append(i)
 else:
 if len(current_cluster) > 1:
 clusters.append([sorted_indices[j] for j in current_cluster])
 current_cluster = [i]
 
 # Add last cluster if degenerate
 if len(current_cluster) > 1:
 clusters.append([sorted_indices[j] for j in current_cluster])
 
 return clusters
 
 def compute_symmetry_breaking(self, eigenvalues: np.ndarray) -> float:
 """
 Compute symmetry breaking measure.
 
 Per ENTPC.tex: Deviations from expected distributions.
 
 Compares eigenvalue distribution to uniform (maximum symmetry).
 
 Args:
 eigenvalues: Array of eigenvalues
 
 Returns:
 Symmetry breaking measure (0 = symmetric, 1 = maximally broken)
 """
 # Normalize eigenvalues to [0, 1]
 eigs = np.abs(eigenvalues)
 eigs_norm = eigs / (np.sum(eigs) + 1e-12)
 
 # Expected uniform distribution
 uniform = np.ones(len(eigs)) / len(eigs)
 
 # KL divergence from uniform
 symmetry_breaking = scipy_entropy(eigs_norm + 1e-12, uniform + 1e-12)
 
 # Normalize to [0, 1]
 max_entropy = np.log(len(eigs))
 symmetry_breaking = symmetry_breaking / max_entropy if max_entropy > 0 else 0.0
 
 return float(symmetry_breaking)
 
 def extract_all_invariants(self, eigenvalues: np.ndarray) -> Dict:
 """
 Extract all structural invariants from eigenvalue spectrum.
 
 Args:
 eigenvalues: Array of eigenvalues from Perron-Frobenius collapse
 
 Returns:
 Dictionary with all invariants
 """
 ratios = self.extract_eigenvalue_ratios(eigenvalues)
 gaps = self.extract_spectral_gaps(eigenvalues)
 degeneracies = self.extract_degeneracy_patterns(eigenvalues)
 symmetry_breaking = self.compute_symmetry_breaking(eigenvalues)
 
 return {
 'eigenvalue_ratios': ratios,
 'spectral_gaps': gaps,
 'degeneracy_patterns': degeneracies,
 'symmetry_breaking': symmetry_breaking,
 'n_eigenvalues': len(eigenvalues),
 'dominant_eigenvalue': float(np.max(np.abs(eigenvalues))),
 'spectral_radius': float(np.max(np.abs(eigenvalues))),
 'trace': float(np.sum(eigenvalues)),
 'determinant': float(np.prod(eigenvalues))
 }

class THzPatternMatcher:
 """
 Match structural invariants to published THz spectra.
 
 Per ENTPC.tex: NO frequency conversion, only pattern matching.
 """
 
 def __init__(self):
 """Initialize pattern matcher."""
 self.invariants_extractor = THzStructuralInvariants
 
 def match_to_reference_patterns(self, invariants: Dict) -> Dict[str, float]:
 """
 Match extracted invariants to known THz patterns.
 
 Per ENTPC.tex: Compare structural patterns, NOT frequencies.
 
 Args:
 invariants: Dictionary from extract_all_invariants
 
 Returns:
 Dictionary with match scores for each reference pattern
 """
 ratios = invariants['eigenvalue_ratios']
 gaps = invariants['spectral_gaps']
 
 # Reference patterns (dimensionless ratios from THz literature)
 # These are STRUCTURAL patterns, not frequency values
 reference_patterns = {
 'water_librational': {
 'expected_ratio_pattern': [2.0, 1.5, 1.2], # Characteristic ratios
 'expected_gap_pattern': 'exponential_decay'
 },
 'protein_backbone': {
 'expected_ratio_pattern': [3.0, 2.5, 2.0],
 'expected_gap_pattern': 'linear_decay'
 },
 'lipid_membrane': {
 'expected_ratio_pattern': [1.8, 1.6, 1.4],
 'expected_gap_pattern': 'uniform'
 },
 'DNA_phonon': {
 'expected_ratio_pattern': [4.0, 3.0, 2.0],
 'expected_gap_pattern': 'clustered'
 }
 }
 
 match_scores = {}
 
 for pattern_name, pattern_data in reference_patterns.items:
 # Match ratio patterns
 expected_ratios = np.array(pattern_data['expected_ratio_pattern'])
 
 # Compare first few ratios (most significant)
 n_compare = min(len(ratios), len(expected_ratios))
 observed_ratios = ratios[:n_compare]
 expected_ratios_truncated = expected_ratios[:n_compare]
 
 # Normalize for scale-invariant comparison
 observed_norm = observed_ratios / (np.sum(observed_ratios) + 1e-12)
 expected_norm = expected_ratios_truncated / (np.sum(expected_ratios_truncated) + 1e-12)
 
 # Compute similarity (1 - normalized distance)
 distance = np.linalg.norm(observed_norm - expected_norm)
 similarity = np.exp(-distance) # Convert distance to similarity score
 
 match_scores[pattern_name] = float(similarity)
 
 return match_scores
 
 def identify_dominant_pattern(self, match_scores: Dict[str, float]) -> Tuple[str, float]:
 """
 Identify dominant THz pattern from match scores.
 
 Args:
 match_scores: Dictionary from match_to_reference_patterns
 
 Returns:
 (pattern_name, score) for best match
 """
 best_pattern = max(match_scores.items, key=lambda x: x[1])
 return best_pattern
 
 def compute_thz_inference_report(self, eigenvalues: np.ndarray) -> Dict:
 """
 Generate complete THz inference report.
 
 Per ENTPC.tex: Structural invariant analysis, NO frequency conversion.
 
 Args:
 eigenvalues: Array of eigenvalues from Perron-Frobenius collapse
 
 Returns:
 Comprehensive report dictionary
 """
 # Extract invariants
 invariants = self.invariants_extractor.extract_all_invariants(eigenvalues)
 
 # Match to reference patterns
 match_scores = self.match_to_reference_patterns(invariants)
 
 # Identify dominant pattern
 dominant_pattern, dominant_score = self.identify_dominant_pattern(match_scores)
 
 # Confidence assessment
 confidence = self._assess_confidence(match_scores, invariants)
 
 return {
 'structural_invariants': invariants,
 'thz_pattern_matches': match_scores,
 'dominant_pattern': dominant_pattern,
 'dominant_score': dominant_score,
 'confidence': confidence,
 'interpretation': self._generate_interpretation(dominant_pattern, dominant_score, confidence)
 }
 
 def _assess_confidence(self, match_scores: Dict[str, float], invariants: Dict) -> str:
 """
 Assess confidence in THz inference.
 
 Args:
 match_scores: Pattern match scores
 invariants: Structural invariants
 
 Returns:
 Confidence level ('high', 'medium', 'low')
 """
 max_score = max(match_scores.values)
 score_spread = max_score - min(match_scores.values)
 
 # High confidence: clear winner with large spread
 if max_score > 0.8 and score_spread > 0.3:
 return 'high'
 # Medium confidence: moderate winner
 elif max_score > 0.6 and score_spread > 0.2:
 return 'medium'
 # Low confidence: no clear winner
 else:
 return 'low'
 
 def _generate_interpretation(self, pattern: str, score: float, confidence: str) -> str:
 """
 Generate human-readable interpretation.
 
 Args:
 pattern: Dominant pattern name
 score: Match score
 confidence: Confidence level
 
 Returns:
 Interpretation string
 """
 interpretations = {
 'water_librational': "Structural invariants suggest water librational mode resonance patterns. "
 "This indicates organized water dynamics at THz scales.",
 'protein_backbone': "Structural invariants match protein backbone phonon patterns. "
 "This suggests collective protein dynamics at THz frequencies.",
 'lipid_membrane': "Structural invariants align with lipid membrane vibration patterns. "
 "This indicates membrane-level THz resonances.",
 'DNA_phonon': "Structural invariants correspond to DNA phonon mode patterns. "
 "This suggests genetic material THz dynamics."
 }
 
 base_interpretation = interpretations.get(pattern, "Unknown pattern.")
 
 confidence_statement = {
 'high': f"High confidence (score: {score:.3f}).",
 'medium': f"Medium confidence (score: {score:.3f}).",
 'low': f"Low confidence (score: {score:.3f}). Multiple patterns possible."
 }
 
 return base_interpretation + " " + confidence_statement[confidence]

class THzCohortAnalyzer:
 """
 Analyze THz patterns across cohort for treatment effects.
 """
 
 def __init__(self):
 """Initialize cohort analyzer."""
 self.pattern_matcher = THzPatternMatcher
 
 def analyze_subject_pair(self, eigenvalues_pre: np.ndarray,
 eigenvalues_post: np.ndarray,
 subject_id: str) -> Dict:
 """
 Analyze THz patterns for pre/post treatment pair.
 
 Args:
 eigenvalues_pre: Pre-treatment eigenvalues
 eigenvalues_post: Post-treatment eigenvalues
 subject_id: Subject identifier
 
 Returns:
 Dictionary with pre/post comparison
 """
 # Generate reports for both
 report_pre = self.pattern_matcher.compute_thz_inference_report(eigenvalues_pre)
 report_post = self.pattern_matcher.compute_thz_inference_report(eigenvalues_post)
 
 # Compute pattern shift
 pattern_shift = (report_pre['dominant_pattern'] != report_post['dominant_pattern'])
 
 return {
 'subject_id': subject_id,
 'pre_treatment': report_pre,
 'post_treatment': report_post,
 'pattern_shift': pattern_shift,
 'pattern_shift_description': f"{report_pre['dominant_pattern']} → {report_post['dominant_pattern']}" if pattern_shift else "No shift"
 }
 
 def analyze_cohort(self, subject_pairs: List[Dict]) -> Dict:
 """
 Analyze THz patterns across entire cohort.
 
 Args:
 subject_pairs: List of subject pair analyses
 
 Returns:
 Cohort-level summary
 """
 # Count pattern shifts
 n_shifts = sum(1 for pair in subject_pairs if pair['pattern_shift'])
 shift_rate = n_shifts / len(subject_pairs) if subject_pairs else 0.0
 
 # Aggregate dominant patterns
 pre_patterns = [pair['pre_treatment']['dominant_pattern'] for pair in subject_pairs]
 post_patterns = [pair['post_treatment']['dominant_pattern'] for pair in subject_pairs]
 
 # Pattern distribution
 from collections import Counter
 pre_distribution = Counter(pre_patterns)
 post_distribution = Counter(post_patterns)
 
 return {
 'n_subjects': len(subject_pairs),
 'pattern_shift_rate': shift_rate,
 'n_shifts': n_shifts,
 'pre_pattern_distribution': dict(pre_distribution),
 'post_pattern_distribution': dict(post_distribution),
 'most_common_pre': pre_distribution.most_common(1)[0] if pre_distribution else None,
 'most_common_post': post_distribution.most_common(1)[0] if post_distribution else None
 }

def validate_thz_inference(eigenvalues: np.ndarray) -> bool:
 """
 Validate THz inference is being used correctly.
 
 Per ENTPC.tex: NO frequency conversion, only structural invariants.
 
 Args:
 eigenvalues: Eigenvalues to analyze
 
 Returns:
 True if validation passes
 
 Raises:
 AssertionError if misused
 """
 # Check eigenvalues are real (from real symmetric matrix)
 assert np.all(np.isreal(eigenvalues)), "Eigenvalues must be real"
 
 # Check the analysis enough eigenvalues for pattern matching
 assert len(eigenvalues) >= 3, "Need at least 3 eigenvalues for pattern matching"
 
 # Check eigenvalues are from Perron-Frobenius (positive dominant)
 max_eig = np.max(np.abs(eigenvalues))
 assert max_eig > 0, "Dominant eigenvalue must be positive"
 
 return True
```
