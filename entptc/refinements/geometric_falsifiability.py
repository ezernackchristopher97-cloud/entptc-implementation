"""
Geometric Falsifiability Criteria for EntPTC Model

Implements PROPER falsifiability testing based on:
- Trajectory geometry (curvature, torsion, path length)
- Attractor structure (basin topology, stability)
- Entropy flow and localization
- Winding numbers and topological invariants

NOT based on:
- Statistical mean-difference tests
- Classical IID hypothesis testing
- Variance comparisons

The EntPTC model is a geometric and dynamical state-space framework.
Falsifiability must be evaluated via trajectory and attractor properties,
not statistical discrimination.

Reference: ENTPC.tex lines 649-734 (Toroidal Manifold and Regimes)

"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.stats import entropy as shannon_entropy
import warnings
warnings.filterwarnings('ignore')

from .toroidal_dynamics import ToroidalState, ToroidalManifold, GridCellDynamics

@dataclass
class GeometricSignature:
 """
 Geometric signature of a trajectory on T³.
 
 Contains all geometric and dynamical properties used for falsifiability.
 """
 # Trajectory properties
 path_length: float
 mean_curvature: float
 max_curvature: float
 curvature_variance: float
 
 # Winding properties
 winding_numbers: np.ndarray # (3,)
 total_winding: float
 
 # Attractor properties
 n_attractors: int
 attractor_basin_sizes: List[float]
 attractor_stability: List[float]
 
 # Entropy properties
 entropy_mean: float
 entropy_variance: float
 entropy_flow: float # Total entropy change
 entropy_production_rate: float
 
 # Localization properties
 spatial_localization: float # Inverse of trajectory spread
 temporal_persistence: float # Time spent near attractors
 
 # Regime properties
 regime_occupancy: Dict[str, float] # Fraction of time in each regime
 regime_transitions: int # Number of regime switches
 
 def __repr__(self):
 return (f"GeometricSignature(\n"
 f" path_length={self.path_length:.3f},\n"
 f" mean_curvature={self.mean_curvature:.3f},\n"
 f" winding={self.winding_numbers},\n"
 f" n_attractors={self.n_attractors},\n"
 f" entropy_flow={self.entropy_flow:.3f}\n"
 f")")

class GeometricFalsifiability:
 """
 Geometric falsifiability testing for EntPTC model.
 
 Tests whether trajectories from different conditions show distinct
 geometric and dynamical properties on T³.
 """
 
 def __init__(self, manifold: ToroidalManifold):
 """
 Initialize geometric falsifiability tester.
 
 Args:
 manifold: Toroidal manifold T³
 """
 self.manifold = manifold
 
 def compute_geometric_signature(self,
 trajectory: List[ToroidalState],
 dynamics: GridCellDynamics) -> GeometricSignature:
 """
 Compute complete geometric signature of a trajectory.
 
 Args:
 trajectory: List of states along trajectory
 dynamics: Grid-cell dynamics object
 
 Returns:
 GeometricSignature object
 """
 # Extract phases and entropies
 phases = np.array([state.phases for state in trajectory])
 entropies = np.array([state.entropy for state in trajectory])
 times = np.array([state.time for state in trajectory])
 
 # 1. Path length
 path_length = self._compute_path_length(phases)
 
 # 2. Curvature statistics
 curvatures = dynamics.compute_trajectory_curvature(trajectory)
 mean_curvature = np.mean(curvatures)
 max_curvature = np.max(curvatures)
 curvature_variance = np.var(curvatures)
 
 # 3. Winding numbers
 winding_numbers = dynamics.compute_winding_number(trajectory)
 total_winding = np.linalg.norm(winding_numbers)
 
 # 4. Attractor analysis
 n_attractors, basin_sizes, stability = self._analyze_attractors(phases, entropies)
 
 # 5. Entropy properties
 entropy_mean = np.mean(entropies)
 entropy_variance = np.var(entropies)
 entropy_flow = entropies[-1] - entropies[0]
 
 # Entropy production rate (change per unit time)
 if len(times) > 1:
 entropy_production_rate = entropy_flow / (times[-1] - times[0])
 else:
 entropy_production_rate = 0.0
 
 # 6. Localization
 spatial_localization = self._compute_spatial_localization(phases)
 temporal_persistence = self._compute_temporal_persistence(phases, entropies)
 
 # 7. Regime analysis
 regime_occupancy, regime_transitions = self._analyze_regimes(trajectory)
 
 return GeometricSignature(
 path_length=path_length,
 mean_curvature=mean_curvature,
 max_curvature=max_curvature,
 curvature_variance=curvature_variance,
 winding_numbers=winding_numbers,
 total_winding=total_winding,
 n_attractors=n_attractors,
 attractor_basin_sizes=basin_sizes,
 attractor_stability=stability,
 entropy_mean=entropy_mean,
 entropy_variance=entropy_variance,
 entropy_flow=entropy_flow,
 entropy_production_rate=entropy_production_rate,
 spatial_localization=spatial_localization,
 temporal_persistence=temporal_persistence,
 regime_occupancy=regime_occupancy,
 regime_transitions=regime_transitions
 )
 
 def _compute_path_length(self, phases: np.ndarray) -> float:
 """Compute total path length on T³."""
 path_length = 0.0
 for i in range(1, len(phases)):
 dist = self.manifold.toroidal_distance(phases[i-1], phases[i])
 path_length += dist
 return path_length
 
 def _analyze_attractors(self, 
 phases: np.ndarray, 
 entropies: np.ndarray) -> Tuple[int, List[float], List[float]]:
 """
 Analyze attractor structure.
 
 Returns:
 n_attractors: Number of distinct attractors visited
 basin_sizes: Relative size of each basin
 stability: Stability of each attractor
 """
 # Find local minima in entropy (attractors)
 attractors = []
 for i in range(1, len(entropies) - 1):
 if entropies[i] < entropies[i-1] and entropies[i] < entropies[i+1]:
 attractors.append(i)
 
 n_attractors = len(attractors)
 
 if n_attractors == 0:
 return 0, [], []
 
 # Cluster trajectory points by nearest attractor
 attractor_phases = phases[attractors]
 
 basin_assignments = []
 for phase in phases:
 distances = [self.manifold.toroidal_distance(phase, att_phase) 
 for att_phase in attractor_phases]
 basin_assignments.append(np.argmin(distances))
 
 # Compute basin sizes
 basin_sizes = []
 for i in range(n_attractors):
 basin_size = np.sum(np.array(basin_assignments) == i) / len(phases)
 basin_sizes.append(basin_size)
 
 # Compute stability (inverse of local entropy variance)
 stability = []
 for i, att_idx in enumerate(attractors):
 # Get points in this basin
 basin_points = [j for j, b in enumerate(basin_assignments) if b == i]
 if len(basin_points) > 1:
 basin_entropy_var = np.var(entropies[basin_points])
 stability.append(1.0 / (basin_entropy_var + 1e-6))
 else:
 stability.append(0.0)
 
 return n_attractors, basin_sizes, stability
 
 def _compute_spatial_localization(self, phases: np.ndarray) -> float:
 """
 Compute spatial localization (inverse of spread).
 
 Higher values = more localized trajectory.
 """
 # Compute mean position
 mean_phase = np.mean(phases, axis=0)
 
 # Compute spread (mean distance from center)
 spread = np.mean([
 self.manifold.toroidal_distance(phase, mean_phase)
 for phase in phases
 ])
 
 # Localization is inverse of spread
 localization = 1.0 / (spread + 1e-6)
 
 return localization
 
 def _compute_temporal_persistence(self, 
 phases: np.ndarray, 
 entropies: np.ndarray) -> float:
 """
 Compute temporal persistence (time spent near attractors).
 
 Higher values = more time in stable states.
 """
 # Define "near attractor" as entropy below median
 median_entropy = np.median(entropies)
 near_attractor = entropies < median_entropy
 
 persistence = np.sum(near_attractor) / len(entropies)
 
 return persistence
 
 def _analyze_regimes(self, trajectory: List[ToroidalState]) -> Tuple[Dict[str, float], int]:
 """
 Analyze regime occupancy and transitions.
 
 Returns:
 regime_occupancy: Dict mapping regime name to fraction of time
 regime_transitions: Number of regime switches
 """
 # Extract entropies
 entropies = np.array([state.entropy for state in trajectory])
 
 # Classify regimes based on entropy
 # (This is a simplified classification; should be based on spectral gap in full model)
 regimes = []
 for entropy in entropies:
 if entropy < -1.0:
 regimes.append('Regime_I') # Local stabilized
 elif entropy < 0.0:
 regimes.append('Regime_II') # Transitional
 else:
 regimes.append('Regime_III') # Global experience
 
 # Compute occupancy
 regime_occupancy = {
 'Regime_I': np.sum(np.array(regimes) == 'Regime_I') / len(regimes),
 'Regime_II': np.sum(np.array(regimes) == 'Regime_II') / len(regimes),
 'Regime_III': np.sum(np.array(regimes) == 'Regime_III') / len(regimes)
 }
 
 # Count transitions
 regime_transitions = 0
 for i in range(1, len(regimes)):
 if regimes[i] != regimes[i-1]:
 regime_transitions += 1
 
 return regime_occupancy, regime_transitions
 
 def test_trajectory_divergence(self,
 signature1: GeometricSignature,
 signature2: GeometricSignature) -> Dict:
 """
 Test 1: Trajectory Divergence
 
 Hypothesis: Different conditions should produce trajectories with
 different geometric properties (path length, curvature).
 
 Args:
 signature1: Geometric signature of condition 1
 signature2: Geometric signature of condition 2
 
 Returns:
 Test results dictionary
 """
 # Compute relative differences
 path_length_diff = abs(signature1.path_length - signature2.path_length) / \
 (signature1.path_length + signature2.path_length + 1e-12)
 
 curvature_diff = abs(signature1.mean_curvature - signature2.mean_curvature) / \
 (signature1.mean_curvature + signature2.mean_curvature + 1e-12)
 
 # Combined divergence metric
 divergence = (path_length_diff + curvature_diff) / 2
 
 # Threshold: 10% difference is significant
 threshold = 0.1
 falsified = divergence < threshold
 
 return {
 'test_name': 'Trajectory Divergence',
 'divergence': divergence,
 'path_length_diff': path_length_diff,
 'curvature_diff': curvature_diff,
 'threshold': threshold,
 'falsified': falsified,
 'interpretation': f'Model falsified if divergence < {threshold}'
 }
 
 def test_attractor_topology(self,
 signature1: GeometricSignature,
 signature2: GeometricSignature) -> Dict:
 """
 Test 2: Attractor Topology
 
 Hypothesis: Different conditions should have different attractor structures.
 
 Args:
 signature1: Geometric signature of condition 1
 signature2: Geometric signature of condition 2
 
 Returns:
 Test results dictionary
 """
 # Compare number of attractors
 attractor_count_diff = abs(signature1.n_attractors - signature2.n_attractors)
 
 # Compare basin structure (if both have attractors)
 if signature1.n_attractors > 0 and signature2.n_attractors > 0:
 # Compute Jensen-Shannon divergence of basin size distributions
 # Pad to same length
 max_len = max(len(signature1.attractor_basin_sizes), 
 len(signature2.attractor_basin_sizes))
 
 basins1 = list(signature1.attractor_basin_sizes) + [0] * (max_len - len(signature1.attractor_basin_sizes))
 basins2 = list(signature2.attractor_basin_sizes) + [0] * (max_len - len(signature2.attractor_basin_sizes))
 
 # Normalize
 basins1 = np.array(basins1) / (np.sum(basins1) + 1e-12)
 basins2 = np.array(basins2) / (np.sum(basins2) + 1e-12)
 
 # Add small constant to avoid log(0)
 basins1 = basins1 + 1e-12
 basins2 = basins2 + 1e-12
 
 # Jensen-Shannon divergence
 from scipy.spatial.distance import jensenshannon
 basin_divergence = jensenshannon(basins1, basins2)
 else:
 basin_divergence = 0.0
 
 # Combined topology difference
 topology_diff = (attractor_count_diff / 5.0) + basin_divergence
 
 # Threshold: 0.2 difference is significant
 threshold = 0.2
 falsified = topology_diff < threshold
 
 return {
 'test_name': 'Attractor Topology',
 'topology_diff': topology_diff,
 'attractor_count_diff': attractor_count_diff,
 'basin_divergence': basin_divergence,
 'threshold': threshold,
 'falsified': falsified,
 'interpretation': f'Model falsified if topology_diff < {threshold}'
 }
 
 def test_winding_structure(self,
 signature1: GeometricSignature,
 signature2: GeometricSignature) -> Dict:
 """
 Test 3: Winding Structure
 
 Hypothesis: Different conditions should show different winding patterns.
 
 Args:
 signature1: Geometric signature of condition 1
 signature2: Geometric signature of condition 2
 
 Returns:
 Test results dictionary
 """
 # Compute winding difference
 winding_diff = np.linalg.norm(signature1.winding_numbers - signature2.winding_numbers)
 
 # Normalize by total winding
 total_winding = signature1.total_winding + signature2.total_winding + 1e-12
 normalized_diff = winding_diff / total_winding
 
 # Threshold: 0.15 difference is significant
 threshold = 0.15
 falsified = normalized_diff < threshold
 
 return {
 'test_name': 'Winding Structure',
 'winding_diff': winding_diff,
 'normalized_diff': normalized_diff,
 'threshold': threshold,
 'falsified': falsified,
 'interpretation': f'Model falsified if normalized_diff < {threshold}'
 }
 
 def test_entropy_flow(self,
 signature1: GeometricSignature,
 signature2: GeometricSignature) -> Dict:
 """
 Test 4: Entropy Flow
 
 Hypothesis: Different conditions should show different entropy dynamics.
 
 Args:
 signature1: Geometric signature of condition 1
 signature2: Geometric signature of condition 2
 
 Returns:
 Test results dictionary
 """
 # Compare entropy production rates
 rate_diff = abs(signature1.entropy_production_rate - signature2.entropy_production_rate)
 
 # Compare entropy flow
 flow_diff = abs(signature1.entropy_flow - signature2.entropy_flow)
 
 # Combined entropy difference
 entropy_diff = (rate_diff + flow_diff) / 2
 
 # Threshold: 0.3 difference is significant
 threshold = 0.3
 falsified = entropy_diff < threshold
 
 return {
 'test_name': 'Entropy Flow',
 'entropy_diff': entropy_diff,
 'rate_diff': rate_diff,
 'flow_diff': flow_diff,
 'threshold': threshold,
 'falsified': falsified,
 'interpretation': f'Model falsified if entropy_diff < {threshold}'
 }
 
 def test_regime_dynamics(self,
 signature1: GeometricSignature,
 signature2: GeometricSignature) -> Dict:
 """
 Test 5: Regime Dynamics
 
 Hypothesis: Different conditions should show different regime occupancy
 and transition patterns.
 
 Args:
 signature1: Geometric signature of condition 1
 signature2: Geometric signature of condition 2
 
 Returns:
 Test results dictionary
 """
 # Compare regime occupancy distributions
 occ1 = np.array([signature1.regime_occupancy['Regime_I'],
 signature1.regime_occupancy['Regime_II'],
 signature1.regime_occupancy['Regime_III']])
 
 occ2 = np.array([signature2.regime_occupancy['Regime_I'],
 signature2.regime_occupancy['Regime_II'],
 signature2.regime_occupancy['Regime_III']])
 
 # Jensen-Shannon divergence
 from scipy.spatial.distance import jensenshannon
 occupancy_divergence = jensenshannon(occ1 + 1e-12, occ2 + 1e-12)
 
 # Compare transition rates
 transition_diff = abs(signature1.regime_transitions - signature2.regime_transitions) / \
 (signature1.regime_transitions + signature2.regime_transitions + 1e-12)
 
 # Combined regime difference
 regime_diff = (occupancy_divergence + transition_diff) / 2
 
 # Threshold: 0.2 difference is significant
 threshold = 0.2
 falsified = regime_diff < threshold
 
 return {
 'test_name': 'Regime Dynamics',
 'regime_diff': regime_diff,
 'occupancy_divergence': occupancy_divergence,
 'transition_diff': transition_diff,
 'threshold': threshold,
 'falsified': falsified,
 'interpretation': f'Model falsified if regime_diff < {threshold}'
 }
 
 def run_all_geometric_tests(self,
 signature1: GeometricSignature,
 signature2: GeometricSignature) -> Dict:
 """
 Run all geometric falsifiability tests.
 
 Args:
 signature1: Geometric signature of condition 1
 signature2: Geometric signature of condition 2
 
 Returns:
 Dictionary with all test results
 """
 tests = [
 self.test_trajectory_divergence,
 self.test_attractor_topology,
 self.test_winding_structure,
 self.test_entropy_flow,
 self.test_regime_dynamics
 ]
 
 results = {}
 for test_func in tests:
 result = test_func(signature1, signature2)
 results[result['test_name']] = result
 
 # Overall assessment
 falsified_count = sum(1 for r in results.values() if r['falsified'])
 total_tests = len(results)
 
 results['overall'] = {
 'falsified_count': falsified_count,
 'total_tests': total_tests,
 'falsification_rate': falsified_count / total_tests,
 'conclusion': 'FALSIFIED' if falsified_count >= total_tests * 0.6 else 'NOT FALSIFIED'
 }
 
 return results

if __name__ == '__main__':
 print("Geometric Falsifiability Module")
 print("=" * 80)
 print("Implements PROPER falsifiability testing:")
 print(" - Trajectory geometry (path length, curvature)")
 print(" - Attractor topology (basin structure, stability)")
 print(" - Winding structure (topological invariants)")
 print(" - Entropy flow (production rate, localization)")
 print(" - Regime dynamics (occupancy, transitions)")
 print()
 print("NOT based on statistical mean-difference tests!")
 print("=" * 80)
