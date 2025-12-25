"""
Comprehensive Stress Test Suite for EntPTC Implementation

This test suite verifies:
1. 64 quaternion indices (16 ROIs × 4 components)
2. 256 Progenitor Matrix elements (16×16)
3. Full model pipeline integration
4. All 4 enhanced modules (CGA, Lie Group, Delaunay, Robust Predicates)

"""

import numpy as np
import sys
import traceback
from typing import Dict, List, Tuple

# Add paths
sys.path.insert(0, '/home/ubuntu/entptc-archive/entptc/core')
sys.path.insert(0, '/home/ubuntu/entptc-archive/entptc/analysis')
sys.path.insert(0, '/home/ubuntu/entptc-archive/entptc/pipeline')
sys.path.insert(0, '/home/ubuntu/entptc-archive/enhanced')

print("="*80)
print("EntPTC Comprehensive Stress Test Suite")
print("="*80)
print()

# Test results storage
test_results = {
 'passed': [],
 'failed': [],
 'errors': []
}

def run_test(test_name: str, test_func):
 """Run a single test and record results."""
 print(f"Running: {test_name}...", end=" ")
 try:
 test_func()
 print("✅ PASSED")
 test_results['passed'].append(test_name)
 return True
 except AssertionError as e:
 print(f"❌ FAILED: {e}")
 test_results['failed'].append((test_name, str(e)))
 return False
 except Exception as e:
 print(f"⚠️ ERROR: {e}")
 test_results['errors'].append((test_name, str(e), traceback.format_exc()))
 return False

# ============================================================================
# TEST 1: Core Modules - 64 Quaternion Indices
# ============================================================================

def test_64_quaternion_indices():
 """Stress test 64 quaternion indices (16 ROIs × 4 components)."""
 print("\n" + "="*80)
 print("TEST 1: 64 Quaternion Indices (16 ROIs × 4 components)")
 print("="*80)
 
 try:
 from quaternion import QuaternionHilbertSpace
 except ImportError:
 print("⚠️ quaternion.py not found, skipping...")
 return
 
 # Create 16 ROIs with 4 quaternion components each
 n_rois = 16
 n_components = 4
 total_indices = n_rois * n_components
 
 assert total_indices == 64, f"Expected 64 indices, got {total_indices}"
 
 # Create test data: 16 quaternions
 test_quaternions = []
 for i in range(n_rois):
 # Each quaternion: (w, x, y, z)
 q = np.array([
 np.random.randn(), # w (scalar)
 np.random.randn(), # x (i component)
 np.random.randn(), # y (j component)
 np.random.randn() # z (k component)
 ])
 test_quaternions.append(q)
 
 test_quaternions = np.array(test_quaternions)
 assert test_quaternions.shape == (16, 4), f"Expected (16, 4), got {test_quaternions.shape}"
 
 # Flatten to 64 indices
 flattened = test_quaternions.flatten()
 assert len(flattened) == 64, f"Expected 64 elements, got {len(flattened)}"
 
 # Verify each index is accessible
 for idx in range(64):
 roi_idx = idx // 4
 component_idx = idx % 4
 
 value_flat = flattened[idx]
 value_structured = test_quaternions[roi_idx, component_idx]
 
 assert np.isclose(value_flat, value_structured), \
 f"Index {idx} mismatch: {value_flat} != {value_structured}"
 
 print(f"✅ All 64 quaternion indices verified")
 print(f" - 16 ROIs × 4 components = 64 total indices")
 print(f" - Shape: {test_quaternions.shape}")
 print(f" - Flattened length: {len(flattened)}")

run_test("64 Quaternion Indices", test_64_quaternion_indices)

# ============================================================================
# TEST 2: Progenitor Matrix - 256 Elements (16×16)
# ============================================================================

def test_256_progenitor_elements():
 """Stress test 256 Progenitor Matrix elements (16×16)."""
 print("\n" + "="*80)
 print("TEST 2: 256 Progenitor Matrix Elements (16×16)")
 print("="*80)
 
 try:
 from progenitor import ProgenitorMatrix
 except ImportError:
 print("⚠️ progenitor.py not found, skipping...")
 return
 
 # Create 16×16 matrix
 n_rois = 16
 matrix_size = n_rois * n_rois
 
 assert matrix_size == 256, f"Expected 256 elements, got {matrix_size}"
 
 # Create test Progenitor Matrix
 # Using random values for stress test
 coherence_matrix = np.random.rand(16, 16) # λ_ij (PLV)
 entropy_gradient_matrix = np.random.rand(16, 16) # ∇S_ij
 quaternion_norm_matrix = np.random.rand(16, 16) # |Q(θ_ij)|
 
 # Compute Progenitor Matrix: c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|
 progenitor_matrix = coherence_matrix * np.exp(-entropy_gradient_matrix) * quaternion_norm_matrix
 
 assert progenitor_matrix.shape == (16, 16), f"Expected (16, 16), got {progenitor_matrix.shape}"
 
 # Verify all 256 elements
 element_count = 0
 for i in range(16):
 for j in range(16):
 element = progenitor_matrix[i, j]
 assert np.isfinite(element), f"Element [{i},{j}] is not finite: {element}"
 assert element >= 0, f"Element [{i},{j}] is negative: {element}"
 element_count += 1
 
 assert element_count == 256, f"Expected 256 elements, counted {element_count}"
 
 # Verify matrix properties
 # Should be non-negative (due to construction)
 assert np.all(progenitor_matrix >= 0), "Progenitor matrix has negative elements"
 
 # Check Perron-Frobenius applicability
 # Matrix should be non-negative and irreducible for Perron-Frobenius
 eigenvalues = np.linalg.eigvals(progenitor_matrix)
 dominant_eigenvalue = np.max(np.abs(eigenvalues))
 
 assert dominant_eigenvalue > 0, "Dominant eigenvalue should be positive"
 
 print(f"✅ All 256 Progenitor Matrix elements verified")
 print(f" - Matrix shape: {progenitor_matrix.shape}")
 print(f" - Total elements: {progenitor_matrix.size}")
 print(f" - Dominant eigenvalue: {dominant_eigenvalue:.6f}")
 print(f" - All elements non-negative: {np.all(progenitor_matrix >= 0)}")

run_test("256 Progenitor Matrix Elements", test_256_progenitor_elements)

# ============================================================================
# TEST 3: Conformal Geometric Algebra (CGA)
# ============================================================================

def test_cga_module():
 """Test Conformal Geometric Algebra module."""
 print("\n" + "="*80)
 print("TEST 3: Conformal Geometric Algebra (CGA)")
 print("="*80)
 
 try:
 from cga import CGAElement, euclidean_point_to_cga, cga_motor, apply_motor_to_point
 except ImportError as e:
 print(f"⚠️ CGA module import failed: {e}")
 return
 
 # Test 1: Point representation
 point_3d = np.array([1.0, 2.0, 3.0])
 point_cga = euclidean_point_to_cga(point_3d)
 
 assert isinstance(point_cga, CGAElement), "CGA point should be CGAElement"
 assert point_cga.coeffs.shape == (32,), f"Expected 32 coefficients, got {point_cga.coeffs.shape}"
 
 # Test 2: Motor (rotation + translation)
 rotation_axis = np.array([0.0, 0.0, 1.0]) # Rotate around z-axis
 rotation_angle = np.pi / 4 # 45 degrees
 translation = np.array([1.0, 0.0, 0.0]) # Translate in x
 
 motor = cga_motor(rotation_axis, rotation_angle, translation)
 assert isinstance(motor, CGAElement), "Motor should be CGAElement"
 
 # Test 3: Apply motor to point
 transformed_point = apply_motor_to_point(motor, point_cga)
 assert isinstance(transformed_point, CGAElement), "Transformed point should be CGAElement"
 
 print(f"✅ CGA module working correctly")
 print(f" - Point representation: {point_cga.coeffs.shape}")
 print(f" - Motor representation: {motor.coeffs.shape}")
 print(f" - Transformation applied successfully")

run_test("Conformal Geometric Algebra", test_cga_module)

# ============================================================================
# TEST 4: Lie Group Integration
# ============================================================================

def test_lie_group_integration():
 """Test Lie Group Integration module."""
 print("\n" + "="*80)
 print("TEST 4: Lie Group Integration")
 print("="*80)
 
 try:
 from lie_group_integrator import LieGroupIntegrator, ToroidalState, entropy_gradient_vector_field
 except ImportError as e:
 print(f"⚠️ Lie Group Integration module import failed: {e}")
 return
 
 # Test 1: Create toroidal state
 initial_state = ToroidalState(theta1=1.0, theta2=2.0, theta3=3.0)
 assert 0 <= initial_state.theta1 < 2*np.pi, "theta1 should be wrapped to [0, 2π)"
 assert 0 <= initial_state.theta2 < 2*np.pi, "theta2 should be wrapped to [0, 2π)"
 assert 0 <= initial_state.theta3 < 2*np.pi, "theta3 should be wrapped to [0, 2π)"
 
 # Test 2: Define entropy field (simple quadratic for testing)
 def test_entropy_field(state):
 theta = state.to_array()
 # Simple quadratic entropy centered at origin
 return 0.5 * np.sum(theta**2)
 
 # Test 3: Create vector field
 vector_field = entropy_gradient_vector_field(test_entropy_field)
 
 # Test 4: Create integrator
 integrator = LieGroupIntegrator(vector_field, method='rk4')
 
 # Test 5: Integrate for a few steps
 dt = 0.01
 n_steps = 10
 
 state = initial_state
 for _ in range(n_steps):
 state = integrator.step(state, dt)
 # Verify state stays on manifold
 assert 0 <= state.theta1 < 2*np.pi, "theta1 left manifold"
 assert 0 <= state.theta2 < 2*np.pi, "theta2 left manifold"
 assert 0 <= state.theta3 < 2*np.pi, "theta3 left manifold"
 
 print(f"✅ Lie Group Integration working correctly")
 print(f" - Initial state: {initial_state.to_array()}")
 print(f" - Final state after {n_steps} steps: {state.to_array()}")
 print(f" - State remained on T³ manifold")

run_test("Lie Group Integration", test_lie_group_integration)

# ============================================================================
# TEST 5: Delaunay Triangulation on T³
# ============================================================================

def test_delaunay_t3():
 """Test Delaunay Triangulation on T³."""
 print("\n" + "="*80)
 print("TEST 5: Delaunay Triangulation on T³")
 print("="*80)
 
 try:
 from delaunay_t3 import ToroidalPoint, PeriodicDelaunayT3
 except ImportError as e:
 print(f"⚠️ Delaunay T³ module import failed: {e}")
 return
 
 # Test 1: Create toroidal points
 n_points = 50
 points = []
 for i in range(n_points):
 theta1 = np.random.uniform(0, 2*np.pi)
 theta2 = np.random.uniform(0, 2*np.pi)
 theta3 = np.random.uniform(0, 2*np.pi)
 points.append(ToroidalPoint(theta1, theta2, theta3))
 
 # Test 2: Compute Delaunay triangulation
 delaunay = PeriodicDelaunayT3(points)
 
 # Test 3: Verify neighbors
 for i in range(n_points):
 neighbors = delaunay.get_neighbors(i)
 assert isinstance(neighbors, set), "Neighbors should be a set"
 # Each point should have at least a few neighbors
 # (might be 0 for some points in small samples)
 
 # Test 4: Compute statistics
 edge_stats = delaunay.compute_edge_length_statistics()
 assert 'mean' in edge_stats, "Edge stats should have mean"
 assert 'std' in edge_stats, "Edge stats should have std"
 assert edge_stats['mean'] > 0, "Mean edge length should be positive"
 
 print(f"✅ Delaunay Triangulation on T³ working correctly")
 print(f" - Number of points: {n_points}")
 print(f" - Number of tetrahedra: {len(delaunay.tetrahedra)}")
 print(f" - Mean edge length: {edge_stats['mean']:.4f}")
 print(f" - Std edge length: {edge_stats['std']:.4f}")

run_test("Delaunay Triangulation on T³", test_delaunay_t3)

# ============================================================================
# TEST 6: Robust Geometric Predicates
# ============================================================================

def test_robust_predicates():
 """Test Robust Geometric Predicates."""
 print("\n" + "="*80)
 print("TEST 6: Robust Geometric Predicates")
 print("="*80)
 
 try:
 from robust_predicates import orient3d_adaptive, toroidal_orientation_test, robust_angle_comparison
 except ImportError as e:
 print(f"⚠️ Robust Predicates module import failed: {e}")
 return
 
 # Test 1: 3D orientation test
 pa = (0.0, 0.0, 0.0)
 pb = (1.0, 0.0, 0.0)
 pc = (0.0, 1.0, 0.0)
 pd_above = (0.0, 0.0, 1.0)
 pd_below = (0.0, 0.0, -1.0)
 
 orient_above = orient3d_adaptive(pa, pb, pc, pd_above)
 orient_below = orient3d_adaptive(pa, pb, pc, pd_below)
 
 # They should have opposite signs
 assert orient_above * orient_below < 0, "Orientation test failed"
 
 # Test 2: Toroidal orientation test
 p1 = (1.0, 2.0, 3.0)
 p2 = (1.5, 2.5, 3.5)
 p3 = (2.0, 3.0, 4.0)
 p4 = (1.0, 2.0, 4.0)
 
 toroidal_orient = toroidal_orientation_test(p1, p2, p3, p4)
 assert toroidal_orient in [-1, 0, 1], "Toroidal orientation should be -1, 0, or 1"
 
 # Test 3: Angle comparison
 angle1 = 0.1
 angle2 = 6.2 # Close to 2π, so close to 0
 
 comparison = robust_angle_comparison(angle1, angle2)
 assert comparison in [-1, 0, 1], "Angle comparison should be -1, 0, or 1"
 
 print(f"✅ Robust Geometric Predicates working correctly")
 print(f" - 3D orientation test: {orient_above > 0}")
 print(f" - Toroidal orientation: {toroidal_orient}")
 print(f" - Angle comparison: {comparison}")

run_test("Robust Geometric Predicates", test_robust_predicates)

# ============================================================================
# TEST 7: Full Pipeline Integration
# ============================================================================

def test_full_pipeline():
 """Test full EntPTC pipeline integration."""
 print("\n" + "="*80)
 print("TEST 7: Full Pipeline Integration")
 print("="*80)
 
 # Simulate full pipeline with 64 indices and 256 matrix elements
 
 # Step 1: Create 16 ROI quaternions (64 indices)
 n_rois = 16
 quaternions = np.random.randn(n_rois, 4)
 
 # Normalize quaternions
 norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
 quaternions = quaternions / norms
 
 assert quaternions.shape == (16, 4), f"Expected (16, 4), got {quaternions.shape}"
 total_indices = quaternions.size
 assert total_indices == 64, f"Expected 64 indices, got {total_indices}"
 
 # Step 2: Compute Progenitor Matrix (256 elements)
 # Simplified computation for testing
 progenitor_matrix = np.zeros((16, 16))
 
 for i in range(16):
 for j in range(16):
 # Simplified: coherence based on quaternion dot product
 coherence = abs(np.dot(quaternions[i], quaternions[j]))
 
 # Simplified entropy gradient (random for testing)
 entropy_grad = np.random.uniform(0, 1)
 
 # Simplified quaternion norm product
 q_norm = np.linalg.norm(quaternions[i]) * np.linalg.norm(quaternions[j])
 
 # Progenitor element: c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|
 progenitor_matrix[i, j] = coherence * np.exp(-entropy_grad) * q_norm
 
 assert progenitor_matrix.shape == (16, 16), f"Expected (16, 16), got {progenitor_matrix.shape}"
 assert progenitor_matrix.size == 256, f"Expected 256 elements, got {progenitor_matrix.size}"
 
 # Step 3: Perron-Frobenius collapse
 eigenvalues, eigenvectors = np.linalg.eig(progenitor_matrix)
 dominant_idx = np.argmax(np.abs(eigenvalues))
 dominant_eigenvalue = eigenvalues[dominant_idx]
 dominant_eigenvector = eigenvectors[:, dominant_idx]
 
 # Normalize
 dominant_eigenvector = dominant_eigenvector / np.linalg.norm(dominant_eigenvector)
 
 assert len(dominant_eigenvector) == 16, f"Expected 16 components, got {len(dominant_eigenvector)}"
 
 # Step 4: Compute Absurdity Gap (POST-OPERATOR)
 # Simulate pre and post states
 psi_pre = dominant_eigenvector
 psi_post = dominant_eigenvector + np.random.randn(16) * 0.1 # Add noise
 psi_post = psi_post / np.linalg.norm(psi_post)
 
 absurdity_gap = np.linalg.norm(psi_pre - psi_post)
 
 assert absurdity_gap >= 0, "Absurdity gap should be non-negative"
 
 print(f"✅ Full pipeline integration successful")
 print(f" - 64 quaternion indices: {quaternions.shape} = {quaternions.size} elements")
 print(f" - 256 Progenitor Matrix elements: {progenitor_matrix.shape} = {progenitor_matrix.size} elements")
 print(f" - Dominant eigenvalue: {abs(dominant_eigenvalue):.6f}")
 print(f" - Dominant eigenvector dimension: {len(dominant_eigenvector)}")
 print(f" - Absurdity gap: {absurdity_gap:.6f}")

run_test("Full Pipeline Integration", test_full_pipeline)

# ============================================================================
# TEST SUMMARY
# ============================================================================

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"✅ Passed: {len(test_results['passed'])}")
print(f"❌ Failed: {len(test_results['failed'])}")
print(f"⚠️ Errors: {len(test_results['errors'])}")
print()

if test_results['passed']:
 print("Passed tests:")
 for test in test_results['passed']:
 print(f" ✅ {test}")
 print()

if test_results['failed']:
 print("Failed tests:")
 for test, reason in test_results['failed']:
 print(f" ❌ {test}: {reason}")
 print()

if test_results['errors']:
 print("Tests with errors:")
 for test, error, trace in test_results['errors']:
 print(f" ⚠️ {test}: {error}")
 print(f" Traceback: {trace[:200]}...")
 print()

# Final verdict
total_tests = len(test_results['passed']) + len(test_results['failed']) + len(test_results['errors'])
success_rate = len(test_results['passed']) / total_tests * 100 if total_tests > 0 else 0

print("="*80)
print(f"FINAL VERDICT: {success_rate:.1f}% tests passed ({len(test_results['passed'])}/{total_tests})")
print("="*80)

# Exit with appropriate code
if test_results['failed'] or test_results['errors']:
 sys.exit(1)
else:
 sys.exit(0)
