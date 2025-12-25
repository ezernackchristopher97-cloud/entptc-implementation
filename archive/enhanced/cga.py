"""
Conformal Geometric Algebra (CGA) Cl(4,1) Implementation for EntPTC

Implements the 5D Conformal Geometric Algebra Cl(4,1), which provides
a unified framework for handling rotations, translations, scaling, and spherical
geometries in a single algebraic structure.

References:
- Ghali (2008), Chapter 18: Conformal Geometry
- Bayro Corrochano (2012), Chapter 5: Conformal Geometric Algebra for Perception-Action
- Dorst, Fontijne, Mann (2007): Geometric Algebra for Computer Science

Per ENTPC.tex: This extends the Cl(3,0) implementation to CGA for enhanced
geometric operations on the toroidal manifold T³.

"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

@dataclass
class CGAElement:
 """
 Element of Conformal Geometric Algebra Cl(4,1).
 
 CGA is a 32-dimensional algebra (2^5) with signature (4,1).
 It extends 3D Euclidean space with two null vectors: e_∞ (point at infinity)
 and e_0 (origin), enabling unified representation of points, lines, planes,
 circles, and spheres.
 
 Basis vectors:
 - e1, e2, e3: Standard Euclidean basis
 - e_plus = e4: Positive null vector (e_plus² = +1)
 - e_minus = e5: Negative null vector (e_minus² = -1)
 - e_∞ = e_minus - e_plus: Point at infinity
 - e_0 = 0.5*(e_minus + e_plus): Origin
 
 For efficiency, storing only the most commonly used components.
 Full 32-dimensional representation can be reconstructed if needed.
 """
 
 # Grade 0: Scalar
 scalar: float = 0.0
 
 # Grade 1: Vectors (5 basis vectors)
 e1: float = 0.0
 e2: float = 0.0
 e3: float = 0.0
 e_plus: float = 0.0
 e_minus: float = 0.0
 
 # Grade 2: Bivectors (10 basis bivectors)
 e12: float = 0.0
 e13: float = 0.0
 e23: float = 0.0
 e1_plus: float = 0.0
 e1_minus: float = 0.0
 e2_plus: float = 0.0
 e2_minus: float = 0.0
 e3_plus: float = 0.0
 e3_minus: float = 0.0
 e_plus_minus: float = 0.0
 
 # Grade 3: Trivectors (10 basis trivectors, storing key ones)
 e123: float = 0.0
 e12_plus: float = 0.0
 e12_minus: float = 0.0
 e13_plus: float = 0.0
 e13_minus: float = 0.0
 e23_plus: float = 0.0
 e23_minus: float = 0.0
 
 # Grade 4: Quadvectors (5 basis quadvectors, storing key ones)
 e123_plus: float = 0.0
 e123_minus: float = 0.0
 
 # Grade 5: Pseudoscalar
 e12345: float = 0.0
 
 def __post_init__(self):
 """Validate CGA element."""
 # Check for NaN or Inf
 for field_name, value in self.__dataclass_fields__.items():
 val = getattr(self, field_name)
 if not np.isfinite(val):
 warnings.warn(f"Non-finite value in CGA element field {field_name}: {val}")
 
 @property
 def coeffs(self) -> np.ndarray:
 """Return all coefficients as a 32-dimensional numpy array (full CGA basis)."""
 # Full 32-dimensional CGA Cl(4,1) representation
 # Store 26 explicitly, pad with zeros for remaining 6 components
 return np.array([
 self.scalar, # Grade 0: 1 component
 self.e1, self.e2, self.e3, self.e_plus, self.e_minus, # Grade 1: 5 components
 self.e12, self.e13, self.e23, self.e1_plus, self.e1_minus, # Grade 2: 10 components (5 shown)
 self.e2_plus, self.e2_minus, self.e3_plus, self.e3_minus, self.e_plus_minus, # Grade 2: (5 more)
 self.e123, self.e12_plus, self.e12_minus, self.e13_plus, self.e13_minus, # Grade 3: 10 components (5 shown)
 self.e23_plus, self.e23_minus, 0.0, 0.0, 0.0, # Grade 3: (2 shown + 3 implicit zeros)
 self.e123_plus, self.e123_minus, 0.0, 0.0, 0.0, # Grade 4: 5 components (2 shown + 3 zeros)
 self.e12345 # Grade 5: 1 component
 ])
 
 def __add__(self, other: 'CGAElement') -> 'CGAElement':
 """Addition of CGA elements."""
 return CGAElement(
 scalar=self.scalar + other.scalar,
 e1=self.e1 + other.e1,
 e2=self.e2 + other.e2,
 e3=self.e3 + other.e3,
 e_plus=self.e_plus + other.e_plus,
 e_minus=self.e_minus + other.e_minus,
 e12=self.e12 + other.e12,
 e13=self.e13 + other.e13,
 e23=self.e23 + other.e23,
 e1_plus=self.e1_plus + other.e1_plus,
 e1_minus=self.e1_minus + other.e1_minus,
 e2_plus=self.e2_plus + other.e2_plus,
 e2_minus=self.e2_minus + other.e2_minus,
 e3_plus=self.e3_plus + other.e3_plus,
 e3_minus=self.e3_minus + other.e3_minus,
 e_plus_minus=self.e_plus_minus + other.e_plus_minus,
 e123=self.e123 + other.e123,
 e12_plus=self.e12_plus + other.e12_plus,
 e12_minus=self.e12_minus + other.e12_minus,
 e13_plus=self.e13_plus + other.e13_plus,
 e13_minus=self.e13_minus + other.e13_minus,
 e23_plus=self.e23_plus + other.e23_plus,
 e23_minus=self.e23_minus + other.e23_minus,
 e123_plus=self.e123_plus + other.e123_plus,
 e123_minus=self.e123_minus + other.e123_minus,
 e12345=self.e12345 + other.e12345
 )
 
 def __sub__(self, other: 'CGAElement') -> 'CGAElement':
 """Subtraction of CGA elements."""
 return CGAElement(
 scalar=self.scalar - other.scalar,
 e1=self.e1 - other.e1,
 e2=self.e2 - other.e2,
 e3=self.e3 - other.e3,
 e_plus=self.e_plus - other.e_plus,
 e_minus=self.e_minus - other.e_minus,
 e12=self.e12 - other.e12,
 e13=self.e13 - other.e13,
 e23=self.e23 - other.e23,
 e1_plus=self.e1_plus - other.e1_plus,
 e1_minus=self.e1_minus - other.e1_minus,
 e2_plus=self.e2_plus - other.e2_plus,
 e2_minus=self.e2_minus - other.e2_minus,
 e3_plus=self.e3_plus - other.e3_plus,
 e3_minus=self.e3_minus - other.e3_minus,
 e_plus_minus=self.e_plus_minus - other.e_plus_minus,
 e123=self.e123 - other.e123,
 e12_plus=self.e12_plus - other.e12_plus,
 e12_minus=self.e12_minus - other.e12_minus,
 e13_plus=self.e13_plus - other.e13_plus,
 e13_minus=self.e13_minus - other.e13_minus,
 e23_plus=self.e23_plus - other.e23_plus,
 e23_minus=self.e23_minus - other.e23_minus,
 e123_plus=self.e123_plus - other.e123_plus,
 e123_minus=self.e123_minus - other.e123_minus,
 e12345=self.e12345 - other.e12345
 )
 
 def __mul__(self, scalar: float) -> 'CGAElement':
 """Scalar multiplication."""
 return CGAElement(
 scalar=self.scalar * scalar,
 e1=self.e1 * scalar,
 e2=self.e2 * scalar,
 e3=self.e3 * scalar,
 e_plus=self.e_plus * scalar,
 e_minus=self.e_minus * scalar,
 e12=self.e12 * scalar,
 e13=self.e13 * scalar,
 e23=self.e23 * scalar,
 e1_plus=self.e1_plus * scalar,
 e1_minus=self.e1_minus * scalar,
 e2_plus=self.e2_plus * scalar,
 e2_minus=self.e2_minus * scalar,
 e3_plus=self.e3_plus * scalar,
 e3_minus=self.e3_minus * scalar,
 e_plus_minus=self.e_plus_minus * scalar,
 e123=self.e123 * scalar,
 e12_plus=self.e12_plus * scalar,
 e12_minus=self.e12_minus * scalar,
 e13_plus=self.e13_plus * scalar,
 e13_minus=self.e13_minus * scalar,
 e23_plus=self.e23_plus * scalar,
 e23_minus=self.e23_minus * scalar,
 e123_plus=self.e123_plus * scalar,
 e123_minus=self.e123_minus * scalar,
 e12345=self.e12345 * scalar
 )
 
 def norm_squared(self) -> float:
 """
 Compute squared norm of CGA element.
 
 For CGA with signature (4,1):
 - e1² = e2² = e3² = e_plus² = +1
 - e_minus² = -1
 """
 # Grade 0
 n2 = self.scalar**2
 
 # Grade 1 (accounting for signature)
 n2 += self.e1**2 + self.e2**2 + self.e3**2
 n2 += self.e_plus**2 - self.e_minus**2 # Signature (4,1)
 
 # Grade 2
 n2 += self.e12**2 + self.e13**2 + self.e23**2
 n2 += self.e1_plus**2 + self.e1_minus**2
 n2 += self.e2_plus**2 + self.e2_minus**2
 n2 += self.e3_plus**2 + self.e3_minus**2
 n2 += self.e_plus_minus**2
 
 # Grade 3
 n2 += self.e123**2
 n2 += self.e12_plus**2 + self.e12_minus**2
 n2 += self.e13_plus**2 + self.e13_minus**2
 n2 += self.e23_plus**2 + self.e23_minus**2
 
 # Grade 4
 n2 += self.e123_plus**2 + self.e123_minus**2
 
 # Grade 5
 n2 += self.e12345**2
 
 return n2
 
 def norm(self) -> float:
 """Compute norm of CGA element."""
 return np.sqrt(abs(self.norm_squared()))

def euclidean_point_to_cga(point) -> CGAElement:
 """
 Convert 3D Euclidean point to CGA representation.
 
 Per Dorst et al. (2007), a point P = (x, y, z) in Euclidean space
 is represented in CGA as:
 
 P_cga = P + 0.5*P²*e_∞ + e_0
 
 where:
 - P = x*e1 + y*e2 + z*e3
 - e_∞ = e_minus - e_plus
 - e_0 = 0.5*(e_minus + e_plus)
 - P² = x² + y² + z²
 
 Args:
 point: 3D Euclidean coordinates as numpy array [x, y, z] or tuple (x, y, z)
 
 Returns:
 CGA representation of the point
 """
 # Handle both array and tuple inputs
 if isinstance(point, (list, tuple)):
 point = np.array(point)
 x, y, z = point[0], point[1], point[2]
 P_squared = x**2 + y**2 + z**2
 
 # e_∞ = e_minus - e_plus
 e_inf_minus = 1.0
 e_inf_plus = -1.0
 
 # e_0 = 0.5*(e_minus + e_plus)
 e_0_minus = 0.5
 e_0_plus = 0.5
 
 # P_cga = P + 0.5*P²*e_∞ + e_0
 return CGAElement(
 e1=x,
 e2=y,
 e3=z,
 e_plus=e_0_plus + 0.5*P_squared*e_inf_plus,
 e_minus=e_0_minus + 0.5*P_squared*e_inf_minus
 )

def cga_to_euclidean_point(P_cga: CGAElement) -> Tuple[float, float, float]:
 """
 Extract 3D Euclidean coordinates from CGA point representation.
 
 Per Dorst et al. (2007), to extract Euclidean coordinates from
 a CGA point, computing:
 
 (x, y, z) = (P_cga · e_∞)^(-1) * (P_cga ∧ e_∞)
 
 Simplified: If P_cga = x*e1 + y*e2 + z*e3 + ..., then
 the Euclidean part is just (e1, e2, e3) components.
 
 Args:
 P_cga: CGA representation of a point
 
 Returns:
 (x, y, z) Euclidean coordinates
 """
 return (P_cga.e1, P_cga.e2, P_cga.e3)

def circle_on_torus_to_cga(center: Tuple[float, float, float], 
 radius: float, 
 normal: Tuple[float, float, float]) -> CGAElement:
 """
 Represent a circle on the toroidal manifold T³ in CGA.
 
 Per Dorst et al. (2007), a circle in CGA is represented as a bivector:
 
 C = P1 ∧ P2 ∧ P3
 
 where P1, P2, P3 are three points on the circle.
 
 For a circle with center c, radius r, and normal n, constructing
 three points and compute their outer product.
 
 Args:
 center: (x, y, z) center of circle
 radius: Circle radius
 normal: (nx, ny, nz) normal vector to circle plane
 
 Returns:
 CGA representation of the circle as a bivector
 """
 cx, cy, cz = center
 nx, ny, nz = normal
 
 # Normalize normal vector
 n_norm = np.sqrt(nx**2 + ny**2 + nz**2)
 nx, ny, nz = nx/n_norm, ny/n_norm, nz/n_norm
 
 # Find two orthogonal vectors in the circle plane
 # First tangent vector (perpendicular to normal)
 if abs(nx) < 0.9:
 t1x, t1y, t1z = 1.0, 0.0, 0.0
 else:
 t1x, t1y, t1z = 0.0, 1.0, 0.0
 
 # Gram-Schmidt orthogonalization
 dot = t1x*nx + t1y*ny + t1z*nz
 t1x -= dot*nx
 t1y -= dot*ny
 t1z -= dot*nz
 t1_norm = np.sqrt(t1x**2 + t1y**2 + t1z**2)
 t1x, t1y, t1z = t1x/t1_norm, t1y/t1_norm, t1z/t1_norm
 
 # Second tangent vector (cross product of normal and first tangent)
 t2x = ny*t1z - nz*t1y
 t2y = nz*t1x - nx*t1z
 t2z = nx*t1y - ny*t1x
 
 # Three points on the circle
 P1 = euclidean_point_to_cga(cx + radius*t1x, cy + radius*t1y, cz + radius*t1z)
 P2 = euclidean_point_to_cga(cx + radius*t2x, cy + radius*t2y, cz + radius*t2z)
 P3 = euclidean_point_to_cga(cx - radius*t1x, cy - radius*t1y, cz - radius*t1z)
 
 # Circle = P1 ∧ P2 ∧ P3 (outer product)
 # For this implementation, returning a simplified bivector representation
 # Full outer product computation would require complete geometric product implementation
 
 return CGAElement(
 e12=(P1.e1*P2.e2 - P1.e2*P2.e1),
 e13=(P1.e1*P3.e3 - P1.e3*P3.e1),
 e23=(P2.e2*P3.e3 - P2.e3*P3.e2),
 e1_plus=(P1.e1*P2.e_plus - P1.e_plus*P2.e1),
 e1_minus=(P1.e1*P2.e_minus - P1.e_minus*P2.e1),
 e2_plus=(P2.e2*P3.e_plus - P2.e_plus*P3.e2),
 e2_minus=(P2.e2*P3.e_minus - P2.e_minus*P3.e2),
 e3_plus=(P3.e3*P1.e_plus - P3.e_plus*P1.e3),
 e3_minus=(P3.e3*P1.e_minus - P3.e_minus*P1.e3)
 )

def cga_motor(rotation_axis: np.ndarray, 
 rotation_angle: float,
 translation: np.ndarray) -> CGAElement:
 """
 Create a CGA motor for combined rotation and translation.
 
 Per Dorst et al. (2007), a motor in CGA is an element that performs
 both rotation and translation in a single operation:
 
 M = T * R
 
 where:
 - R = e^(-θB/2) is a rotor (rotation)
 - T = e^(-t*e_∞/2) is a translator
 - θ is rotation angle
 - B is rotation bivector
 - t is translation vector
 
 Args:
 rotation_axis: (ax, ay, az) unit vector defining rotation axis
 rotation_angle: Rotation angle in radians
 translation: (tx, ty, tz) translation vector
 
 Returns:
 CGA motor element
 """
 # Handle array inputs
 if isinstance(translation, (list, tuple)):
 translation = np.array(translation)
 if isinstance(rotation_axis, (list, tuple)):
 rotation_axis = np.array(rotation_axis)
 
 tx, ty, tz = translation[0], translation[1], translation[2]
 ax, ay, az = rotation_axis[0], rotation_axis[1], rotation_axis[2]
 
 # Normalize rotation axis
 axis_norm = np.sqrt(ax**2 + ay**2 + az**2)
 if axis_norm > 0:
 ax, ay, az = ax/axis_norm, ay/axis_norm, az/axis_norm
 
 # Translator: T = 1 - 0.5*t*e_∞
 # where t*e_∞ = tx*e1*e_∞ + ty*e2*e_∞ + tz*e3*e_∞
 translator = CGAElement(
 scalar=1.0,
 e1_minus=-0.5*tx, # e1*e_∞ has e_minus component
 e1_plus=0.5*tx, # e1*e_∞ has e_plus component
 e2_minus=-0.5*ty,
 e2_plus=0.5*ty,
 e3_minus=-0.5*tz,
 e3_plus=0.5*tz
 )
 
 # Rotor: R = cos(θ/2) - sin(θ/2)*B
 # where B is bivector from rotation axis: B = ax*e23 + ay*e31 + az*e12
 half_angle = rotation_angle / 2.0
 rotor = CGAElement(
 scalar=np.cos(half_angle),
 e12=-np.sin(half_angle) * az, # e12 component
 e13=-np.sin(half_angle) * ay, # e13 component (note: e31 = -e13)
 e23=-np.sin(half_angle) * ax # e23 component
 )
 
 # Motor = Translator * Rotor (simplified composition)
 # Full geometric product would be needed for exact computation
 motor = CGAElement(
 scalar=translator.scalar * rotor.scalar,
 e1_minus=translator.e1_minus,
 e1_plus=translator.e1_plus,
 e2_minus=translator.e2_minus,
 e2_plus=translator.e2_plus,
 e3_minus=translator.e3_minus,
 e3_plus=translator.e3_plus,
 e12=rotor.e12,
 e13=rotor.e13,
 e23=rotor.e23
 )
 
 return motor

def apply_motor_to_point(motor: CGAElement, point: CGAElement) -> CGAElement:
 """
 Apply a CGA motor to a point.
 
 Per Dorst et al. (2007), to transform a point P by a motor M:
 
 P' = M * P * M^†
 
 where M^† is the reverse of M.
 
 Args:
 motor: CGA motor
 point: CGA point
 
 Returns:
 Transformed point
 """
 # Simplified transformation (full geometric product needed for exact computation)
 # Extract translation and rotation components
 
 # Translation part
 tx = -2.0 * (motor.e1_minus - motor.e1_plus)
 ty = -2.0 * (motor.e2_minus - motor.e2_plus)
 tz = -2.0 * (motor.e3_minus - motor.e3_plus)
 
 # Apply translation
 transformed = CGAElement(
 e1=point.e1 + tx,
 e2=point.e2 + ty,
 e3=point.e3 + tz,
 e_plus=point.e_plus,
 e_minus=point.e_minus
 )
 
 # Rotation part (simplified)
 # Full implementation would apply rotation using geometric product
 
 return transformed

# Integration with EntPTC toroidal manifold

def toroidal_angle_to_cga_circle(theta: float, 
 torus_major_radius: float = 2.0,
 torus_minor_radius: float = 1.0,
 circle_index: int = 0) -> CGAElement:
 """
 Convert a toroidal angle θ to a CGA circle representation.
 
 For T³ = S¹ × S¹ × S¹, each S¹ can be represented as a circle in CGA.
 This enables unified geometric operations on the toroidal manifold.
 
 Args:
 theta: Angle in [0, 2π)
 torus_major_radius: Major radius of torus
 torus_minor_radius: Minor radius of torus
 circle_index: Which S¹ component (0, 1, or 2)
 
 Returns:
 CGA circle representation
 """
 # Center of circle depends on which S¹ component
 if circle_index == 0:
 center = (torus_major_radius * np.cos(theta), 
 torus_major_radius * np.sin(theta), 
 0.0)
 normal = (np.cos(theta), np.sin(theta), 0.0)
 elif circle_index == 1:
 center = (0.0, 
 torus_major_radius * np.cos(theta), 
 torus_major_radius * np.sin(theta))
 normal = (0.0, np.cos(theta), np.sin(theta))
 else: # circle_index == 2
 center = (torus_major_radius * np.sin(theta), 
 0.0, 
 torus_major_radius * np.cos(theta))
 normal = (np.sin(theta), 0.0, np.cos(theta))
 
 return circle_on_torus_to_cga(center, torus_minor_radius, normal)

# Summary and integration notes

"""
CGA Integration with EntPTC:

1. **Unified Transformations**: CGA motors can perform rotation + translation
 in a single operation, simplifying the two-stage quaternion-Clifford process.

2. **Direct Circle Representation**: The S¹ components of T³ can be represented
 directly as circles in CGA, simplifying geodesic computations.

3. **Extensibility**: CGA naturally extends to non-Euclidean geometries, enabling
 future research directions.

4. **Backward Compatibility**: The existing Cl(3,0) implementation is a subalgebra
 of CGA, so all existing code remains valid.

Next steps for full integration:
- Implement complete geometric product for CGA
- Update entropy.py to use CGA circles for T³ representation
- Update progenitor.py to use CGA motors for transformations
- Benchmark performance vs. existing implementation
"""
