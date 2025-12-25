"""
Quaternionic Hilbert Space H_H Implementation

Reference: ENTPC.tex Definition 2.1-2.2 (lines 230-240)

From ENTPC.tex:

Definition 2.1 (Quaternionic Hilbert Space): Let H_H denote the quaternionic 
Hilbert space of dimension n over the quaternions H. For q = a + bi + cj + dk ∈ H,
the conjugate is q* = a - bi - cj - dk, and the norm is |q| = √(qq*).

Definition 2.2 (Local Quaternionic Filtering): The local filtering operator 
F_H: H_H → H_H acts on quaternionic vectors to stabilize context-dependent states.
This represents the first stage of the two-stage EntPTC process, providing local
stability before global Clifford embedding.

The quaternionic structure provides:
1. Local stability through quaternionic filtering
2. Context-dependent state stabilization
3. Foundation for transition to global Clifford algebra

Transition to Clifford: Quaternions are isomorphic to the even subalgebra of 
Cl(3,0), enabling the mapping Π: H_H → Cl(3,0) via Π(q) = e^(-B/2) where B is
a bivector derived from semantic context.
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Quaternion:
 """
 Quaternion q = a + bi + cj + dk.
 
 Per ENTPC.tex Definition 2.1:
 - Conjugate: q* = a - bi - cj - dk
 - Norm: |q| = √(qq*)
 - Multiplication: non-commutative, i²=j²=k²=ijk=-1
 """
 a: float = 0.0 # scalar (real) part
 b: float = 0.0 # i component
 c: float = 0.0 # j component
 d: float = 0.0 # k component
 
 def __post_init__(self):
 """Ensure all components are float."""
 self.a = float(self.a)
 self.b = float(self.b)
 self.c = float(self.c)
 self.d = float(self.d)
 
 def to_array(self) -> np.ndarray:
 """Convert to 4-element array [a, b, c, d]."""
 return np.array([self.a, self.b, self.c, self.d])
 
 @classmethod
 def from_array(cls, arr: np.ndarray) -> 'Quaternion':
 """Create from 4-element array."""
 assert len(arr) == 4, f"Expected 4 elements, got {len(arr)}"
 return cls(a=float(arr[0]), b=float(arr[1]), c=float(arr[2]), d=float(arr[3]))
 
 def conjugate(self) -> 'Quaternion':
 """Return quaternion conjugate q* = a - bi - cj - dk."""
 return Quaternion(a=self.a, b=-self.b, c=-self.c, d=-self.d)
 
 def norm_squared(self) -> float:
 """Compute squared norm qq*."""
 return self.a**2 + self.b**2 + self.c**2 + self.d**2
 
 def norm(self) -> float:
 """Compute norm |q| = √(qq*)."""
 return np.sqrt(self.norm_squared())
 
 def normalize(self) -> 'Quaternion':
 """Return unit quaternion q/|q|."""
 n = self.norm()
 if n < 1e-12:
 return Quaternion(a=1.0) # default to identity
 return Quaternion(
 a=self.a/n, b=self.b/n, c=self.c/n, d=self.d/n
 )
 
 def inverse(self) -> 'Quaternion':
 """Return multiplicative inverse q^(-1) = q*/|q|²."""
 n_sq = self.norm_squared()
 assert n_sq > 1e-12, "Cannot invert zero quaternion"
 conj = self.conjugate()
 return Quaternion(
 a=conj.a/n_sq, b=conj.b/n_sq, c=conj.c/n_sq, d=conj.d/n_sq
 )
 
 def __mul__(self, other: 'Quaternion') -> 'Quaternion':
 """
 Quaternion multiplication (non-commutative).
 
 Hamilton's rules:
 i² = j² = k² = ijk = -1
 ij = k, jk = i, ki = j
 ji = -k, kj = -i, ik = -j
 """
 a1, b1, c1, d1 = self.a, self.b, self.c, self.d
 a2, b2, c2, d2 = other.a, other.b, other.c, other.d
 
 return Quaternion(
 a = a1*a2 - b1*b2 - c1*c2 - d1*d2,
 b = a1*b2 + b1*a2 + c1*d2 - d1*c2,
 c = a1*c2 - b1*d2 + c1*a2 + d1*b2,
 d = a1*d2 + b1*c2 - c1*b2 + d1*a2
 )
 
 def __add__(self, other: 'Quaternion') -> 'Quaternion':
 """Quaternion addition (component-wise)."""
 return Quaternion(
 a=self.a + other.a,
 b=self.b + other.b,
 c=self.c + other.c,
 d=self.d + other.d
 )
 
 def __sub__(self, other: 'Quaternion') -> 'Quaternion':
 """Quaternion subtraction (component-wise)."""
 return Quaternion(
 a=self.a - other.a,
 b=self.b - other.b,
 c=self.c - other.c,
 d=self.d - other.d
 )
 
 def __rmul__(self, scalar: float) -> 'Quaternion':
 """Scalar multiplication."""
 return Quaternion(
 a=scalar * self.a,
 b=scalar * self.b,
 c=scalar * self.c,
 d=scalar * self.d
 )
 
 def scalar_part(self) -> float:
 """Extract scalar (real) part."""
 return self.a
 
 def vector_part(self) -> np.ndarray:
 """Extract vector (imaginary) part as 3D array [b, c, d]."""
 return np.array([self.b, self.c, self.d])
 
 def to_rotation_matrix(self) -> np.ndarray:
 """
 Convert unit quaternion to 3×3 rotation matrix.
 
 Used for quaternionic rotation operators in Progenitor matrix.
 Per ENTPC.tex: Q(θ) represents quaternionic rotation.
 """
 # Normalize first
 q = self.normalize()
 a, b, c, d = q.a, q.b, q.c, q.d
 
 # Rotation matrix formula
 R = np.array([
 [1 - 2*(c**2 + d**2), 2*(b*c - a*d), 2*(b*d + a*c)],
 [ 2*(b*c + a*d), 1 - 2*(b**2 + d**2), 2*(c*d - a*b)],
 [ 2*(b*d - a*c), 2*(c*d + a*b), 1 - 2*(b**2 + c**2)]
 ])
 
 return R
 
 @classmethod
 def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
 """
 Create quaternion from axis-angle representation.
 
 q = cos(θ/2) + sin(θ/2)(u_x i + u_y j + u_z k)
 where u is the unit axis vector.
 """
 axis = np.array(axis, dtype=float)
 axis_norm = np.linalg.norm(axis)
 assert axis_norm > 1e-12, "Axis must be non-zero"
 
 u = axis / axis_norm
 half_angle = angle / 2.0
 
 return cls(
 a=np.cos(half_angle),
 b=np.sin(half_angle) * u[0],
 c=np.sin(half_angle) * u[1],
 d=np.sin(half_angle) * u[2]
 )
 
 def __repr__(self) -> str:
 return f"Quaternion({self.a:.4f} + {self.b:.4f}i + {self.c:.4f}j + {self.d:.4f}k)"

class QuaternionicHilbertSpace:
 """
 Quaternionic Hilbert Space H_H of dimension n.
 
 Per ENTPC.tex Definition 2.1:
 - Elements are n-dimensional vectors over quaternions H
 - Inner product: ⟨q, p⟩ = Σ q_i* p_i
 - Norm: ||q|| = √⟨q, q⟩
 
 Per ENTPC.tex Definition 2.2:
 - Local filtering operator F_H provides context-dependent stabilization
 - First stage of two-stage EntPTC process
 """
 
 def __init__(self, dimension: int):
 """
 Initialize quaternionic Hilbert space of given dimension.
 
 Args:
 dimension: Dimension n of H_H (number of quaternions in vector)
 """
 assert dimension > 0, "Dimension must be positive"
 self.dimension = dimension
 
 def inner_product(self, q: np.ndarray, p: np.ndarray) -> Quaternion:
 """
 Quaternionic inner product ⟨q, p⟩ = Σ q_i* p_i.
 
 Args:
 q: Array of Quaternion objects (length n)
 p: Array of Quaternion objects (length n)
 
 Returns:
 Quaternion representing the inner product
 """
 assert len(q) == self.dimension, f"Expected {self.dimension} quaternions, got {len(q)}"
 assert len(p) == self.dimension, f"Expected {self.dimension} quaternions, got {len(p)}"
 
 result = Quaternion()
 for qi, pi in zip(q, p):
 result = result + (qi.conjugate() * pi)
 
 return result
 
 def norm(self, q: np.ndarray) -> float:
 """
 Compute norm ||q|| = √⟨q, q⟩.
 
 Args:
 q: Array of Quaternion objects (length n)
 
 Returns:
 Real-valued norm
 """
 inner = self.inner_product(q, q)
 # Inner product of vector with itself should be real (scalar part only)
 return np.sqrt(inner.scalar_part())
 
 def normalize(self, q: np.ndarray) -> np.ndarray:
 """
 Normalize quaternionic vector to unit norm.
 
 Args:
 q: Array of Quaternion objects (length n)
 
 Returns:
 Normalized array of Quaternion objects
 """
 n = self.norm(q)
 if n < 1e-12:
 # Return identity quaternions if zero vector
 return np.array([Quaternion(a=1.0) for _ in range(self.dimension)])
 
 return np.array([Quaternion(
 a=qi.a/n, b=qi.b/n, c=qi.c/n, d=qi.d/n
 ) for qi in q])

class LocalQuaternionicFilter:
 """
 Local Quaternionic Filtering Operator F_H: H_H → H_H
 
 Per ENTPC.tex Definition 2.2:
 - Provides context-dependent state stabilization
 - First stage of two-stage EntPTC process
 - Prepares states for global Clifford embedding
 
 Implementation:
 - Applies quaternionic smoothing based on local coherence
 - Stabilizes rapid fluctuations while preserving structure
 - Context-dependent: filter strength adapts to local signal properties
 """
 
 def __init__(self, dimension: int, coherence_threshold: float = 0.5):
 """
 Initialize local quaternionic filter.
 
 Args:
 dimension: Dimension of quaternionic Hilbert space
 coherence_threshold: Threshold for adaptive filtering (0 to 1)
 """
 self.dimension = dimension
 self.coherence_threshold = coherence_threshold
 self.hilbert_space = QuaternionicHilbertSpace(dimension)
 
 def compute_local_coherence(self, q: np.ndarray) -> float:
 """
 Compute local coherence measure for adaptive filtering.
 
 Coherence measures the alignment of quaternions in the vector.
 High coherence → less filtering needed (stable state)
 Low coherence → more filtering needed (unstable state)
 
 Args:
 q: Array of Quaternion objects (length n)
 
 Returns:
 Coherence value in [0, 1]
 """
 # Normalize all quaternions
 q_normalized = np.array([qi.normalize() for qi in q])
 
 # Compute pairwise alignment (scalar part of q_i* q_j)
 coherence_sum = 0.0
 count = 0
 
 for i in range(len(q_normalized)):
 for j in range(i+1, len(q_normalized)):
 alignment = (q_normalized[i].conjugate() * q_normalized[j]).scalar_part()
 coherence_sum += abs(alignment)
 count += 1
 
 if count == 0:
 return 1.0
 
 return coherence_sum / count
 
 def filter(self, q: np.ndarray, strength: Optional[float] = None) -> np.ndarray:
 """
 Apply local quaternionic filtering F_H(q).
 
 Per ENTPC.tex Definition 2.2: Context-dependent state stabilization.
 
 Implementation:
 - Adaptive filtering based on local coherence
 - Smooths quaternionic vector while preserving structure
 - Strength adapts to signal properties
 
 Args:
 q: Array of Quaternion objects (length n)
 strength: Optional manual filter strength (0 to 1)
 If None, computed adaptively from coherence
 
 Returns:
 Filtered array of Quaternion objects
 """
 assert len(q) == self.dimension, f"Expected {self.dimension} quaternions, got {len(q)}"
 
 # Compute adaptive filter strength if not provided
 if strength is None:
 coherence = self.compute_local_coherence(q)
 # Low coherence → high filtering strength
 strength = max(0.0, min(1.0, 1.0 - coherence))
 
 # Compute mean quaternion (geometric mean approximation)
 mean_components = np.mean([qi.to_array() for qi in q], axis=0)
 mean_q = Quaternion.from_array(mean_components).normalize()
 
 # Apply adaptive smoothing: q_filtered = (1-α)q + α*mean
 filtered = []
 for qi in q:
 # Linear interpolation in quaternion space
 filtered_components = (1 - strength) * qi.to_array() + strength * mean_q.to_array()
 filtered_q = Quaternion.from_array(filtered_components)
 filtered.append(filtered_q)
 
 return np.array(filtered)
 
 def filter_sequence(self, Q: np.ndarray, window_size: int = 3) -> np.ndarray:
 """
 Apply temporal filtering to sequence of quaternionic vectors.
 
 Used for EEG time series: each time point has n quaternions.
 
 Args:
 Q: Array of shape (T, n) where T is time steps, n is dimension
 Each Q[t] is an array of Quaternion objects
 window_size: Size of temporal smoothing window
 
 Returns:
 Filtered array of shape (T, n)
 """
 T = len(Q)
 filtered = []
 
 for t in range(T):
 # Define temporal window
 t_start = max(0, t - window_size // 2)
 t_end = min(T, t + window_size // 2 + 1)
 
 # Average over temporal window
 window_quaternions = []
 for tau in range(t_start, t_end):
 window_quaternions.extend(Q[tau])
 
 # Compute mean quaternion for window
 mean_components = np.mean([qi.to_array() for qi in window_quaternions], axis=0)
 mean_q = Quaternion.from_array(mean_components).normalize()
 
 # Apply filtering to current time point
 filtered_t = []
 for qi in Q[t]:
 # Adaptive blend with temporal mean
 coherence = abs((qi.conjugate() * mean_q).scalar_part())
 strength = max(0.0, min(1.0, 1.0 - coherence))
 
 filtered_components = (1 - strength) * qi.to_array() + strength * mean_q.to_array()
 filtered_q = Quaternion.from_array(filtered_components)
 filtered_t.append(filtered_q)
 
 filtered.append(np.array(filtered_t))
 
 return np.array(filtered)

def quaternion_from_eeg_channels(channels: np.ndarray) -> Quaternion:
 """
 Construct quaternion from 4 EEG channels.
 
 Maps 4-dimensional real signal to quaternion space:
 q = c1 + c2*i + c3*j + c4*k
 
 Args:
 channels: Array of 4 channel values
 
 Returns:
 Quaternion constructed from channels
 """
 assert len(channels) == 4, f"Expected 4 channels, got {len(channels)}"
 return Quaternion(
 a=float(channels[0]),
 b=float(channels[1]),
 c=float(channels[2]),
 d=float(channels[3])
 )

def eeg_to_quaternionic_vector(eeg_data: np.ndarray) -> np.ndarray:
 """
 Convert EEG data to quaternionic vector.
 
 Per ENTPC.tex: 64 EEG channels → 16 quaternions (64/4 = 16)
 Each group of 4 channels forms one quaternion.
 
 Args:
 eeg_data: Array of shape (64,) representing 64 EEG channels
 
 Returns:
 Array of 16 Quaternion objects
 """
 assert len(eeg_data) == 64, f"Expected 64 channels, got {len(eeg_data)}"
 
 quaternions = []
 for i in range(0, 64, 4):
 q = quaternion_from_eeg_channels(eeg_data[i:i+4])
 quaternions.append(q)
 
 assert len(quaternions) == 16, f"Expected 16 quaternions, got {len(quaternions)}"
 return np.array(quaternions)

def quaternionic_vector_to_matrix(q_vector: np.ndarray) -> np.ndarray:
 """
 Convert quaternionic vector to 4×4 matrix representation.
 
 Used for Progenitor matrix construction (16 quaternions → 4×4 blocks).
 
 Args:
 q_vector: Array of 16 Quaternion objects
 
 Returns:
 4×4 matrix where each entry is a quaternion norm
 """
 assert len(q_vector) == 16, f"Expected 16 quaternions, got {len(q_vector)}"
 
 matrix = np.zeros((4, 4))
 for i in range(4):
 for j in range(4):
 idx = i * 4 + j
 matrix[i, j] = q_vector[idx].norm()
 
 return matrix
