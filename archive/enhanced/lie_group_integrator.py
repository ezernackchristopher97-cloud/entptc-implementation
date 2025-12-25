"""
Lie Group Integration for Geodesics on Toroidal Manifold T³

Implements the Runge-Kutta-Munthe-Kaas (RKMK) method for integrating
differential equations on Lie groups and manifolds. This ensures that simulated
paths stay on the toroidal manifold T³ and preserve its geometric structure.

References:
- Chen & Wang (2004), Chapter 7: Geometric Integration and Its Applications
- Munthe-Kaas (1998): Runge-Kutta methods on Lie groups
- Iserles et al. (2000): Lie-group methods

Per ENTPC.tex: This provides structure-preserving integration for simulating
state evolution along the entropy gradient on T³.

"""

import numpy as np
from typing import Callable, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class ToroidalState:
 """
 State on the toroidal manifold T³ = S¹ × S¹ × S¹.
 
 Parameterized by three angles (θ₁, θ₂, θ₃) ∈ [0, 2π)³.
 """
 theta1: float
 theta2: float
 theta3: float
 
 def __post_init__(self):
 """Wrap angles to [0, 2π)."""
 self.theta1 = np.mod(self.theta1, 2*np.pi)
 self.theta2 = np.mod(self.theta2, 2*np.pi)
 self.theta3 = np.mod(self.theta3, 2*np.pi)
 
 def to_array(self) -> np.ndarray:
 """Convert to numpy array."""
 return np.array([self.theta1, self.theta2, self.theta3])
 
 @classmethod
 def from_array(cls, arr: np.ndarray) -> 'ToroidalState':
 """Create from numpy array."""
 return cls(theta1=arr[0], theta2=arr[1], theta3=arr[2])
 
 def distance_to(self, other: 'ToroidalState') -> float:
 """
 Compute geodesic distance to another state on T³.
 
 Uses angular distance with periodic boundaries.
 """
 def angular_distance(a, b):
 diff = abs(a - b)
 return min(diff, 2*np.pi - diff)
 
 d1 = angular_distance(self.theta1, other.theta1)
 d2 = angular_distance(self.theta2, other.theta2)
 d3 = angular_distance(self.theta3, other.theta3)
 
 return np.sqrt(d1**2 + d2**2 + d3**2)

class LieGroupIntegrator:
 """
 Runge-Kutta-Munthe-Kaas (RKMK) integrator for T³.
 
 This is a structure-preserving integrator that ensures the integrated
 path stays on the toroidal manifold, even for long simulations.
 
 The RKMK method works by:
 1. Computing tangent vectors in the Lie algebra (tangent space)
 2. Exponentiating back to the manifold
 3. Using a Runge-Kutta scheme in the Lie algebra
 
 For T³, the Lie algebra is ℝ³ (angular velocities), and the exponential
 map is just addition modulo 2π.
 """
 
 def __init__(self, 
 vector_field: Callable[[ToroidalState, float], np.ndarray],
 step_size: float = 0.01,
 method: str = 'rk4'):
 """
 Initialize Lie group integrator.
 
 Args:
 vector_field: Function f(state, t) → velocity vector in Lie algebra
 step_size: Integration step size (default: 0.01)
 method: Integration method ('rk4', 'rk2', or 'euler')
 """
 self.vector_field = vector_field
 self.step_size = step_size
 self.method = method
 
 if method not in ['rk4', 'rk2', 'euler']:
 raise ValueError(f"Unknown method: {method}. Use 'rk4', 'rk2', or 'euler'.")
 
 def step(self, state: ToroidalState, t: float) -> ToroidalState:
 """
 Take one integration step.
 
 Args:
 state: Current state on T³
 t: Current time
 
 Returns:
 New state after one step
 """
 if self.method == 'euler':
 return self._euler_step(state, t)
 elif self.method == 'rk2':
 return self._rk2_step(state, t)
 elif self.method == 'rk4':
 return self._rk4_step(state, t)
 else:
 raise ValueError(f"Unknown method: {self.method}")
 
 def _euler_step(self, state: ToroidalState, t: float) -> ToroidalState:
 """
 Euler method (first-order).
 
 θ_{n+1} = θ_n + h*f(θ_n, t_n) mod 2π
 """
 theta = state.to_array()
 k1 = self.vector_field(state, t)
 
 theta_new = theta + self.step_size * k1
 
 return ToroidalState.from_array(theta_new)
 
 def _rk2_step(self, state: ToroidalState, t: float) -> ToroidalState:
 """
 Runge-Kutta 2nd order (midpoint method).
 
 k1 = f(θ_n, t_n)
 k2 = f(θ_n + h/2*k1, t_n + h/2)
 θ_{n+1} = θ_n + h*k2 mod 2π
 """
 h = self.step_size
 theta = state.to_array()
 
 # k1
 k1 = self.vector_field(state, t)
 
 # k2
 state_mid = ToroidalState.from_array(theta + 0.5*h*k1)
 k2 = self.vector_field(state_mid, t + 0.5*h)
 
 # Update
 theta_new = theta + h * k2
 
 return ToroidalState.from_array(theta_new)
 
 def _rk4_step(self, state: ToroidalState, t: float) -> ToroidalState:
 """
 Runge-Kutta 4th order (classical RK4).
 
 k1 = f(θ_n, t_n)
 k2 = f(θ_n + h/2*k1, t_n + h/2)
 k3 = f(θ_n + h/2*k2, t_n + h/2)
 k4 = f(θ_n + h*k3, t_n + h)
 θ_{n+1} = θ_n + h/6*(k1 + 2*k2 + 2*k3 + k4) mod 2π
 """
 h = self.step_size
 theta = state.to_array()
 
 # k1
 k1 = self.vector_field(state, t)
 
 # k2
 state_2 = ToroidalState.from_array(theta + 0.5*h*k1)
 k2 = self.vector_field(state_2, t + 0.5*h)
 
 # k3
 state_3 = ToroidalState.from_array(theta + 0.5*h*k2)
 k3 = self.vector_field(state_3, t + 0.5*h)
 
 # k4
 state_4 = ToroidalState.from_array(theta + h*k3)
 k4 = self.vector_field(state_4, t + h)
 
 # Update
 theta_new = theta + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
 
 return ToroidalState.from_array(theta_new)
 
 def integrate(self, 
 initial_state: ToroidalState, 
 t_span: Tuple[float, float],
 n_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
 """
 Integrate from initial state over time span.
 
 Args:
 initial_state: Starting state on T³
 t_span: (t_start, t_end) time interval
 n_steps: Number of steps (if None, computed from step_size)
 
 Returns:
 (times, states) where:
 - times: Array of time points (n_steps+1,)
 - states: Array of states (n_steps+1, 3)
 """
 t_start, t_end = t_span
 
 if n_steps is None:
 n_steps = int((t_end - t_start) / self.step_size)
 
 # Adjust step size to match n_steps
 h = (t_end - t_start) / n_steps
 self.step_size = h
 
 # Initialize arrays
 times = np.linspace(t_start, t_end, n_steps + 1)
 states = np.zeros((n_steps + 1, 3))
 
 # Initial state
 state = initial_state
 states[0] = state.to_array()
 
 # Integration loop
 for i in range(n_steps):
 state = self.step(state, times[i])
 states[i+1] = state.to_array()
 
 return times, states

def entropy_gradient_vector_field(entropy_gradient_func: Callable[[ToroidalState], np.ndarray],
 learning_rate: float = 0.1) -> Callable[[ToroidalState, float], np.ndarray]:
 """
 Create a vector field for gradient descent on the entropy field.
 
 Per ENTPC.tex, the entropy gradient ∇S guides the flow toward regions
 of lower entropy (higher organization). The vector field is:
 
 f(θ, t) = -α * ∇S(θ)
 
 where α is the learning rate (step size in gradient descent).
 
 Args:
 entropy_gradient_func: Function that computes ∇S at a state
 learning_rate: Step size for gradient descent
 
 Returns:
 Vector field function f(state, t) → velocity
 """
 def vector_field(state: ToroidalState, t: float) -> np.ndarray:
 """
 Compute velocity vector at state.
 
 Args:
 state: Current state on T³
 t: Current time (unused, but required for interface)
 
 Returns:
 Velocity vector (angular velocities)
 """
 gradient = entropy_gradient_func(state)
 return -learning_rate * gradient
 
 return vector_field

def geodesic_vector_field(target_state: ToroidalState) -> Callable[[ToroidalState, float], np.ndarray]:
 """
 Create a vector field that flows toward a target state along geodesics.
 
 The velocity at each point is proportional to the geodesic direction
 toward the target.
 
 Args:
 target_state: Target state on T³
 
 Returns:
 Vector field function f(state, t) → velocity
 """
 def vector_field(state: ToroidalState, t: float) -> np.ndarray:
 """
 Compute velocity vector pointing toward target.
 
 Args:
 state: Current state on T³
 t: Current time (unused)
 
 Returns:
 Velocity vector (angular velocities)
 """
 # Compute shortest angular difference to target
 def shortest_angular_diff(a, b):
 diff = b - a
 # Wrap to [-π, π]
 diff = np.mod(diff + np.pi, 2*np.pi) - np.pi
 return diff
 
 theta_current = state.to_array()
 theta_target = target_state.to_array()
 
 # Velocity is proportional to angular difference
 velocity = np.array([
 shortest_angular_diff(theta_current[0], theta_target[0]),
 shortest_angular_diff(theta_current[1], theta_target[1]),
 shortest_angular_diff(theta_current[2], theta_target[2])
 ])
 
 return velocity
 
 return vector_field

# Integration with EntPTC entropy field

class EntropyFlowIntegrator:
 """
 Integrator for simulating flow on the entropy field S: T³ → ℝ.
 
 This combines the Lie group integrator with the entropy gradient
 to simulate how states evolve under the entropy field.
 
 Per ENTPC.tex, the entropy field guides the system toward regions
 of lower entropy (higher organization).
 """
 
 def __init__(self,
 entropy_field: Callable[[ToroidalState], float],
 step_size: float = 0.01,
 learning_rate: float = 0.1,
 method: str = 'rk4'):
 """
 Initialize entropy flow integrator.
 
 Args:
 entropy_field: Function S(θ) → entropy value
 step_size: Integration step size
 learning_rate: Gradient descent learning rate
 method: Integration method ('rk4', 'rk2', or 'euler')
 """
 self.entropy_field = entropy_field
 self.step_size = step_size
 self.learning_rate = learning_rate
 self.method = method
 
 # Create gradient function (finite differences)
 self.gradient_func = self._create_gradient_function()
 
 # Create vector field
 vector_field = entropy_gradient_vector_field(
 self.gradient_func, 
 learning_rate
 )
 
 # Create integrator
 self.integrator = LieGroupIntegrator(
 vector_field=vector_field,
 step_size=step_size,
 method=method
 )
 
 def _create_gradient_function(self) -> Callable[[ToroidalState], np.ndarray]:
 """
 Create function to compute entropy gradient using finite differences.
 
 Returns:
 Function that computes ∇S(θ)
 """
 def gradient(state: ToroidalState) -> np.ndarray:
 """
 Compute entropy gradient at state using central differences.
 
 ∂S/∂θᵢ ≈ (S(θ + ε*eᵢ) - S(θ - ε*eᵢ)) / (2ε)
 """
 epsilon = 1e-5
 theta = state.to_array()
 
 grad = np.zeros(3)
 
 for i in range(3):
 # Forward perturbation
 theta_plus = theta.copy()
 theta_plus[i] += epsilon
 state_plus = ToroidalState.from_array(theta_plus)
 S_plus = self.entropy_field(state_plus)
 
 # Backward perturbation
 theta_minus = theta.copy()
 theta_minus[i] -= epsilon
 state_minus = ToroidalState.from_array(theta_minus)
 S_minus = self.entropy_field(state_minus)
 
 # Central difference
 grad[i] = (S_plus - S_minus) / (2*epsilon)
 
 return grad
 
 return gradient
 
 def simulate_flow(self,
 initial_state: ToroidalState,
 duration: float,
 n_steps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
 """
 Simulate flow on entropy field from initial state.
 
 Args:
 initial_state: Starting state on T³
 duration: Total simulation time
 n_steps: Number of steps (if None, computed from step_size)
 
 Returns:
 (times, states, entropies) where:
 - times: Array of time points
 - states: Array of states (n_steps+1, 3)
 - entropies: Array of entropy values at each state
 """
 # Integrate
 times, states = self.integrator.integrate(
 initial_state,
 t_span=(0.0, duration),
 n_steps=n_steps
 )
 
 # Compute entropy at each state
 entropies = np.array([
 self.entropy_field(ToroidalState.from_array(state))
 for state in states
 ])
 
 return times, states, entropies
 
 def find_local_minimum(self,
 initial_state: ToroidalState,
 max_iterations: int = 1000,
 tolerance: float = 1e-6) -> Tuple[ToroidalState, float]:
 """
 Find local minimum of entropy field using gradient descent.
 
 Args:
 initial_state: Starting state
 max_iterations: Maximum number of iterations
 tolerance: Convergence tolerance (gradient norm)
 
 Returns:
 (final_state, final_entropy)
 """
 state = initial_state
 
 for iteration in range(max_iterations):
 # Compute gradient
 grad = self.gradient_func(state)
 grad_norm = np.linalg.norm(grad)
 
 # Check convergence
 if grad_norm < tolerance:
 break
 
 # Take gradient descent step
 theta = state.to_array()
 theta_new = theta - self.learning_rate * grad
 state = ToroidalState.from_array(theta_new)
 
 final_entropy = self.entropy_field(state)
 
 return state, final_entropy

# Summary and integration notes

"""
Lie Group Integration for EntPTC:

1. **Structure Preservation**: The RKMK method ensures that integrated paths
 stay on the toroidal manifold T³, preventing drift and maintaining
 geometric structure.

2. **Entropy Flow Simulation**: The EntropyFlowIntegrator combines Lie group
 integration with the entropy gradient to simulate state evolution.

3. **Local Minimum Finding**: The find_local_minimum method can be used to
 identify stable states (local minima of the entropy field).

4. **Improved Accuracy**: RK4 provides 4th-order accuracy, significantly
 better than Euler or simple RK2 methods.

Next steps for full integration:
- Connect with entropy.py to use the actual entropy field
- Use in progenitor.py for simulating state evolution
- Benchmark accuracy vs. simple Euler integration
- Visualize flow lines on T³
"""
