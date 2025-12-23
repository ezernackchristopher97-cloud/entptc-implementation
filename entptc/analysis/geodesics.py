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
                    theta_plus = theta.copy()
                    theta_minus = theta.copy()
                    
                    # ∂_i g_jl
                    theta_plus[i] += epsilon
                    theta_minus[i] -= epsilon
                    g_plus = self.metric_tensor(theta_plus)
                    g_minus = self.metric_tensor(theta_minus)
                    dg_jl_di = (g_plus[j, :] - g_minus[j, :]) / (2 * epsilon)
                    
                    # ∂_j g_il
                    theta_plus = theta.copy()
                    theta_minus = theta.copy()
                    theta_plus[j] += epsilon
                    theta_minus[j] -= epsilon
                    g_plus = self.metric_tensor(theta_plus)
                    g_minus = self.metric_tensor(theta_minus)
                    dg_il_dj = (g_plus[i, :] - g_minus[i, :]) / (2 * epsilon)
                    
                    # ∂_l g_ij
                    dg_ij_dl = np.zeros(3)
                    for l in range(3):
                        theta_plus = theta.copy()
                        theta_minus = theta.copy()
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
            end: Ending point (θ₁', θ₂', θ₃')
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
                method='RK45',
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
        result = minimize(objective, initial_v, method='BFGS', options={'maxiter': 50})
        optimal_v0 = result.x
        
        # Solve with optimal initial velocity
        y0 = np.concatenate([start, optimal_v0])
        sol = solve_ivp(
            self.geodesic_equation,
            (0, 1),
            y0,
            method='RK45',
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
            end: Ending point (θ₁', θ₂', θ₃')
        
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
            flow_time: Total flow time
            num_steps: Number of time steps
        
        Returns:
            Array of shape (n, num_steps, 3) with trajectories
        """
        n = len(start_points)
        trajectories = np.zeros((n, num_steps, 3))
        
        for i, start in enumerate(start_points):
            # Initial velocity: gradient descent on entropy
            grad_S = self.solver.entropy_field.gradient(start[0], start[1], start[2])
            initial_v = -grad_S  # Flow toward lower entropy
            
            # Solve geodesic equation
            y0 = np.concatenate([start, initial_v])
            sol = solve_ivp(
                self.solver.geodesic_equation,
                (0, flow_time),
                y0,
                method='RK45',
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
            Dictionary with 'minima', 'maxima', 'saddles' lists
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
                                point_pp = point.copy()
                                point_pm = point.copy()
                                point_mp = point.copy()
                                point_mm = point.copy()
                                
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
            'minima': np.array(minima) if minima else np.array([]).reshape(0, 3),
            'maxima': np.array(maxima) if maxima else np.array([]).reshape(0, 3),
            'saddles': np.array(saddles) if saddles else np.array([]).reshape(0, 3)
        }
    
    def compute_regime_transitions(self, regime_points: dict) -> dict:
        """
        Compute geodesic paths between regime representative points.
        
        Per ENTPC.tex: Reveals transition pathways between regimes I, II, III.
        
        Args:
            regime_points: Dictionary with 'regime_I', 'regime_II', 'regime_III'
                          each containing representative point (θ₁, θ₂, θ₃)
        
        Returns:
            Dictionary with transition paths and distances
        """
        transitions = {}
        
        # Compute all pairwise transitions
        regime_names = ['regime_I', 'regime_II', 'regime_III']
        
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
                        'path': path,
                        'velocities': velocities,
                        'distance': distance,
                        'start': start,
                        'end': end
                    }
        
        return transitions
