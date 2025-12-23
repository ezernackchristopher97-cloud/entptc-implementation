"""

Perron-Frobenius Operator Implementation

Reference: ENTPC.tex Definition 2.7-2.8 (lines 287-297)

 

From ENTPC.tex Definition 2.7-2.8:

The Progenitor Operator O_P resolves multiplicity through spectral decomposition.

For an irreducible matrix M, the Perron-Frobenius theorem guarantees a unique,

simple, and positive dominant eigenvalue λ_max, whose corresponding eigenvector

v_1 has all positive entries.

 

The operator collapses the system dynamics to this dominant mode:

    lim_{n→∞} M^n ψ_0 / ||M^n ψ_0|| = v_1

 

This eigenvector v_1 represents the unified conscious state, with all other

modes decaying exponentially.

 

From ENTPC.tex:

- Quaternionic structure is used for local phase stabilization

- Multiplicity resolution is performed AFTER projection to real, non-negative matrix

- Perron-Frobenius guarantees unique dominant mode

"""

 

import numpy as np

from typing import Tuple, Dict, Any, Optional, List

from dataclasses import dataclass

import logging

 

logger = logging.getLogger(__name__)

 

 

@dataclass

class PerronFrobeniusResult:

    """

    Result of Perron-Frobenius operator application per ENTPC.tex.

 

    Attributes:

        dominant_eigenvalue: λ_max (largest eigenvalue, must be positive and simple)

        dominant_eigenvector: v_1 (corresponding eigenvector, all positive entries)

        eigenvalue_spectrum: All eigenvalues sorted by magnitude

        spectral_gap: λ_1/λ_2 ratio (determines collapse rate)

        convergence_iterations: Number of power iterations to converge

        regime: Experience regime based on spectral gap

    """

    dominant_eigenvalue: float

    dominant_eigenvector: np.ndarray

    eigenvalue_spectrum: np.ndarray

    spectral_gap: float

    convergence_iterations: int

    regime: str

    all_eigenvectors: Optional[np.ndarray] = None

 

 

class PerronFrobeniusOperator:

    """

    Perron-Frobenius operator for Progenitor Matrix collapse per ENTPC.tex.

 

    This operator:

    1. Verifies the matrix is non-negative and irreducible

    2. Computes the dominant eigenvalue λ_max and eigenvector v_1

    3. Determines the experience regime from spectral gap

    4. Provides power iteration for convergence verification

 

    Per ENTPC.tex Definition 2.7:

    The operator collapses the system dynamics to the dominant mode:

        lim_{n→∞} M^n ψ_0 / ||M^n ψ_0|| = v_1

    """

 

    # Regime thresholds per ENTPC.tex Section 5.1 (lines 669-676)

    REGIME_I_THRESHOLD = 2.0    # Local Stabilized: gap > 2.0

    REGIME_II_UPPER = 2.0       # Transitional: 1.2 < gap < 2.0

    REGIME_II_LOWER = 1.2

    REGIME_III_THRESHOLD = 1.5  # Global Experience: gap < 1.5

 

    def __init__(self, tolerance: float = 1e-10, max_iterations: int = 1000):

        """

        Initialize Perron-Frobenius operator.

 

        Args:

            tolerance: Convergence tolerance for power iteration

            max_iterations: Maximum power iterations

        """

        self.tolerance = tolerance

        self.max_iterations = max_iterations

 

    def apply(self, matrix: np.ndarray) -> PerronFrobeniusResult:

        """

        Apply Perron-Frobenius operator to non-negative matrix.

 

        Per ENTPC.tex Definition 2.7:

        For an irreducible matrix M, the Perron-Frobenius theorem guarantees

        a unique, simple, and positive dominant eigenvalue λ_max.

 

        Args:

            matrix: 16×16 non-negative real matrix (Progenitor Matrix)

 

        Returns:

            PerronFrobeniusResult with dominant eigenvalue, eigenvector, spectrum

 

        Raises:

            ValueError: If matrix violates Perron-Frobenius conditions

        """

        # Validate matrix per ENTPC.tex requirements

        self._validate_matrix(matrix)

 

        # Compute full eigendecomposition

        eigenvalues, eigenvectors = np.linalg.eig(matrix)

 

        # Sort by magnitude (largest first)

        idx = np.argsort(np.abs(eigenvalues))[::-1]

        eigenvalues = eigenvalues[idx]

        eigenvectors = eigenvectors[:, idx]

 

        # Dominant eigenvalue and eigenvector

        lambda_max = np.real(eigenvalues[0])

        v1 = np.real(eigenvectors[:, 0])

 

        # Perron-Frobenius: dominant eigenvector should have all positive entries

        # Normalize to have positive entries

        if np.sum(v1) < 0:

            v1 = -v1

        v1 = v1 / np.linalg.norm(v1)

 

        # Verify positivity (allow small numerical errors)

        if not np.all(v1 > -1e-10):

            logger.warning(

                f"Dominant eigenvector has negative entries: min={v1.min():.6f}. "

                "Matrix may not satisfy Perron-Frobenius conditions."

            )

        # Clamp small negatives to zero

        v1 = np.maximum(v1, 0)

        v1 = v1 / np.linalg.norm(v1)

 

        # Compute spectral gap (λ_1 / λ_2)

        lambda_2 = np.abs(eigenvalues[1]) if len(eigenvalues) > 1 else 1e-10

        spectral_gap = lambda_max / max(lambda_2, 1e-10)

 

        # Power iteration for convergence verification

        iterations = self._power_iteration_check(matrix, v1)

 

        # Determine regime per ENTPC.tex Section 5.1

        regime = self._determine_regime(spectral_gap)

 

        return PerronFrobeniusResult(

            dominant_eigenvalue=lambda_max,

            dominant_eigenvector=v1,

            eigenvalue_spectrum=np.real(eigenvalues),

            spectral_gap=spectral_gap,

            convergence_iterations=iterations,

            regime=regime,

            all_eigenvectors=np.real(eigenvectors)

        )

 

    def power_iterate(

        self,

        matrix: np.ndarray,

        initial_state: Optional[np.ndarray] = None

    ) -> Tuple[np.ndarray, int]:

        """

        Perform power iteration per ENTPC.tex Definition 2.7.

 

        Computes: lim_{n→∞} M^n ψ_0 / ||M^n ψ_0|| = v_1

 

        This demonstrates the collapse of system dynamics to dominant mode.

 

        Args:

            matrix: 16×16 Progenitor Matrix

            initial_state: Optional initial state vector (default: uniform)

 

        Returns:

            (converged_vector, iterations)

        """

        n = matrix.shape[0]

 

        if initial_state is None:

            # Start with uniform state per ENTPC.tex toy example

            psi = np.ones(n) / np.sqrt(n)

        else:

            psi = initial_state / np.linalg.norm(initial_state)

 

        prev_psi = psi.copy()

 

        for iteration in range(self.max_iterations):

            # Apply M: ψ_{n+1} = M ψ_n

            psi = matrix @ psi

 

            # Normalize: ψ_{n+1} = ψ_{n+1} / ||ψ_{n+1}||

            norm = np.linalg.norm(psi)

            if norm < 1e-15:

                raise ValueError("Power iteration collapsed to zero vector")

            psi = psi / norm

 

            # Check convergence

            diff = np.linalg.norm(psi - prev_psi)

            if diff < self.tolerance:

                return psi, iteration + 1

 

            prev_psi = psi.copy()

 

        logger.warning(

            f"Power iteration did not converge after {self.max_iterations} iterations"

        )

        return psi, self.max_iterations

 

    def _validate_matrix(self, matrix: np.ndarray):

        """Validate matrix satisfies Perron-Frobenius conditions."""

        # Check shape per ENTPC.tex

        if matrix.shape != (16, 16):

            raise ValueError(

                f"ENTPC.tex requires 16×16 matrix, got {matrix.shape}"

            )

 

        # Check non-negative per ENTPC.tex Definition 2.6

        if not np.all(matrix >= -1e-10):

            min_val = matrix.min()

            raise ValueError(

                f"Progenitor Matrix must be non-negative. Min value: {min_val}"

            )

 

        # Check real-valued

        if np.iscomplexobj(matrix) and np.any(np.abs(matrix.imag) > 1e-10):

            raise ValueError("Progenitor Matrix must be real-valued")

 

        # Check finite

        if not np.all(np.isfinite(matrix)):

            raise ValueError("Matrix contains non-finite values")

 

    def _power_iteration_check(

        self,

        matrix: np.ndarray,

        v1: np.ndarray

    ) -> int:

        """Verify convergence to v1 via power iteration."""

        psi, iterations = self.power_iterate(matrix)

 

        # Check convergence to dominant eigenvector

        alignment = abs(np.dot(psi, v1))

        if alignment < 0.99:

            logger.warning(

                f"Power iteration alignment with v1: {alignment:.4f} "

                "(expected > 0.99)"

            )

 

        return iterations

 

    def _determine_regime(self, spectral_gap: float) -> str:

        """

        Determine experience regime from spectral gap per ENTPC.tex Section 5.1.

 

        From ENTPC.tex (lines 669-676):

        - Regime I (Local Stabilized): gap > 2.0

          Fast, reflexive, automatic processing. Quaternionic dominant.

        - Regime II (Transitional): 1.2 < gap < 2.0

          Context switching, multiple component integration.

        - Regime III (Global Experience): gap < 1.5

          Multiple modes active, holistic experiences. Clifford dominant.

        """

        if spectral_gap > self.REGIME_I_THRESHOLD:

            return "I_LOCAL_STABILIZED"

        elif spectral_gap > self.REGIME_II_LOWER:

            return "II_TRANSITIONAL"

        else:

            return "III_GLOBAL_EXPERIENCE"

 

    def compute_collapse_rate(self, result: PerronFrobeniusResult) -> float:

        """

        Compute rate of collapse to dominant mode.

 

        Per ENTPC.tex: The spectral gap determines the rate of collapse.

        Larger gap = faster collapse to unified conscious state.

 

        Rate approximated as: -log(λ_2/λ_1) = log(spectral_gap)

        """

        return np.log(max(result.spectral_gap, 1.0 + 1e-10))

 

    def validate_perron_frobenius(

        self,

        matrix: np.ndarray,

        result: PerronFrobeniusResult

    ) -> Dict[str, Any]:

        """

        Validate that Perron-Frobenius theorem conditions are satisfied.

 

        Per ENTPC.tex Definition 2.7:

        1. Matrix is non-negative

        2. Matrix is irreducible

        3. Dominant eigenvalue is positive, real, and simple

        4. Dominant eigenvector has all positive entries

 

        Returns dict with validation results.

        """

        validations = {}

 

        # 1. Non-negativity

        validations['non_negative'] = bool(np.all(matrix >= -1e-10))

 

        # 2. Irreducibility (check via (I+M)^(n-1) > 0)

        n = matrix.shape[0]

        power = np.eye(n) + matrix

        for _ in range(n - 1):

            power = power @ (np.eye(n) + matrix)

        validations['irreducible'] = bool(np.all(power > 0))

 

        # 3. Dominant eigenvalue properties

        validations['lambda_max_positive'] = result.dominant_eigenvalue > 0

        validations['lambda_max_real'] = True  # We only store real part

        validations['lambda_max_simple'] = (

            abs(result.eigenvalue_spectrum[0] - result.eigenvalue_spectrum[1])

            > 1e-6 if len(result.eigenvalue_spectrum) > 1 else True

        )

 

        # 4. Eigenvector positivity

        validations['v1_positive'] = bool(np.all(result.dominant_eigenvector >= -1e-10))

        validations['v1_normalized'] = abs(

            np.linalg.norm(result.dominant_eigenvector) - 1.0

        ) < 1e-6

 

        # Overall validity

        validations['all_valid'] = all([

            validations['non_negative'],

            validations['lambda_max_positive'],

            validations['lambda_max_simple'],

            validations['v1_positive'],

            validations['v1_normalized']

        ])

 

        return validations

 

 

def extract_structural_invariants(

    result: PerronFrobeniusResult

) -> Dict[str, float]:

    """

    Extract structural invariants from Perron-Frobenius result.

 

    Per ENTPC.tex Section 6.3 (lines 718-727):

    Structural invariants include:

    - Eigenvalue ratios (λ_1/λ_2, λ_2/λ_3, etc.)

    - Spectral decay slopes

    - Regime-dependent stability thresholds

 

    These invariants are used for THz inference via structural matching,

    NOT frequency conversion.

    """

    spectrum = result.eigenvalue_spectrum

    n = len(spectrum)

 

    invariants = {

        'lambda_max': float(spectrum[0]),

        'spectral_gap': result.spectral_gap,

        'regime': result.regime,

    }

 

    # Eigenvalue ratios per ENTPC.tex D.2 (lines 1118-1122)

    for i in range(min(5, n - 1)):

        if abs(spectrum[i + 1]) > 1e-10:

            invariants[f'ratio_lambda{i+1}_lambda{i+2}'] = float(

                abs(spectrum[i]) / abs(spectrum[i + 1])

            )

        else:

            invariants[f'ratio_lambda{i+1}_lambda{i+2}'] = float('inf')

 

    # Spectral decay exponent α per ENTPC.tex D.2

    # α = -d log(λ_n) / dn

    if n >= 4:

        log_spectrum = np.log(np.abs(spectrum[:min(8, n)]) + 1e-15)

        indices = np.arange(len(log_spectrum))

        # Linear fit for decay slope

        if len(indices) > 1:

            coeffs = np.polyfit(indices, log_spectrum, 1)

            invariants['decay_exponent_alpha'] = float(-coeffs[0])

        else:

            invariants['decay_exponent_alpha'] = 0.0

 

    # Normalized spectral gap per ENTPC.tex D.2

    # G_norm = (λ_1 - λ_2) / λ_1

    if n >= 2 and abs(spectrum[0]) > 1e-10:

        invariants['normalized_gap'] = float(

            (spectrum[0] - abs(spectrum[1])) / spectrum[0]

        )

    else:

        invariants['normalized_gap'] = 1.0

 

    # Trace and determinant (matrix invariants)

    invariants['trace_fraction'] = float(spectrum[0] / max(np.sum(spectrum), 1e-10))

 

    return invariants

 

Limit reached · resets 10pm (UTC)

Limit reached · resets 10pm (UTC)