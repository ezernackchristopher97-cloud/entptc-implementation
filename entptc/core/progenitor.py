"""

Progenitor Matrix Implementation

Reference: ENTPC.tex Definition 2.6 (lines 266-285), Section 3 (lines 330-437)

 

From ENTPC.tex Definition 2.6:

The Progenitor Matrix, denoted M or C_16, is a 16×16 real, non-negative matrix

that generates the dynamics of experience. Each entry c_ij is given by:

 

    c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|

 

where:

- λ_ij is the coherence amplitude (derived from PLV coherence matrix)

- ∇S_ij is the entropy gradient between subsystems i and j

- Q(θ_ij) is a quaternionic rotation operator

- The norm |·| ensures a real-valued matrix entry

 

The matrix has a specific 16×16 block structure organized into four 4×4 quadrants,

each representing a different cognitive subsystem:

- Diagonal blocks A_pp: intra-subsystem coherence

- Off-diagonal blocks A_pq (p≠q): inter-subsystem coupling

 

From ENTPC.tex Section 3.1 (lines 332-334):

Two-stage process:

1. Quaternionic filtering applied to raw EEG data

2. Projection to real-valued matrix via modulus/norm operation

Then Perron-Frobenius collapse yields dominant eigenmode.

"""

 

import numpy as np

from typing import Tuple, Optional, Dict, Any, List

from dataclasses import dataclass

from .quaternion import Quaternion, QuaternionicHilbertSpace

 

 

@dataclass

class ProgenitorMatrixConfig:

    """Configuration for Progenitor Matrix construction per ENTPC.tex."""

    dimension: int = 16  # Must be 16 per ENTPC.tex

    num_quadrants: int = 4  # 4×4 block structure

    quadrant_size: int = 4  # Each quadrant is 4×4

 

    def __post_init__(self):

        """Validate configuration matches ENTPC.tex requirements."""

        assert self.dimension == 16, \

            f"ENTPC.tex requires 16×16 matrix, got {self.dimension}"

        assert self.num_quadrants == 4, \

            f"ENTPC.tex requires 4 quadrants, got {self.num_quadrants}"

        assert self.quadrant_size == 4, \

            f"ENTPC.tex requires 4×4 quadrants, got {self.quadrant_size}"

        assert self.num_quadrants * self.quadrant_size == self.dimension, \

            "Quadrant structure inconsistent with matrix dimension"

 

 

class ProgenitorMatrix:

    """

    16×16 Progenitor Matrix per ENTPC.tex Definition 2.6.

 

    The matrix is constructed from:

    1. Coherence matrix (16×16 PLV from EEG, aggregated from 64 channels to 16 ROIs)

    2. Entropy gradients between subsystems

    3. Quaternionic rotation operators

 

    Entry formula: c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|

    """

 

    def __init__(self, config: Optional[ProgenitorMatrixConfig] = None):

        """Initialize Progenitor Matrix structure."""

        self.config = config or ProgenitorMatrixConfig()

        self._matrix: Optional[np.ndarray] = None

        self._coherence_matrix: Optional[np.ndarray] = None

        self._entropy_gradients: Optional[np.ndarray] = None

        self._quaternion_norms: Optional[np.ndarray] = None

        self._is_constructed = False

 

    def construct_from_coherence(

        self,

        coherence_matrix: np.ndarray,

        entropy_field: np.ndarray,

        quaternion_phases: np.ndarray

    ) -> np.ndarray:

        """

        Construct Progenitor Matrix from coherence data per ENTPC.tex Eq. (6).

 

        c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|

 

        Args:

            coherence_matrix: 16×16 complex coherence (PLV) matrix

            entropy_field: 16-element entropy values for each ROI

            quaternion_phases: 16×3 array of toroidal coordinates (θ1, θ2, θ3)

                               per ROI for quaternionic rotation

 

        Returns:

            16×16 real non-negative Progenitor Matrix

        """

        # Validate inputs

        assert coherence_matrix.shape == (16, 16), \

            f"Coherence matrix must be 16×16, got {coherence_matrix.shape}"

        assert len(entropy_field) == 16, \

            f"Entropy field must have 16 elements, got {len(entropy_field)}"

        assert quaternion_phases.shape == (16, 3), \

            f"Quaternion phases must be 16×3, got {quaternion_phases.shape}"

 

        # Store inputs for later analysis

        self._coherence_matrix = coherence_matrix.copy()

 

        # 1. Compute coherence amplitudes λ_ij (magnitude of complex coherence)

        # Per ENTPC.tex: derived from PLV coherence

        lambda_ij = np.abs(coherence_matrix)

        assert lambda_ij.shape == (16, 16)

        assert np.all(lambda_ij >= 0), "Coherence amplitudes must be non-negative"

 

        # 2. Compute entropy gradients ∇S_ij between subsystems

        # Per ENTPC.tex Definition 2.5: ∇S = ∂_μ S

        # Gradient between i and j approximated as |S_i - S_j|

        self._entropy_gradients = np.zeros((16, 16))

        for i in range(16):

            for j in range(16):

                self._entropy_gradients[i, j] = abs(entropy_field[i] - entropy_field[j])

 

        # Exponential decay factor e^(-∇S_ij)

        exp_neg_grad_S = np.exp(-self._entropy_gradients)

        assert np.all(exp_neg_grad_S > 0), "Exponential must be positive"

        assert np.all(exp_neg_grad_S <= 1), "Exponential must be ≤1 for non-negative gradients"

 

        # 3. Compute quaternionic rotation norms |Q(θ_ij)|

        # Per ENTPC.tex: Q(θ_ij) is quaternionic rotation operator

        # We compute relative rotation between ROIs and take norm

        self._quaternion_norms = np.zeros((16, 16))

        for i in range(16):

            for j in range(16):

                # Quaternion from relative phase difference

                theta_diff = quaternion_phases[i] - quaternion_phases[j]

 

                # Create quaternion from Euler-like angles

                # q = cos(θ/2) + sin(θ/2)(n1*i + n2*j + n3*k)

                angle = np.linalg.norm(theta_diff)

                if angle > 1e-12:

                    axis = theta_diff / angle

                    q = Quaternion.from_axis_angle(axis, angle)

                else:

                    q = Quaternion(1.0, 0.0, 0.0, 0.0)  # Identity

 

                self._quaternion_norms[i, j] = q.norm()

 

        # 4. Construct Progenitor Matrix: c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|

        self._matrix = lambda_ij * exp_neg_grad_S * self._quaternion_norms

 

        # Validate result per ENTPC.tex requirements

        assert self._matrix.shape == (16, 16), \

            f"Matrix must be 16×16, got {self._matrix.shape}"

        assert np.all(self._matrix >= 0), \

            "Progenitor matrix must be non-negative"

        assert np.all(np.isfinite(self._matrix)), \

            "Matrix contains non-finite values"

 

        self._is_constructed = True

        return self._matrix

 

    def get_quadrant(self, p: int, q: int) -> np.ndarray:

        """

        Get A_pq quadrant (4×4 submatrix) per ENTPC.tex block structure.

 

        Per ENTPC.tex Definition 2.6:

        - Diagonal blocks A_pp: intra-subsystem coherence

        - Off-diagonal blocks A_pq: inter-subsystem coupling

 

        Args:

            p: Row quadrant index (0-3)

            q: Column quadrant index (0-3)

 

        Returns:

            4×4 submatrix A_pq

        """

        self._check_constructed()

        assert 0 <= p < 4, f"Quadrant row index must be 0-3, got {p}"

        assert 0 <= q < 4, f"Quadrant col index must be 0-3, got {q}"

 

        row_start = p * 4

        col_start = q * 4

        return self._matrix[row_start:row_start+4, col_start:col_start+4].copy()

 

    def get_intra_coherence(self) -> List[np.ndarray]:

        """

        Get diagonal blocks (intra-subsystem coherence) per ENTPC.tex.

 

        Returns list of four 4×4 matrices [A_11, A_22, A_33, A_44].

        """

        return [self.get_quadrant(i, i) for i in range(4)]

 

    def get_inter_coupling(self) -> Dict[Tuple[int, int], np.ndarray]:

        """

        Get off-diagonal blocks (inter-subsystem coupling) per ENTPC.tex.

 

        Returns dict mapping (p, q) to A_pq for p ≠ q.

        """

        coupling = {}

        for p in range(4):

            for q in range(4):

                if p != q:

                    coupling[(p, q)] = self.get_quadrant(p, q)

        return coupling

 

    @property

    def matrix(self) -> np.ndarray:

        """Get the 16×16 Progenitor Matrix."""

        self._check_constructed()

        return self._matrix.copy()

 

    @property

    def is_irreducible(self) -> bool:

        """

        Check if matrix is irreducible (required for Perron-Frobenius).

 

        Per ENTPC.tex Definition 2.7:

        For an irreducible matrix M, the Perron-Frobenius theorem guarantees

        a unique, simple, and positive dominant eigenvalue.

 

        A matrix is irreducible iff its associated directed graph is strongly connected.

        """

        self._check_constructed()

 

        # Matrix is irreducible if (I + M)^(n-1) has all positive entries

        n = self._matrix.shape[0]

        power = np.eye(n) + self._matrix

        for _ in range(n - 1):

            power = power @ (np.eye(n) + self._matrix)

 

        return np.all(power > 0)

 

    def _check_constructed(self):

        """Verify matrix has been constructed."""

        if not self._is_constructed or self._matrix is None:

            raise RuntimeError(

                "Progenitor Matrix not yet constructed. "

                "Call construct_from_coherence() first."

            )

 

    def validate_structure(self) -> Dict[str, Any]:

        """

        Validate matrix structure per ENTPC.tex requirements.

 

        Returns dict with validation results.

        """

        self._check_constructed()

 

        results = {

            'shape_valid': self._matrix.shape == (16, 16),

            'non_negative': bool(np.all(self._matrix >= 0)),

            'finite': bool(np.all(np.isfinite(self._matrix))),

            'irreducible': self.is_irreducible,

            'diagonal_sum': float(np.trace(self._matrix)),

            'matrix_sum': float(np.sum(self._matrix)),

            'frobenius_norm': float(np.linalg.norm(self._matrix, 'fro')),

        }

 

        # Check quadrant structure

        for p in range(4):

            for q in range(4):

                quad = self.get_quadrant(p, q)

                key = f'quadrant_A{p+1}{q+1}_norm'

                results[key] = float(np.linalg.norm(quad, 'fro'))

 

        return results

 

 

def construct_coherence_from_plv(

    plv_matrix: np.ndarray,

    roi_mapping: Optional[np.ndarray] = None

) -> np.ndarray:

    """

    Construct 16×16 coherence matrix from Phase Locking Value data.

 

    Per ENTPC.tex Section 6.1 (lines 696-703):

    1. The 64 channels are aggregated into 16 anatomically defined ROIs

    2. PLV is computed for each pair of ROIs

 

    Args:

        plv_matrix: Complex PLV matrix (64×64 or 16×16)

        roi_mapping: Optional 64-element array mapping channels to ROIs (0-15)

                     If None and input is 64×64, uses default 4-channel per ROI

 

    Returns:

        16×16 complex coherence matrix

    """

    if plv_matrix.shape == (16, 16):

        # Already 16×16

        return plv_matrix.astype(np.complex128)

 

    if plv_matrix.shape == (64, 64):

        # Aggregate 64 channels to 16 ROIs

        if roi_mapping is None:

            # Default: 4 consecutive channels per ROI

            roi_mapping = np.repeat(np.arange(16), 4)

 

        assert len(roi_mapping) == 64, \

            f"ROI mapping must have 64 elements, got {len(roi_mapping)}"

        assert np.all((roi_mapping >= 0) & (roi_mapping < 16)), \

            "ROI indices must be 0-15"

 

        coherence_16 = np.zeros((16, 16), dtype=np.complex128)

        counts = np.zeros((16, 16))

 

        for i in range(64):

            for j in range(64):

                ri, rj = roi_mapping[i], roi_mapping[j]

                coherence_16[ri, rj] += plv_matrix[i, j]

                counts[ri, rj] += 1

 

        # Average within each ROI pair

        counts[counts == 0] = 1  # Avoid division by zero

        coherence_16 /= counts

 

        return coherence_16

 

    raise ValueError(

        f"PLV matrix must be 64×64 or 16×16, got {plv_matrix.shape}"

    )

 

 

def compute_plv(

    signals: np.ndarray,

    fs: float,

    freq_band: Tuple[float, float] = (1.0, 50.0)

) -> np.ndarray:

    """

    Compute Phase Locking Value matrix from multichannel signals.

 

    Per ENTPC.tex Section 6.1:

    "The Phase Locking Value (PLV) is computed for each pair of ROIs to

    generate a 16x16 complex coherence matrix for each condition."

 

    Args:

        signals: Array of shape (n_channels, n_samples)

        fs: Sampling frequency in Hz

        freq_band: Frequency band for filtering (default 1-50 Hz per ENTPC.tex)

 

    Returns:

        Complex PLV matrix of shape (n_channels, n_channels)

    """

    from scipy.signal import butter, filtfilt, hilbert

 

    n_channels, n_samples = signals.shape

 

    # Bandpass filter per ENTPC.tex: 1-50 Hz

    nyq = fs / 2

    low = max(freq_band[0] / nyq, 0.001)

    high = min(freq_band[1] / nyq, 0.999)

 

    b, a = butter(4, [low, high], btype='band')

    filtered = np.zeros_like(signals)

    for ch in range(n_channels):

        filtered[ch] = filtfilt(b, a, signals[ch])

 

    # Hilbert transform to get analytic signal

    analytic = np.zeros((n_channels, n_samples), dtype=np.complex128)

    for ch in range(n_channels):

        analytic[ch] = hilbert(filtered[ch])

 

    # Extract instantaneous phase

    phases = np.angle(analytic)

 

    # Compute PLV: PLV_ij = |<e^{i(φ_i - φ_j)}>|

    # We compute the complex average for full phase information

    plv_matrix = np.zeros((n_channels, n_channels), dtype=np.complex128)

 

    for i in range(n_channels):

        for j in range(n_channels):

            phase_diff = phases[i] - phases[j]

            plv_matrix[i, j] = np.mean(np.exp(1j * phase_diff))

 

    return plv_matrix