"""
Fixed PAC Computation with Proper Windowing
============================================

Phase-Amplitude Coupling (PAC) for sub-delta frequencies (0.14-0.33 Hz)
requires long windows: 120-300 seconds minimum.

Implements:
1. PAC with variable window lengths (sweep 60-300 sec)
2. Sanity checks for phase/amplitude extraction
3. Null distribution via surrogate data
4. PAC vs window length curve

"""

import numpy as np
import scipy.signal as signal
from scipy.stats import circmean
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict

# Set random seed
np.random.seed(42)

# ============================================================================
# PAC COMPUTATION WITH PROPER WINDOWING
# ============================================================================

def compute_pac_with_windowing(data: np.ndarray, fs: float,
 phase_freq: Tuple[float, float] = (0.14, 0.33),
 amp_freq: Tuple[float, float] = (30, 50),
 window_lengths: list = None) -> Dict:
 """
 Compute PAC with multiple window lengths.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 phase_freq: (low, high) for phase-providing frequency
 amp_freq: (low, high) for amplitude-providing frequency
 window_lengths: list of window lengths in seconds
 Default: [60, 120, 180, 240, 300]
 
 Returns:
 results: dict with PAC values for each window length
 """
 if window_lengths is None:
 window_lengths = [60, 120, 180, 240, 300]
 
 n_rois, n_samples = data.shape
 duration_sec = n_samples / fs
 
 print(f"Data duration: {duration_sec:.1f} seconds")
 print(f"Phase frequency: {phase_freq[0]}-{phase_freq[1]} Hz")
 print(f"Amplitude frequency: {amp_freq[0]}-{amp_freq[1]} Hz")
 print(f"Window lengths to test: {window_lengths} seconds")
 
 results = {
 'window_lengths': window_lengths,
 'pac_values': [],
 'pac_null_values': [],
 'pac_std_values': [],
 'n_windows': [],
 'cycles_per_window': []
 }
 
 for window_length_sec in window_lengths:
 # Check if data is long enough
 if window_length_sec > duration_sec:
 print(f"\n⚠️ Window length {window_length_sec}s exceeds data duration {duration_sec:.1f}s - skipping")
 results['pac_values'].append(np.nan)
 results['pac_null_values'].append(np.nan)
 results['pac_std_values'].append(np.nan)
 results['n_windows'].append(0)
 results['cycles_per_window'].append(0)
 continue
 
 # Compute cycles per window
 cycles = window_length_sec * phase_freq[0]
 results['cycles_per_window'].append(cycles)
 
 print(f"\n--- Window length: {window_length_sec}s ({cycles:.1f} cycles at {phase_freq[0]} Hz) ---")
 
 # Compute PAC
 pac, pac_null, pac_std, n_windows = _compute_pac_single_window(
 data, fs, phase_freq, amp_freq, window_length_sec
 )
 
 results['pac_values'].append(pac)
 results['pac_null_values'].append(pac_null)
 results['pac_std_values'].append(pac_std)
 results['n_windows'].append(n_windows)
 
 print(f"PAC: {pac:.6f}")
 print(f"PAC null: {pac_null:.6f} ± {pac_std:.6f}")
 print(f"Number of windows: {n_windows}")
 
 if pac_null > 0:
 z_score = (pac - pac_null) / pac_std if pac_std > 0 else 0
 print(f"Z-score: {z_score:.2f}")
 
 return results

def _compute_pac_single_window(data: np.ndarray, fs: float,
 phase_freq: Tuple[float, float],
 amp_freq: Tuple[float, float],
 window_length_sec: float) -> Tuple[float, float, float, int]:
 """
 Compute PAC for a single window length.
 
 Returns:
 pac: mean PAC across ROIs and windows
 pac_null: mean null PAC
 pac_std: std of null PAC
 n_windows: number of windows used
 """
 n_rois, n_samples = data.shape
 window_length_samples = int(window_length_sec * fs)
 overlap = 0.5
 step = int(window_length_samples * (1 - overlap))
 
 # Bandpass filters
 sos_phase = signal.butter(4, phase_freq, btype='band', fs=fs, output='sos')
 sos_amp = signal.butter(4, amp_freq, btype='band', fs=fs, output='sos')
 
 pac_values = []
 pac_null_values = []
 
 for roi in range(n_rois):
 # Filter
 phase_signal = signal.sosfiltfilt(sos_phase, data[roi])
 amp_signal = signal.sosfiltfilt(sos_amp, data[roi])
 
 # Extract phase and amplitude
 phase = np.angle(signal.hilbert(phase_signal))
 amplitude = np.abs(signal.hilbert(amp_signal))
 
 # Sanity check: phase and amplitude should have variance
 if np.var(phase) == 0 or np.var(amplitude) == 0:
 print(f" ⚠️ ROI {roi}: Zero variance in phase or amplitude")
 continue
 
 # Compute PAC in sliding windows
 for start in range(0, n_samples - window_length_samples, step):
 end = start + window_length_samples
 
 phase_window = phase[start:end]
 amp_window = amplitude[start:end]
 
 # PAC: Mean Vector Length (MVL)
 pac = np.abs(np.mean(amp_window * np.exp(1j * phase_window)))
 pac_values.append(pac)
 
 # Null PAC: shuffle amplitude
 amp_shuffled = np.random.permutation(amp_window)
 pac_null = np.abs(np.mean(amp_shuffled * np.exp(1j * phase_window)))
 pac_null_values.append(pac_null)
 
 if len(pac_values) == 0:
 return 0.0, 0.0, 0.0, 0
 
 pac_mean = np.mean(pac_values)
 pac_null_mean = np.mean(pac_null_values)
 pac_null_std = np.std(pac_null_values)
 n_windows = len(pac_values) // n_rois if n_rois > 0 else 0
 
 return pac_mean, pac_null_mean, pac_null_std, n_windows

# ============================================================================
# SANITY CHECKS FOR PHASE/AMPLITUDE EXTRACTION
# ============================================================================

def sanity_check_pac_pipeline(data: np.ndarray, fs: float,
 phase_freq: Tuple[float, float] = (0.14, 0.33),
 amp_freq: Tuple[float, float] = (30, 50),
 roi_idx: int = 0,
 output_dir: Path = None):
 """
 Sanity check PAC pipeline with visualizations.
 
 Generates plots:
 1. Raw signal
 2. Filtered signals (phase and amplitude bands)
 3. Extracted phase and amplitude
 4. Phase-amplitude histogram
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 phase_freq: (low, high) for phase
 amp_freq: (low, high) for amplitude
 roi_idx: ROI index to visualize
 output_dir: directory to save plots
 """
 print(f"\n=== PAC Pipeline Sanity Check (ROI {roi_idx}) ===")
 
 # Extract ROI signal
 signal_raw = data[roi_idx]
 
 # Bandpass filters
 sos_phase = signal.butter(4, phase_freq, btype='band', fs=fs, output='sos')
 sos_amp = signal.butter(4, amp_freq, btype='band', fs=fs, output='sos')
 
 # Filter
 signal_phase = signal.sosfiltfilt(sos_phase, signal_raw)
 signal_amp = signal.sosfiltfilt(sos_amp, signal_raw)
 
 # Extract phase and amplitude
 phase = np.angle(signal.hilbert(signal_phase))
 amplitude = np.abs(signal.hilbert(signal_amp))
 
 # Print statistics
 print(f"Raw signal: min={signal_raw.min():.6e}, max={signal_raw.max():.6e}, std={signal_raw.std():.6e}")
 print(f"Phase signal: min={signal_phase.min():.6e}, max={signal_phase.max():.6e}, std={signal_phase.std():.6e}")
 print(f"Amp signal: min={signal_amp.min():.6e}, max={signal_amp.max():.6e}, std={signal_amp.std():.6e}")
 print(f"Phase: min={phase.min():.3f}, max={phase.max():.3f}, std={phase.std():.3f}")
 print(f"Amplitude: min={amplitude.min():.6e}, max={amplitude.max():.6e}, std={amplitude.std():.6e}")
 
 # Check for artifacts
 if signal_phase.std() == 0:
 print("❌ ERROR: Phase signal has zero variance (filter artifact?)")
 if signal_amp.std() == 0:
 print("❌ ERROR: Amplitude signal has zero variance (filter artifact?)")
 if phase.std() == 0:
 print("❌ ERROR: Phase has zero variance (Hilbert artifact?)")
 if amplitude.std() == 0:
 print("❌ ERROR: Amplitude has zero variance (Hilbert artifact?)")
 
 # Visualize
 if output_dir is not None:
 output_dir.mkdir(parents=True, exist_ok=True)
 
 fig, axes = plt.subplots(4, 1, figsize=(12, 10))
 
 # Plot first 10 seconds
 t_max = min(10, len(signal_raw) / fs)
 n_samples_plot = int(t_max * fs)
 t = np.arange(n_samples_plot) / fs
 
 # Raw signal
 axes[0].plot(t, signal_raw[:n_samples_plot])
 axes[0].set_title(f'Raw Signal (ROI {roi_idx})')
 axes[0].set_ylabel('Amplitude')
 
 # Filtered signals
 axes[1].plot(t, signal_phase[:n_samples_plot], label=f'Phase band ({phase_freq[0]}-{phase_freq[1]} Hz)')
 axes[1].plot(t, signal_amp[:n_samples_plot], label=f'Amp band ({amp_freq[0]}-{amp_freq[1]} Hz)', alpha=0.7)
 axes[1].set_title('Filtered Signals')
 axes[1].set_ylabel('Amplitude')
 axes[1].legend()
 
 # Phase and amplitude
 axes[2].plot(t, phase[:n_samples_plot], label='Phase')
 axes[2].set_title('Extracted Phase')
 axes[2].set_ylabel('Phase (rad)')
 axes[2].set_ylim([-np.pi, np.pi])
 
 axes[3].plot(t, amplitude[:n_samples_plot], label='Amplitude', color='orange')
 axes[3].set_title('Extracted Amplitude')
 axes[3].set_ylabel('Amplitude')
 axes[3].set_xlabel('Time (s)')
 
 plt.tight_layout()
 plot_path = output_dir / f'pac_sanity_check_roi{roi_idx}.png'
 plt.savefig(plot_path, dpi=150)
 plt.close()
 
 print(f"✅ Sanity check plot saved to {plot_path}")

# ============================================================================
# PAC VS WINDOW LENGTH CURVE
# ============================================================================

def plot_pac_vs_window_length(results: Dict, output_dir: Path):
 """
 Plot PAC vs window length curve.
 
 Args:
 results: dict from compute_pac_with_windowing
 output_dir: directory to save plot
 """
 output_dir.mkdir(parents=True, exist_ok=True)
 
 window_lengths = results['window_lengths']
 pac_values = results['pac_values']
 pac_null_values = results['pac_null_values']
 cycles_per_window = results['cycles_per_window']
 
 fig, axes = plt.subplots(2, 1, figsize=(10, 8))
 
 # PAC vs window length
 axes[0].plot(window_lengths, pac_values, 'o-', label='PAC (real)', linewidth=2)
 axes[0].plot(window_lengths, pac_null_values, 's--', label='PAC (null)', alpha=0.7)
 axes[0].axhline(0, color='k', linestyle=':', alpha=0.3)
 axes[0].set_xlabel('Window Length (seconds)')
 axes[0].set_ylabel('PAC (MVL)')
 axes[0].set_title('Phase-Amplitude Coupling vs Window Length')
 axes[0].legend()
 axes[0].grid(True, alpha=0.3)
 
 # PAC vs cycles per window
 axes[1].plot(cycles_per_window, pac_values, 'o-', label='PAC (real)', linewidth=2)
 axes[1].plot(cycles_per_window, pac_null_values, 's--', label='PAC (null)', alpha=0.7)
 axes[1].axvline(3, color='r', linestyle='--', alpha=0.5, label='3 cycles (minimum)')
 axes[1].axhline(0, color='k', linestyle=':', alpha=0.3)
 axes[1].set_xlabel('Cycles per Window')
 axes[1].set_ylabel('PAC (MVL)')
 axes[1].set_title('PAC vs Cycles per Window')
 axes[1].legend()
 axes[1].grid(True, alpha=0.3)
 
 plt.tight_layout()
 plot_path = output_dir / 'pac_vs_window_length.png'
 plt.savefig(plot_path, dpi=150)
 plt.close()
 
 print(f"✅ PAC vs window length plot saved to {plot_path}")

if __name__ == '__main__':
 print("Fixed PAC Computation with Proper Windowing")
 print("Use: from stage_c_pac_fixed import compute_pac_with_windowing")
