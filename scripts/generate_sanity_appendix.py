"""
Sanity Appendix Generator
==========================

Generates 5 diagnostic plots to prove metrics are not artifacts:
1. Phase histograms per ROI
2. Amplitude distributions per frequency band
3. Window-by-window variance (phase and amplitude)
4. PAC surrogate tests (real vs null distribution)
5. Regime dwell-time distributions

"""

import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import json

# Set random seed
np.random.seed(42)

# ============================================================================
# PLOT 1: PHASE HISTOGRAMS PER ROI
# ============================================================================

def plot_phase_histograms(data: np.ndarray, fs: float, freq_range: Tuple[float, float],
 output_path: Path, n_rois_plot: int = 4):
 """
 Plot phase histograms for first N ROIs to verify phase coverage.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 freq_range: (low, high) frequency range for bandpass
 output_path: path to save plot
 n_rois_plot: number of ROIs to plot
 """
 print("\n=== Plot 1: Phase Histograms ===")
 
 n_rois = min(n_rois_plot, data.shape[0])
 
 # Bandpass filter
 sos = signal.butter(4, freq_range, btype='band', fs=fs, output='sos')
 
 fig, axes = plt.subplots(2, 2, figsize=(12, 10))
 axes = axes.flatten()
 
 for i in range(n_rois):
 # Filter and extract phase
 filtered = signal.sosfiltfilt(sos, data[i])
 phase = np.angle(signal.hilbert(filtered))
 
 # Plot histogram
 axes[i].hist(phase, bins=50, density=True, alpha=0.7, edgecolor='black')
 axes[i].axhline(1/(2*np.pi), color='r', linestyle='--', label='Uniform')
 axes[i].set_xlabel('Phase (rad)')
 axes[i].set_ylabel('Density')
 axes[i].set_title(f'ROI {i} Phase Distribution')
 axes[i].set_xlim([-np.pi, np.pi])
 axes[i].legend()
 axes[i].grid(True, alpha=0.3)
 
 # Print statistics
 print(f"ROI {i}: mean={np.mean(phase):.3f}, std={np.std(phase):.3f}, unique={len(np.unique(np.round(phase, 3)))}")
 
 plt.suptitle(f'Phase Histograms ({freq_range[0]}-{freq_range[1]} Hz)', fontsize=14, fontweight='bold')
 plt.tight_layout()
 plt.savefig(output_path, dpi=150)
 plt.close()
 
 print(f"✅ Saved to {output_path}")

# ============================================================================
# PLOT 2: AMPLITUDE DISTRIBUTIONS PER FREQUENCY BAND
# ============================================================================

def plot_amplitude_distributions(data: np.ndarray, fs: float,
 freq_bands: dict, output_path: Path, roi_idx: int = 0):
 """
 Plot amplitude distributions for multiple frequency bands.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 freq_bands: dict of {name: (low, high)} frequency bands
 output_path: path to save plot
 roi_idx: ROI index to analyze
 """
 print("\n=== Plot 2: Amplitude Distributions ===")
 
 fig, axes = plt.subplots(2, 2, figsize=(12, 10))
 axes = axes.flatten()
 
 for idx, (band_name, (low, high)) in enumerate(freq_bands.items()):
 if idx >= 4:
 break
 
 # Bandpass filter
 sos = signal.butter(4, [low, high], btype='band', fs=fs, output='sos')
 filtered = signal.sosfiltfilt(sos, data[roi_idx])
 
 # Extract amplitude
 amplitude = np.abs(signal.hilbert(filtered))
 
 # Plot histogram
 axes[idx].hist(amplitude, bins=50, density=True, alpha=0.7, edgecolor='black')
 axes[idx].set_xlabel('Amplitude')
 axes[idx].set_ylabel('Density')
 axes[idx].set_title(f'{band_name} ({low}-{high} Hz)')
 axes[idx].grid(True, alpha=0.3)
 
 # Print statistics
 print(f"{band_name}: mean={np.mean(amplitude):.6e}, std={np.std(amplitude):.6e}, min={amplitude.min():.6e}, max={amplitude.max():.6e}")
 
 plt.suptitle(f'Amplitude Distributions (ROI {roi_idx})', fontsize=14, fontweight='bold')
 plt.tight_layout()
 plt.savefig(output_path, dpi=150)
 plt.close()
 
 print(f"✅ Saved to {output_path}")

# ============================================================================
# PLOT 3: WINDOW-BY-WINDOW VARIANCE
# ============================================================================

def plot_window_variance(data: np.ndarray, fs: float, freq_range: Tuple[float, float],
 output_path: Path, window_length_sec: float = 10.0, roi_idx: int = 0):
 """
 Plot window-by-window variance for phase and amplitude.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 freq_range: (low, high) frequency range
 output_path: path to save plot
 window_length_sec: window length in seconds
 roi_idx: ROI index to analyze
 """
 print("\n=== Plot 3: Window-by-Window Variance ===")
 
 # Bandpass filter
 sos = signal.butter(4, freq_range, btype='band', fs=fs, output='sos')
 filtered = signal.sosfiltfilt(sos, data[roi_idx])
 
 # Extract phase and amplitude
 phase = np.angle(signal.hilbert(filtered))
 amplitude = np.abs(signal.hilbert(filtered))
 
 # Sliding window variance
 window_length_samples = int(window_length_sec * fs)
 overlap = 0.5
 step = int(window_length_samples * (1 - overlap))
 
 n_samples = len(filtered)
 n_windows = (n_samples - window_length_samples) // step + 1
 
 phase_vars = []
 amp_vars = []
 window_centers = []
 
 for i in range(n_windows):
 start = i * step
 end = start + window_length_samples
 
 if end > n_samples:
 break
 
 phase_window = phase[start:end]
 amp_window = amplitude[start:end]
 
 phase_vars.append(np.var(phase_window))
 amp_vars.append(np.var(amp_window))
 window_centers.append((start + end) / 2 / fs)
 
 # Plot
 fig, axes = plt.subplots(2, 1, figsize=(12, 8))
 
 axes[0].plot(window_centers, phase_vars, 'o-', linewidth=2)
 axes[0].axhline(np.mean(phase_vars), color='r', linestyle='--', label=f'Mean = {np.mean(phase_vars):.3f}')
 axes[0].set_ylabel('Phase Variance')
 axes[0].set_title(f'Phase Variance per Window (ROI {roi_idx}, {freq_range[0]}-{freq_range[1]} Hz)')
 axes[0].legend()
 axes[0].grid(True, alpha=0.3)
 
 axes[1].plot(window_centers, amp_vars, 'o-', linewidth=2, color='orange')
 axes[1].axhline(np.mean(amp_vars), color='r', linestyle='--', label=f'Mean = {np.mean(amp_vars):.3e}')
 axes[1].set_xlabel('Time (s)')
 axes[1].set_ylabel('Amplitude Variance')
 axes[1].set_title('Amplitude Variance per Window')
 axes[1].legend()
 axes[1].grid(True, alpha=0.3)
 
 plt.tight_layout()
 plt.savefig(output_path, dpi=150)
 plt.close()
 
 print(f"Phase variance: mean={np.mean(phase_vars):.6f}, min={np.min(phase_vars):.6f}, max={np.max(phase_vars):.6f}")
 print(f"Amplitude variance: mean={np.mean(amp_vars):.6e}, min={np.min(amp_vars):.6e}, max={np.max(amp_vars):.6e}")
 print(f"✅ Saved to {output_path}")

# ============================================================================
# PLOT 4: PAC SURROGATE TESTS
# ============================================================================

def plot_pac_surrogate_tests(data: np.ndarray, fs: float,
 phase_freq: Tuple[float, float],
 amp_freq: Tuple[float, float],
 output_path: Path, n_surrogates: int = 100, roi_idx: int = 0):
 """
 Plot PAC surrogate test: real PAC vs null distribution.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 phase_freq: (low, high) for phase
 amp_freq: (low, high) for amplitude
 output_path: path to save plot
 n_surrogates: number of surrogate iterations
 roi_idx: ROI index to analyze
 """
 print("\n=== Plot 4: PAC Surrogate Tests ===")
 
 # Bandpass filters
 sos_phase = signal.butter(4, phase_freq, btype='band', fs=fs, output='sos')
 sos_amp = signal.butter(4, amp_freq, btype='band', fs=fs, output='sos')
 
 # Filter
 phase_signal = signal.sosfiltfilt(sos_phase, data[roi_idx])
 amp_signal = signal.sosfiltfilt(sos_amp, data[roi_idx])
 
 # Extract phase and amplitude
 phase = np.angle(signal.hilbert(phase_signal))
 amplitude = np.abs(signal.hilbert(amp_signal))
 
 # Real PAC
 pac_real = np.abs(np.mean(amplitude * np.exp(1j * phase)))
 
 # Surrogate PAC (amplitude shuffling)
 pac_surrogates = []
 for _ in range(n_surrogates):
 amp_shuffled = np.random.permutation(amplitude)
 pac_surrogate = np.abs(np.mean(amp_shuffled * np.exp(1j * phase)))
 pac_surrogates.append(pac_surrogate)
 
 pac_surrogates = np.array(pac_surrogates)
 
 # Plot
 fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
 # Histogram
 axes[0].hist(pac_surrogates, bins=30, density=True, alpha=0.7, edgecolor='black', label='Null distribution')
 axes[0].axvline(pac_real, color='r', linewidth=2, label=f'Real PAC = {pac_real:.6f}')
 axes[0].set_xlabel('PAC (MVL)')
 axes[0].set_ylabel('Density')
 axes[0].set_title('PAC: Real vs Null Distribution')
 axes[0].legend()
 axes[0].grid(True, alpha=0.3)
 
 # Percentile plot
 percentile = (np.sum(pac_surrogates < pac_real) / len(pac_surrogates)) * 100
 z_score = (pac_real - np.mean(pac_surrogates)) / np.std(pac_surrogates) if np.std(pac_surrogates) > 0 else 0
 
 axes[1].boxplot(pac_surrogates, vert=True, widths=0.5)
 axes[1].scatter([1], [pac_real], color='r', s=100, zorder=10, label=f'Real PAC (p={100-percentile:.1f}%)')
 axes[1].set_ylabel('PAC (MVL)')
 axes[1].set_title(f'PAC Boxplot (Z={z_score:.2f})')
 axes[1].set_xticks([1])
 axes[1].set_xticklabels(['Null'])
 axes[1].legend()
 axes[1].grid(True, alpha=0.3, axis='y')
 
 plt.suptitle(f'PAC Surrogate Test (ROI {roi_idx})\nPhase: {phase_freq[0]}-{phase_freq[1]} Hz, Amp: {amp_freq[0]}-{amp_freq[1]} Hz',
 fontsize=14, fontweight='bold')
 plt.tight_layout()
 plt.savefig(output_path, dpi=150)
 plt.close()
 
 print(f"Real PAC: {pac_real:.6f}")
 print(f"Null PAC: {np.mean(pac_surrogates):.6f} ± {np.std(pac_surrogates):.6f}")
 print(f"Percentile: {percentile:.1f}%")
 print(f"Z-score: {z_score:.2f}")
 print(f"✅ Saved to {output_path}")

# ============================================================================
# PLOT 5: REGIME DWELL-TIME DISTRIBUTIONS
# ============================================================================

def plot_regime_dwell_times(data: np.ndarray, fs: float, output_path: Path,
 threshold_percentile: float = 50, min_dwell_windows: int = 3):
 """
 Plot regime dwell-time distribution.
 
 Args:
 data: (n_rois, n_samples) array
 fs: sampling rate (Hz)
 output_path: path to save plot
 threshold_percentile: percentile for regime threshold
 min_dwell_windows: minimum dwell time in windows
 """
 print("\n=== Plot 5: Regime Dwell-Time Distributions ===")
 
 # Compute spectral gap time series
 n_rois, n_samples = data.shape
 window_length_sec = 10.0
 window_length_samples = int(window_length_sec * fs)
 overlap = 0.5
 step = int(window_length_samples * (1 - overlap))
 
 spectral_gaps = []
 
 for start in range(0, n_samples - window_length_samples, step):
 end = start + window_length_samples
 window_data = data[:, start:end]
 
 # Covariance matrix
 cov = np.cov(window_data)
 
 # Eigenvalues
 eigenvalues = np.linalg.eigvalsh(cov)
 eigenvalues = np.sort(eigenvalues)[::-1]
 
 # Spectral gap
 if len(eigenvalues) >= 2:
 gap = eigenvalues[0] - eigenvalues[1]
 spectral_gaps.append(gap)
 
 spectral_gaps = np.array(spectral_gaps)
 
 # Threshold
 threshold = np.percentile(spectral_gaps, threshold_percentile)
 
 # Assign regime labels
 regime_labels_raw = (spectral_gaps > threshold).astype(int)
 
 # Apply minimum dwell time filter
 regime_labels = regime_labels_raw.copy()
 current_regime = regime_labels[0]
 dwell_count = 1
 
 for i in range(1, len(regime_labels)):
 if regime_labels_raw[i] == current_regime:
 dwell_count += 1
 else:
 if dwell_count >= min_dwell_windows:
 current_regime = regime_labels_raw[i]
 dwell_count = 1
 else:
 regime_labels[i] = current_regime
 dwell_count += 1
 
 # Compute dwell times
 dwell_times = []
 current_regime = regime_labels[0]
 dwell_count = 1
 
 for i in range(1, len(regime_labels)):
 if regime_labels[i] == current_regime:
 dwell_count += 1
 else:
 dwell_times.append(dwell_count)
 current_regime = regime_labels[i]
 dwell_count = 1
 dwell_times.append(dwell_count) # Last regime
 
 dwell_times = np.array(dwell_times)
 dwell_times_sec = dwell_times * window_length_sec * (1 - overlap)
 
 # Plot
 fig, axes = plt.subplots(2, 1, figsize=(12, 8))
 
 # Spectral gap time series with regime labels
 t = np.arange(len(spectral_gaps)) * window_length_sec * (1 - overlap)
 axes[0].plot(t, spectral_gaps, linewidth=1, alpha=0.7, label='Spectral gap')
 axes[0].axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}th percentile)')
 
 # Color by regime
 for i in range(len(regime_labels)):
 color = 'lightblue' if regime_labels[i] == 0 else 'lightcoral'
 axes[0].axvspan(t[i], t[i] + window_length_sec * (1 - overlap), alpha=0.3, color=color)
 
 axes[0].set_xlabel('Time (s)')
 axes[0].set_ylabel('Spectral Gap')
 axes[0].set_title('Spectral Gap Time Series with Regime Labels')
 axes[0].legend()
 axes[0].grid(True, alpha=0.3)
 
 # Dwell time histogram
 axes[1].hist(dwell_times_sec, bins=20, edgecolor='black', alpha=0.7)
 axes[1].axvline(np.mean(dwell_times_sec), color='r', linestyle='--', linewidth=2, label=f'Mean = {np.mean(dwell_times_sec):.1f}s')
 axes[1].set_xlabel('Dwell Time (seconds)')
 axes[1].set_ylabel('Count')
 axes[1].set_title('Regime Dwell-Time Distribution')
 axes[1].legend()
 axes[1].grid(True, alpha=0.3)
 
 plt.tight_layout()
 plt.savefig(output_path, dpi=150)
 plt.close()
 
 print(f"Number of regimes: {len(dwell_times)}")
 print(f"Dwell times: mean={np.mean(dwell_times_sec):.1f}s, min={np.min(dwell_times_sec):.1f}s, max={np.max(dwell_times_sec):.1f}s")
 print(f"✅ Saved to {output_path}")

# ============================================================================
# MAIN SANITY APPENDIX GENERATOR
# ============================================================================

def generate_sanity_appendix(data_path: Path, output_dir: Path):
 """
 Generate complete sanity appendix with 5 diagnostic plots.
 
 Args:
 data_path: path to MAT file with preprocessed data
 output_dir: directory to save plots
 """
 output_dir.mkdir(parents=True, exist_ok=True)
 
 print("="*80)
 print("SANITY APPENDIX GENERATOR")
 print("="*80)
 
 # Load data
 print(f"\nLoading {data_path.name}...")
 mat = sio.loadmat(data_path)
 data = mat['eeg_data']
 fs = float(mat['fs'][0, 0])
 
 print(f"Data shape: {data.shape}")
 print(f"Sampling rate: {fs} Hz")
 print(f"Duration: {data.shape[1] / fs:.1f} seconds")
 
 # Frequency bands
 freq_bands = {
 'Sub-delta': (0.14, 0.33),
 'Delta': (0.5, 4.0),
 'Theta': (4.0, 8.0),
 'Gamma': (30.0, 50.0)
 }
 
 # Plot 1: Phase histograms
 plot_phase_histograms(data, fs, freq_bands['Sub-delta'],
 output_dir / 'sanity_plot1_phase_histograms.png')
 
 # Plot 2: Amplitude distributions
 plot_amplitude_distributions(data, fs, freq_bands,
 output_dir / 'sanity_plot2_amplitude_distributions.png')
 
 # Plot 3: Window-by-window variance
 plot_window_variance(data, fs, freq_bands['Sub-delta'],
 output_dir / 'sanity_plot3_window_variance.png')
 
 # Plot 4: PAC surrogate tests
 plot_pac_surrogate_tests(data, fs, freq_bands['Sub-delta'], freq_bands['Gamma'],
 output_dir / 'sanity_plot4_pac_surrogate.png')
 
 # Plot 5: Regime dwell-time distributions
 plot_regime_dwell_times(data, fs, output_dir / 'sanity_plot5_regime_dwell_times.png')
 
 print("\n" + "="*80)
 print("SANITY APPENDIX COMPLETE")
 print("="*80)
 print(f"All plots saved to {output_dir}")

if __name__ == '__main__':
 # Example usage
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 output_dir = Path('/home/ubuntu/entptc-implementation/outputs/sanity_appendix')
 
 if data_path.exists():
 generate_sanity_appendix(data_path, output_dir)
 else:
 print(f"Data file not found: {data_path}")
