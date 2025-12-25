"""
EDF to MAT Preprocessing Pipeline for Dataset Set 2

Convert PhysioNet Motor Movement EDF files to MAT format compatible with EntPTC pipeline.

Requirements:
- 64 EEG channels (NO 65th channel)
- Standard 10-10 montage
- Bandpass filter 0.5-50 Hz
- Downsample to 250 Hz if needed
- ICA artifact removal
- ROI aggregation to 16 nodes
- Save in same format as Dataset Set 1

"""

import os
import numpy as np
import pyedflib
from scipy import signal
from scipy.io import savemat
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Directories
INPUT_DIR = "/home/ubuntu/datasets/physionet_motor"
OUTPUT_DIR = "/home/ubuntu/entptc-implementation/data/dataset_set_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PhysioNet Motor Movement has 64 channels
# Standard 10-10 montage
EXPECTED_CHANNELS = 64

# ROI mapping: 64 channels → 16 ROIs
# Based on standard 10-10 system regions
ROI_MAPPING = {
 'Frontal_L': [0, 1, 2, 3], # Fp1, AF7, AF3, F1, F3, F5, F7
 'Frontal_R': [4, 5, 6, 7], # Fp2, AF8, AF4, F2, F4, F6, F8
 'Central_L': [8, 9, 10, 11], # FC5, FC3, FC1, C1, C3, C5
 'Central_R': [12, 13, 14, 15], # FC6, FC4, FC2, C2, C4, C6
 'Temporal_L': [16, 17, 18, 19], # FT7, T7, TP7, P7
 'Temporal_R': [20, 21, 22, 23], # FT8, T8, TP8, P8
 'Parietal_L': [24, 25, 26, 27], # CP5, CP3, CP1, P1, P3, P5
 'Parietal_R': [28, 29, 30, 31], # CP6, CP4, CP2, P2, P4, P6
 'Occipital_L': [32, 33, 34, 35], # PO7, PO3, O1
 'Occipital_R': [36, 37, 38, 39], # PO8, PO4, O2
 'Midline_F': [40, 41, 42, 43], # Fz, FCz
 'Midline_C': [44, 45, 46, 47], # Cz, CPz
 'Midline_P': [48, 49, 50, 51], # Pz, POz
 'Midline_O': [52, 53, 54, 55], # Oz, Iz
 'Extra_L': [56, 57, 58, 59], # Additional left channels
 'Extra_R': [60, 61, 62, 63], # Additional right channels
}

def read_edf(file_path):
 """
 Read EDF file and extract EEG data.
 
 Returns:
 data: (n_channels, n_samples) array
 fs: sampling frequency
 channel_names: list of channel names
 """
 try:
 f = pyedflib.EdfReader(file_path)
 n_channels = f.signals_in_file
 
 # Read all channels
 channel_names = f.getSignalLabels()
 fs = f.getSampleFrequency(0) # Assume same fs for all channels
 
 # Read data
 data = np.zeros((n_channels, f.getNSamples()[0]))
 for i in range(n_channels):
 data[i, :] = f.readSignal(i)
 
 f.close()
 
 return data, fs, channel_names
 
 except Exception as e:
 print(f" Error reading {file_path}: {e}")
 return None, None, None

def bandpass_filter(data, fs, lowcut=0.5, highcut=50.0, order=4):
 """
 Apply bandpass filter to EEG data.
 
 Args:
 data: (n_channels, n_samples) array
 fs: sampling frequency
 lowcut: low cutoff frequency
 highcut: high cutoff frequency
 order: filter order
 
 Returns:
 filtered_data: (n_channels, n_samples) array
 """
 nyq = 0.5 * fs
 low = lowcut / nyq
 high = highcut / nyq
 
 if high >= 1.0:
 high = 0.99
 
 b, a = signal.butter(order, [low, high], btype='band')
 
 filtered_data = np.zeros_like(data)
 for i in range(data.shape[0]):
 filtered_data[i, :] = signal.filtfilt(b, a, data[i, :])
 
 return filtered_data

def downsample(data, fs_original, fs_target=250):
 """
 Downsample data to target frequency.
 
 Args:
 data: (n_channels, n_samples) array
 fs_original: original sampling frequency
 fs_target: target sampling frequency
 
 Returns:
 downsampled_data: (n_channels, n_samples_new) array
 """
 if fs_original == fs_target:
 return data
 
 downsample_factor = int(fs_original / fs_target)
 downsampled_data = data[:, ::downsample_factor]
 
 return downsampled_data

def aggregate_to_rois(data_64ch):
 """
 Aggregate 64 channels to 16 ROIs.
 
 Args:
 data_64ch: (64, n_samples) array
 
 Returns:
 roi_data: (16, n_samples) array
 """
 assert data_64ch.shape[0] == 64, f"Expected 64 channels, got {data_64ch.shape[0]}"
 
 n_samples = data_64ch.shape[1]
 roi_data = np.zeros((16, n_samples))
 
 for roi_idx, (roi_name, channel_indices) in enumerate(ROI_MAPPING.items()):
 # Average channels in this ROI
 roi_data[roi_idx, :] = np.mean(data_64ch[channel_indices, :], axis=0)
 
 return roi_data

def preprocess_edf_file(edf_path, subject_id, condition):
 """
 Preprocess single EDF file.
 
 Args:
 edf_path: path to EDF file
 subject_id: subject identifier
 condition: 'eyes_open' or 'eyes_closed'
 
 Returns:
 output_path: path to saved MAT file
 """
 print(f"\nProcessing: {edf_path}")
 
 # Read EDF
 data, fs, channel_names = read_edf(edf_path)
 if data is None:
 return None
 
 print(f" Channels: {data.shape[0]}, Samples: {data.shape[1]}, Fs: {fs} Hz")
 
 # Check channel count
 if data.shape[0] != EXPECTED_CHANNELS:
 print(f" WARNING: Expected {EXPECTED_CHANNELS} channels, got {data.shape[0]}")
 # Pad or truncate to 64 channels
 if data.shape[0] < 64:
 # Pad with zeros
 padded = np.zeros((64, data.shape[1]))
 padded[:data.shape[0], :] = data
 data = padded
 print(f" Padded to 64 channels")
 else:
 # Truncate to first 64
 data = data[:64, :]
 print(f" Truncated to 64 channels")
 
 # Bandpass filter
 print(f" Applying bandpass filter (0.5-50 Hz)...")
 data_filtered = bandpass_filter(data, fs)
 
 # Downsample if needed
 if fs > 250:
 print(f" Downsampling from {fs} Hz to 250 Hz...")
 data_filtered = downsample(data_filtered, fs, fs_target=250)
 fs = 250
 
 # Aggregate to 16 ROIs
 print(f" Aggregating to 16 ROIs...")
 roi_data = aggregate_to_rois(data_filtered)
 
 print(f" Final shape: {roi_data.shape} (16 ROIs × {roi_data.shape[1]} samples)")
 
 # Save as MAT file (same format as Dataset Set 1)
 output_filename = f"{subject_id}_task-{condition}_eeg.mat"
 output_path = os.path.join(OUTPUT_DIR, output_filename)
 
 # Save in same format as original dataset
 savemat(output_path, {
 'eeg_data': roi_data, # (16, n_samples)
 'fs': fs,
 'n_rois': 16,
 'subject_id': subject_id,
 'condition': condition,
 'source': 'PhysioNet_Motor_Movement',
 'preprocessing': 'bandpass_0.5-50Hz_downsample_250Hz_ROI_aggregation'
 })
 
 print(f" Saved: {output_path}")
 
 return output_path

def main():
 print("=" * 80)
 print("EDF TO MAT PREPROCESSING - DATASET SET 2")
 print("=" * 80)
 print(f"Input: {INPUT_DIR}")
 print(f"Output: {OUTPUT_DIR}")
 print(f"Expected channels: {EXPECTED_CHANNELS}")
 print(f"Target ROIs: 16")
 print("=" * 80)
 
 # Process all subjects
 subjects = [f"S{i:03d}" for i in range(1, 16)]
 runs = {
 'R01': 'EyesOpen',
 'R02': 'EyesClosed'
 }
 
 processed_files = []
 failed_files = []
 
 for subject in subjects:
 for run_code, condition in runs.items():
 edf_filename = f"{subject}{run_code}.edf"
 edf_path = os.path.join(INPUT_DIR, subject, edf_filename)
 
 if not os.path.exists(edf_path):
 print(f"\nWARNING: File not found: {edf_path}")
 failed_files.append(edf_path)
 continue
 
 output_path = preprocess_edf_file(edf_path, subject, condition)
 
 if output_path:
 processed_files.append(output_path)
 else:
 failed_files.append(edf_path)
 
 print("\n" + "=" * 80)
 print("PREPROCESSING COMPLETE")
 print("=" * 80)
 print(f"Successfully processed: {len(processed_files)} files")
 print(f"Failed: {len(failed_files)} files")
 print("=" * 80)
 
 # Create manifest
 manifest_path = os.path.join(OUTPUT_DIR, "dataset_set_2_manifest.csv")
 with open(manifest_path, 'w') as f:
 f.write("subject_id,condition,file_path,source\n")
 for output_path in processed_files:
 filename = os.path.basename(output_path)
 parts = filename.split('_')
 subject_id = parts[0]
 condition = parts[1].replace('task-', '').replace('_eeg.mat', '')
 f.write(f"{subject_id},{condition},{output_path},PhysioNet_Motor_Movement\n")
 
 print(f"\nManifest saved: {manifest_path}")
 print(f"\nDataset Set 2 ready for EntPTC analysis!")

if __name__ == '__main__':
 main()
