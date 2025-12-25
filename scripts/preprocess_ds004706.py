#!/usr/bin/env python3.11
"""
Preprocess ds004706 Spatial Navigation EEG Data
================================================

Preprocesses BDF files from OpenNeuro ds004706 dataset:
1. Load BDF files using MNE-Python
2. Extract 64-channel EEG
3. Downsample to 160 Hz
4. Bandpass filter 0.5-50 Hz
5. Aggregate to 16 ROIs (4x4 toroidal grid)
6. Extract task events (navigation, object encoding, recall)
7. Save to MAT format for Stage C analysis

"""

import mne
import numpy as np
import scipy.io
from pathlib import Path
import json
import pandas as pd

# Paths
DATA_DIR = Path("/home/ubuntu/entptc-implementation/stage_c_datasets/ds004706_data")
OUTPUT_DIR = Path("/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
TARGET_FS = 160 # Hz (match Stage A/B/C1 pipeline)
LOWCUT = 0.5 # Hz
HIGHCUT = 50.0 # Hz

# 64-channel to 16-ROI mapping (same as PhysioNet preprocessing)
# Standard 10-20 system aggregation
ROI_MAPPING = {
 0: [0, 1, 2, 3], # Frontal Left
 1: [4, 5, 6, 7], # Frontal Center-Left
 2: [8, 9, 10, 11], # Frontal Center-Right
 3: [12, 13, 14, 15], # Frontal Right
 4: [16, 17, 18, 19], # Central Left
 5: [20, 21, 22, 23], # Central Center-Left
 6: [24, 25, 26, 27], # Central Center-Right
 7: [28, 29, 30, 31], # Central Right
 8: [32, 33, 34, 35], # Parietal Left
 9: [36, 37, 38, 39], # Parietal Center-Left
 10: [40, 41, 42, 43], # Parietal Center-Right
 11: [44, 45, 46, 47], # Parietal Right
 12: [48, 49, 50, 51], # Occipital Left
 13: [52, 53, 54, 55], # Occipital Center-Left
 14: [56, 57, 58, 59], # Occipital Center-Right
 15: [60, 61, 62, 63], # Occipital Right
}

def load_bdf_file(bdf_path, max_duration_minutes=10):
 """Load BDF file and extract EEG data (first N minutes only to save memory)"""
 print(f"Loading {bdf_path.name} (first {max_duration_minutes} minutes)...")
 
 # Load raw BDF file WITHOUT preloading (streaming mode)
 raw = mne.io.read_raw_bdf(bdf_path, preload=False, verbose=False)
 
 # Get sampling rate
 fs = raw.info['sfreq']
 print(f" Original sampling rate: {fs} Hz")
 
 # Pick EEG channels only (exclude external channels)
 raw.pick_types(eeg=True, exclude='bads')
 
 # Get number of channels
 n_channels = len(raw.ch_names)
 print(f" Number of EEG channels: {n_channels}")
 
 # If more than 64 channels, take first 64
 if n_channels > 64:
 raw.pick_channels(raw.ch_names[:64])
 print(f" Selected first 64 channels")
 
 # Calculate how many samples to load (first N minutes)
 max_samples = int(max_duration_minutes * 60 * fs)
 
 # Load only first N minutes
 print(f" Loading first {max_duration_minutes} minutes ({max_samples} samples)...")
 raw.crop(tmax=max_duration_minutes * 60)
 
 # NOW preload the cropped data
 raw.load_data(verbose=False)
 
 # Bandpass filter
 print(f" Bandpass filtering {LOWCUT}-{HIGHCUT} Hz...")
 raw.filter(LOWCUT, HIGHCUT, fir_design='firwin', verbose=False)
 
 # Resample to target sampling rate
 if fs != TARGET_FS:
 print(f" Resampling to {TARGET_FS} Hz...")
 raw.resample(TARGET_FS, npad='auto', verbose=False)
 
 # Get data
 eeg_data = raw.get_data() # Shape: (n_channels, n_samples)
 
 # Extract events if available
 events = None
 try:
 events, event_id = mne.events_from_annotations(raw, verbose=False)
 print(f" Found {len(events)} events")
 except:
 print(f" No events found in BDF file")
 
 return eeg_data, TARGET_FS, events, raw.ch_names

def aggregate_to_rois(eeg_64ch):
 """Aggregate 64 channels to 16 ROIs"""
 n_samples = eeg_64ch.shape[1]
 eeg_16roi = np.zeros((16, n_samples))
 
 for roi_idx, ch_indices in ROI_MAPPING.items():
 # Handle case where the analysis fewer than 64 channels
 valid_indices = [i for i in ch_indices if i < eeg_64ch.shape[0]]
 if valid_indices:
 eeg_16roi[roi_idx, :] = np.mean(eeg_64ch[valid_indices, :], axis=0)
 
 return eeg_16roi

def extract_task_events(bdf_path, events):
 """Extract task events from accompanying TSV files"""
 # Look for events TSV file
 events_tsv = bdf_path.parent / bdf_path.name.replace('_eeg.bdf', '_events.tsv')
 
 task_events = {
 'navigation_starts': [],
 'object_encodings': [],
 'location_recalls': [],
 'regime_transitions': []
 }
 
 if events_tsv.exists():
 print(f" Loading task events from {events_tsv.name}...")
 df = pd.read_csv(events_tsv, sep='\t')
 
 # Extract relevant events (adapt based on actual event structure)
 if 'trial_type' in df.columns:
 task_events['navigation_starts'] = df[df['trial_type'].str.contains('nav', case=False, na=False)]['onset'].tolist()
 task_events['object_encodings'] = df[df['trial_type'].str.contains('encode', case=False, na=False)]['onset'].tolist()
 task_events['location_recalls'] = df[df['trial_type'].str.contains('recall', case=False, na=False)]['onset'].tolist()
 
 print(f" Navigation starts: {len(task_events['navigation_starts'])}")
 print(f" Object encodings: {len(task_events['object_encodings'])}")
 print(f" Location recalls: {len(task_events['location_recalls'])}")
 else:
 print(f" No events TSV file found")
 
 return task_events

def process_single_file(bdf_path):
 """Process a single BDF file"""
 try:
 # Load BDF file
 eeg_64ch, fs, events, ch_names = load_bdf_file(bdf_path)
 
 # Aggregate to 16 ROIs
 print(f" Aggregating to 16 ROIs...")
 eeg_16roi = aggregate_to_rois(eeg_64ch)
 
 # Extract task events
 task_events = extract_task_events(bdf_path, events)
 
 # Prepare output filename
 subject = bdf_path.parts[-4] # e.g., sub-LTP448
 session = bdf_path.parts[-3] # e.g., ses-0
 output_filename = f"{subject}_{session}_task-SpatialNav_eeg.mat"
 output_path = OUTPUT_DIR / output_filename
 
 # Save to MAT file
 print(f" Saving to {output_filename}...")
 scipy.io.savemat(
 output_path,
 {
 'eeg_data': eeg_16roi, # Shape: (16 ROIs, n_samples)
 'fs': fs,
 'n_rois': 16,
 'subject_id': subject,
 'session_id': session,
 'task': 'SpatialNavigation',
 'source': 'OpenNeuro_ds004706',
 'preprocessing': {
 'bandpass': f'{LOWCUT}-{HIGHCUT} Hz',
 'sampling_rate': f'{fs} Hz',
 'roi_aggregation': '64ch -> 16 ROIs (4x4 grid)',
 'original_channels': len(ch_names)
 },
 'task_events': task_events,
 'duration_seconds': eeg_16roi.shape[1] / fs
 }
 )
 
 print(f" ✅ Success! Duration: {eeg_16roi.shape[1]/fs:.1f} seconds")
 return True, output_path
 
 except Exception as e:
 print(f" ❌ Error: {e}")
 return False, None

def main():
 print("="*80)
 print("DS004706 SPATIAL NAVIGATION EEG PREPROCESSING")
 print("="*80)
 
 # Find all ReadOnly BDF files
 bdf_files = sorted(DATA_DIR.glob("sub-*/ses-*/eeg/*ReadOnly*.bdf"))
 
 print(f"\nFound {len(bdf_files)} BDF files to process\n")
 
 results = []
 for i, bdf_file in enumerate(bdf_files, 1):
 print(f"[{i}/{len(bdf_files)}] Processing {bdf_file.name}")
 success, output_path = process_single_file(bdf_file)
 results.append({
 'input': str(bdf_file),
 'output': str(output_path) if output_path else None,
 'success': success
 })
 print()
 
 # Summary
 print("="*80)
 print("PREPROCESSING SUMMARY")
 print("="*80)
 n_success = sum(1 for r in results if r['success'])
 print(f"Successfully processed: {n_success}/{len(results)} files")
 print(f"Output directory: {OUTPUT_DIR}")
 print(f"Total output size: {sum(p.stat().st_size for p in OUTPUT_DIR.glob('*.mat')) / 1e9:.2f} GB")
 print("="*80)
 
 # Save processing log
 log_path = OUTPUT_DIR / "preprocessing_log.json"
 with open(log_path, 'w') as f:
 json.dump({
 'date': '2025-12-24',
 'n_files': len(results),
 'n_success': n_success,
 'parameters': {
 'target_fs': TARGET_FS,
 'lowcut': LOWCUT,
 'highcut': HIGHCUT,
 'n_rois': 16
 },
 'results': results
 }, f, indent=2)
 
 print(f"\nProcessing log saved to {log_path}")

if __name__ == "__main__":
 main()
