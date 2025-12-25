"""
Preprocess PhysioNet Motor Movement/Imagery Dataset (Set 2)

Download and preprocess 15 subjects from the PhysioNet EEG Motor Movement/Imagery Database.
Focus on baseline runs (R01 = eyes open, R02 = eyes closed).

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
Subjects: 109 total, we'll select first 15
Channels: 64 EEG channels (10-10 system)
Sampling rate: 160 Hz
Runs: R01 (eyes open), R02 (eyes closed)

Output: 64-channel preprocessed EEG → 16 ROI aggregation
"""

import os
import numpy as np
import subprocess
from pathlib import Path

# Base URL for PhysioNet
BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0"

# Output directories
DATA_DIR = "/home/ubuntu/datasets/physionet_motor"
OUTPUT_DIR = "/home/ubuntu/entptc-implementation/data/dataset_set_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select 15 subjects
SUBJECTS = [f"S{i:03d}" for i in range(1, 16)] # S001 to S015

# Runs to download
# R01 = Baseline, eyes open
# R02 = Baseline, eyes closed
RUNS = ["R01", "R02"]

def download_file(url, output_path):
 """Download file using wget."""
 if os.path.exists(output_path):
 print(f" Already exists: {output_path}")
 return True
 
 try:
 cmd = f"wget -q -O {output_path} {url}"
 result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
 if result.returncode == 0:
 print(f" Downloaded: {output_path}")
 return True
 else:
 print(f" Failed to download: {url}")
 return False
 except Exception as e:
 print(f" Error downloading {url}: {e}")
 return False

def download_subject_data(subject):
 """Download baseline runs for a subject."""
 print(f"\nDownloading {subject}...")
 subject_dir = os.path.join(DATA_DIR, subject)
 os.makedirs(subject_dir, exist_ok=True)
 
 downloaded_files = []
 
 for run in RUNS:
 # Download EDF file
 edf_filename = f"{subject}{run}.edf"
 edf_url = f"{BASE_URL}/{subject}/{edf_filename}"
 edf_path = os.path.join(subject_dir, edf_filename)
 
 if download_file(edf_url, edf_path):
 downloaded_files.append(edf_path)
 
 return downloaded_files

def create_subject_manifest():
 """Create manifest of all subjects to process."""
 manifest_path = os.path.join(OUTPUT_DIR, "subject_manifest.csv")
 
 with open(manifest_path, 'w') as f:
 f.write("subject_id,run,condition,file_path\n")
 
 for subject in SUBJECTS:
 for run in RUNS:
 condition = "eyes_open" if run == "R01" else "eyes_closed"
 edf_filename = f"{subject}{run}.edf"
 file_path = os.path.join(DATA_DIR, subject, edf_filename)
 f.write(f"{subject},{run},{condition},{file_path}\n")
 
 print(f"\nSubject manifest created: {manifest_path}")
 return manifest_path

def main():
 print("=" * 80)
 print("PHYSIONET MOTOR MOVEMENT DATASET - DOWNLOAD")
 print("=" * 80)
 print(f"Dataset: EEG Motor Movement/Imagery Database")
 print(f"URL: {BASE_URL}")
 print(f"Subjects to download: {len(SUBJECTS)}")
 print(f"Runs per subject: {len(RUNS)} (R01=eyes open, R02=eyes closed)")
 print(f"Total files: {len(SUBJECTS) * len(RUNS)}")
 print("=" * 80)
 
 # Download all subjects
 all_files = []
 for subject in SUBJECTS:
 files = download_subject_data(subject)
 all_files.extend(files)
 
 print("\n" + "=" * 80)
 print(f"DOWNLOAD COMPLETE")
 print(f"Total files downloaded: {len(all_files)}")
 print("=" * 80)
 
 # Create manifest
 manifest_path = create_subject_manifest()
 
 print("\n" + "=" * 80)
 print("NEXT STEPS")
 print("=" * 80)
 print("1. Install pyedflib: pip3 install pyedflib")
 print("2. Run preprocessing script to convert EDF → 64 channels → 16 ROIs")
 print("3. Apply toroidal constraint")
 print("4. Run EntPTC pipeline")
 print("=" * 80)
 
 # Save download log
 log_path = os.path.join(OUTPUT_DIR, "download_log.txt")
 with open(log_path, 'w') as f:
 f.write("PhysioNet Motor Movement Dataset - Download Log\n")
 f.write("=" * 80 + "\n")
 f.write(f"Date: December 24, 2025\n")
 f.write(f"Subjects: {len(SUBJECTS)}\n")
 f.write(f"Files: {len(all_files)}\n")
 f.write("=" * 80 + "\n\n")
 f.write("Downloaded files:\n")
 for file_path in all_files:
 f.write(f" {file_path}\n")
 
 print(f"\nDownload log saved: {log_path}")

if __name__ == '__main__':
 main()
