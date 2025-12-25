#!/usr/bin/env python3.11
"""
extract_cohort.py - Reproducible extraction of 40-subject cohort from ds005385

Usage:
 python3.11 extract_cohort.py /path/to/ds005385

Outputs:
 - cohort_40_manifest.csv
 - subject_summary.csv
 - validation_report.md

Requirements:
 - Python 3.11+
 - ds005385 dataset cloned via datalad with EDF files extracted
"""

import os
import sys
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

def compute_sha256(filepath):
 """Compute SHA256 checksum of a file."""
 sha256_hash = hashlib.sha256()
 with open(filepath, 'rb') as f:
 for chunk in iter(lambda: f.read(8192), b''):
 sha256_hash.update(chunk)
 return sha256_hash.hexdigest()

def read_edf_header(filepath):
 """Extract metadata from EDF header."""
 metadata = {
 'duration': '',
 'sampling_rate': '',
 'channels_count': ''
 }
 
 try:
 with open(filepath, 'rb') as f:
 header = f.read(256)
 
 if len(header) >= 256:
 # Number of signals (channels)
 num_signals = int(header[252:256].decode('ascii').strip())
 metadata['channels_count'] = str(num_signals)
 
 # Number of data records
 num_records = int(header[236:244].decode('ascii').strip())
 
 # Duration of each record
 record_duration = float(header[244:252].decode('ascii').strip())
 
 # Total duration
 metadata['duration'] = str(num_records * record_duration)
 
 # Read signal headers for sampling rate
 signal_header_start = 256
 f.seek(signal_header_start + num_signals * 216)
 samples_per_record = f.read(8).decode('ascii', errors='ignore').strip()
 if samples_per_record:
 metadata['sampling_rate'] = str(int(samples_per_record) / record_duration)
 except Exception as e:
 print(f"Warning: Could not read EDF header for {filepath}: {e}")
 
 return metadata

def main(dataset_path):
 """Extract 40-subject cohort from ds005385."""
 
 dataset_path = Path(dataset_path)
 if not dataset_path.exists():
 print(f"Error: Dataset path does not exist: {dataset_path}")
 sys.exit(1)
 
 print("=" * 80)
 print("EXTRACTING 40-SUBJECT COHORT FROM ds005385")
 print("=" * 80)
 
 # Load participants.tsv
 print("\n[1/6] Loading participants...")
 participants_file = dataset_path / "participants.tsv"
 if not participants_file.exists():
 print(f"Error: participants.tsv not found at {participants_file}")
 sys.exit(1)
 
 participants = {}
 with open(participants_file, 'r') as f:
 reader = csv.DictReader(f, delimiter='\t')
 for row in reader:
 participants[row['participant_id']] = row
 
 print(f" Loaded {len(participants)} participants")
 
 # Find subjects with both sessions
 print("\n[2/6] Finding subjects with both sessions...")
 subjects_with_both = []
 for subj_id, data in participants.items():
 if data.get('session1') == 'yes' and data.get('session2') == 'yes':
 subjects_with_both.append(subj_id)
 
 print(f" Found {len(subjects_with_both)} subjects with both sessions")
 
 # Select 40-subject cohort (deterministic)
 cohort_subjects = sorted(subjects_with_both)[:40]
 print(f" Selected {len(cohort_subjects)} subjects for cohort")
 
 # Walk dataset to find EDF files
 print("\n[3/6] Scanning for EDF files...")
 edf_files = []
 for subject_dir in dataset_path.glob("sub-*"):
 if not subject_dir.is_dir():
 continue
 
 subject_id = subject_dir.name
 if subject_id not in cohort_subjects:
 continue
 
 for edf_file in subject_dir.rglob("*.edf"):
 if edf_file.is_file():
 edf_files.append(edf_file)
 
 print(f" Found {len(edf_files)} EDF files for cohort")
 
 # Build manifest
 print("\n[4/6] Building manifest...")
 manifest_records = []
 
 for edf_path in edf_files:
 # Parse BIDS filename
 # Format: sub-XXX_ses-X_task-XXXX_acq-XXX_eeg.edf
 filename = edf_path.name
 parts = filename.replace('.edf', '').split('_')
 
 subject_id = None
 session_id = None
 task = None
 acq = None
 
 for part in parts:
 if part.startswith('sub-'):
 subject_id = part
 elif part.startswith('ses-'):
 session_id = part
 elif part.startswith('task-'):
 task = part.replace('task-', '')
 elif part.startswith('acq-'):
 acq = part.replace('acq-', '')
 
 if not subject_id or not session_id:
 print(f" Warning: Could not parse {filename}")
 continue
 
 # Determine pre/post
 if acq == 'pre':
 pre_or_post = 'pre'
 treatment_label = 'baseline'
 elif acq == 'post':
 pre_or_post = 'post'
 treatment_label = 'post_treatment'
 else:
 pre_or_post = 'UNKNOWN'
 treatment_label = 'UNKNOWN'
 
 # Compute checksum
 edf_sha256 = compute_sha256(edf_path)
 
 # Get file size
 edf_bytes = edf_path.stat().st_size
 
 # Read EDF metadata
 edf_metadata = read_edf_header(edf_path)
 
 # Get relative path
 edf_relpath = str(edf_path.relative_to(dataset_path))
 
 # Get participant data
 part_data = participants.get(subject_id, {})
 
 # Build record
 record = {
 'subject_id': subject_id,
 'session_id': session_id,
 'condition': pre_or_post,
 'pre_or_post': pre_or_post,
 'treatment_label': treatment_label,
 'recording_modality': 'EEG',
 'task': task,
 'run': '1',
 'edf_filename': filename,
 'edf_relpath': edf_relpath,
 'edf_sha256': edf_sha256,
 'edf_bytes': edf_bytes,
 'start_time_if_available': '',
 'duration_if_available': edf_metadata['duration'],
 'sampling_rate_if_available': edf_metadata['sampling_rate'],
 'channels_count_if_available': edf_metadata['channels_count'],
 'dataset_source': 'openneuro',
 'dataset_id': 'ds005385',
 'notes': f"Age={part_data.get('age', 'N/A')}, Sex={part_data.get('sex', 'N/A')}"
 }
 
 manifest_records.append(record)
 
 print(f" Created {len(manifest_records)} manifest records")
 
 # Save manifest
 print("\n[5/6] Saving cohort_40_manifest.csv...")
 with open('cohort_40_manifest.csv', 'w', newline='') as f:
 fieldnames = [
 'subject_id', 'session_id', 'condition', 'pre_or_post', 'treatment_label',
 'recording_modality', 'task', 'run', 'edf_filename', 'edf_relpath',
 'edf_sha256', 'edf_bytes', 'start_time_if_available', 'duration_if_available',
 'sampling_rate_if_available', 'channels_count_if_available',
 'dataset_source', 'dataset_id', 'notes'
 ]
 writer = csv.DictWriter(f, fieldnames=fieldnames)
 writer.writeheader()
 writer.writerows(manifest_records)
 
 print(" ✅ Saved cohort_40_manifest.csv")
 
 # Create subject summary
 print("\n[6/6] Creating subject_summary.csv...")
 subject_data = defaultdict(lambda: {'pre': [], 'post': []})
 for rec in manifest_records:
 subject = rec['subject_id']
 condition = rec['pre_or_post']
 subject_data[subject][condition].append(rec)
 
 summary_records = []
 for subject in sorted(subject_data.keys()):
 pre_files = subject_data[subject]['pre']
 post_files = subject_data[subject]['post']
 
 has_pre = 'yes' if len(pre_files) > 0 else 'no'
 has_post = 'yes' if len(post_files) > 0 else 'no'
 
 pre_selected = sorted(pre_files, key=lambda x: x['task'])[0] if pre_files else None
 post_selected = sorted(post_files, key=lambda x: x['task'])[0] if post_files else None
 
 flags = []
 if not pre_selected:
 flags.append('MISSING_PRE')
 if not post_selected:
 flags.append('MISSING_POST')
 if len(pre_files) > 1:
 flags.append(f'MULTIPLE_PRE({len(pre_files)})')
 if len(post_files) > 1:
 flags.append(f'MULTIPLE_POST({len(post_files)})')
 
 summary_record = {
 'subject_id': subject,
 'has_pre': has_pre,
 'has_post': has_post,
 'pre_file': pre_selected['edf_filename'] if pre_selected else '',
 'post_file': post_selected['edf_filename'] if post_selected else '',
 'pre_sha256': pre_selected['edf_sha256'] if pre_selected else '',
 'post_sha256': post_selected['edf_sha256'] if post_selected else '',
 'pre_duration': pre_selected['duration_if_available'] if pre_selected else '',
 'post_duration': post_selected['duration_if_available'] if post_selected else '',
 'pre_sampling_rate': pre_selected['sampling_rate_if_available'] if pre_selected else '',
 'post_sampling_rate': post_selected['sampling_rate_if_available'] if post_selected else '',
 'channel_count_pre': pre_selected['channels_count_if_available'] if pre_selected else '',
 'channel_count_post': post_selected['channels_count_if_available'] if post_selected else '',
 'flags': '; '.join(flags) if flags else 'OK',
 'exclusion_reason_if_any': ''
 }
 
 summary_records.append(summary_record)
 
 with open('subject_summary.csv', 'w', newline='') as f:
 fieldnames = [
 'subject_id', 'has_pre', 'has_post', 'pre_file', 'post_file',
 'pre_sha256', 'post_sha256', 'pre_duration', 'post_duration',
 'pre_sampling_rate', 'post_sampling_rate',
 'channel_count_pre', 'channel_count_post',
 'flags', 'exclusion_reason_if_any'
 ]
 writer = csv.DictWriter(f, fieldnames=fieldnames)
 writer.writeheader()
 writer.writerows(summary_records)
 
 print(" ✅ Saved subject_summary.csv")
 
 print("\n" + "=" * 80)
 print("EXTRACTION COMPLETE")
 print("=" * 80)
 print(f"Cohort subjects: {len(cohort_subjects)}")
 print(f"Total EDF files: {len(manifest_records)}")
 print(f"Outputs: cohort_40_manifest.csv, subject_summary.csv")
 print()

if __name__ == "__main__":
 if len(sys.argv) != 2:
 print("Usage: python3.11 extract_cohort.py /path/to/ds005385")
 sys.exit(1)
 
 main(sys.argv[1])
