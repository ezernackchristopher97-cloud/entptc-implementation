#!/usr/bin/env python3
"""Test script to process a single file and identify issues."""

import numpy as np
import h5py
import sys
sys.path.insert(0, '/home/ubuntu/entptc-implementation')

from entptc.core.progenitor import ProgenitorMatrix

# Load one file
mat_file = 'data/sub-001_ses-1_task-EyesClosed_acq-post_eeg.mat'
print(f'Loading {mat_file}...')

with h5py.File(mat_file, 'r') as f:
 data_matrix = np.array(f['data_matrix'])

print(f'Original shape: {data_matrix.shape}')

# Transpose if needed
if data_matrix.shape[0] > data_matrix.shape[1]:
 data_matrix = data_matrix.T

print(f'After transpose: {data_matrix.shape}')
assert data_matrix.shape[0] == 64, f"Expected 64 channels, got {data_matrix.shape[0]}"

# Aggregate to 16 ROIs
ROI_MAP = {
 0: [0, 1, 2, 3], 1: [4, 5, 6, 7], 2: [8, 9, 10, 11], 3: [12, 13, 14, 15],
 4: [16, 17, 18, 19], 5: [20, 21, 22, 23], 6: [24, 25, 26, 27], 7: [28, 29, 30, 31],
 8: [32, 33, 34, 35], 9: [36, 37, 38, 39], 10: [40, 41, 42, 43], 11: [44, 45, 46, 47],
 12: [48, 49, 50, 51], 13: [52, 53, 54, 55], 14: [56, 57, 58, 59], 15: [60, 61, 62, 63]
}

n_rois = 16
n_timepoints = data_matrix.shape[1]
roi_data = np.zeros((n_rois, n_timepoints))

for roi_idx, channel_indices in ROI_MAP.items():
 roi_data[roi_idx, :] = np.mean(data_matrix[channel_indices, :], axis=0)

print(f'ROI data shape: {roi_data.shape}')

# Build Progenitor Matrix
print('Building Progenitor Matrix...')
progenitor = ProgenitorMatrix()

try:
 P = progenitor.construct_from_eeg_data(roi_data)
 print(f'Progenitor Matrix shape: {P.shape}')
 print(f'Progenitor Matrix norm: {np.linalg.norm(P):.4f}')
 print('SUCCESS!')
except Exception as e:
 print(f'ERROR: {e}')
 import traceback
 traceback.print_exc()
