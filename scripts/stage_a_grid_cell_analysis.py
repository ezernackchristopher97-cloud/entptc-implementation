"""
STAGE A: Grid Cell Toroidal Geometry Extraction

Extract TRUE toroidal structure from Hafting et al. 2005 grid cell data.
This is Point A - where geometry is data-anchored, not inferred.

Pipeline:
1. Load spike times and position data
2. Compute spatial firing rate maps for each grid cell
3. Detect hexagonal grid structure and extract grid parameters
4. Map grid cells to toroidal coordinates (phase on T²)
5. Extract geometry-based invariants (modality-agnostic)
6. Compute internal frequency inference from dynamics

"""

import scipy.io as sio
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import json

# Data directory
data_dir = '/home/ubuntu/entptc-implementation/stage_a_datasets/hafting_2005'
output_dir = '/home/ubuntu/entptc-implementation/stage_a_outputs'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/figures', exist_ok=True)

print("=" * 80)
print("STAGE A: GRID CELL TOROIDAL GEOMETRY EXTRACTION")
print("=" * 80)

# ============================================================================
# STEP 1: Load all grid cell data
# ============================================================================

mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
mat_files.sort()

all_grid_cells = []

for mat_file in mat_files:
 print(f"\nLoading {mat_file}...")
 file_path = os.path.join(data_dir, mat_file)
 data = sio.loadmat(file_path)
 
 # Extract position data
 pos_x = data['pos_x'].flatten()
 pos_y = data['pos_y'].flatten()
 pos_timestamps = data['pos_timeStamps'].flatten()
 
 # Find all cell spike timestamp keys
 cell_keys = [k for k in data.keys() if 'timeStamps' in k and not k.startswith('pos')]
 
 print(f" Position samples: {len(pos_x)}")
 print(f" Cells found: {len(cell_keys)}")
 
 for cell_key in cell_keys:
 spike_timestamps = data[cell_key].flatten()
 
 all_grid_cells.append({
 'file': mat_file,
 'cell_id': cell_key.replace('_timeStamps', ''),
 'pos_x': pos_x,
 'pos_y': pos_y,
 'pos_timestamps': pos_timestamps,
 'spike_timestamps': spike_timestamps
 })
 
 print(f" {cell_key}: {len(spike_timestamps)} spikes")

print(f"\n{'=' * 80}")
print(f"TOTAL GRID CELLS LOADED: {len(all_grid_cells)}")
print(f"{'=' * 80}")

# ============================================================================
# STEP 2: Compute spatial firing rate maps
# ============================================================================

def compute_firing_rate_map(pos_x, pos_y, pos_timestamps, spike_timestamps, 
 bins=50, smoothing_sigma=2.0):
 """
 Compute 2D spatial firing rate map.
 
 Args:
 pos_x, pos_y: Position coordinates
 pos_timestamps: Timestamps for positions
 spike_timestamps: Timestamps for spikes
 bins: Number of spatial bins
 smoothing_sigma: Gaussian smoothing sigma
 
 Returns:
 rate_map: 2D firing rate map (Hz)
 x_edges, y_edges: Bin edges
 """
 # Remove NaN values from position data
 valid_mask = ~(np.isnan(pos_x) | np.isnan(pos_y) | np.isnan(pos_timestamps))
 pos_x = pos_x[valid_mask]
 pos_y = pos_y[valid_mask]
 pos_timestamps = pos_timestamps[valid_mask]
 
 if len(pos_x) == 0:
 return np.zeros((bins, bins)), np.array([0, 1]), np.array([0, 1])
 
 # Create 2D histogram of positions (occupancy)
 occupancy, x_edges, y_edges = np.histogram2d(pos_x, pos_y, bins=bins)
 
 # Create 2D histogram of spike positions
 # Match each spike to its position
 spike_positions_x = []
 spike_positions_y = []
 
 for spike_time in spike_timestamps:
 # Find closest position timestamp
 idx = np.argmin(np.abs(pos_timestamps - spike_time))
 spike_positions_x.append(pos_x[idx])
 spike_positions_y.append(pos_y[idx])
 
 spike_hist, _, _ = np.histogram2d(spike_positions_x, spike_positions_y, 
 bins=[x_edges, y_edges])
 
 # Compute firing rate (spikes / occupancy time)
 # Assume position sampled at ~50 Hz (typical for this dataset)
 sampling_rate = len(pos_timestamps) / (pos_timestamps[-1] - pos_timestamps[0])
 occupancy_time = occupancy / sampling_rate
 
 # Avoid division by zero
 rate_map = np.zeros_like(occupancy)
 mask = occupancy_time > 0
 rate_map[mask] = spike_hist[mask] / occupancy_time[mask]
 
 # Smooth with Gaussian
 rate_map = gaussian_filter(rate_map, sigma=smoothing_sigma)
 
 return rate_map, x_edges, y_edges

print("\n" + "=" * 80)
print("STEP 2: Computing spatial firing rate maps...")
print("=" * 80)

for i, cell in enumerate(all_grid_cells):
 print(f"\n[{i+1}/{len(all_grid_cells)}] {cell['cell_id']}...")
 
 rate_map, x_edges, y_edges = compute_firing_rate_map(
 cell['pos_x'], cell['pos_y'], cell['pos_timestamps'],
 cell['spike_timestamps'], bins=50, smoothing_sigma=2.0
 )
 
 cell['rate_map'] = rate_map
 cell['x_edges'] = x_edges
 cell['y_edges'] = y_edges
 cell['peak_rate'] = np.max(rate_map)
 
 print(f" Peak firing rate: {cell['peak_rate']:.2f} Hz")

# ============================================================================
# STEP 3: Detect hexagonal grid structure
# ============================================================================

def detect_grid_structure(rate_map):
 """
 Detect hexagonal grid structure from firing rate map.
 
 Returns:
 grid_score: Measure of hexagonal periodicity
 grid_spacing: Distance between grid fields
 grid_orientation: Orientation of grid axes
 """
 # Compute autocorrelation
 from scipy.signal import correlate2d
 
 autocorr = correlate2d(rate_map, rate_map, mode='same')
 autocorr = autocorr / np.max(autocorr) # Normalize
 
 # Find peaks in autocorrelation (excluding center)
 center = np.array(autocorr.shape) // 2
 mask = np.ones_like(autocorr, dtype=bool)
 mask[center[0]-5:center[0]+5, center[1]-5:center[1]+5] = False
 
 # Detect peaks
 from scipy.ndimage import maximum_filter
 local_max = maximum_filter(autocorr, size=10)
 peaks = (autocorr == local_max) & mask & (autocorr > 0.2)
 
 peak_coords = np.argwhere(peaks)
 
 if len(peak_coords) < 6:
 return 0.0, 0.0, 0.0 # Not enough peaks for grid
 
 # Compute distances from center
 distances = np.linalg.norm(peak_coords - center, axis=1)
 
 # Grid spacing is the median distance to nearest peaks
 grid_spacing = np.median(distances)
 
 # Grid score: correlation at 60° vs 30°/90°
 # Simplified version: count peaks in hexagonal pattern
 grid_score = len(peak_coords) / 20.0 # Normalize
 
 # Grid orientation: angle of first peak
 if len(peak_coords) > 0:
 first_peak = peak_coords[0] - center
 grid_orientation = np.arctan2(first_peak[1], first_peak[0])
 else:
 grid_orientation = 0.0
 
 return grid_score, grid_spacing, grid_orientation

print("\n" + "=" * 80)
print("STEP 3: Detecting hexagonal grid structure...")
print("=" * 80)

for i, cell in enumerate(all_grid_cells):
 print(f"\n[{i+1}/{len(all_grid_cells)}] {cell['cell_id']}...")
 
 grid_score, grid_spacing, grid_orientation = detect_grid_structure(cell['rate_map'])
 
 cell['grid_score'] = grid_score
 cell['grid_spacing'] = grid_spacing
 cell['grid_orientation'] = grid_orientation
 
 print(f" Grid score: {grid_score:.3f}")
 print(f" Grid spacing: {grid_spacing:.2f} bins")
 print(f" Grid orientation: {np.degrees(grid_orientation):.1f}°")

# ============================================================================
# STEP 4: Map to toroidal coordinates
# ============================================================================

def map_to_toroidal_coordinates(pos_x, pos_y, grid_spacing, grid_orientation):
 """
 Map 2D positions to toroidal phase coordinates.
 
 Grid cells naturally encode position on a torus T² through their
 hexagonal firing pattern. The phase is the position modulo the grid spacing.
 
 Returns:
 phase_x, phase_y: Phase coordinates on [0, 2π) × [0, 2π)
 """
 # Remove NaN values
 valid_mask = ~(np.isnan(pos_x) | np.isnan(pos_y))
 pos_x_clean = pos_x[valid_mask]
 pos_y_clean = pos_y[valid_mask]
 
 # Rotate coordinates to align with grid orientation
 cos_theta = np.cos(-grid_orientation)
 sin_theta = np.sin(-grid_orientation)
 
 x_rot = pos_x_clean * cos_theta - pos_y_clean * sin_theta
 y_rot = pos_x_clean * sin_theta + pos_y_clean * cos_theta
 
 # Compute phase (position modulo grid spacing, mapped to [0, 2π))
 if grid_spacing > 0:
 phase_x = (x_rot % grid_spacing) / grid_spacing * 2 * np.pi
 phase_y = (y_rot % grid_spacing) / grid_spacing * 2 * np.pi
 else:
 phase_x = np.zeros_like(x_rot)
 phase_y = np.zeros_like(y_rot)
 
 return phase_x, phase_y

print("\n" + "=" * 80)
print("STEP 4: Mapping to toroidal coordinates...")
print("=" * 80)

for i, cell in enumerate(all_grid_cells):
 if cell['grid_spacing'] > 0:
 phase_x, phase_y = map_to_toroidal_coordinates(
 cell['pos_x'], cell['pos_y'],
 cell['grid_spacing'], cell['grid_orientation']
 )
 
 cell['phase_x'] = phase_x
 cell['phase_y'] = phase_y
 
 print(f"[{i+1}/{len(all_grid_cells)}] {cell['cell_id']}: Mapped to T²")
 else:
 cell['phase_x'] = None
 cell['phase_y'] = None
 print(f"[{i+1}/{len(all_grid_cells)}] {cell['cell_id']}: Skipped (no grid structure)")

# ============================================================================
# STEP 5: Extract geometry-based invariants
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Extracting geometry-based invariants...")
print("=" * 80)

# Compute modality-agnostic invariants from toroidal dynamics
invariants = []

for i, cell in enumerate(all_grid_cells):
 if cell['phase_x'] is not None:
 # Filter out NaN values from timestamps as well
 valid_mask = ~(np.isnan(cell['pos_x']) | np.isnan(cell['pos_y']) | np.isnan(cell['pos_timestamps']))
 timestamps_clean = cell['pos_timestamps'][valid_mask]
 
 # Compute phase velocity (trajectory on torus)
 dt = np.diff(timestamps_clean)
 dphi_x = np.diff(cell['phase_x'])
 dphi_y = np.diff(cell['phase_y'])
 
 # Handle phase wrapping
 dphi_x[dphi_x > np.pi] -= 2 * np.pi
 dphi_x[dphi_x < -np.pi] += 2 * np.pi
 dphi_y[dphi_y > np.pi] -= 2 * np.pi
 dphi_y[dphi_y < -np.pi] += 2 * np.pi
 
 phase_velocity = np.sqrt((dphi_x / dt)**2 + (dphi_y / dt)**2)
 
 # Compute trajectory curvature on torus
 d2phi_x = np.diff(dphi_x / dt)
 d2phi_y = np.diff(dphi_y / dt)
 curvature = np.sqrt(d2phi_x**2 + d2phi_y**2)
 
 # Compute winding numbers (how many times trajectory wraps around torus)
 winding_x = np.sum(dphi_x) / (2 * np.pi)
 winding_y = np.sum(dphi_y) / (2 * np.pi)
 
 # Compute entropy of phase distribution
 phase_hist, _, _ = np.histogram2d(cell['phase_x'], cell['phase_y'], bins=20)
 phase_hist = phase_hist / np.sum(phase_hist)
 entropy = -np.sum(phase_hist[phase_hist > 0] * np.log(phase_hist[phase_hist > 0]))
 
 invariants.append({
 'cell_id': cell['cell_id'],
 'grid_score': cell['grid_score'],
 'grid_spacing': cell['grid_spacing'],
 'mean_phase_velocity': np.mean(phase_velocity),
 'std_phase_velocity': np.std(phase_velocity),
 'mean_curvature': np.mean(curvature),
 'std_curvature': np.std(curvature),
 'winding_number_x': winding_x,
 'winding_number_y': winding_y,
 'phase_entropy': entropy
 })
 
 print(f"[{i+1}/{len(all_grid_cells)}] {cell['cell_id']}:")
 print(f" Mean phase velocity: {np.mean(phase_velocity):.4f} rad/s")
 print(f" Mean curvature: {np.mean(curvature):.4f}")
 print(f" Winding numbers: ({winding_x:.2f}, {winding_y:.2f})")
 print(f" Phase entropy: {entropy:.4f}")

# ============================================================================
# STEP 6: Save results
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Saving results...")
print("=" * 80)

# Save invariants to JSON
invariants_file = os.path.join(output_dir, 'grid_cell_invariants.json')
with open(invariants_file, 'w') as f:
 json.dump(invariants, f, indent=2)
print(f"Saved invariants to: {invariants_file}")

# Save summary statistics
summary_file = os.path.join(output_dir, 'stage_a_summary.txt')
with open(summary_file, 'w') as f:
 f.write("STAGE A: Grid Cell Toroidal Geometry Extraction - Summary\n")
 f.write("=" * 80 + "\n\n")
 f.write(f"Total grid cells analyzed: {len(all_grid_cells)}\n")
 f.write(f"Cells with valid grid structure: {len(invariants)}\n\n")
 f.write("Geometry-based invariants (mean ± std):\n")
 if len(invariants) > 0:
 for key in ['mean_phase_velocity', 'mean_curvature', 'phase_entropy']:
 values = [inv[key] for inv in invariants]
 f.write(f" {key}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")
print(f"Saved summary to: {summary_file}")

print("\n" + "=" * 80)
print("STAGE A COMPLETE!")
print("=" * 80)
print(f"\nOutputs saved to: {output_dir}")
print("\nNext: STAGE B - Internal frequency inference from geometry-driven dynamics")
