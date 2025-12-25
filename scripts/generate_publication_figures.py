"""
Publication Figure Generation
==============================

Generates publication-quality figures for EntPTC paper.
All figures derived from committed data and matrix-first pipeline.

Requirements:
- Vector or high-resolution raster (PDF preferred, >=600 dpi if raster)
- Clean axes, labeled quantities
- Neutral scientific captions
- Reproducible from committed code

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json
import scipy.io as sio
from matrix_first_pipeline import construct_progenitor_matrix, extract_collapse_object
from entptc.utils.grid_utils import create_toroidal_grid

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
 'font.size': 10,
 'axes.labelsize': 10,
 'axes.titlesize': 11,
 'xtick.labelsize': 9,
 'ytick.labelsize': 9,
 'legend.fontsize': 9,
 'figure.titlesize': 11,
 'font.family': 'serif',
 'font.serif': ['Times New Roman'],
 'text.usetex': False,
 'figure.dpi': 600,
 'savefig.dpi': 600,
 'savefig.format': 'pdf',
 'savefig.bbox': 'tight'
})

# Output directory
output_dir = Path('/home/ubuntu/entptc-implementation/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FIGURE 1: Geometry → Operator → Collapse Schematic
# ============================================================================

def generate_fig01_schematic():
 """
 Figure 1: Deterministic flow diagram
 T³ → operator → eigenstructure → invariants → projections
 """
 fig, ax = plt.subplots(figsize=(8, 4))
 ax.axis('off')
 
 # Define boxes
 boxes = [
 ('Toroidal\nGeometry\n(T³)', 0.5, 0.5),
 ('Progenitor\nMatrix', 2.0, 0.5),
 ('Operator\nEigendecomp.', 3.5, 0.5),
 ('Collapse\nStructure', 5.0, 0.5),
 ('Invariants', 6.5, 0.5),
 ('Projections\n(EEG/fMRI)', 8.0, 0.5)
 ]
 
 # Draw boxes
 for label, x, y in boxes:
 rect = mpatches.FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
 boxstyle="round,pad=0.05",
 edgecolor='black', facecolor='lightgray',
 linewidth=1.5)
 ax.add_patch(rect)
 ax.text(x, y, label, ha='center', va='center', fontsize=9, weight='bold')
 
 # Draw arrows
 for i in range(len(boxes)-1):
 x1 = boxes[i][1] + 0.4
 x2 = boxes[i+1][1] - 0.4
 y = boxes[i][2]
 ax.arrow(x1, y, x2-x1-0.1, 0, head_width=0.15, head_length=0.1,
 fc='black', ec='black', linewidth=1.5)
 
 ax.set_xlim(0, 8.5)
 ax.set_ylim(0, 1)
 ax.set_aspect('equal')
 
 plt.savefig(output_dir / 'fig01_schematic.pdf')
 plt.close()
 print("✓ Figure 1 generated: fig01_schematic.pdf")

# ============================================================================
# FIGURE 2: Eigenvalue Spectrum & Collapse
# ============================================================================

def generate_fig02_eigenspectrum():
 """
 Figure 2: Full eigenvalue distribution
 Dominant eigenvalue highlighted, spectral gap shown
 """
 # Load data
 data_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_3_ds004706/sub-LTP448_ses-0_task-SpatialNav_eeg.mat')
 if not data_path.exists():
 print(f"⚠ Data not found: {data_path}")
 return
 
 mat = sio.loadmat(data_path)
 data = mat['eeg_data']
 
 # Construct progenitor and extract collapse
 adjacency = create_toroidal_grid(4)
 progenitor = construct_progenitor_matrix(data, adjacency)
 collapse = extract_collapse_object(progenitor)
 
 eigenvalues = np.array(collapse['eigenmode_profile'])
 
 fig, ax = plt.subplots(figsize=(6, 4))
 
 # Plot eigenvalues
 indices = np.arange(len(eigenvalues))
 ax.bar(indices, eigenvalues, color='steelblue', edgecolor='black', linewidth=0.5)
 
 # Highlight dominant eigenvalue
 ax.bar(0, eigenvalues[0], color='darkred', edgecolor='black', linewidth=0.5,
 label=f'Dominant λ₁ = {eigenvalues[0]:.3f}')
 
 # Show spectral gap
 spectral_gap = eigenvalues[0] - eigenvalues[1]
 ax.annotate('', xy=(0, eigenvalues[1]), xytext=(0, eigenvalues[0]),
 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
 ax.text(0.5, (eigenvalues[0] + eigenvalues[1])/2,
 f'Gap = {spectral_gap:.3f}',
 fontsize=9, color='red', weight='bold')
 
 ax.set_xlabel('Eigenmode Index')
 ax.set_ylabel('Eigenvalue')
 ax.set_title('Eigenvalue Spectrum (Intact Condition)')
 ax.legend(loc='upper right')
 ax.grid(True, alpha=0.3)
 
 plt.savefig(output_dir / 'fig02_eigenspectrum.pdf')
 plt.close()
 print("✓ Figure 2 generated: fig02_eigenspectrum.pdf")

# ============================================================================
# FIGURE 3: Entropy & Participation Metrics
# ============================================================================

def generate_fig03_entropy_participation():
 """
 Figure 3: Von Neumann entropy and participation ratio
 Shown across conditions (task vs EO/EC)
 """
 # Load EO/EC data
 eo_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_2/S001_task-EyesOpen_eeg.mat')
 ec_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_2/S001_task-EyesClosed_eeg.mat')
 
 if not eo_path.exists() or not ec_path.exists():
 print("⚠ EO/EC data not found")
 return
 
 # Extract collapse objects
 adjacency = create_toroidal_grid(4)
 
 mat_eo = sio.loadmat(eo_path)
 data_eo = mat_eo['eeg_data']
 progenitor_eo = construct_progenitor_matrix(data_eo, adjacency)
 collapse_eo = extract_collapse_object(progenitor_eo)
 
 mat_ec = sio.loadmat(ec_path)
 data_ec = mat_ec['eeg_data']
 progenitor_ec = construct_progenitor_matrix(data_ec, adjacency)
 collapse_ec = extract_collapse_object(progenitor_ec)
 
 # Plot
 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
 
 # Entropy
 conditions = ['Eyes Open', 'Eyes Closed']
 entropies = [collapse_eo['von_neumann_entropy'], collapse_ec['von_neumann_entropy']]
 
 ax1.bar(conditions, entropies, color=['steelblue', 'darkgreen'], edgecolor='black', linewidth=1)
 ax1.set_ylabel('Von Neumann Entropy')
 ax1.set_title('Entropy Across Conditions')
 ax1.grid(True, alpha=0.3, axis='y')
 
 # Participation ratio
 participation = [collapse_eo['participation_ratio'], collapse_ec['participation_ratio']]
 
 ax2.bar(conditions, participation, color=['steelblue', 'darkgreen'], edgecolor='black', linewidth=1)
 ax2.set_ylabel('Participation Ratio')
 ax2.set_title('Participation Ratio Across Conditions')
 ax2.grid(True, alpha=0.3, axis='y')
 
 plt.tight_layout()
 plt.savefig(output_dir / 'fig03_entropy_participation.pdf')
 plt.close()
 print("✓ Figure 3 generated: fig03_entropy_participation.pdf")

# ============================================================================
# FIGURE 4: EO vs EC Stability
# ============================================================================

def generate_fig04_eoec_stability():
 """
 Figure 4: Percent change bars
 Dominant eigenvalue, entropy, participation ratio, spectral gap
 """
 # Load EO/EC data
 eo_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_2/S001_task-EyesOpen_eeg.mat')
 ec_path = Path('/home/ubuntu/entptc-implementation/data/dataset_set_2/S001_task-EyesClosed_eeg.mat')
 
 if not eo_path.exists() or not ec_path.exists():
 print("⚠ EO/EC data not found")
 return
 
 # Extract collapse objects
 adjacency = create_toroidal_grid(4)
 
 mat_eo = sio.loadmat(eo_path)
 data_eo = mat_eo['eeg_data']
 progenitor_eo = construct_progenitor_matrix(data_eo, adjacency)
 collapse_eo = extract_collapse_object(progenitor_eo)
 
 mat_ec = sio.loadmat(ec_path)
 data_ec = mat_ec['eeg_data']
 progenitor_ec = construct_progenitor_matrix(data_ec, adjacency)
 collapse_ec = extract_collapse_object(progenitor_ec)
 
 # Compute percent changes
 components = [
 'Dominant\nEigenvalue',
 'Von Neumann\nEntropy',
 'Participation\nRatio',
 'Spectral\nGap'
 ]
 
 eo_values = [
 collapse_eo['dominant_eigenvalue'],
 collapse_eo['von_neumann_entropy'],
 collapse_eo['participation_ratio'],
 collapse_eo['spectral_gap']
 ]
 
 ec_values = [
 collapse_ec['dominant_eigenvalue'],
 collapse_ec['von_neumann_entropy'],
 collapse_ec['participation_ratio'],
 collapse_ec['spectral_gap']
 ]
 
 percent_changes = [
 100 * (ec - eo) / (eo + 1e-10) for eo, ec in zip(eo_values, ec_values)
 ]
 
 # Plot
 fig, ax = plt.subplots(figsize=(8, 5))
 
 colors = ['green' if abs(pc) < 10 else 'orange' for pc in percent_changes]
 bars = ax.barh(components, percent_changes, color=colors, edgecolor='black', linewidth=1)
 
 # Add threshold lines
 ax.axvline(10, color='red', linestyle='--', linewidth=1, label='±10% threshold')
 ax.axvline(-10, color='red', linestyle='--', linewidth=1)
 
 ax.set_xlabel('Percent Change (EO → EC)')
 ax.set_title('Collapse Structure Stability: Eyes Open vs Eyes Closed')
 ax.legend(loc='lower right')
 ax.grid(True, alpha=0.3, axis='x')
 
 # Add value labels
 for i, (bar, pc) in enumerate(zip(bars, percent_changes)):
 ax.text(pc + 1 if pc > 0 else pc - 1, i, f'{pc:.1f}%',
 va='center', ha='left' if pc > 0 else 'right', fontsize=9)
 
 plt.tight_layout()
 plt.savefig(output_dir / 'fig04_eoec_stability.pdf')
 plt.close()
 print("✓ Figure 4 generated: fig04_eoec_stability.pdf")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
 print("\n" + "="*80)
 print("GENERATING PUBLICATION FIGURES")
 print("="*80)
 
 generate_fig01_schematic()
 generate_fig02_eigenspectrum()
 generate_fig03_entropy_participation()
 generate_fig04_eoec_stability()
 
 print("\n" + "="*80)
 print("ALL FIGURES GENERATED")
 print(f"Output directory: {output_dir}")
 print("="*80)
