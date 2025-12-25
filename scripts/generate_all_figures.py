#!/usr/bin/env python3
"""
Generate All Figures for EntPTC Analysis
Uses matplotlib for professional-quality visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from pathlib import Path
import seaborn as sns

# Set style for professional figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load results
results_path = Path('/home/ubuntu/entptc-implementation/outputs/master_results.csv')
df = pd.read_csv(results_path)

# Create figures directory
fig_dir = Path('/home/ubuntu/entptc-implementation/outputs/figures')
fig_dir.mkdir(exist_ok=True)

print("=" * 80)
print("GENERATING ALL FIGURES")
print("=" * 80)

# Figure 1: Eigenvalue Spectrum
print("\n[1/10] Generating eigenvalue spectrum...")
fig, ax = plt.subplots(figsize=(12, 7))

eig_cols = [f'eigenvalue_{i}' for i in range(16)]
eig_means = [df[col].mean() for col in eig_cols]
eig_stds = [df[col].std() for col in eig_cols]
x = np.arange(16)

ax.errorbar(x, eig_means, yerr=eig_stds, fmt='o-', linewidth=2, 
 markersize=8, capsize=5, capthick=2, label='Mean ± Std')
ax.set_xlabel('Eigenvalue Index', fontsize=14, fontweight='bold')
ax.set_ylabel('Eigenvalue Magnitude', fontsize=14, fontweight='bold')
ax.set_title('Eigenvalue Spectrum (n=150 recordings)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'λ_{i}' for i in range(16)])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(fig_dir / 'eigenvalue_spectrum.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Spectral Gap Distribution
print("[2/10] Generating spectral gap distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

spectral_gaps = df['spectral_gap'].dropna()
ax.hist(spectral_gaps, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(spectral_gaps.mean(), color='red', linestyle='--', linewidth=2, 
 label=f'Mean = {spectral_gaps.mean():.2f}')
ax.axvline(2.0, color='green', linestyle='--', linewidth=2, 
 label='Regime I/II boundary (2.0)')
ax.set_xlabel('Spectral Gap (λ₁/λ₂)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Spectral Gap Distribution', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'spectral_gap_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Regime Distribution
print("[3/10] Generating regime distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

regime_counts = df['regime'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax.bar(range(len(regime_counts)), regime_counts.values, color=colors[:len(regime_counts)])

ax.set_xlabel('Regime', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Regime Distribution (n=150)', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(regime_counts)))
ax.set_xticklabels([r.replace('_', ' ') for r in regime_counts.index], rotation=15, ha='right')

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, regime_counts.values)):
 height = bar.get_height()
 ax.text(bar.get_x() + bar.get_width()/2., height,
 f'{count}\n({count/len(df)*100:.1f}%)',
 ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(fig_dir / 'regime_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Absurdity Gap - Eyes Open vs Eyes Closed
print("[4/10] Generating absurdity gap comparison...")
fig, ax = plt.subplots(figsize=(10, 7))

eyes_closed = df[df['task'] == 'EyesClosed']['absurdity_gap_L2'].dropna()
eyes_open = df[df['task'] == 'EyesOpen']['absurdity_gap_L2'].dropna()

positions = [1, 2]
data = [eyes_closed, eyes_open]
bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
 showmeans=True, meanline=True)

# Color the boxes
colors_box = ['#3498db', '#e74c3c']
for patch, color in zip(bp['boxes'], colors_box):
 patch.set_facecolor(color)
 patch.set_alpha(0.7)

ax.set_xlabel('Condition', fontsize=14, fontweight='bold')
ax.set_ylabel('Absurdity Gap (L2 Norm)', fontsize=14, fontweight='bold')
ax.set_title('Absurdity Gap: Eyes Open vs Eyes Closed\n(Falsifiability Test)', 
 fontsize=16, fontweight='bold')
ax.set_xticks(positions)
ax.set_xticklabels(['Eyes Closed', 'Eyes Open'])

# Add statistical annotation
t_stat, p_value = stats.ttest_ind(eyes_closed, eyes_open)
y_max = max(eyes_closed.max(), eyes_open.max())
ax.plot([1, 2], [y_max * 1.1, y_max * 1.1], 'k-', linewidth=2)
ax.text(1.5, y_max * 1.15, f'p = {p_value:.3f}', ha='center', fontsize=12, fontweight='bold')

# Add means
ax.text(1, eyes_closed.mean(), f'{eyes_closed.mean():.3f}', 
 ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.text(2, eyes_open.mean(), f'{eyes_open.mean():.3f}', 
 ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(fig_dir / 'absurdity_gap_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Pre vs Post Treatment Comparison
print("[5/10] Generating pre/post treatment comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
 ('lambda_max', 'λ_max (Dominant Eigenvalue)'),
 ('spectral_gap', 'Spectral Gap (λ₁/λ₂)'),
 ('entropy_mean', 'Entropy (Mean)'),
 ('absurdity_gap_L2', 'Absurdity Gap (L2)')
]

for ax, (metric, title) in zip(axes.flat, metrics):
 pre_vals = df[df['timepoint'] == 'pre'][metric].dropna()
 post_vals = df[df['timepoint'] == 'post'][metric].dropna()
 
 data = [pre_vals, post_vals]
 bp = ax.boxplot(data, positions=[1, 2], widths=0.6, patch_artist=True,
 showmeans=True, meanline=True)
 
 # Color the boxes
 for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
 patch.set_facecolor(color)
 patch.set_alpha(0.7)
 
 ax.set_xlabel('Timepoint', fontsize=12, fontweight='bold')
 ax.set_ylabel(title, fontsize=12, fontweight='bold')
 ax.set_xticks([1, 2])
 ax.set_xticklabels(['Pre', 'Post'])
 
 # Add statistical test
 t_stat, p_value = stats.ttest_ind(pre_vals, post_vals)
 significance = '*' if p_value < 0.05 else 'ns'
 ax.set_title(f'{title}\np = {p_value:.4f} {significance}', fontsize=11, fontweight='bold')
 
 ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Pre vs Post Treatment Comparison', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(fig_dir / 'pre_post_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Entropy Distribution
print("[6/10] Generating entropy distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

entropy_vals = df['entropy_mean'].dropna()
ax.hist(entropy_vals, bins=30, edgecolor='black', alpha=0.7, color='#9b59b6')
ax.axvline(entropy_vals.mean(), color='red', linestyle='--', linewidth=2,
 label=f'Mean = {entropy_vals.mean():.3f}')
ax.set_xlabel('Shannon Entropy (Mean)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Entropy Distribution', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'entropy_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 7: Dominant Eigenvalue Distribution
print("[7/10] Generating dominant eigenvalue distribution...")
fig, ax = plt.subplots(figsize=(10, 6))

lambda_max = df['lambda_max'].dropna()
ax.hist(lambda_max, bins=30, edgecolor='black', alpha=0.7, color='#e67e22')
ax.axvline(lambda_max.mean(), color='red', linestyle='--', linewidth=2,
 label=f'Mean = {lambda_max.mean():.3f}')
ax.axvline(12.6, color='green', linestyle='--', linewidth=2,
 label='Predicted (12.6)')
ax.set_xlabel('λ_max (Dominant Eigenvalue)', fontsize=14, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
ax.set_title('Dominant Eigenvalue Distribution', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'lambda_max_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 8: Correlation Matrix (Key Metrics)
print("[8/10] Generating correlation matrix...")
fig, ax = plt.subplots(figsize=(12, 10))

key_metrics = ['lambda_max', 'spectral_gap', 'entropy_mean', 
 'absurdity_gap_L2', 'quaternion_norm', 'symmetry_breaking']
corr_data = df[key_metrics].corr()

im = ax.imshow(corr_data, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(np.arange(len(key_metrics)))
ax.set_yticks(np.arange(len(key_metrics)))
ax.set_xticklabels([m.replace('_', ' ').title() for m in key_metrics], rotation=45, ha='right')
ax.set_yticklabels([m.replace('_', ' ').title() for m in key_metrics])

# Add correlation values
for i in range(len(key_metrics)):
 for j in range(len(key_metrics)):
 text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
 ha="center", va="center", color="black", fontsize=10, fontweight='bold')

ax.set_title('Correlation Matrix: Key EntPTC Metrics', fontsize=16, fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation Coefficient')
plt.tight_layout()
plt.savefig(fig_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 9: Task Comparison (All Metrics)
print("[9/10] Generating task comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, (metric, title) in zip(axes.flat, metrics):
 eyes_closed_vals = df[df['task'] == 'EyesClosed'][metric].dropna()
 eyes_open_vals = df[df['task'] == 'EyesOpen'][metric].dropna()
 
 data = [eyes_closed_vals, eyes_open_vals]
 bp = ax.boxplot(data, positions=[1, 2], widths=0.6, patch_artist=True,
 showmeans=True, meanline=True)
 
 for patch, color in zip(bp['boxes'], ['#3498db', '#e74c3c']):
 patch.set_facecolor(color)
 patch.set_alpha(0.7)
 
 ax.set_xlabel('Task', fontsize=12, fontweight='bold')
 ax.set_ylabel(title, fontsize=12, fontweight='bold')
 ax.set_xticks([1, 2])
 ax.set_xticklabels(['Eyes Closed', 'Eyes Open'])
 
 t_stat, p_value = stats.ttest_ind(eyes_closed_vals, eyes_open_vals)
 significance = '*' if p_value < 0.05 else 'ns'
 ax.set_title(f'{title}\np = {p_value:.4f} {significance}', fontsize=11, fontweight='bold')
 
 ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Task Comparison: Eyes Open vs Eyes Closed', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(fig_dir / 'task_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 10: Eigenvalue Decay (Log Scale)
print("[10/10] Generating eigenvalue decay plot...")
fig, ax = plt.subplots(figsize=(12, 7))

# Plot individual recordings (sample)
sample_indices = np.random.choice(len(df), size=min(20, len(df)), replace=False)
for idx in sample_indices:
 eig_vals = [df.iloc[idx][f'eigenvalue_{i}'] for i in range(16)]
 ax.plot(range(16), eig_vals, alpha=0.3, color='gray', linewidth=1)

# Plot mean
ax.plot(range(16), eig_means, 'o-', linewidth=3, markersize=10, 
 color='red', label='Mean', zorder=10)

ax.set_xlabel('Eigenvalue Index', fontsize=14, fontweight='bold')
ax.set_ylabel('Eigenvalue Magnitude (log scale)', fontsize=14, fontweight='bold')
ax.set_title('Eigenvalue Decay Profile (n=150)', fontsize=16, fontweight='bold')
ax.set_xticks(range(16))
ax.set_xticklabels([f'λ_{i}' for i in range(16)])
ax.set_yscale('log')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'eigenvalue_decay.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 80)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\nFigures saved to: {fig_dir}")
print("\nGenerated figures:")
for i, fig_name in enumerate([
 'eigenvalue_spectrum.png',
 'spectral_gap_distribution.png',
 'regime_distribution.png',
 'absurdity_gap_comparison.png',
 'pre_post_comparison.png',
 'entropy_distribution.png',
 'lambda_max_distribution.png',
 'correlation_matrix.png',
 'task_comparison.png',
 'eigenvalue_decay.png'
], 1):
 print(f" [{i:2d}] {fig_name}")
