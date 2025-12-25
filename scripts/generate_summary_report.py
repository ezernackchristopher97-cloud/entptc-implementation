#!/usr/bin/env python3
"""
Generate Comprehensive Summary Report for EntPTC Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# Load results
results_path = Path('/home/ubuntu/entptc-implementation/outputs/master_results.csv')
df = pd.read_csv(results_path)

print("=" * 80)
print("ENTPC ANALYSIS - COMPREHENSIVE SUMMARY REPORT")
print("=" * 80)

# Basic statistics
print(f"\n{'='*80}")
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total recordings processed: {len(df)}")
print(f"Unique subjects: {df['subject_id'].nunique()}")
print(f"Unique sessions: {df.groupby('subject_id')['session'].nunique().sum()}")

print(f"\nBy Task:")
task_counts = df['task'].value_counts()
for task, count in task_counts.items():
 print(f" {task}: {count}")

print(f"\nBy Timepoint:")
timepoint_counts = df['timepoint'].value_counts()
for tp, count in timepoint_counts.items():
 print(f" {tp}: {count}")

print(f"\nBy Regime:")
regime_counts = df['regime'].value_counts()
for regime, count in regime_counts.items():
 print(f" {regime}: {count}")

# Key metrics
print(f"\n{'='*80}")
print("KEY METRICS (Mean ± Std)")
print("=" * 80)

metrics = {
 'λ_max (Dominant Eigenvalue)': 'lambda_max',
 'Spectral Gap (λ₁/λ₂)': 'spectral_gap',
 'Entropy (Mean)': 'entropy_mean',
 'Absurdity Gap (L1)': 'absurdity_gap_L1',
 'Absurdity Gap (L2)': 'absurdity_gap_L2',
 'Absurdity Gap (L∞)': 'absurdity_gap_Linf',
 'Absurdity Gap (Overlap)': 'absurdity_gap_overlap',
 'Absurdity Gap (Info Loss)': 'absurdity_gap_info_loss',
 'Quaternion Norm': 'quaternion_norm',
 'Clifford Multivector Norm': 'clifford_multivector_norm',
 'THz Symmetry Breaking': 'symmetry_breaking',
 'THz Spectral Radius': 'spectral_radius'
}

for name, col in metrics.items():
 if col in df.columns:
 mean_val = df[col].mean()
 std_val = df[col].std()
 print(f"{name:40s}: {mean_val:12.6f} ± {std_val:12.6f}")

# Regime-specific analysis
print(f"\n{'='*80}")
print("REGIME-SPECIFIC ANALYSIS")
print("=" * 80)

for regime in df['regime'].unique():
 regime_df = df[df['regime'] == regime]
 print(f"\n{regime} (n={len(regime_df)}):")
 print(f" λ_max: {regime_df['lambda_max'].mean():.6f} ± {regime_df['lambda_max'].std():.6f}")
 print(f" Spectral Gap: {regime_df['spectral_gap'].mean():.6f} ± {regime_df['spectral_gap'].std():.6f}")
 print(f" Entropy: {regime_df['entropy_mean'].mean():.6f} ± {regime_df['entropy_mean'].std():.6f}")
 print(f" Absurd. Gap: {regime_df['absurdity_gap_L2'].mean():.6f} ± {regime_df['absurdity_gap_L2'].std():.6f}")

# Falsifiability test
print(f"\n{'='*80}")
print("FALSIFIABILITY TEST (ENTPC.tex line 663)")
print("=" * 80)

eyes_closed = df[df['task'] == 'EyesClosed']['absurdity_gap_L2'].dropna()
eyes_open = df[df['task'] == 'EyesOpen']['absurdity_gap_L2'].dropna()

if len(eyes_closed) > 0 and len(eyes_open) > 0:
 t_stat, p_value = stats.ttest_ind(eyes_closed, eyes_open)
 
 print(f"\nAbsurdity Gap (L2) Comparison:")
 print(f" Eyes Closed: {eyes_closed.mean():.6f} ± {eyes_closed.std():.6f} (n={len(eyes_closed)})")
 print(f" Eyes Open: {eyes_open.mean():.6f} ± {eyes_open.std():.6f} (n={len(eyes_open)})")
 print(f" Difference: {eyes_closed.mean() - eyes_open.mean():.6f}")
 print(f" t-statistic: {t_stat:.6f}")
 print(f" p-value: {p_value:.6f}")
 
 if p_value < 0.05:
 print(f"\n✓ FALSIFIABILITY TEST PASSED")
 print(f" Significant difference detected (p < 0.05)")
 print(f" Model is NOT falsified.")
 else:
 print(f"\n✗ FALSIFIABILITY TEST FAILED")
 print(f" No significant difference (p >= 0.05)")
 print(f" WARNING: Model may be falsified per ENTPC.tex line 663")

# Pre/Post comparison
print(f"\n{'='*80}")
print("PRE vs POST TREATMENT COMPARISON")
print("=" * 80)

pre_df = df[df['timepoint'] == 'pre']
post_df = df[df['timepoint'] == 'post']

comparison_metrics = ['lambda_max', 'spectral_gap', 'entropy_mean', 'absurdity_gap_L2']

for metric in comparison_metrics:
 if metric in df.columns:
 pre_vals = pre_df[metric].dropna()
 post_vals = post_df[metric].dropna()
 
 if len(pre_vals) > 0 and len(post_vals) > 0:
 t_stat, p_value = stats.ttest_ind(pre_vals, post_vals)
 
 print(f"\n{metric}:")
 print(f" Pre: {pre_vals.mean():.6f} ± {pre_vals.std():.6f} (n={len(pre_vals)})")
 print(f" Post: {post_vals.mean():.6f} ± {post_vals.std():.6f} (n={len(post_vals)})")
 print(f" Δ: {post_vals.mean() - pre_vals.mean():.6f}")
 print(f" p: {p_value:.6f} {'*' if p_value < 0.05 else ''}")

# Eigenvalue spectrum summary
print(f"\n{'='*80}")
print("EIGENVALUE SPECTRUM SUMMARY")
print("=" * 80)

eig_cols = [f'eigenvalue_{i}' for i in range(16)]
eig_means = [df[col].mean() for col in eig_cols]
eig_stds = [df[col].std() for col in eig_cols]

print(f"\n{'Index':<8} {'Mean':<15} {'Std':<15} {'Mean/λ_max':<15}")
print("-" * 60)
for i in range(16):
 ratio = eig_means[i] / eig_means[0] if eig_means[0] > 0 else 0
 print(f"{i:<8} {eig_means[i]:<15.6f} {eig_stds[i]:<15.6f} {ratio:<15.6f}")

# Save summary to file
output_path = Path('/home/ubuntu/entptc-implementation/outputs/ANALYSIS_SUMMARY.txt')
with open(output_path, 'w') as f:
 f.write("ENTPC ANALYSIS - COMPREHENSIVE SUMMARY REPORT\n")
 f.write("=" * 80 + "\n\n")
 f.write(f"Total recordings: {len(df)}\n")
 f.write(f"Unique subjects: {df['subject_id'].nunique()}\n\n")
 
 f.write("KEY FINDINGS:\n")
 f.write(f"- Mean λ_max: {df['lambda_max'].mean():.6f}\n")
 f.write(f"- Mean Spectral Gap: {df['spectral_gap'].mean():.6f}\n")
 f.write(f"- Mean Absurdity Gap (L2): {df['absurdity_gap_L2'].mean():.6f}\n\n")
 
 f.write(f"FALSIFIABILITY TEST:\n")
 f.write(f"- Eyes Closed Absurdity Gap: {eyes_closed.mean():.6f}\n")
 f.write(f"- Eyes Open Absurdity Gap: {eyes_open.mean():.6f}\n")
 f.write(f"- p-value: {p_value:.6f}\n")
 f.write(f"- Result: {'PASSED' if p_value < 0.05 else 'FAILED'}\n")

print(f"\n{'='*80}")
print(f"Summary saved to: {output_path}")
print("=" * 80)
