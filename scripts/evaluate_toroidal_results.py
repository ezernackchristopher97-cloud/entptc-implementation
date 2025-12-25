"""
Evaluate Toroidal EntPTC Results

Determine which case (A-E) applies based on Dataset Set 2 results.

"""

import pandas as pd
import numpy as np
from scipy import stats

# Load results
df = pd.read_csv('/home/ubuntu/entptc-implementation/outputs/toroidal_analysis/dataset_set_2_toroidal.csv')

print("=" * 80)
print("TOROIDAL ENTPTC RESULTS EVALUATION")
print("=" * 80)
print(f"Dataset: Dataset Set 2 (PhysioNet Motor Movement)")
print(f"Total recordings: {len(df)}")
print(f"Subjects: {df['subject_id'].nunique()}")
print(f"Conditions: {df['condition'].unique()}")
print("=" * 80)

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

metrics = ['lambda_max', 'spectral_gap', 'entropy', 'absurdity_gap_l2']

for metric in metrics:
 mean = df[metric].mean()
 std = df[metric].std()
 print(f"{metric:20s}: {mean:8.4f} ± {std:6.4f}")

# Regime distribution
print("\n" + "=" * 80)
print("REGIME DISTRIBUTION")
print("=" * 80)

regime_counts = df['regime'].value_counts()
for regime, count in regime_counts.items():
 pct = 100 * count / len(df)
 print(f"{regime:15s}: {count:3d} / {len(df):3d} ({pct:5.1f}%)")

# Eyes Open vs Eyes Closed comparison
print("\n" + "=" * 80)
print("EYES OPEN VS EYES CLOSED COMPARISON")
print("=" * 80)

eyes_closed = df[df['condition'] == 'EyesClosed']
eyes_open = df[df['condition'] == 'EyesOpen']

print(f"\nEyes Closed: n = {len(eyes_closed)}")
print(f"Eyes Open: n = {len(eyes_open)}")

comparison_results = {}

for metric in metrics:
 closed_vals = eyes_closed[metric].values
 open_vals = eyes_open[metric].values
 
 # Paired t-test
 t_stat, p_value = stats.ttest_rel(closed_vals, open_vals)
 
 closed_mean = closed_vals.mean()
 open_mean = open_vals.mean()
 delta = closed_mean - open_mean
 
 print(f"\n{metric}:")
 print(f" Eyes Closed: {closed_mean:.4f}")
 print(f" Eyes Open: {open_mean:.4f}")
 print(f" Δ (C - O): {delta:+.4f}")
 print(f" t-statistic: {t_stat:.4f}")
 print(f" p-value: {p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
 
 comparison_results[metric] = {
 'closed_mean': closed_mean,
 'open_mean': open_mean,
 'delta': delta,
 'p_value': p_value,
 'significant': p_value < 0.05
 }

# Case determination
print("\n" + "=" * 80)
print("CASE DETERMINATION (A-E)")
print("=" * 80)

# Check for significant differences
significant_metrics = [m for m, r in comparison_results.items() if r['significant']]

print(f"\nSignificant metrics (p < 0.05): {len(significant_metrics)}/{len(metrics)}")
if significant_metrics:
 print(f" - {', '.join(significant_metrics)}")
else:
 print(" - None")

# Determine case
if len(significant_metrics) >= 2:
 print("\n✅ CASE A: Toroidal improves discrimination")
 print(" → Validate geometric hypothesis")
 print(" → Compute trajectory curvature, winding numbers")
 print(" → Check attractor stability")
 case = "A"
elif len(significant_metrics) == 1:
 print("\n⚠️ CASE B: Weak but present signal")
 print(" → Reframe around trajectory geometry")
 print(" → Focus on the one sensitive metric")
 case = "B"
elif comparison_results['spectral_gap']['delta'] > 0.1:
 print("\n⚠️ CASE D: Entropy works, geometry doesn't")
 print(" → Preserve structure, entropy dominant")
 print(" → Toroidal topology may not be critical")
 case = "D"
else:
 print("\n❌ CASE C or E: No structure detected")
 print(" → Verify implementation")
 print(" → Check if dataset limitation")
 case = "C_or_E"

# Toroidal constraint effectiveness
print("\n" + "=" * 80)
print("TOROIDAL CONSTRAINT ANALYSIS")
print("=" * 80)

print(f"\nConstraint strength: {df['constraint_strength'].iloc[0]}")
print(f"All recordings constrained: {df['toroidal_constrained'].all()}")

# Check if spectral gap distribution changed
print(f"\nSpectral gap range: [{df['spectral_gap'].min():.3f}, {df['spectral_gap'].max():.3f}]")
print(f"Mean spectral gap: {df['spectral_gap'].mean():.3f}")
print(f"Std spectral gap: {df['spectral_gap'].std():.3f}")

# Regime transitions
regime_transitions = 0
for subject_id in df['subject_id'].unique():
 subject_data = df[df['subject_id'] == subject_id]
 if len(subject_data['regime'].unique()) > 1:
 regime_transitions += 1

print(f"\nSubjects with regime transitions: {regime_transitions}/{df['subject_id'].nunique()}")

# Final assessment
print("\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

print(f"\n**CASE: {case}**")

if case == "A":
 print("\n✅ Toroidal constraints show promise")
 print(" Next steps:")
 print(" 1. Compute geometric metrics (trajectory curvature, winding)")
 print(" 2. Validate attractor structure")
 print(" 3. Test on Dataset Set 3")
elif case == "B":
 print("\n⚠️ Weak signal detected")
 print(" Next steps:")
 print(" 1. Focus on sensitive metric")
 print(" 2. Reframe around trajectory geometry")
 print(" 3. Consider alternative datasets")
elif case == "D":
 print("\n⚠️ Entropy-dominated dynamics")
 print(" Next steps:")
 print(" 1. Simplify model around entropy")
 print(" 2. Question necessity of toroidal topology")
 print(" 3. Test with task-based data")
else:
 print("\n❌ No clear structure")
 print(" Next steps:")
 print(" 1. Verify implementation correctness")
 print(" 2. Test with task-based EEG (working memory, cognitive control)")
 print(" 3. Consider dataset limitation hypothesis")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)

# Save evaluation summary
summary = {
 'case': case,
 'significant_metrics': significant_metrics,
 'n_significant': len(significant_metrics),
 'regime_transitions': regime_transitions,
 'mean_spectral_gap': df['spectral_gap'].mean(),
 'mean_entropy': df['entropy'].mean(),
 'mean_absurdity_gap': df['absurdity_gap_l2'].mean()
}

import json
with open('/home/ubuntu/entptc-implementation/outputs/toroidal_analysis/evaluation_summary.json', 'w') as f:
 json.dump(summary, f, indent=2)

print(f"\nEvaluation summary saved to: evaluation_summary.json")
