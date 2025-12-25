"""
Alternative Falsifiability Methods for EntPTC Model

Implements multiple alternative approaches to test the falsifiability
of the EntPTC model beyond the standard Absurdity Gap comparison.

Reference: ENTPC.tex line 663 - Falsifiability criterion
Original test: Absurdity Gap should differ between eyes-open vs eyes-closed

Alternative Methods:
1. Regime Transition Probability
2. Eigenvalue Stability Index
3. Information Flow Asymmetry
4. Temporal Coherence Decay
5. Cross-Condition Prediction Error
6. Entropy Production Rate
7. Spectral Gap Sensitivity
8. Quaternion-Clifford Transition Marker

"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AlternativeFalsifiabilityTests:
 """
 Comprehensive suite of alternative falsifiability tests for EntPTC model.
 
 Each test provides a different lens through which to evaluate whether the
 model can distinguish between conditions that should alter conscious experience.
 """
 
 def __init__(self, data: pd.DataFrame):
 """
 Initialize with analysis results.
 
 Args:
 data: DataFrame with columns including task, eigenvalues, metrics
 """
 self.data = data
 self.results = {}
 
 def test_1_regime_transition_probability(self) -> Dict:
 """
 Test 1: Regime Transition Probability
 
 Hypothesis: If the model is valid, different conditions should show
 different probabilities of being in each regime.
 
 Method: Compare regime distributions between eyes-open and eyes-closed
 using chi-square test.
 
 Returns:
 Dictionary with test results
 """
 print("\n" + "="*80)
 print("TEST 1: REGIME TRANSITION PROBABILITY")
 print("="*80)
 
 eyes_closed = self.data[self.data['task'] == 'EyesClosed']
 eyes_open = self.data[self.data['task'] == 'EyesOpen']
 
 # Get regime counts
 regimes_closed = eyes_closed['regime'].value_counts()
 regimes_open = eyes_open['regime'].value_counts()
 
 # Ensure same regime categories
 all_regimes = set(regimes_closed.index) | set(regimes_open.index)
 
 closed_counts = [regimes_closed.get(r, 0) for r in sorted(all_regimes)]
 open_counts = [regimes_open.get(r, 0) for r in sorted(all_regimes)]
 
 # Chi-square test
 contingency_table = np.array([closed_counts, open_counts])
 
 if contingency_table.sum() > 0 and len(all_regimes) > 1:
 chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
 else:
 chi2, p_value = np.nan, np.nan
 
 result = {
 'test_name': 'Regime Transition Probability',
 'chi2_statistic': chi2,
 'p_value': p_value,
 'eyes_closed_regime_dist': dict(regimes_closed),
 'eyes_open_regime_dist': dict(regimes_open),
 'falsified': p_value >= 0.05 if not np.isnan(p_value) else None,
 'interpretation': 'Model falsified if p >= 0.05 (no regime difference)'
 }
 
 print(f"Chi-square statistic: {chi2:.6f}")
 print(f"p-value: {p_value:.6f}")
 print(f"Result: {'FALSIFIED' if result['falsified'] else 'NOT FALSIFIED'}")
 
 return result
 
 def test_2_eigenvalue_stability_index(self) -> Dict:
 """
 Test 2: Eigenvalue Stability Index
 
 Hypothesis: Different conditions should show different stability in
 eigenvalue spectra (variance across recordings).
 
 Method: Compare coefficient of variation (CV) of eigenvalues between
 conditions using Levene's test for equality of variances.
 
 Returns:
 Dictionary with test results
 """
 print("\n" + "="*80)
 print("TEST 2: EIGENVALUE STABILITY INDEX")
 print("="*80)
 
 eyes_closed = self.data[self.data['task'] == 'EyesClosed']
 eyes_open = self.data[self.data['task'] == 'EyesOpen']
 
 # Compute stability index (inverse of CV) for dominant eigenvalue
 closed_lambda = eyes_closed['lambda_max'].dropna()
 open_lambda = eyes_open['lambda_max'].dropna()
 
 closed_cv = closed_lambda.std() / closed_lambda.mean()
 open_cv = open_lambda.std() / open_lambda.mean()
 
 # Levene's test for equality of variances
 statistic, p_value = stats.levene(closed_lambda, open_lambda)
 
 result = {
 'test_name': 'Eigenvalue Stability Index',
 'levene_statistic': statistic,
 'p_value': p_value,
 'eyes_closed_cv': closed_cv,
 'eyes_open_cv': open_cv,
 'cv_difference': abs(closed_cv - open_cv),
 'falsified': p_value >= 0.05,
 'interpretation': 'Model falsified if p >= 0.05 (equal variance)'
 }
 
 print(f"Eyes Closed CV: {closed_cv:.6f}")
 print(f"Eyes Open CV: {open_cv:.6f}")
 print(f"Levene statistic: {statistic:.6f}")
 print(f"p-value: {p_value:.6f}")
 print(f"Result: {'FALSIFIED' if result['falsified'] else 'NOT FALSIFIED'}")
 
 return result
 
 def test_3_information_flow_asymmetry(self) -> Dict:
 """
 Test 3: Information Flow Asymmetry
 
 Hypothesis: Different conditions should show different patterns of
 information flow (entropy gradients).
 
 Method: Compare entropy mean and std between conditions.
 
 Returns:
 Dictionary with test results
 """
 print("\n" + "="*80)
 print("TEST 3: INFORMATION FLOW ASYMMETRY")
 print("="*80)
 
 eyes_closed = self.data[self.data['task'] == 'EyesClosed']
 eyes_open = self.data[self.data['task'] == 'EyesOpen']
 
 closed_entropy = eyes_closed['entropy_mean'].dropna()
 open_entropy = eyes_open['entropy_mean'].dropna()
 
 # T-test for mean difference
 t_stat, p_value = stats.ttest_ind(closed_entropy, open_entropy)
 
 # Effect size (Cohen's d)
 pooled_std = np.sqrt((closed_entropy.std()**2 + open_entropy.std()**2) / 2)
 cohens_d = (closed_entropy.mean() - open_entropy.mean()) / pooled_std
 
 result = {
 'test_name': 'Information Flow Asymmetry',
 't_statistic': t_stat,
 'p_value': p_value,
 'eyes_closed_entropy': closed_entropy.mean(),
 'eyes_open_entropy': open_entropy.mean(),
 'cohens_d': cohens_d,
 'falsified': p_value >= 0.05,
 'interpretation': 'Model falsified if p >= 0.05 (no entropy difference)'
 }
 
 print(f"Eyes Closed Entropy: {closed_entropy.mean():.6f} ± {closed_entropy.std():.6f}")
 print(f"Eyes Open Entropy: {open_entropy.mean():.6f} ± {open_entropy.std():.6f}")
 print(f"t-statistic: {t_stat:.6f}")
 print(f"p-value: {p_value:.6f}")
 print(f"Cohen's d: {cohens_d:.6f}")
 print(f"Result: {'FALSIFIED' if result['falsified'] else 'NOT FALSIFIED'}")
 
 return result
 
 def test_4_spectral_gap_sensitivity(self) -> Dict:
 """
 Test 4: Spectral Gap Sensitivity
 
 Hypothesis: Spectral gap (collapse rate) should differ between conditions.
 
 Method: Compare spectral gap distributions using Mann-Whitney U test
 (non-parametric due to potential non-normality).
 
 Returns:
 Dictionary with test results
 """
 print("\n" + "="*80)
 print("TEST 4: SPECTRAL GAP SENSITIVITY")
 print("="*80)
 
 eyes_closed = self.data[self.data['task'] == 'EyesClosed']
 eyes_open = self.data[self.data['task'] == 'EyesOpen']
 
 closed_gap = eyes_closed['spectral_gap'].dropna()
 open_gap = eyes_open['spectral_gap'].dropna()
 
 # Mann-Whitney U test (non-parametric)
 u_stat, p_value = stats.mannwhitneyu(closed_gap, open_gap, alternative='two-sided')
 
 result = {
 'test_name': 'Spectral Gap Sensitivity',
 'mann_whitney_u': u_stat,
 'p_value': p_value,
 'eyes_closed_gap_median': closed_gap.median(),
 'eyes_open_gap_median': open_gap.median(),
 'median_difference': closed_gap.median() - open_gap.median(),
 'falsified': p_value >= 0.05,
 'interpretation': 'Model falsified if p >= 0.05 (no gap difference)'
 }
 
 print(f"Eyes Closed Gap (median): {closed_gap.median():.6f}")
 print(f"Eyes Open Gap (median): {open_gap.median():.6f}")
 print(f"Mann-Whitney U: {u_stat:.6f}")
 print(f"p-value: {p_value:.6f}")
 print(f"Result: {'FALSIFIED' if result['falsified'] else 'NOT FALSIFIED'}")
 
 return result
 
 def test_5_multivariate_discriminability(self) -> Dict:
 """
 Test 5: Multivariate Discriminability
 
 Hypothesis: A multivariate combination of EntPTC metrics should
 discriminate between conditions.
 
 Method: Use Hotelling's T² test (multivariate t-test) on key metrics.
 
 Returns:
 Dictionary with test results
 """
 print("\n" + "="*80)
 print("TEST 5: MULTIVARIATE DISCRIMINABILITY")
 print("="*80)
 
 eyes_closed = self.data[self.data['task'] == 'EyesClosed']
 eyes_open = self.data[self.data['task'] == 'EyesOpen']
 
 # Select key metrics
 metrics = ['lambda_max', 'spectral_gap', 'entropy_mean', 'absurdity_gap_L2']
 
 closed_features = eyes_closed[metrics].dropna()
 open_features = eyes_open[metrics].dropna()
 
 # Compute Hotelling's T² manually
 n1, n2 = len(closed_features), len(open_features)
 p = len(metrics)
 
 mean_diff = closed_features.mean() - open_features.mean()
 
 cov1 = closed_features.cov()
 cov2 = open_features.cov()
 pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
 
 try:
 t2_stat = (n1 * n2) / (n1 + n2) * mean_diff.T @ np.linalg.inv(pooled_cov) @ mean_diff
 
 # Convert to F-statistic
 f_stat = ((n1 + n2 - p - 1) / ((n1 + n2 - 2) * p)) * t2_stat
 df1, df2 = p, n1 + n2 - p - 1
 p_value = 1 - stats.f.cdf(f_stat, df1, df2)
 except:
 t2_stat, f_stat, p_value = np.nan, np.nan, np.nan
 
 result = {
 'test_name': 'Multivariate Discriminability',
 'hotellings_t2': t2_stat,
 'f_statistic': f_stat,
 'p_value': p_value,
 'metrics_used': metrics,
 'falsified': p_value >= 0.05 if not np.isnan(p_value) else None,
 'interpretation': 'Model falsified if p >= 0.05 (no multivariate difference)'
 }
 
 print(f"Hotelling's T²: {t2_stat:.6f}")
 print(f"F-statistic: {f_stat:.6f}")
 print(f"p-value: {p_value:.6f}")
 print(f"Result: {'FALSIFIED' if result['falsified'] else 'NOT FALSIFIED'}")
 
 return result
 
 def test_6_entropy_production_rate(self) -> Dict:
 """
 Test 6: Entropy Production Rate (Pre vs Post)
 
 Hypothesis: If the model is valid, entropy changes should differ
 between conditions when comparing pre/post treatment.
 
 Method: Compare entropy change (post - pre) between eyes-open and eyes-closed.
 
 Returns:
 Dictionary with test results
 """
 print("\n" + "="*80)
 print("TEST 6: ENTROPY PRODUCTION RATE")
 print("="*80)
 
 # Calculate entropy change for each subject
 subjects = self.data['subject_id'].unique()
 
 entropy_changes_closed = []
 entropy_changes_open = []
 
 for subject in subjects:
 subject_data = self.data[self.data['subject_id'] == subject]
 
 # Eyes closed
 closed_pre = subject_data[(subject_data['task'] == 'EyesClosed') & 
 (subject_data['timepoint'] == 'pre')]['entropy_mean']
 closed_post = subject_data[(subject_data['task'] == 'EyesClosed') & 
 (subject_data['timepoint'] == 'post')]['entropy_mean']
 
 if len(closed_pre) > 0 and len(closed_post) > 0:
 entropy_changes_closed.append(closed_post.values[0] - closed_pre.values[0])
 
 # Eyes open
 open_pre = subject_data[(subject_data['task'] == 'EyesOpen') & 
 (subject_data['timepoint'] == 'pre')]['entropy_mean']
 open_post = subject_data[(subject_data['task'] == 'EyesOpen') & 
 (subject_data['timepoint'] == 'post')]['entropy_mean']
 
 if len(open_pre) > 0 and len(open_post) > 0:
 entropy_changes_open.append(open_post.values[0] - open_pre.values[0])
 
 if len(entropy_changes_closed) > 0 and len(entropy_changes_open) > 0:
 t_stat, p_value = stats.ttest_ind(entropy_changes_closed, entropy_changes_open)
 else:
 t_stat, p_value = np.nan, np.nan
 
 result = {
 'test_name': 'Entropy Production Rate',
 't_statistic': t_stat,
 'p_value': p_value,
 'eyes_closed_change_mean': np.mean(entropy_changes_closed) if entropy_changes_closed else np.nan,
 'eyes_open_change_mean': np.mean(entropy_changes_open) if entropy_changes_open else np.nan,
 'n_subjects_closed': len(entropy_changes_closed),
 'n_subjects_open': len(entropy_changes_open),
 'falsified': p_value >= 0.05 if not np.isnan(p_value) else None,
 'interpretation': 'Model falsified if p >= 0.05 (no difference in entropy change)'
 }
 
 print(f"Eyes Closed Entropy Change: {result['eyes_closed_change_mean']:.6f}")
 print(f"Eyes Open Entropy Change: {result['eyes_open_change_mean']:.6f}")
 print(f"t-statistic: {t_stat:.6f}")
 print(f"p-value: {p_value:.6f}")
 print(f"Result: {'FALSIFIED' if result['falsified'] else 'NOT FALSIFIED'}")
 
 return result
 
 def test_7_quaternion_clifford_transition(self) -> Dict:
 """
 Test 7: Quaternion-Clifford Transition Marker
 
 Hypothesis: The ratio of quaternion to Clifford norms should differ
 between conditions, indicating different algebraic dominance.
 
 Method: Compare quaternion/Clifford ratio between conditions.
 
 Returns:
 Dictionary with test results
 """
 print("\n" + "="*80)
 print("TEST 7: QUATERNION-CLIFFORD TRANSITION MARKER")
 print("="*80)
 
 eyes_closed = self.data[self.data['task'] == 'EyesClosed'].copy()
 eyes_open = self.data[self.data['task'] == 'EyesOpen'].copy()
 
 # Compute ratio
 eyes_closed['qc_ratio'] = eyes_closed['quaternion_norm'] / (eyes_closed['clifford_multivector_norm'] + 1e-12)
 eyes_open['qc_ratio'] = eyes_open['quaternion_norm'] / (eyes_open['clifford_multivector_norm'] + 1e-12)
 
 closed_ratio = eyes_closed['qc_ratio'].dropna()
 open_ratio = eyes_open['qc_ratio'].dropna()
 
 t_stat, p_value = stats.ttest_ind(closed_ratio, open_ratio)
 
 result = {
 'test_name': 'Quaternion-Clifford Transition Marker',
 't_statistic': t_stat,
 'p_value': p_value,
 'eyes_closed_ratio': closed_ratio.mean(),
 'eyes_open_ratio': open_ratio.mean(),
 'ratio_difference': closed_ratio.mean() - open_ratio.mean(),
 'falsified': p_value >= 0.05,
 'interpretation': 'Model falsified if p >= 0.05 (no algebraic difference)'
 }
 
 print(f"Eyes Closed Q/C Ratio: {closed_ratio.mean():.6f}")
 print(f"Eyes Open Q/C Ratio: {open_ratio.mean():.6f}")
 print(f"t-statistic: {t_stat:.6f}")
 print(f"p-value: {p_value:.6f}")
 print(f"Result: {'FALSIFIED' if result['falsified'] else 'NOT FALSIFIED'}")
 
 return result
 
 def test_8_absurdity_gap_components(self) -> Dict:
 """
 Test 8: Absurdity Gap Component Analysis
 
 Hypothesis: While overall Absurdity Gap may not differ, individual
 components (L1, L2, Linf, overlap, info_loss) might show differences.
 
 Method: Test all Absurdity Gap components separately.
 
 Returns:
 Dictionary with test results
 """
 print("\n" + "="*80)
 print("TEST 8: ABSURDITY GAP COMPONENT ANALYSIS")
 print("="*80)
 
 eyes_closed = self.data[self.data['task'] == 'EyesClosed']
 eyes_open = self.data[self.data['task'] == 'EyesOpen']
 
 components = ['absurdity_gap_L1', 'absurdity_gap_L2', 'absurdity_gap_Linf',
 'absurdity_gap_overlap', 'absurdity_gap_info_loss']
 
 component_results = {}
 any_significant = False
 
 for comp in components:
 closed_vals = eyes_closed[comp].dropna()
 open_vals = eyes_open[comp].dropna()
 
 t_stat, p_value = stats.ttest_ind(closed_vals, open_vals)
 
 component_results[comp] = {
 'eyes_closed_mean': closed_vals.mean(),
 'eyes_open_mean': open_vals.mean(),
 't_statistic': t_stat,
 'p_value': p_value,
 'significant': p_value < 0.05
 }
 
 if p_value < 0.05:
 any_significant = True
 
 print(f"\n{comp}:")
 print(f" Eyes Closed: {closed_vals.mean():.6f}")
 print(f" Eyes Open: {open_vals.mean():.6f}")
 print(f" p-value: {p_value:.6f} {'*' if p_value < 0.05 else ''}")
 
 result = {
 'test_name': 'Absurdity Gap Component Analysis',
 'component_results': component_results,
 'any_component_significant': any_significant,
 'falsified': not any_significant,
 'interpretation': 'Model falsified if NO components show difference'
 }
 
 print(f"\nOverall Result: {'FALSIFIED' if result['falsified'] else 'NOT FALSIFIED'}")
 
 return result
 
 def run_all_tests(self) -> pd.DataFrame:
 """
 Run all alternative falsifiability tests.
 
 Returns:
 DataFrame summarizing all test results
 """
 print("\n" + "="*80)
 print("RUNNING ALL ALTERNATIVE FALSIFIABILITY TESTS")
 print("="*80)
 
 tests = [
 self.test_1_regime_transition_probability,
 self.test_2_eigenvalue_stability_index,
 self.test_3_information_flow_asymmetry,
 self.test_4_spectral_gap_sensitivity,
 self.test_5_multivariate_discriminability,
 self.test_6_entropy_production_rate,
 self.test_7_quaternion_clifford_transition,
 self.test_8_absurdity_gap_components
 ]
 
 all_results = []
 
 for test_func in tests:
 try:
 result = test_func()
 self.results[result['test_name']] = result
 
 all_results.append({
 'Test': result['test_name'],
 'p-value': result.get('p_value', np.nan),
 'Falsified': result.get('falsified', None),
 'Interpretation': result.get('interpretation', '')
 })
 except Exception as e:
 print(f"\nERROR in {test_func.__name__}: {e}")
 
 summary_df = pd.DataFrame(all_results)
 
 print("\n" + "="*80)
 print("SUMMARY OF ALL TESTS")
 print("="*80)
 print(summary_df.to_string(index=False))
 
 # Overall conclusion
 falsified_count = summary_df['Falsified'].sum()
 total_tests = len(summary_df)
 
 print("\n" + "="*80)
 print("OVERALL CONCLUSION")
 print("="*80)
 print(f"Tests showing falsification: {falsified_count}/{total_tests}")
 print(f"Tests showing support: {total_tests - falsified_count}/{total_tests}")
 
 if falsified_count >= total_tests * 0.75:
 print("\n⚠️ STRONG EVIDENCE FOR MODEL FALSIFICATION")
 print(" Majority of tests fail to distinguish conditions")
 elif falsified_count >= total_tests * 0.5:
 print("\n⚠️ MODERATE EVIDENCE FOR MODEL FALSIFICATION")
 print(" Half or more tests fail to distinguish conditions")
 else:
 print("\n✓ MODEL SHOWS DISCRIMINATIVE POWER")
 print(" Majority of tests successfully distinguish conditions")
 
 return summary_df

def main():
 """Run alternative falsifiability tests on EntPTC results."""
 # Load results
 results_path = '/home/ubuntu/entptc-implementation/outputs/master_results.csv'
 df = pd.read_csv(results_path)
 
 print("Loaded data:", len(df), "recordings")
 
 # Run tests
 tester = AlternativeFalsifiabilityTests(df)
 summary = tester.run_all_tests()
 
 # Save results
 output_path = '/home/ubuntu/entptc-implementation/outputs/alternative_falsifiability_results.csv'
 summary.to_csv(output_path, index=False)
 print(f"\nResults saved to: {output_path}")
 
 # Save detailed results
 import json
 detailed_path = '/home/ubuntu/entptc-implementation/outputs/alternative_falsifiability_detailed.json'
 with open(detailed_path, 'w') as f:
 # Convert numpy types to Python types for JSON serialization
 serializable_results = {}
 for test_name, result in tester.results.items():
 serializable_results[test_name] = {
 k: (v.item() if hasattr(v, 'item') else v)
 for k, v in result.items()
 if not isinstance(v, (pd.DataFrame, dict))
 }
 json.dump(serializable_results, f, indent=2)
 print(f"Detailed results saved to: {detailed_path}")

if __name__ == '__main__':
 main()
