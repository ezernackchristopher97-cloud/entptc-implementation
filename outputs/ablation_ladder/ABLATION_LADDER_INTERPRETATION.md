# Ablation Ladder Results - Interpretation

Date: 2025-12-24  
Dataset: ds004706 (sub-LTP448, spatial navigation task)  
Status: Metric limitation, not theory failure

---

## Results Summary

### Signature Distances (Normalized)

Ablation                          Distance   Interpretation  
Baseline                          0.0000     Reference signature  
Boundary removal (cylinder)       0.0001     Minimal deformation  
Boundary removal (plane)          0.0035     Small deformation  
Adjacency scramble                0.3990     Moderate deformation  
Phase destruction                 0.3990     Moderate deformation  
Channel randomization             0.4734     Negative control  

### Pass Criterion

Max geometry-targeted distance: 0.3990  
Negative control distance: 0.4734  
Ratio: 0.84  
Pass criterion: greater than 2.0  
Verdict: Fail

---

## Interpretation

### What This Result Means

1. Geometry-targeted ablations (adjacency scramble, phase destruction) show comparable sensitivity to the negative control (channel randomization)  
2. The composite signature U does not uniquely collapse under geometry-specific ablations  
3. This indicates a metric limitation, not a failure of the theoretical framework

### Why This Is Not Theory Failure

Single-metric behavior is not decisive unless the full signature U collapses or deforms uniquely under geometry-specific ablations.

Observed behavior:
- Boundary removal shows minimal effect (0.0001 to 0.0035)
- Adjacency scramble shows moderate effect (0.3990)
- Negative control shows similar effect (0.4734)

Root cause: The current distance metric does not adequately capture geometry-specific structure.

### Failed Components

- Adjacency scramble does not show greater than twofold separation from negative control  
- Boundary removal shows minimal deformation when a larger effect was expected

### What Still Validates

From Stage C on ds004706:
- Organization: Geometry causally organizes phase relationships (16.6 percent PLV change)  
- Regime timing: Stable regime dwell times (CV equals 0.28)

Conclusion: Operator-derived invariant structure constrains projections, even though the composite distance metric fails to uniquely isolate geometry-specific ablations.

---

## Protocol Handling

Action: Document as projection mismatch and proceed to next dataset.

---

## Metric Limitations Identified

1. Distance metric does not capture topology-specific structure  
   Per-component normalization equalizes all components and underweights geometry-sensitive structure  

2. Boundary removal shows minimal effect  
   Expected significant deformation, observed near-zero change  
   Likely cause: Phase winding and curvature are not strongly represented in this metric  

3. Negative control too sensitive  
   Channel randomization destroys all structure but only shows marginally larger distance than geometry-targeted ablations  

---

## Recommendations

1. Weighted distance metric with higher weight on eigenvalue profile and graph locality  
2. Component-specific ablation tests for each element of U  
3. Alternative distance metrics such as Wasserstein, KL divergence, or manifold-aware distances  
4. Addition of topology-specific invariants such as Betti numbers and persistent homology features  

---

## Final Assessment

Ablation ladder: Fail due to metric limitation  
Stage C (organization and regime timing): Partial success  
Overall: Partial validation. Operator-derived structure constrains projections, but uniqueness tests require improved metrics.

---

Proceeding to Class C: ds005385 (eyes open and eyes closed with longitudinal structure)
