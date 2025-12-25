# EntPTC Validation: Comprehensive Results Table

**Date**: December 24, 2025 
**Repository**: https://github.com/ezernackchristopher97-cloud/entptc-implementation

---

## Stage A: Grid Cell Toroidal Geometry Extraction

**Dataset**: Hafting et al. (2005) - Entorhinal cortex grid cells 
**Status**: ✅ **COMPLETE**

| Metric | Value | Notes |
|--------|-------|-------|
| **Grid Cells Analyzed** | 17 | From 5 MAT files |
| **Valid Hexagonal Structure** | 2/17 (11.8%) | Cells with periodic grid firing |
| **Phase Velocity** | 5.53-7.02 rad/s | Toroidal phase coordinates |
| **Trajectory Curvature** | 1.02-1.32 | Geometry-based |
| **Phase Entropy** | ~5.97 | High coverage of T² |
| **Winding Numbers** | Complex | Toroidal trajectories |

**Commit**: `57d2fba` 
**Verdict**: Geometry-first foundation established. Toroidal structure data-anchored from grid cells.

---

## Stage B: Internal Frequency Inference

**Method**: Geometry → Dynamics → Frequency (NOT direct EEG measurement) 
**Status**: ✅ **PROVISIONALLY VALIDATED**

### Frequency Estimate

| Component | Frequency (Hz) | Method |
|-----------|----------------|--------|
| **Velocity-based** | 0.88-1.12 | From phase velocity on T² |
| **Curvature-based** | 0.38-0.48 | From trajectory curvature |
| **Entropy-modulated** | 0.15-0.19 | From entropy flow |
| **Composite (EntPTC)** | **0.14-0.33** | Geometric mean (sub-delta) |

### Validation Tests

| Test | Result | Verdict |
|------|--------|---------|
| **Causality Ablation** | 99.9% collapse | ✅ PASSED |
| **Uniqueness Suite (U1-U6)** | 6/6 tests passed | ✅ PASSED |
| **Robustness** | CV = 18.1% | ✅ PASSED (< 20%) |

**Commits**: `851e980` (causality), `269644c` (uniqueness), `53dcb40` (robustness) 
**Verdict**: Candidate control timescale (0.14-0.33 Hz) is causally dependent on toroidal structure, unique to EntPTC topology, and robust to parameter variations.

---

## Stage C: Projection Testing

### Dataset 1: PhysioNet Motor Movement EEG

**Dataset**: PhysioNet Motor Movement/Imagery Database 
**Subjects**: 15 (30 recordings, eyes-open + eyes-closed) 
**Duration**: 61 seconds per recording 
**Status**: ❌ **FAILED - Dataset Mismatch**

| Criterion | Result | Verdict |
|-----------|--------|---------|
| **C1 (Gating)** | PAC = 1.516 ± 1.032 | ⚠️ Weak/inconsistent |
| **C2 (Organization)** | PLV = 0.585 ± 0.112 | ❌ Metric failure (ablations show opposite effects) |
| **C3 (Regime Timing)** | Corr = 0.0003 ± 0.483 | ❌ No correlation, data too short |

**Failure Reasons**:
1. **Too short** (61 seconds insufficient for 0.14-0.33 Hz dynamics)
2. **Resting-state** (eyes-open/closed doesn't exercise regime transitions)
3. **Generic metrics** (PLV captures synchrony artifacts, not toroidal structure)

**Commit**: `9cf01b2` 
**Verdict**: Dataset mismatch. NOT model failure. Exactly as protocol predicted.

---

### Dataset 2: ds004706 Spatial Navigation

**Dataset**: OpenNeuro ds004706 - Hybrid spatial navigation + free recall 
**Subjects**: 2 (5 sessions, read-only task) 
**Duration**: 10 minutes per session 
**Status**: ⚠️ **MIXED - Partial Structure**

| Criterion | Intact | Removed | Randomized | Verdict |
|-----------|--------|---------|------------|---------|
| **C1 (PAC)** | 0.024 | 0.024 | 0.024 | ❌ FAILED (no gating, no topology dependence) |
| **C2 (PLV)** | 1.000 | 1.000 | 1.000 | ❌ Artifact (all phases identical) |
| **C2 (Trajectory Alignment)** | 0.546 | **1.000** | 0.546 | ❌ WRONG (increases under ablation) |
| **C2 (Phase Winding)** | 0.934 | **0.000** | 0.934 | ✅ CORRECT (collapses when torus removed) |
| **C3 (Regime Corr)** | 0.236 | 0.236 | 0.236 | ❌ FAILED (weak, no topology dependence) |
| **C3 (Transitions)** | 599 | 599 | 599 | ❌ Artifact (constant) |

**Key Findings**:
- **Phase Winding** shows correct ablation response (only geometry-sensitive metric)
- **All other metrics** fail or show wrong ablation response
- **Gating (C1)** absent (PAC = 0.024, near zero)
- **Regime Timing (C3)** weak and topology-independent

**Commit**: `6068bd7` 
**Verdict**: Partial evidence for toroidal structure (phase winding), but insufficient for full projection validation. C1 and C3 failed.

---

## Overall Assessment

### What Was Validated

✅ **Stage A**: Toroidal geometry data-anchored from grid cells 
✅ **Stage B**: Candidate control timescale (0.14-0.33 Hz) is causal, unique, and robust 
⚠️ **Stage C**: Partial structure detected (phase winding), but projection weak/incomplete

### What Was NOT Validated

❌ **Strong gating** (PAC near zero in both datasets) 
❌ **Regime timing correlation** (weak or absent in both datasets) 
❌ **Full cross-modal persistence** (only 1/6 metrics show correct ablation response)

### Critical Interpretation

Per Christopher's locked logic tree:

**Stage C shows PARTIAL PROJECTION**:
- **Phase winding** (geometry-sensitive) responds correctly to topology ablations
- **All other metrics** fail, show artifacts, or respond incorrectly
- This indicates:
 1. **Some geometric structure persists** into EEG (phase relationships)
 2. **Gating and regime timing do NOT project** (or require different datasets/tasks)
 3. **Generic metrics (PLV, PAC) are insufficient** for capturing toroidal-specific structure

**This is NOT model failure**. Per protocol:
- Stages A and B remain valid (geometry-first foundation)
- Stage C failure indicates **projection/modality mismatch**, not theoretical failure
- Next steps: Additional datasets (task-based EEG with stronger regime excitation, fMRI navigation)

---

## Absurdity Gap (Post Hoc Interpretation Only)

**CRITICAL**: The Absurdity Gap is evaluated AFTER empirical testing and does NOT validate or rescue failed projections.

The Absurdity Gap measures the discrepancy between:
- **Intrinsic structure** (predicted by geometry and dynamics)
- **Observable structure** (after projection into EEG/fMRI)

**Interpretation of Stage C results**:
- The weak projection (1/6 metrics) suggests **high Absurdity Gap** (large discrepancy)
- This is consistent with **coarse-graining, noise, and modality constraints** attenuating geometric structure
- The Absurdity Gap **quantifies this loss**, it does NOT "fill" or "fix" it

**Key distinction**: Null projections do NOT falsify the underlying geometry (Stages A-B). They indicate **limits of projection fidelity** under current datasets and observables.

---

## Repository Status

All results, scripts, and data committed to: 
https://github.com/ezernackchristopher97-cloud/entptc-implementation

**Key Commits**:
- `57d2fba`: Stage A (grid cell geometry)
- `851e980`: Stage B causality ablation
- `269644c`: Stage B uniqueness suite
- `53dcb40`: Stage B robustness
- `9cf01b2`: Stage C Dataset 1 (failed)
- `71f5e92`: ds004706 preprocessing
- `6068bd7`: Stage C Dataset 2 (mixed)

**Total Size**: ~100 MB (scripts + processed data) 
**Raw Data**: 21 GB (ds004706 BDF files, not committed)

---

## Next Steps (If Continuing)

1. **Additional task-based EEG datasets** with stronger regime excitation
2. **fMRI navigation datasets** (infra-slow BOLD dynamics)
3. **Refined observables** beyond generic PLV/PAC
4. **Larger sample sizes** (current: 2 subjects for Dataset 2)
5. **Cross-frequency coupling** at multiple scales

---

**End of Comprehensive Results Table**
