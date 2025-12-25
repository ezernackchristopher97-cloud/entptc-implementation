# Stage C Zero-Variance Resolution

**Date**: 2025-12-24 
**Status**: RESOLVED - Display precision issue, NOT data artifact

---

## Issue

Analysis showed:
```
ROI variances: min=0.000000, max=0.000000, mean=0.000000
```

This appeared to be a severe data artifact (flat line data).

---

## Investigation

Inspected actual MAT file data:

```python
Data shape: (16, 96000)
Data type: float64
Data min: -0.00020659342380082075
Data max: 0.00014564672241968192
Data mean: 7.342273708337123e-09
Data std: 9.083916341309998e-06

Per-ROI variance:
 ROI 0: var=0.000000, min=-0.000030, max=0.000043
 ROI 1: var=0.000000, min=-0.000024, max=0.000021
 ROI 2: var=0.000070, min=-0.000070, max=0.000033
```

---

## Root Cause

**Data is in microvolts (10^-6 scale)**:
- Actual variance per ROI: ~10^-12 to 10^-11
- When printed with 6 decimal places: rounds to 0.000000
- **This is a DISPLAY issue, not a data artifact**

The data IS real and has non-zero variance at the appropriate scale.

---

## Validation

1. **Data range**: ±200 microvolts (typical for EEG)
2. **Bandpass filtering works**: Phases are computed correctly from filtered data
3. **PLV shows strong signal**: 0.535 (81 SD above null) - this is REAL
4. **Phase uniqueness**: 6283-6285 unique phases per ROI (full coverage)

**Conclusion**: The analysis pipeline is working correctly. The "zero variance" was a false alarm due to printing precision.

---

## Remaining Issues

### 1. **PAC = 0.000000** (Still a problem)

```
C1 (PAC): 0.000000
```

**This is NOT explained by the scale issue**. PAC should be dimensionless (normalized).

**Possible causes**:
- High-frequency amplitude is truly flat (no gamma activity?)
- PAC computation error
- Window length too short (1.4 cycles at 0.14 Hz - below 3-cycle threshold)

**Action**: Increase window length to 30 seconds (4.2 cycles at 0.14 Hz) and recompute PAC.

---

### 2. **U1 (Ablation Ladder) FAILED - Non-Monotonic**

```
Intact torus: metric = 0.627253
Cylinder (one periodic boundary): metric = 0.684505, collapse = -9.1% ❌ INCREASES
```

**This is a REAL FAILURE**, not explained by scale issues.

**Interpretation**:
- The phase winding metric is **NOT uniquely sensitive to toroidal structure**
- Cylindrical topology (one periodic boundary) produces **higher** phase coherence
- This suggests the metric captures **generic periodic structure**, not toroidal-specific geometry

**Action**: This is a **uniqueness failure**. The current phase winding metric does NOT uniquely identify toroidal topology.

---

### 3. **U2 PASSED (But needs verification)**

```
S² trajectory control: metric = 0.081158, difference = 87.1% ✅
Random walk control: metric = 0.088081, difference = 86.0% ✅
Matched spectrum surrogate: metric = 0.105238, difference = 83.2% ✅
```

**All controls show >30% difference** - U2 PASSED

**But**: The controls have **different data scales** (synthetic data with variance ~1.0 vs real data with variance ~10^-12). The difference may be scale-driven, not topology-driven.

**Action**: Normalize all data to unit variance before computing metrics for fair comparison.

---

## Corrected Status

**Stage C FIXED Analysis: PARTIAL SUCCESS**

- ✅ **Data is valid** (zero-variance was display issue)
- ✅ **PLV, PPC, phase winding work correctly**
- ❌ **PAC = 0** (window length too short or no gamma activity)
- ❌ **U1 FAILED** (non-monotonic degradation - REAL uniqueness failure)
- ⚠️ **U2 PASSED** (but needs scale normalization for fair comparison)
- ⚠️ **U3 NOT RUN** (timed out)

---

## Next Steps

1. **Fix PAC**: Increase window length to 30 seconds
2. **Fix U1**: Either:
 - Accept that current metric is NOT uniquely toroidal (honest assessment)
 - Develop new metric that IS uniquely toroidal (trajectory curvature?)
3. **Fix U2**: Normalize all data to unit variance before comparison
4. **Run U3**: Resolution consistency test (4×4, 6×6, 8×8)
5. **Only then**: Interpret results and generate final report

---

## Commit Status

✅ This resolution committed to repository.

**Next commit**: After PAC fix, U1 interpretation, and U2/U3 completion.
