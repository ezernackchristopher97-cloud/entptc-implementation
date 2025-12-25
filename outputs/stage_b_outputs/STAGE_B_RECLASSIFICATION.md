# STAGE B RECLASSIFICATION - FORMAL PROTOCOL

**Date**: December 24, 2025 
**Status**: PROVISIONAL (PENDING ABLATION AND UNIQUENESS TESTS)

---

## Reclassification Notice

The Stage B output (≈0.42 Hz internal frequency inferred from toroidal geometry) is hereby **reclassified as PROVISIONAL**.

### Correct Classification

The ≈0.42 Hz frequency is:
- ✓ A **candidate control timescale**
- ✓ **Derived** from geometry, not measured
- ✓ **Not yet causal** (ablation required)
- ✓ "Modality-agnostic" means **derived without EEG**, not validated across modalities

### What It Is NOT (Until Tests Complete)

- ✗ A confirmed invariant
- ✗ A unique EntPTC signature
- ✗ A demonstrated operant-control mechanism
- ✗ Evidence of THz or specific frequency bands

---

## Mandatory Tests Required

### 1. Causality Check (Ablation A)

**Procedure**:
- Run identical Stage A → Stage B pipeline
- Break toroidal structure in controlled ways:
 - Remove periodic boundary conditions
 - Randomize neighbor adjacency
 - Destroy grid-cell phase coherence
- Re-infer internal frequencies under each condition

**Interpretation Rules**:
- If ≈0.4 Hz collapses/shifts → toroidal geometry is causal ✓
- If ≈0.4 Hz persists unchanged → generic infra-slow dynamics, reinterpret

**Status**: PENDING

---

### 2. Robustness Checks

The Stage B frequency estimate is invalid unless shown to be robust to:

- Grid resolution (4×4 vs higher-resolution toroidal discretizations)
- Trajectory smoothing / interpolation method
- Curvature estimator choice
- Window length and segmentation

**Requirement**: Report sensitivity ranges, not single point estimate

**Status**: PENDING

---

### 3. Uniqueness & Identifiability Tests

#### 3.1 Non-Toroidal Geometry Controls

Replace toroidal topology with:
- Cylindrical topology
- Planar grid without periodic closure
- Random manifold with matched dimensionality

Re-run Stage B frequency inference.

**Interpretation**:
- If ≈0.4 Hz specific to torus → uniqueness supported
- If appears across geometries → uniqueness fails

**Status**: PENDING

#### 3.2 Geometry-Frequency Decoupling Test

- Randomize geometric phase relationships while preserving marginal statistics
- Verify whether inferred frequency survives phase scrambling

**Interpretation**:
- Survival → generic slow dynamics
- Collapse → geometry-dependent control structure

**Status**: PENDING

#### 3.3 Parameter-Scaling Test

Systematically vary:
- Grid scale
- Curvature strength
- Adjacency weighting

Test whether inferred control frequency:
- Rescales predictably with geometry (supports mechanistic link)
- Remains fixed regardless of geometry (fails uniqueness)

**Status**: PENDING

---

## Reviewer-Safe Language (MANDATORY)

Until ablation, robustness, and uniqueness tests are complete, Stage B **MUST** be described strictly as:

> "A geometry-derived candidate control timescale emerging from toroidal grid-cell dynamics under current modeling assumptions."

It **MUST NOT** be described as:
- A confirmed invariant
- A unique EntPTC signature
- A demonstrated operant-control mechanism
- Evidence of THz or specific frequency bands

---

## Stage C Authorization (Conditional)

Stage C is authorized, but only under strict constraints.

### Stage C Question (CORRECT)

Does the geometry-derived mode:
- Gate higher-frequency activity?
- Organize phase relationships?
- Time regime transitions or task events?
- Does breaking toroidal structure remove that organization?

### Stage C Question (INCORRECT)

- ✗ "Does ≈0.4 Hz appear in EEG or fMRI?"

**Presence alone proves nothing.** EEG/fMRI are projection layers, not generators.

---

## Stage C Failure Interpretation

If Stage C fails cleanly, the **CORRECT** conclusion is:

> "Geometry-derived control dynamics do not project reliably into macroscopic observables under the tested tasks."

This outcome:
- ✓ Does NOT invalidate Stage A
- ✓ Does NOT invalidate Stage B
- ✓ Indicates projection or task-excitation mismatch

**Any attempt to reframe Stage C failure as theory falsification is INCORRECT.**

---

## Locked Logic Tree

All outcomes **MUST** be interpreted using this logic:

### 1. Stage B + ablation + uniqueness tests succeed
→ Geometry is causal and specific → proceed to consolidation

### 2. Stage B robust, ablation fails, uniqueness fails
→ Dynamics are generic infra-slow → reinterpret control layer, not geometry

### 3. Stage B robust, uniqueness holds, Stage C projects successfully
→ Cross-modal structural validation

### 4. Stage B robust, uniqueness holds, Stage C fails
→ Projection/modality mismatch → restrict macroscopic claims

### 5. Stage B unstable under robustness checks
→ Modeling artifact → revise estimators and geometry encoding

**No other interpretation paths are valid.**

---

## Current Status

**Stage B**: PROVISIONAL 
**Ablation**: PENDING 
**Robustness**: PENDING 
**Uniqueness**: PENDING 
**Stage C**: AUTHORIZED (under constraints)

---

## Next Steps

1. Perform toroidal causality ablation
2. Execute robustness checks
3. Run uniqueness tests
4. Interpret using locked logic tree
5. Proceed to Stage C with strict constraints
6. Generate final report with reviewer-safe language

---

**END OF RECLASSIFICATION DOCUMENT**
