# Claude Code Handoff - EntPTC Implementation

## Context

This document captures the work that Claude Code was doing before running out of time. The implementation is being continued to match ENTPC.tex exactly.

## What Claude Code Completed

### 1. Clifford Algebra Implementation (`entptc/core/clifford.py`)
- **Lines:** 888
- **Status:** Implemented
- **Reference:** ENTPC.tex Definition 2.3 (lines 242-252)
- **Features:**
  - CliffordElement dataclass with 8-dimensional basis (scalar, e1, e2, e3, e12, e23, e31, e123)
  - Full multiplication table for geometric product
  - Wedge (outer) and inner products
  - Bivector exponential for rotor operations
  - Quaternion to Clifford mapping (even subalgebra isomorphism)
  - Semantic bivector encoding

### 2. Progenitor Matrix Implementation (`entptc/core/progenitor.py`)
- **Lines:** 740
- **Status:** Implemented
- **Reference:** ENTPC.tex Definition 2.6 (lines 266-285), Section 3 (lines 330-437)
- **Features:**
  - 16×16 matrix construction per ENTPC.tex formula: c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|
  - Coherence matrix (PLV) integration
  - Entropy gradient computation
  - Quaternionic rotation operators
  - 4×4 quadrant block structure
  - Validation and irreducibility checks

### 3. Perron-Frobenius Operator (`entptc/core/perron_frobenius.py`)
- **Lines:** 776
- **Status:** Implemented
- **Reference:** ENTPC.tex Definition 2.7-2.8 (lines 287-297)
- **Features:**
  - Dominant eigenvalue and eigenvector computation
  - Power iteration method
  - Spectral gap calculation
  - Regime determination (I: Local Stabilized, II: Transitional, III: Global Experience)
  - Structural invariants extraction for THz inference
  - Full validation per Perron-Frobenius theorem

## What Still Needs Implementation

### Critical Missing Components

1. **Quaternionic Hilbert Space Filtering** (`entptc/core/quaternion.py`)
   - Reference: ENTPC.tex Definition 2.1-2.2
   - Quaternion class and operations
   - Hilbert space structure
   - Local filtering operations

2. **Entropy Field on T³** (`entptc/core/entropy.py`)
   - Reference: ENTPC.tex Definition 2.4, Section 2.2 (lines 258-262)
   - Toroidal manifold T³ definition
   - Entropy field S: T³ → ℝ
   - Gradient computation ∇S

3. **Geodesic Computation** (`entptc/analysis/geodesics.py`)
   - Reference: ENTPC.tex Section 6.2 (lines 678-687)
   - Euler-Lagrange formulation
   - Geodesic path computation on T³

4. **Absurdity Gap** (`entptc/analysis/absurdity_gap.py`)
   - Reference: ENTPC.tex Section 5.2 (lines 649-659, 728-733)
   - Post-operator only (applied AFTER Perron-Frobenius collapse)
   - Gap computation and interpretation

5. **THz Structural Invariants** (`entptc/analysis/thz_inference.py`)
   - Reference: ENTPC.tex Section 6.3 (lines 713-727)
   - NO frequency conversion
   - Structural invariant matching only
   - Eigenvalue ratio patterns

6. **EDF Pipeline** (`entptc/pipeline/edf_processor.py`)
   - Reference: ENTPC.tex lines 696-703
   - EDF file ingestion
   - 65→64 channel dimension reduction (explicit, logged, principled)
   - 64→16 ROI aggregation
   - Git LFS pointer detection and rejection
   - Symlink detection and rejection

7. **Subject Selection** (`entptc/pipeline/subject_selector.py`)
   - Deterministic, rule-based selection
   - Explicit inclusion/exclusion criteria
   - Logging of decisions
   - Integration with cohort_40_manifest.csv

8. **Main Pipeline** (`entptc/pipeline/main_pipeline.py`)
   - Orchestrates entire pipeline in correct order per ENTPC.tex
   - Assertions at every step
   - Real data validation

## Hard Constraints (NON-NEGOTIABLE)

1. **No Synthetic Data**
   - No placeholder data
   - No mock results
   - No randomly generated signals
   - Code must fail loudly if real data not available

2. **No GHz to THz Conversion**
   - THz behavior via structural invariants only
   - No frequency mapping invented

3. **Pipeline Order Must Match ENTPC.tex**
   - EDF ingestion → Quaternionic filtering → Clifford embedding → Progenitor matrix → Perron-Frobenius collapse → Entropy field → Geodesics → Absurdity gap → THz inference

4. **Dataset Handling**
   - Real EDF files from OpenNeuro ds005385 only
   - Detect and reject Git LFS pointer files
   - Detect and reject broken symlinks

5. **Dimensionality Handling**
   - 65→64 channels: explicit, logged, principled rule
   - No silent truncation
   - Assert all matrix shapes

6. **All Assertions Explicit**
   - Matrix shapes
   - Eigenvalue properties
   - Entropy field dimensions
   - All invariants

7. **No AI References**
   - No mention of AI systems, LLMs, or tools in code/comments/logs

## Deliverables Required

1. **Corrected and refactored code** matching ENTPC.tex exactly
2. **TeX-to-code mapping document** (sections/equations → modules/functions)
3. **Updated figures** from real computations only
4. **Full analysis report** documenting:
   - Which parts of tex implemented where
   - Subject selection logic and results
   - Validation checks and outcomes
   - Progenitor matrix properties
   - Entropy field summaries
5. **CSV outputs** for subject selection and intermediate validation results

## Current Repository Structure

```
entptc-archive/
├── entptc/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── clifford.py          ✅ DONE (888 lines)
│   │   ├── progenitor.py        ✅ DONE (740 lines)
│   │   ├── perron_frobenius.py  ✅ DONE (776 lines)
│   │   ├── quaternion.py        ❌ TODO
│   │   └── entropy.py           ❌ TODO
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── edf_processor.py     ❌ TODO
│   │   ├── subject_selector.py  ❌ TODO
│   │   └── main_pipeline.py     ❌ TODO
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── geodesics.py         ❌ TODO
│   │   ├── absurdity_gap.py     ❌ TODO
│   │   └── thz_inference.py     ❌ TODO
│   └── utils/
│       └── __init__.py
├── reference/
│   └── ENTPC.tex                (1296 lines - AUTHORITATIVE SPEC)
├── data_archives/               (EDF files via Git LFS)
├── cohort_40_manifest.csv       (284 files, SHA256 checksums)
├── subject_summary.csv          (40 subjects, pre/post pairs)
├── validation_report.md
└── extract_cohort.py
```

## Next Steps

1. Implement quaternion.py (Def 2.1-2.2)
2. Implement entropy.py (Def 2.4, Sec 2.2)
3. Implement geodesics.py (Sec 6.2)
4. Implement absurdity_gap.py (Sec 5.2)
5. Implement thz_inference.py (Sec 6.3)
6. Implement edf_processor.py with 65→64 handling
7. Implement subject_selector.py with explicit rules
8. Implement main_pipeline.py
9. Create TeX-to-code mapping document
10. Generate analysis report
11. Validate with real ds005385 data

## Critical Reference

**ENTPC.tex is the ONLY authoritative specification.**

Every mathematical object, operator, ordering, dimensionality, and transformation in the code must correspond one-to-one with ENTPC.tex. If something appears in code but not in ENTPC.tex, it must be removed. If something appears in ENTPC.tex but is missing in code, it must be implemented.

## Status

- **Started:** Claude Code (ran out of time)
- **Continuing:** Manus Agent
- **Target:** Complete implementation matching ENTPC.tex exactly, ready for real ds005385 data analysis
