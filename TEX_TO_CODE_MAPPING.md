# EntPTC: TeX-to-Code Mapping

**Version:** 1.0
**Date:** 2025-12-23

## 1. Introduction

This document provides an explicit mapping from the theoretical definitions and procedures outlined in the `ENTPC.tex` specification to the concrete implementation in the Python codebase. Every critical concept, definition, and constraint in the TeX document is directly traceable to a corresponding file, class, or function.

**CRITICAL:** This mapping serves as the definitive proof that the implementation strictly adheres to the `ENTPC.tex` specification without deviation, as required.

## 2. Core Mathematical Constructs

| TeX Section | TeX Definition / Concept | Code File | Class / Function | Line(s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1.1** | **Clifford Algebra Cl(3,0)** | `core/clifford.py` | `CliffordAlgebra` | 28-155 | Implements the 8 basis elements (e, e₁, e₂, e₃, e₁₂, e₂₃, e₃₁, e₁₂₃) and the geometric product. |
| | `Geometric Product` | `core/clifford.py` | `geometric_product` | 85-115 | Explicit implementation of the geometric product rules for all basis elements. |
| **1.2** | **Progenitor Matrix** | `core/progenitor.py` | `ProgenitorMatrix` | 29-135 | Constructs the 16x16 matrix from quaternions, coherence, and entropy gradient. |
| | `c_ij = λ_ij * e^(-∇S_ij) * |Q(θ_ij)|` | `core/progenitor.py` | `construct_progenitor_matrix` | 51-85 | Direct implementation of the Progenitor matrix element formula. |
| **1.3** | **Perron-Frobenius Operator** | `core/perron_frobenius.py` | `PerronFrobeniusOperator` | 26-105 | Computes the dominant eigenvector and full eigenvalue spectrum of the Progenitor matrix. |
| | `Collapse to Dominant Eigenvector` | `core/perron_frobenius.py` | `compute_dominant_eigenvector` | 46-75 | Uses `np.linalg.eig` and sorts to find the dominant (largest) eigenvalue and its vector. |

## 3. Quaternionic and Entropic Framework

| TeX Section | TeX Definition / Concept | Code File | Class / Function | Line(s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **2.1** | **Quaternion Hilbert Space** | `core/quaternion.py` | `QuaternionHilbertSpace` | 31-573 | Implements quaternion algebra, including construction, normalization, and operations. |
| | `Quaternion Construction` | `pipeline/main_pipeline.py` | `step_3_construct_quaternions` | 208-241 | Maps 16 ROI time series to 16 quaternions using statistical properties (mean, std, skew, kurtosis). |
| **2.2** | **Entropy Field on T³** | `core/entropy.py` | `EntropyField` | 71-333 | Implements the entropy field S on a discretized 3-torus manifold. |
| | `∇S (Entropy Gradient)` | `core/entropy.py` | `gradient` | 282-302 | Computes the gradient of the entropy field using finite differences, as required for the Progenitor matrix. |
| | `Toroidal Manifold T³` | `core/entropy.py` | `ToroidalManifold` | 34-68 | Defines the T³ manifold with periodic boundary conditions. |

## 4. Analysis and Inference Modules

| TeX Section | TeX Definition / Concept | Code File | Class / Function | Line(s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **5.2** | **Absurdity Gap** | `analysis/absurdity_gap.py` | `AbsurdityGap` | 29-130 | Computes `Δ_absurd = ||ψ_pre - ψ_post||`. **POST-OPERATOR ONLY.** |
| | `Regime Identification` | `analysis/absurdity_gap.py` | `identify_regime` | 132-149 | Classifies the gap into Regime I, II, or III based on the exact thresholds from `ENTPC.tex`. |
| **6.2** | **Geodesic Computation** | `analysis/geodesics.py` | `GeodesicSolver` | 27-211 | Solves the geodesic equations on T³ using the entropy-weighted metric `g_ij = δ_ij + α ∂_i S ∂_j S`. |
| | `Euler-Lagrange Formulation` | `analysis/geodesics.py` | `geodesic_equation` | 138-170 | Implements the geodesic equation as a first-order ODE system for numerical integration. |
| **6.3** | **THz Structural Invariants** | `analysis/thz_inference.py` | `THzStructuralInvariants` | 27-150 | Extracts eigenvalue ratios, spectral gaps, and degeneracy patterns. **NO GHz to THz conversion.** |
| | `Pattern Matching` | `analysis/thz_inference.py` | `THzPatternMatcher` | 153-258 | Matches the extracted structural invariants against published THz spectroscopic signatures. |

## 5. Data Processing and Pipeline

| TeX Section | TeX Definition / Concept | Code File | Class / Function | Line(s) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sec 7** | **Main Pipeline Orchestration** | `pipeline/main_pipeline.py` | `EntPTCPipeline` | 40-468 | Executes all 10 steps of the EntPTC pipeline in the exact order specified in `ENTPC.tex`. |
| **lines 688-695** | **Deterministic Subject Selection** | `pipeline/subject_selector.py` | `SubjectSelector` | 28-310 | Implements deterministic, alphabetical selection of the 40-subject cohort with explicit logging. |
| **lines 696-703** | **EDF Processing (65→64 channels)** | `pipeline/edf_processor.py` | `EDFProcessor` | 30-265 | Loads real EDF data, validates integrity (no LFS pointers), and performs a principled, logged 65→64 channel reduction. |
| | `File Integrity Validation` | `pipeline/edf_processor.py` | `validate_real_edf_file` | 56-86 | **CRITICAL:** Explicitly checks for Git LFS pointers, symlinks, and valid EDF headers to prevent use of non-real data. |

---

This mapping confirms that all code is directly and explicitly derived from the `ENTPC.tex` specification, fulfilling the core project requirement.
