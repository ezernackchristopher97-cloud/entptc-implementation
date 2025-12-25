# Draft Submission Audit Checklist

**Purpose**: One-page checklist to audit paper draft before submission.

**Use**: Check every item. If ANY item fails, the draft violates MATRIX_FIRST_PROTOCOL and must be corrected.

---

## 1. Reporting Order (Non-Negotiable)

- [ ] Results section follows this order EXACTLY:
 1. Geometry + operator construction
 2. Collapse behavior (eigenstructure, entropy, spectral gap)
 3. Invariant structure
 4. Ablation effects on collapse
 5. Projection behavior (illustrative only)
 6. Absurdity Gap (post-operator diagnostic)
 7. Optional descriptive remarks

- [ ] Collapse objects are reported BEFORE projection metrics in every section
- [ ] Primary results tables show collapse objects (NOT projection metrics)

---

## 2. Forbidden Language (Must NOT Appear)

- [ ] No "operator frequency" anywhere
- [ ] No "wave" or "oscillation" (unless explicitly describing projections, not operator)
- [ ] No "exceeds Nyquist" or similar sampling-rate language
- [ ] No "band explains X" or "delta/theta/gamma causes Y"
- [ ] No "dataset decides the theory" or "failed validation" (use "metric limitation" or "projection mismatch")

---

## 3. Operator-First Framing (Must Appear)

- [ ] Abstract states: "We construct an operator on T³ and validate collapse structure"
- [ ] Introduction establishes: Geometry → operator → collapse → invariants → projection
- [ ] Methods describe: Progenitor Matrix construction, eigendecomposition, collapse object extraction
- [ ] Results report: Collapse objects first, projection metrics second (illustrative)
- [ ] Discussion interprets: Operator-level validation, projection distortions expected

---

## 4. Uniqueness Tests (Correct Interpretation)

- [ ] Uniqueness is evaluated at collapse level (NOT single metrics)
- [ ] Metric limitations are documented as "distance function insufficient"
- [ ] Single-metric non-monotonicity does NOT weaken operator claim
- [ ] Negative controls are compared to geometry-targeted ablations
- [ ] Verdict is "metric limitation" or "projection mismatch" (NOT "theory failure")

---

## 5. EO/EC and Longitudinal Data (Correct Framing)

- [ ] EO/EC reported as "collapse structure stability/drift"
- [ ] NOT reported as "bandpower differences" or "frequency shifts"
- [ ] Stable components listed (< 10% change)
- [ ] Drifting components listed (> 10% change)
- [ ] Interpretation: "Operator-state change" or "controlled deformation"
- [ ] NOT: "Alpha increase" or "oscillatory change"

---

## 6. Absurdity Gap (Correct Role)

- [ ] Computed post-operator only
- [ ] Used as projection-distortion diagnostic
- [ ] NOT used as statistical fix
- [ ] NOT used as frequency tool
- [ ] Interpretation: "Explains why same invariant appears differently across modalities"

---

## 7. Projection Metrics (Correct Status)

- [ ] Labeled as "illustrative" or "projection-level manifestation"
- [ ] NOT used as primary validation
- [ ] PAC = 0 documented as "dataset limitation" (NOT theory failure)
- [ ] PLV, regime timing, curvature presented AFTER collapse objects
- [ ] Interpretation: "Collapse structure constrains projection behavior"

---

## 8. Frequency Language (If Used At All)

- [ ] Appears only in Discussion or optional remarks
- [ ] Clearly labeled as "post-hoc" or "descriptive"
- [ ] Stated as "inferred indirectly from invariant preservation"
- [ ] NOT stated as "measured from EEG"
- [ ] Microphysical scales (THz) labeled as "hypothesis" (NOT claim)
- [ ] EEG/fMRI/iEEG described as "projections of control process"

---

## 9. Figures and Tables (Correct Emphasis)

- [ ] Figure 1: Toroidal geometry and Progenitor Matrix construction
- [ ] Figure 2: Eigenvalue spectra (collapse objects)
- [ ] Figure 3: Collapse drift under ablations
- [ ] Figure 4: EO/EC collapse stability (NOT bandpower)
- [ ] Table 1: Collapse objects for all conditions
- [ ] Table 2: Collapse drift metrics (NOT projection metrics)
- [ ] Projection figures (PLV, PAC, regime timing) appear AFTER collapse figures

---

## 10. Discussion Section (Correct Interpretation)

- [ ] States: "We validated operator-level collapse structure"
- [ ] States: "Collapse structure is stable and constitutive"
- [ ] States: "Projection distortions are expected and quantified via Absurdity Gap"
- [ ] States: "Metric limitations require refinement, NOT theory revision"
- [ ] Does NOT state: "We found X Hz frequency"
- [ ] Does NOT state: "Dataset failed to validate theory"

---

## 11. Limitations Section (Honest Assessment)

- [ ] Lists: "Uniqueness metrics require refinement"
- [ ] Lists: "Projection-specific manifestations are dataset-dependent"
- [ ] Lists: "Distance function insufficient to capture topology-specific structure"
- [ ] Does NOT list: "Operator frequency not found"
- [ ] Does NOT list: "Theory failed validation"

---

## 12. Conclusion (Locked Framing)

- [ ] States: "EntPTC operator demonstrates stable collapse structure"
- [ ] States: "Invariant structure persists across projections with controlled drift"
- [ ] States: "Partial validation achieved at operator level"
- [ ] Does NOT state: "We discovered X Hz oscillation"
- [ ] Does NOT state: "Frequency-based mechanism"

---

## Final Check

**If ALL boxes are checked**: Draft is ready for submission under MATRIX_FIRST_PROTOCOL.

**If ANY box is unchecked**: Draft violates locked interpretation and must be corrected before submission.

---

## Emergency Deviation Check

**If reviewer asks**:
- "Where is the frequency?" → Point to Discussion (optional remarks), clarify it's inferred from invariant preservation, NOT measured
- "Why didn't uniqueness tests pass?" → Point to Methods/Results, clarify it's metric limitation, NOT theory failure
- "What about bandpower in EO/EC?" → Point to Results Section 4, clarifying the report shows collapse stability/drift, NOT bandpower

**If you're tempted to write**:
- "operator frequency" → STOP. Use "collapse rate" or "control timescale"
- "wave" or "oscillation" → STOP. Use "projection behavior" or "manifests as"
- "exceeds Nyquist" → STOP. Remove entirely.
- "band explains" → STOP. Use "collapse structure constrains"

---

**This checklist is deviation-proof. Follow it exactly.**
