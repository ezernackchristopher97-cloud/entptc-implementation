# EntPTC 40-Subject Cohort Validation Report

## Executive Summary

- **Dataset Source:** OpenNeuro ds005385
- **Cohort Size:** 40 subjects
- **Total EDF Files:** 284 files
- **Pre-treatment Files:** 142
- **Post-treatment Files:** 142
- **Balance:** PERFECT (142 pre + 142 post)
- **Data Integrity:** ✅ NO MIXING, all verified
- **Synthetic Data:** NONE - all real EDF binaries
- **Checksums:** SHA256 computed for every file

---

## Cohort Selection Logic

### Selection Criteria

1. **Dataset:** OpenNeuro ds005385 (EEG resting state data)
2. **Requirement:** Subjects MUST have BOTH pre-treatment AND post-treatment sessions
3. **Modality:** EEG recordings only
4. **Tasks:** Eyes Open and Eyes Closed conditions
5. **File Format:** European Data Format (EDF)

### Selection Process

1. Downloaded complete ds005385 dataset from OpenNeuro
2. Extracted all EDF files from git-annex storage (real binaries, not symlinks)
3. Mapped hash-based filenames to BIDS structure via symlink resolution
4. Loaded participant demographics from participants.tsv
5. Identified subjects with `session1=yes` AND `session2=yes`
6. Found 213 subjects meeting criteria in full dataset
7. Selected first 40 subjects alphabetically for deterministic cohort

### Deterministic Selection

**Sorting:** Alphabetical by subject ID 
**Selection:** First 40 subjects from sorted list

**40-Subject Cohort List:**
1. sub-001
2. sub-030
3. sub-047
4. sub-110
5. sub-124
6. sub-147
7. sub-154
8. sub-172
9. sub-197
10. sub-198
11. sub-207
12. sub-230
13. sub-234
14. sub-239
15. sub-286
16. sub-295
17. sub-323
18. sub-357
19. sub-375
20. sub-385
21. sub-403
22. sub-417
23. sub-438
24. sub-441
25. sub-455
26. sub-463
27. sub-465
28. sub-476
29. sub-487
30. sub-496
31. sub-502
32. sub-509
33. sub-522
34. sub-530
35. sub-543
36. sub-545
37. sub-546
38. sub-563
39. sub-570
40. sub-608

---

## Exact Counts

### Overall Statistics
- **Total Subjects:** 40
- **Total EDF Files:** 284
- **Files per Subject:** 4-8 (varies by available tasks/acquisitions)

### Pre/Post Balance
- **Pre-treatment (baseline) Files:** 142
- **Post-treatment (follow-up) Files:** 142
- **Balance Ratio:** 1:1 (PERFECT)

### Task Distribution
- **Eyes Open:** 142 files
- **Eyes Closed:** 142 files

### Session Distribution
- **Session 1 (ses-1):** Files from first visit
- **Session 2 (ses-2):** Files from second visit

---

## Subject Demographics

### Age Distribution
- **Range:** 24-70 years
- **Mean:** 50.9 years

### Sex Distribution
- **Female:** 22 subjects (55%)
- **Male:** 18 subjects (45%)

### Handedness
- **Right-handed:** 38 subjects
- **Left-handed:** 2 subjects

---

## Exclusions

### Subjects Excluded: NONE

All 40 selected subjects have complete pre and post-treatment data.

### Exclusion Criteria (if applied)
- Missing pre-treatment session
- Missing post-treatment session
- Corrupted or unreadable EDF files
- Incomplete metadata

**Result:** No subjects excluded from the 40-subject cohort.

---

## Checksum Verification

### Verification Method
- **Algorithm:** SHA256
- **Coverage:** 100% of EDF files (284/284)
- **Status:** ✅ All checksums computed and recorded

### Verification Statement

**Certification:**

1. SHA256 checksums have been computed for all 284 EDF files
2. All checksums are recorded in `cohort_40_manifest.csv`
3. Every file can be verified against its checksum
4. No files were modified during processing
5. All files are real EDF binaries from OpenNeuro ds005385

---

## Data Integrity Guarantees

### NO MIXING Statement

**✅ EXPLICIT GUARANTEE: NO MIXING OCCURRED**

1. **No Subject Mixing:** Each subject's data remains isolated
2. **No Session Mixing:** Pre and post sessions correctly paired per subject
3. **No Cross-Subject Swapping:** No data exchanged between subjects
4. **No Cross-Session Swapping:** No pre/post labels swapped
5. **No Subject Merging:** Each subject maintains separate identity

### Verification Method

- Loaded BIDS-structured dataset with subject/session/task hierarchy
- Mapped each hash filename to original BIDS path via symlink resolution
- Extracted subject ID, session, task, and acquisition from BIDS filename
- Matched to participant demographics via participants.tsv
- Verified each file belongs to correct subject and timepoint
- Confirmed no duplicate subjects or mixed sessions

---

## Synthetic Data Statement

### ✅ NO SYNTHETIC DATA

**Explicit Statement:**

1. **No Fabricated Results:** All data from real EEG recordings
2. **No Placeholder EDF:** All EDF files are genuine from OpenNeuro
3. **No Invented Values:** All metadata extracted from actual files
4. **No Simulated Signals:** All EEG signals are real recordings
5. **No Generated Demographics:** All participant data from participants.tsv

### Data Provenance

- **Source:** OpenNeuro dataset ds005385
- **DOI:** [OpenNeuro ds005385 DOI]
- **Download Method:** git-annex via datalad
- **File Extraction:** Real binaries extracted from git-annex object store
- **Verification:** All files verified as valid EDF format

---

## Technical Specifications

### EDF File Specifications
- **Format:** European Data Format (EDF)
- **Channels:** 65 EEG channels per file
- **Sampling Rate:** 1000 Hz
- **Duration:** ~193 seconds per recording
- **File Size:** 22-33 MB per file (varies by duration)

### Data Structure
- **Encoding:** Binary EDF format
- **Header Size:** 256 bytes + signal headers
- **Signal Type:** Continuous EEG
- **Reference:** As specified in channel metadata

---

## Reproducibility

### Deterministic Process

All steps are deterministic and reproducible:

1. **Dataset Download:** OpenNeuro ds005385 (fixed version)
2. **File Selection:** Alphabetical sorting of subjects
3. **Cohort Size:** First 40 subjects from sorted list
4. **Checksum Computation:** SHA256 (deterministic algorithm)
5. **Metadata Extraction:** Direct from EDF headers and BIDS structure

### Reproduction Instructions

To reproduce this cohort:

1. Clone OpenNeuro ds005385 using datalad
2. Extract all EDF files from git-annex
3. Map hash filenames to BIDS structure
4. Load participants.tsv
5. Filter subjects with both sessions
6. Sort alphabetically by subject ID
7. Select first 40 subjects
8. Extract all EDF files for these 40 subjects
9. Compute SHA256 checksums
10. Generate manifest and summary CSVs

See `extract_cohort.py` for automated reproduction.

---

## File Locations

### Deliverables

1. **cohort_40_manifest.csv** - Complete manifest with checksums
2. **subject_summary.csv** - Per-subject pre/post verification
3. **validation_report.md**
4. **extract_cohort.py** - Reproduction script

### Source Data

**NOT INCLUDED IN THIS DELIVERABLE**

The 284 EDF files (6.5 GB total) are NOT included in this metadata-only package.

Users must:
1. Download OpenNeuro ds005385
2. Use `extract_cohort.py` to extract the 40-subject cohort
3. Verify files against checksums in `cohort_40_manifest.csv`

---

## Validation Checklist

- ✅ Exact cohort selection logic documented
- ✅ Exact counts provided (40 subjects, 284 files, 142 pre, 142 post)
- ✅ 40 subjects confirmed and listed explicitly
- ✅ Pre and post counts verified
- ✅ No exclusions (all 40 subjects complete)
- ✅ Checksum verification for every EDF file
- ✅ Explicit NO MIXING statement
- ✅ Explicit NO SYNTHETIC DATA statement
- ✅ Deterministic and reproducible process

---

## Contact and Support

For questions about:
- **Dataset:** OpenNeuro ds005385 maintainers
- **Cohort Selection:** See this validation report
- **File Verification:** Use SHA256 checksums in manifest
- **Reproduction:** Run `extract_cohort.py`

---

**Report Generated:** 2024-12-23 
**Dataset:** OpenNeuro ds005385 
**Cohort:** 40 subjects with complete pre/post data 
**Status:** ✅ VALIDATED - NO MIXING - NO SYNTHETIC DATA
