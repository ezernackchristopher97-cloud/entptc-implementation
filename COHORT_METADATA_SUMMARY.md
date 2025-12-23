# EntPTC 40-Subject Cohort - Metadata Summary

## Overview
This document summarizes the metadata deliverables for the EntPTC 40-subject cohort from OpenNeuro dataset ds005385.

**Total Size:** ~160 KB (GitHub-friendly, NO Git LFS required)

## Key Files

### 1. cohort_40_manifest.csv (117 KB)
Complete manifest of all 284 EDF files in the 40-subject cohort with SHA256 checksums for verification.

### 2. subject_summary.csv (16 KB)
Per-subject summary showing pre/post file pairs for all 40 subjects.

### 3. validation_report.md (8 KB)
Comprehensive validation report documenting cohort selection logic, criteria, and data guarantees.

### 4. extract_cohort.py (12 KB)
Standalone Python script to reproduce the cohort extraction from ds005385.

### 5. EntPTC_40-Subject_Cohort_-_Metadata_Deliverables.pdf (131 KB)
Complete documentation package with usage examples and data guarantees.

## Cohort Statistics

- **Subjects:** 40
- **Total Files:** 284
- **Pre-treatment:** 142 files
- **Post-treatment:** 142 files
- **Balance:** Perfect 1:1
- **Age Range:** 24-70 years
- **Sex:** 22 Female, 18 Male
- **Channels:** 65 EEG channels per file
- **Sampling Rate:** 1000 Hz
- **Duration:** ~193 seconds per recording

## Data Guarantees

### ✅ NO MIXING
- No subject data mixed
- No session swapping
- Pre and post correctly paired per subject
- All verified via checksums

### ✅ NO SYNTHETIC DATA
- All data from real EEG recordings
- No fabricated results
- No placeholder files
- All from OpenNeuro ds005385

### ✅ DETERMINISTIC
- Alphabetical subject selection
- Reproducible via extract_cohort.py
- SHA256 checksums for all files
- Complete provenance documented

## Critical Notes for Implementation

1. **65 Channels → 64 Channels Reduction Required**
   - Dataset has 65 EEG channels
   - EntPTC quaternion framework requires 64 channels
   - Must implement explicit, logged dimension reduction rule
   - No silent truncation allowed

2. **Real Data Only**
   - Code must detect and reject Git LFS pointer files
   - Code must detect and reject broken symlinks
   - Must fail loudly if real EDF data not available

3. **Subject Selection**
   - Deterministic, rule-based selection
   - First 40 subjects alphabetically with both pre/post sessions
   - Explicit logging of inclusion/exclusion criteria

## Status
✅ **VALIDATED - READY FOR GITHUB UPLOAD**

Generated: 2024-12-23
Version: 1.0
