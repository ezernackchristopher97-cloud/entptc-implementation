# EntPTC 40-Subject Cohort Validation Report

## Executive Summary

- Dataset Source: OpenNeuro ds005385  
- Cohort Size: 40 subjects  
- Total EDF Files: 284  
- Pre-treatment Files: 142  
- Post-treatment Files: 142  
- Data Balance: 1:1 pre to post  
- Data Integrity: All files verified, no mixing  
- Synthetic Data: None  
- Checksums: SHA256 computed for every file  

---

## Cohort Selection Logic

### Selection Criteria

1. Dataset: OpenNeuro ds005385 (EEG resting state data)  
2. Subjects must have both pre-treatment and post-treatment sessions  
3. EEG modality only  
4. Tasks: Eyes Open and Eyes Closed  
5. File format: European Data Format (EDF)  

### Selection Process

1. Downloaded the complete ds005385 dataset from OpenNeuro  
2. Extracted all EDF files from git-annex storage as real binaries  
3. Mapped hash-based filenames to BIDS structure via symlink resolution  
4. Loaded participant demographics from participants.tsv  
5. Identified subjects with both session1 and session2 present  
6. Found 213 subjects meeting criteria in the full dataset  
7. Selected the first 40 subjects alphabetically for a deterministic cohort  

### Deterministic Selection

Sorting: Alphabetical by subject ID  
Selection: First 40 subjects from sorted list  

40-Subject Cohort List:

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

- Total Subjects: 40  
- Total EDF Files: 284  
- Files per Subject: 4 to 8 depending on available tasks and acquisitions  

### Pre and Post Distribution

- Pre-treatment Files: 142  
- Post-treatment Files: 142  
- Balance Ratio: 1:1  

### Task Distribution

- Eyes Open: 142 files  
- Eyes Closed: 142 files  

### Session Distribution

- Session 1: First visit recordings  
- Session 2: Second visit recordings  

---

## Subject Demographics

### Age Distribution

- Range: 24 to 70 years  
- Mean: 50.9 years  

### Sex Distribution

- Female: 22 subjects  
- Male: 18 subjects  

### Handedness

- Right-handed: 38 subjects  
- Left-handed: 2 subjects  

---

## Exclusions

No subjects were excluded. All 40 selected subjects have complete pre and post-treatment data.

Exclusion criteria were defined as:
- Missing pre-treatment session  
- Missing post-treatment session  
- Corrupted or unreadable EDF files  
- Incomplete metadata  

None of these conditions were met for the selected cohort.

---

## Checksum Verification

### Method

- Algorithm: SHA256  
- Coverage: 284 out of 284 EDF files  

All checksums were computed and recorded in `cohort_40_manifest.csv`.

Each file can be verified against its corresponding checksum. No files were modified during processing. All files are real EDF binaries from OpenNeuro ds005385.

---

## Data Integrity

The following checks were performed:

- Each file was mapped to its original BIDS path using symlink resolution  
- Subject ID, session, task, and acquisition were extracted from filenames  
- Metadata was matched against participants.tsv  
- Pre and post sessions were verified per subject  
- No duplicate subjects were found  
- No cross-subject or cross-session mixing was observed  

Each subject’s data remains isolated and correctly labeled.

---

## Synthetic Data Statement

No synthetic, simulated, or generated data is included.

All EEG signals are real recordings from the OpenNeuro ds005385 dataset.  
All metadata originates from participants.tsv and BIDS headers.  
No fabricated values, placeholder files, or generated signals are present.

---

## Data Provenance

- Source: OpenNeuro ds005385  
- Download Method: datalad with git-annex  
- File Extraction: Real binaries extracted from git-annex object store  
- Verification: All files validated as proper EDF format  

---

## Technical Specifications

### EDF Files

- Format: European Data Format  
- Channels: 65 EEG channels per file  
- Sampling Rate: 1000 Hz  
- Duration: Approximately 193 seconds per recording  
- File Size: 22 to 33 MB per file  

### Structure

- Encoding: Binary EDF  
- Signal Type: Continuous EEG  
- Header: Standard EDF header with channel metadata  

---

## Reproducibility

All steps are deterministic and reproducible.

1. Download OpenNeuro ds005385  
2. Extract all EDF files from git-annex  
3. Map hash filenames to BIDS structure  
4. Load participants.tsv  
5. Filter subjects with both sessions present  
6. Sort subjects alphabetically  
7. Select the first 40 subjects  
8. Extract all EDF files for these subjects  
9. Compute SHA256 checksums  
10. Generate manifest and summary files  

The full process is implemented in `extract_cohort.py`.

---

## File Locations

### Deliverables

- cohort_40_manifest.csv  
- subject_summary.csv  
- validation_report.md  
- extract_cohort.py  

### Source Data

The 284 EDF files are not included in this repository.

To reproduce the dataset:

1. Download OpenNeuro ds005385  
2. Run `extract_cohort.py`  
3. Verify files using the checksums in `cohort_40_manifest.csv`  

---

## Validation Checklist

- Cohort selection logic documented  
- Exact counts provided  
- Subject list explicitly defined  
- Pre and post balance verified  
- No exclusions  
- Checksums computed for all files  
- No data mixing  
- No synthetic data  
- Deterministic and reproducible process  

---

Report Generated: 2024-12-23  
Dataset: OpenNeuro ds005385  
Cohort: 40 subjects with complete pre and post data  
Status: Validated

