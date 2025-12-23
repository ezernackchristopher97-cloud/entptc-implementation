# HANDOFF INSTRUCTIONS FOR GITHUB UPLOAD

## MISSION
Upload the EntPTC 40-subject cohort metadata deliverables to the user's GitHub account.

## CONTEXT
The user has linked their GitHub account to Manus. All metadata files are ready at `/home/ubuntu/deliverables_metadata_only/`. These are METADATA-ONLY files (168 KB total) - NO large EDF binaries, NO Git LFS required.

## FILES TO UPLOAD
Location: `/home/ubuntu/deliverables_metadata_only/`

1. **README.md** (7 KB) - Main documentation
2. **cohort_40_manifest.csv** (117 KB) - 284 EDF files with SHA256 checksums
3. **subject_summary.csv** (16 KB) - 40 subjects with pre/post pairs
4. **validation_report.md** (8 KB) - Complete validation documentation
5. **extract_cohort.py** (12 KB) - Reproduction script

Total: 168 KB (GitHub-friendly, NO Git LFS needed)

## WHAT THESE FILES ARE

### cohort_40_manifest.csv
- Complete manifest of 284 EDF files from 40 subjects
- Each row = 1 EDF file with metadata
- Columns: subject_id, session_id, pre_or_post, task, edf_filename, edf_sha256, duration, sampling_rate, channels_count, etc.
- **CRITICAL:** Contains SHA256 checksums for verification
- **NO MIXING:** All files correctly paired pre/post per subject

### subject_summary.csv
- Per-subject summary (40 rows, 1 per subject)
- Shows which pre/post files belong to each subject
- Columns: subject_id, has_pre, has_post, pre_file, post_file, pre_sha256, post_sha256, etc.
- **Verification:** All 40 subjects have BOTH pre and post

### validation_report.md
- Complete validation documentation
- Cohort selection logic (alphabetical, first 40 subjects with both sessions)
- Explicit guarantees: NO MIXING, NO SYNTHETIC DATA
- Full subject list
- Checksum verification statement

### extract_cohort.py
- Standalone Python script for reproducibility
- Takes ds005385 path as input
- Generates cohort_40_manifest.csv and subject_summary.csv
- Allows others to reproduce the cohort extraction

### README.md
- User-facing documentation
- Quick start guide
- Code examples for loading and verifying data
- Usage instructions

## REPOSITORY DETAILS

**Suggested Repository Name:** `entptc-cohort-metadata`

**Description:** "EntPTC 40-subject cohort metadata from OpenNeuro ds005385 - Metadata-only deliverable with SHA256 checksums for 284 EDF files (142 pre + 142 post treatment). NO mixing, NO synthetic data, deterministic and reproducible."

**Topics/Tags:** `eeg`, `neuroscience`, `openneuro`, `metadata`, `consciousness`, `entptc`, `validation`

**Visibility:** Public (or Private if user prefers)

## EXACT STEPS TO EXECUTE

### Step 1: Verify GitHub Authentication
```bash
gh auth status
```

If not authenticated, STOP and ask user to authenticate.

### Step 2: Create Local Git Repository
```bash
cd /home/ubuntu/deliverables_metadata_only
git init
git add .
git commit -m "Initial commit: EntPTC 40-subject cohort metadata

- 40 subjects from OpenNeuro ds005385
- 284 EDF files (142 pre + 142 post treatment)
- Complete metadata with SHA256 checksums
- NO mixing, NO synthetic data
- Deterministic and reproducible
- Total size: 168 KB (GitHub-friendly)"
```

### Step 3: Create GitHub Repository
```bash
gh repo create entptc-cohort-metadata \
  --public \
  --description "EntPTC 40-subject cohort metadata from OpenNeuro ds005385 - Metadata-only deliverable with SHA256 checksums" \
  --source=. \
  --push
```

**Alternative if user wants private:**
```bash
gh repo create entptc-cohort-metadata \
  --private \
  --description "EntPTC 40-subject cohort metadata from OpenNeuro ds005385 - Metadata-only deliverable with SHA256 checksums" \
  --source=. \
  --push
```

### Step 4: Verify Upload
```bash
gh repo view --web
```

This will open the repository in the browser for verification.

### Step 5: Confirm to User
Report back:
- Repository URL
- All 5 files uploaded successfully
- Total size (should be ~168 KB)
- Confirmation that NO Git LFS was needed

## CRITICAL CONSTRAINTS

1. **NO large files** - All files are metadata only (168 KB total)
2. **NO Git LFS** - Not needed, everything under GitHub's limits
3. **NO EDF binaries** - The actual 6.5 GB of EDF files are NOT included
4. **NO mixing** - All metadata verified, pre/post correctly paired
5. **NO synthetic data** - All real data from OpenNeuro ds005385

## DATA INTEGRITY GUARANTEES

These files represent:
- **40 subjects** (alphabetically selected from 213 available)
- **284 EDF files** (142 pre-treatment + 142 post-treatment)
- **Perfect balance** (1:1 pre/post ratio)
- **Complete checksums** (SHA256 for every file)
- **Deterministic selection** (reproducible via extract_cohort.py)

## TROUBLESHOOTING

### If GitHub authentication fails:
Ask user to run `gh auth login` in their terminal and follow prompts.

### If repository already exists:
Ask user if they want to:
- Use a different name
- Delete the existing repo and recreate
- Push to the existing repo

### If files are missing:
All files should be at `/home/ubuntu/deliverables_metadata_only/`
Verify with: `ls -lh /home/ubuntu/deliverables_metadata_only/`

### If upload fails:
Fall back to providing the tar.gz:
`/home/ubuntu/entptc_cohort_metadata.tar.gz` (30 KB)
User can manually upload as GitHub release.

## SUCCESS CRITERIA

✅ Repository created on user's GitHub account
✅ All 5 files uploaded and visible
✅ README.md displays properly on GitHub
✅ Total repository size ~168 KB
✅ NO Git LFS required
✅ User can access and share the repository

## ADDITIONAL CONTEXT

This is the culmination of downloading 284 real EDF files (6.5 GB) from OpenNeuro ds005385, verifying them, matching them to subject demographics, and creating metadata-only deliverables. The user specifically requested:

1. NO large file archives
2. NO compression of EDF files
3. Metadata only for GitHub
4. Easy to read and interpret
5. NO mixing of subjects/sessions
6. Deterministic and reproducible

All requirements have been met. The files are ready for upload.

## FINAL NOTES

- The user has been working on this for multiple sessions
- They specifically want metadata only (not the 6.5 GB of EDF files)
- This package allows others to download ds005385 and verify files via checksums
- The extract_cohort.py script enables full reproducibility
- All data integrity has been verified: NO MIXING, NO SYNTHETIC DATA

Execute the steps above and report success to the user with the repository URL.
