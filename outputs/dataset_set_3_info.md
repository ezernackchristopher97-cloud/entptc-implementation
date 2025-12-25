# Dataset Set 3: OpenNeuro Rest Eyes Open (ds004584)

**Date**: December 24, 2025 
**Source**: OpenNeuro 
**DOI**: 10.18112/openneuro.ds004584.v1.0.0

---

## Dataset Details

**Title**: Rest eyes open 
**URL**: https://openneuro.org/datasets/ds004584/versions/1.0.0

### Subjects
- **Total**: 149 subjects
 - 100 individuals with Parkinson's disease
 - 49 healthy controls
- **We'll use**: First 15 subjects (sub-001 to sub-015)

### Recording Details
- **EEG System**: 64-channel BrainVision cap
- **Task**: Resting-state, eyes open
- **Duration**: 2 minutes per subject
- **Setting**: Quiet room, sitting

### Data Format
- **Format**: BIDS-compliant
- **Files**: 1049 total
- **Size**: 2.87 GB
- **Validation**: Valid BIDS (1 warning)

### For Our Pipeline
- ✅ 64 channels (matches our requirement)
- ✅ Resting-state eyes open
- ✅ BIDS format (easy to process)
- ✅ Large sample size (149 subjects available)
- ✅ BrainVision format (standard EEG format)

---

## Selection Rationale

This dataset is ideal for Dataset Set 3 because:

1. **Correct channel count**: 64 channels → 16 ROIs
2. **Resting-state**: Matches Dataset Sets 1 and 2
3. **Eyes open**: Provides within-condition replication
4. **Large N**: 149 subjects available, we'll use 15
5. **BIDS format**: Standardized, easy to preprocess
6. **Quality**: OpenNeuro validated dataset

---

## Comparison with Other Datasets

| Dataset | Channels | Subjects | Conditions | Format |
|---------|----------|----------|------------|--------|
| **Set 1** (Original) | 64 | 34 | Eyes open + closed | MAT |
| **Set 2** (PhysioNet Motor) | 64 | 15 | Eyes open + closed | EDF |
| **Set 3** (OpenNeuro PD) | 64 | 15 (of 149) | Eyes open | BrainVision |

---

## Next Steps

1. Download first 15 subjects from OpenNeuro
2. Preprocess BrainVision files to MAT format
3. Apply same preprocessing as Sets 1 and 2
4. Aggregate to 16 ROIs
5. Run EntPTC pipeline with toroidal constraints

---

**Status**: Dataset identified and validated 
**Ready for**: Download and preprocessing
