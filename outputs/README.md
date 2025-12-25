# EntPTC Analysis - Complete Results Package

**Date**: December 24, 2025 
**Dataset**: 150 EEG recordings from 34 subjects 
**Analysis Duration**: ~10 minutes 
**Success Rate**: 100% (150/150 files processed)

---

## 📁 Files in This Directory

### 📊 Primary Results

**`master_results.csv`** (160 KB)
- Complete analysis results for all 150 recordings
- 55 columns including all EntPTC metrics
- Columns: subject_id, session, task, timepoint, eigenvalues (16), spectral gap, regime, THz invariants, entropy, absurdity gap metrics, quaternion/Clifford norms
- Ready for statistical analysis, visualization, or further processing

### 📄 Reports and Documentation

**`../ENTPTC_SUPPLEMENTARY_MATERIALS.md`** (17 KB) ⭐ **START HERE**
- Comprehensive final report with all findings
- Executive summary, key metrics, regime analysis
- Falsifiability test results (FAILED, p=0.411)
- Pre/post treatment comparison
- Eigenvalue spectrum analysis
- Limitations and future directions
- **Most important document for understanding results**

**`CODE_VS_TEX_VALIDATION.md`** (21 KB)
- Detailed validation of code vs ENTPC.tex specification
- Line-by-line comparison of mathematical formulas
- Verification of all critical components
- Confirms implementation correctness

**`ANALYSIS_SUMMARY.txt`** (410 bytes)
- Quick summary of key findings
- Dataset overview
- Falsifiability test result

**`TEX_TO_CODE_MAPPING.md`** (5.4 KB)
- Mapping between ENTPC.tex definitions and code modules
- Reference for understanding implementation

### 📋 Metadata and Logs

**`subject_manifest.csv`** (15 KB)
- Complete catalog of all 150 MAT files
- Columns: filename, subject_id, session, task, timepoint, file_size, file_path
- Useful for tracking data provenance

**`summary_report_output.log`** (4.8 KB)
- Console output from summary report generation
- Includes all statistical tests and comparisons

### 🐍 Analysis Scripts

**`run_entptc_fast.py`** (12 KB)
- Main analysis script that processed all 150 files
- Implements all EntPTC components per ENTPC.tex
- Can be rerun for reproducibility

**`generate_summary_report.py`** (6.1 KB)
- Script that generated statistical summaries
- Performs falsifiability test and comparisons

---

## 🔑 Key Findings Summary

### ✅ Successfully Implemented
- 16×16 Progenitor Matrix (ENTPC.tex lines 266-285)
- Perron-Frobenius operator (lines 287-297)
- Three-regime classification (lines 669-676)
- THz structural invariants (lines 713-727, NO frequency conversion)
- Absurdity Gap post-operator (lines 728-734)
- Falsifiability test (line 663)

### 📈 Core Metrics
- **λ_max**: 13.30 ± 1.49 (vs predicted ~12.6)
- **Spectral Gap**: 10.05 ± 4.42 (vs predicted 1.47-3.78)
- **Entropy**: 1.17 ± 0.78
- **Absurdity Gap (L2)**: 1.27 ± 0.62

### 🎯 Regime Distribution
- **Regime I** (Local Stabilized): 149/150 (99.3%)
- **Regime II** (Transitional): 1/150 (0.7%)
- **Regime III** (Global Experience): 0/150 (0.0%)

### ⚠️ Critical Finding: Falsifiability Test FAILED
- Eyes Closed Absurdity Gap: 1.313 ± 0.625
- Eyes Open Absurdity Gap: 1.229 ± 0.619
- Difference: 0.084
- **p-value: 0.411** (not significant)
- **Per ENTPC.tex line 663**: Model may be falsified

### 📊 Significant Pre/Post Difference
- **Entropy decreased significantly** post-treatment (p < 0.000001)
- Pre: 1.470 ± 0.688
- Post: 0.866 ± 0.762
- Suggests treatment effect on neural dynamics

---

## 🚀 Quick Start

### View Results
```bash
# View final report
cat ../ENTPTC_SUPPLEMENTARY_MATERIALS.md

# Load results in Python
import pandas as pd
df = pd.read_csv('master_results.csv')
print(df.head)
print(df.describe)
```

### Reproduce Analysis
```bash
# Run analysis on all 150 files
python3.11 run_entptc_fast.py

# Generate summary report
python3.11 generate_summary_report.py
```

---

## 📚 Data Provenance

### Source
- Repository: ezernackchristopher97-cloud/entptc-implementation
- Data: `/data/` directory (1.6 GB, 150 MAT files)
- Format: HDF5 (.mat), 64 channels × ~193,000 timepoints

### Processing Pipeline
1. Load MAT file (64 channels)
2. Aggregate to 16 ROIs (4 channels per ROI)
3. Compute 16×16 Progenitor Matrix via PLV coherence
4. Apply Perron-Frobenius operator (eigendecomposition)
5. Extract all EntPTC metrics
6. Save to CSV

### Validation
- ✅ All 150 files processed successfully
- ✅ 64-channel constraint enforced (assertions)
- ✅ 16×16 matrix dimension validated
- ✅ Code matches ENTPC.tex specification

---

## 🔬 Next Steps

### Recommended Actions
1. **Review falsifiability test failure** - Consider alternative tasks or metrics
2. **Test on task-based EEG** - Working memory, attention, meditation
3. **Increase sample size** - Improve statistical power
4. **Revise preliminary estimates** - Update ENTPC.tex based on observed data
5. **Explore Regime II/III** - Design tasks to elicit transitional/global states

### Figures (Not Yet Generated)
Per user preference, figures should be created **after** reviewing the data. Suggested figures:
1. Eigenvalue spectrum (mean ± std)
2. Regime distribution (bar chart)
3. Absurdity Gap comparison (eyes-open vs eyes-closed)
4. Pre/post entropy comparison
5. Spectral gap distribution (histogram)

---

## 📞 Contact and Support

For questions about:
- **ENTPC.tex specification**: See reference/ENTPC.tex in repository
- **Code implementation**: See CODE_VS_TEX_VALIDATION.md
- **Results interpretation**: See ../ENTPTC_SUPPLEMENTARY_MATERIALS.md
- **Data provenance**: See subject_manifest.csv

---

## ✅ Validation Checklist

- [x] All 150 files processed successfully
- [x] Code validated against ENTPC.tex
- [x] 64-channel constraint enforced
- [x] 16×16 Progenitor Matrix constructed
- [x] Perron-Frobenius operator applied
- [x] Three regimes classified
- [x] THz invariants extracted (dimensionless only)
- [x] Absurdity Gap computed post-operator
- [x] Falsifiability test performed
- [x] Statistical comparisons completed
- [x] Comprehensive reports generated

---

**Analysis Complete**: December 24, 2025 
**Status**: ✅ Technical Success, ⚠️ Falsifiability Test Failed 
**Recommendation**: Review findings and plan next experiments
