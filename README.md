# EntPTC Implementation Repository

**Entropic Toroidal Progenitor Theory of Consciousness**

Christopher Ezernack
University of Texas at Dallas

## Overview

This repository contains the complete implementation, validation, and analysis pipeline for the EntPTC consciousness model. The mathematical framework proposes that conscious experience emerges from recursive entropy gradients on a toroidal manifold, mediated by quaternionic filtering operations.

## Repository Structure

```
entptc-implementation/
├── ENTPC_FINAL_FIXED.tex          # Main manuscript (LaTeX source)
├── ENTPC_FINAL_FIXED.pdf          # Compiled manuscript
├── entptcref.bib                  # Bibliography (254 references)
├── README.md                      # This file
├── LICENSE                        # MIT License
├── CITATION.cff                   # Citation metadata
├── requirements.txt               # Python dependencies
│
├── entptc/                        # Core Python package
│   ├── core/                      # Mathematical primitives
│   │   ├── progenitor.py          # Progenitor matrix construction
│   │   ├── quaternion.py          # Quaternion operations
│   │   ├── clifford.py            # Clifford algebra Cl(3,0)
│   │   ├── perron_frobenius.py    # Spectral analysis
│   │   ├── entropy.py             # Entropy gradient computation
│   │   └── thz_inference.py       # Structural inference
│   ├── analysis/                  # Analysis modules
│   ├── pipeline/                  # Processing pipelines
│   └── utils/                     # Utility functions
│
├── scripts/                       # Analysis scripts
│   ├── run_entptc_*.py            # Main execution scripts
│   ├── stage_a_*.py               # Grid cell analysis
│   ├── stage_b_*.py               # Frequency inference
│   ├── stage_c_*.py               # EEG projection
│   └── generate_*.py              # Figure generation
│
├── data/                          # EEG datasets
│   ├── sub-*_*.mat                # Processed MAT files (40 subjects)
│   ├── dataset_set_2/             # Secondary dataset
│   └── dataset_set_3_ds004706/    # Tertiary dataset
│
├── figures/                       # Publication figures
│   ├── fig01_schematic.pdf
│   ├── fig02_eigenspectrum.pdf
│   ├── fig03_entropy_participation.pdf
│   └── fig04_eoec_stability.pdf
│
├── outputs/                       # Analysis results
│   ├── stage_a_outputs/           # Grid cell results
│   ├── stage_b_outputs/           # Frequency inference results
│   ├── stage_c_outputs/           # EEG projection results
│   └── *.csv                      # Feature matrices
│
├── metadata/                      # Dataset metadata
│   ├── cohort_40_manifest.csv     # Subject manifest
│   ├── subject_manifest.csv       # Subject details
│   └── README.md                  # Metadata documentation
│
├── docs/                          # Documentation
│   ├── METHODS.md                 # Detailed methods
│   ├── FIGURE_CATALOG.md          # Figure descriptions
│   └── *.md                       # Technical notes
│
└── archive/                       # Archived materials
    ├── tex_versions/              # Previous TeX versions
    ├── bib_versions/              # Previous bib versions
    ├── pdf_versions/              # Previous PDF versions
    └── tests/                     # Test files
```

## Installation

```bash
git clone https://github.com/ezernackchristopher97-cloud/entptc-implementation.git
cd entptc-implementation
pip install -r requirements.txt
```

## Dependencies

Python 3.8+ with the following packages:
- numpy
- scipy
- matplotlib
- pandas
- seaborn
- h5py
- mne
- pyedflib
- tqdm

## Usage

Run the main analysis pipeline:

```bash
cd scripts
python run_entptc_full_analysis.py
```

Generate publication figures:

```bash
python generate_publication_figures.py
```

## Key Results

The validation pipeline demonstrates:

1. **Eigenvalue Stability**: Perron eigenvalue λ₁ = 1.0000 ± 0.0001 across all subjects
2. **Entropy Gradients**: Consistent collapse structure with H_collapse = 0.847 ± 0.023
3. **EOEC Differentiation**: Clear separation between eyes-open and eyes-closed conditions
4. **Cross-Dataset Replication**: Results replicate across three independent EEG datasets

## Theoretical Framework

The EntPTC model proposes:

- Quaternionic Hilbert space formulation for conscious states
- Toroidal manifold embedding (T³) for spatial representation
- Recursive entropy filtering through progenitor matrices
- Perron-Frobenius collapse dynamics for state selection

## Data Sources

- OpenNeuro ds005385: 64-channel EEG, resting state
- OpenNeuro ds004706: Grid cell recordings, spatial navigation
- 40-subject cohort with eyes-open/eyes-closed conditions

## Compilation

LaTeX compilation:
```bash
pdflatex ENTPC_FINAL_FIXED.tex
bibtex ENTPC_FINAL_FIXED
pdflatex ENTPC_FINAL_FIXED.tex
pdflatex ENTPC_FINAL_FIXED.tex
```

## Citation

```bibtex
@article{ezernack2024entptc,
  title={EntPTC Paper I: Mathematical Model of Experience through Recursive Entropy and Quaternionic Filtering},
  author={Ezernack, Christopher},
  year={2024},
  institution={University of Texas at Dallas}
}
```

## License

MIT License. See LICENSE file for details.

## Contact

Christopher Ezernack
University of Texas at Dallas
Christopher.Ezernack@utdallas.edu
