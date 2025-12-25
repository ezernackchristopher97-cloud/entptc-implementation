#!/bin/bash
# Fix all THz/frequency violations in ENTPC_FINAL_FIXED.tex

cd /home/ubuntu/entptc-implementation

# Backup original
cp ENTPC_FINAL_FIXED.tex ENTPC_FINAL_FIXED.tex.backup

# Remove Figure 6 (THz hypothesis figure) - lines 590-597
sed -i '590,597d' ENTPC_FINAL_FIXED.tex

# Fix Figure 5 caption (eigenvalue spectrum) - remove "terahertz signatures"
sed -i 's/Eigenvalue spectrum decay of the progenitor matrix showing characteristic frequencies mapping to terahertz signatures\./Eigenvalue spectrum decay of the progenitor matrix showing the collapse structure and spectral gap./g' ENTPC_FINAL_FIXED.tex

# Fix text before figures that mentions THz
sed -i 's/The THz control layer is inferred through structural invariant matching, not through direct frequency conversion\. The eigenvalue ratios and spectral gaps from the Progenitor Matrix collapse are compared to dimensionless structural patterns predicted by microtubule THz resonance physics\. When these scale-invariant structures align, the model infers the presence of a THz control layer operating at characteristic frequencies of 1\.0, 1\.4, 2\.2, and 2\.4 THz, consistent with THz spectroscopy studies of microtubular structures/The operator collapse structure is characterized by eigenvalue ratios and spectral gaps that reflect the recursive entropy filtering mechanism. These dimensionless structural patterns emerge from the Progenitor Matrix dynamics and define the control timescales governing conscious state evolution/g' ENTPC_FINAL_FIXED.tex

# Fix Section 6 text - remove terahertz mentions
sed -i 's/consistent with an inferred control layer operating in the terahertz frequency range/consistent with operator-level control structure/g' ENTPC_FINAL_FIXED.tex
sed -i 's/(Figure~\\ref{fig:thz_frequencies})//g' ENTPC_FINAL_FIXED.tex
sed -i 's/terahertz frequency range/operator control timescales/g' ENTPC_FINAL_FIXED.tex
sed -i 's/terahertz stimulation/operator perturbation/g' ENTPC_FINAL_FIXED.tex
sed -i 's/terahertz spectroscopy/structural analysis/g' ENTPC_FINAL_FIXED.tex
sed -i 's/terahertz signatures/collapse structure signatures/g' ENTPC_FINAL_FIXED.tex

echo "Violations fixed"
