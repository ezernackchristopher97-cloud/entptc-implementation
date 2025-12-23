# EntPTC Archive and Reference Files

This repo includes:
- entptc-FINAL.tar.gz (main archive)
- data_archives/entptc_edf_part01.tar.gz
- data_archives/entptc_edf_part02.tar.gz
- data_archives/entptc_edf_part03.tar.gz
- data_archives/entptc_edf_part04.tar.gz
- ENTPC.tex (paper source)
- edf_metadata_complete (1).csv (EDF metadata)

Notes
- Screenshots are intentionally NOT included.
- Large archives are tracked via Git LFS.

Instructions for Claude Code
- Use ENTPC.tex as the source of truth for the mathematical model and required components.
- Use the EDF metadata CSV to validate dataset ingestion and preprocessing.
- Ensure the implementation matches the paper sections: quaternionic filtering, progenitor matrix, Perron-Frobenius collapse, toroidal entropy field, geodesic/control trajectories, absurdity gap, and THz inference via structural invariants (no frequency conversion).
