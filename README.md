EntPTC Archive

This repository contains:
- EDF dataset archives split into multiple tar.gz parts (Git LFS)
- ENTPC.tex (model and mathematical reference)
- EDF metadata CSV

Notes
- Screenshots are intentionally excluded
- Large files are stored using Git LFS

Instructions for Claude Code
- Use ENTPC.tex as the authoritative model specification
- Ensure code matches the paper sections exactly
- Do not invent missing components

Claude Code Instructions (do not ignore)
1. ENTPC.tex is the authoritative model specification. Code must match it exactly.
2. Do not invent missing components. If something is missing in code, implement it only if it exists in the TeX model.
3. Do not reference any AI tools or meta language in code comments or documentation.
4. THz inference is via structural invariants only. No frequency conversion.
5. Absurdity Gap is post operator only.
