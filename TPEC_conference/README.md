# TPEC conference paper and results

The six-page IEEE conference source is [eig_power_grid_conference.tex](eig_power_grid_conference.tex), with its independent [conference_refs.bib](conference_refs.bib).
Build from this folder with pdfLaTeX, BibTeX, then pdfLaTeX twice. The source uses IEEEtran conference mode with standard margins and fonts. Three author placeholders are provided; remove the third block and its preceding `\and` for two authors.

## Paper organization

- Pages 1–3: purpose, power-grid formulation, methods/pseudocode and experimental settings.
- Page 4: aligned IEEE9/IEEE14 information tables, IEEE9 online timing and information-versus-computation plots.
- Page 5: IEEE14 online/offline computation, grid validation histories and limitations.
- Page 6: supplementary SIR-ODE protocol/results, conclusion and references.

All result tables use Random, Fixed, Myopic, DAD, RL-sBOED, Step-DAD row order and T=3,4,5 columns. IEEE9 Step-DAD T=5 remains pending until all three training-seed evaluations finish. Grid results currently contain three evaluation seeds; the five-seed target is incomplete. SIR has five evaluation seeds and a different finite-particle entropy estimator and historical training protocol, explicitly disclosed in the supplementary subsection. SIR Fixed uses evaluation-seed SD; grid Fixed uses training-seed SD.

## Result collection on LabPC

Open [the result index](results/README.md), then select:

- [SIR-ODE](results/sir_ode/README.md)
- [IEEE9](results/ieee9/README.md)
- [IEEE14](results/ieee14/README.md)

Each benchmark has `01_results`, `02_configuration`, `03_models`, `04_runs`, and `05_provenance`. These views link to preserved, verified data. Original grid campaign paths under `results/experiments` and the original `sir ode result` package remain unchanged. The running collector refreshes the grid views after the two remaining jobs complete. It uses this Mac's authenticated SSH sessions; keep it online for collection.

Raw results, models and generated previews are excluded from GitHub. Tables/plots are embedded in the tracked LaTeX, so the paper builds without the local result collection. The preview is `output/pdf/eig_power_grid_conference.pdf`. The general project manuscript remains in `documents/`.
