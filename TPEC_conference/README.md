# TPEC conference paper and results

The six-page IEEE conference source is [eig_power_grid_conference.tex](eig_power_grid_conference.tex), with its independent [conference_refs.bib](conference_refs.bib).
Build from this folder with pdfLaTeX, BibTeX, then pdfLaTeX twice. The source uses IEEEtran conference mode with standard margins and fonts. Three author placeholders are provided; remove the third block and its preceding `\and` for two authors.

## Paper organization

The paper flows naturally across six pages, without forced page or column breaks. IEEEtran margins and font sizes remain unchanged; display spacing is not stretched to fill columns, and the last page uses balanced columns.

- Problem formulation: unknown M/K and grid dynamics, probe/observation model, feasible adaptive policies, then EIG and its finite-contrast approximation.
- Results: aligned IEEE9/IEEE14 information tables and measured online/offline computation, followed by limitations and the supplementary SIR-ODE comparison.
- Figure 1: directly labeled information and runtime panels at T=3, with the same method order throughout. Information panels show mean ± SD; runtime panels use a logarithmic axis and labels in ms/s. Tables retain T=3,4,5. Hardware is identified separately for each system.
- The former Figure 2 is an optional optimization diagnostic, preserved as [validation histories](results/diagnostics/validation_histories.pdf) in the local result collection. It is omitted from the main six-page paper.

All result tables use Random, Fixed, Myopic, DAD, RL-sBOED, Step-DAD row order and T=3,4,5 columns. IEEE9 Step-DAD T=5 remains pending until all three training-seed evaluations finish. Grid results currently contain three evaluation seeds; the five-seed target is incomplete. SIR has five evaluation seeds and a different finite-particle entropy estimator and historical training protocol, explicitly disclosed in the supplementary subsection. SIR Fixed uses evaluation-seed SD; grid Fixed uses training-seed SD.

## Result collection on LabPC

Open [the result index](results/README.md), then select:

- [SIR-ODE](results/sir_ode/README.md)
- [IEEE9](results/ieee9/README.md)
- [IEEE14](results/ieee14/README.md)

Each benchmark has `01_results`, `02_configuration`, `03_models`, `04_runs`, and `05_provenance`. These views link to preserved, verified data. Original grid campaign paths under `results/experiments` and the original `sir ode result` package remain unchanged. The running collector refreshes the grid views after the two remaining jobs complete. It uses this Mac's authenticated SSH sessions; keep it online for collection.

Raw results, models and generated previews are excluded from GitHub. Tables/plots are embedded in the tracked LaTeX, so the paper builds without the local result collection. The preview is `output/pdf/eig_power_grid_conference.pdf`. The general project manuscript remains in `documents/`.
