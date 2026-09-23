# TPEC conference paper and results

The six-page IEEE conference source is [eig_power_grid_conference.tex](eig_power_grid_conference.tex), with its independent [conference_refs.bib](conference_refs.bib).
Build from this folder with pdfLaTeX, BibTeX, then pdfLaTeX twice. The source uses IEEEtran conference mode with standard margins and fonts. Three author placeholders are provided; remove the third block and its preceding `\and` for two authors.

## Paper organization

The paper flows naturally across six pages, without forced page or column breaks. IEEEtran margins and font sizes remain unchanged; display spacing is not stretched to fill columns, and the last page uses balanced columns.

- Methods: nonadaptive Random/Fixed, online Myopic, offline-trained DAD/RL-sBOED, and online-refined Step-DAD. The experiment-workflow pseudocode in Experimental Settings uses a floating block so it does not reserve an unfillable space in the preceding column.
- The workflow summarizes offline preparation, paired evaluation, probe selection/application/observation, state carry, and separate scoring/timing. Its underlying evaluation order was checked against the archived IEEE14 conference `continuous_eig.py` (`evaluate`, `refine`, `myopic_duration`) and the current implementation. Planner assimilation occurs before the next action; no terminal planner update is required for EIG. Step-DAD retains within-episode refinements and starts each episode from DAD.
- Final-page columns are balanced. Unused space below the final text is intentional; no margins, fonts or equation spacing are enlarged to fill it.
- Problem formulation: unknown M/K and grid dynamics, probe/observation model, feasible adaptive policies, then EIG and its finite-contrast approximation.
- Results: aligned IEEE9/IEEE14 information tables and measured online/offline computation, followed by limitations and the supplementary SIR-ODE comparison.
- All eight tables use the same full IEEE-column width, small font, row spacing and booktabs rules. Complete EIG tables cover IEEE9, IEEE14 and SIR-ODE; four computation tables report offline hours and online seconds for IEEE9/IEEE14 at T=3,4,5. The former computation figures have been removed.
- Hardware details are confined to the experimental-settings paragraph, not figures or tables. LabPC reports RTX 4090 (not RTX 4090 Ti). The paragraph distinguishes project development hardware from reported timing runs and notes that the aggregate timing is not a controlled runtime comparison between networks.
- Optional [validation histories](results/diagnostics/validation_histories.pdf) remain in the local result collection, outside the main paper.

All result tables use Random, Fixed, Myopic, DAD, RL-sBOED, Step-DAD row order and T=3,4,5 columns. IEEE9 and IEEE14 are complete, including Step-DAD T=5. The accepted grid protocol requires evaluation seeds 1001–1003; 1004/1005 are not required. SIR has five evaluation seeds and a different finite-particle entropy estimator and historical training protocol, explicitly disclosed in the supplementary subsection. SIR Fixed uses evaluation-seed SD; grid Fixed uses training-seed SD.

## Result collection on LabPC

Open [the result index](results/README.md), then select:

- [SIR-ODE](results/sir_ode/README.md)
- [IEEE9](results/ieee9/README.md)
- [IEEE14](results/ieee14/README.md)

Each benchmark has `01_results`, `02_configuration`, `03_models`, `04_runs`, and `05_provenance`. These views link to preserved, verified data. Original grid campaign paths under `results/experiments` and the original `sir ode result` package remain unchanged. Collection is complete and all navigation links resolve within TPEC_conference. No live SSH session is required to use the collection. The verification report is `results/FINAL_VERIFICATION.json`; the accepted scope is `results/CONFERENCE_PROTOCOL.json`.

Raw results, models and generated previews are excluded from GitHub. Tables/plots are embedded in the tracked LaTeX, so the paper builds without the local result collection. The preview is `output/pdf/eig_power_grid_conference.pdf`. The general project manuscript remains in `documents/`.

The complete result collection includes 54 grid and 18 SIR saved model files, raw evaluations, configurations, training histories, source snapshots, checksum records, and aggregate EIG/online/offline summaries. IEEE9 Step-DAD T=5 is 2.6621 ± 0.0145 nats across three training-seed means. The six-page draft incorporates the completed results; author placeholders remain.
