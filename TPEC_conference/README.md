# TPEC conference

The six-method IEEE9/IEEE14 EIG draft is [eig_power_grid_conference.tex](eig_power_grid_conference.tex).
It uses `\documentclass[conference]{IEEEtran}` and the [official IEEE conference template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn), with default class margins, two-column layout and IEEE bibliography style.

Build from this folder with pdfLaTeX, BibTeX, then pdfLaTeX twice. The independent bibliography is `conference_refs.bib` in this folder. Upload the conference `.tex` and `.bib` files to Overleaf and select the conference source as the main document. The bibliography was initialized from the general document and can be edited independently. The general manuscript remains in `documents/`.

The draft compiles to six pages including references with default IEEEtran formatting:

- Pages 1–3: abstract, introduction, model, information objective, six methods, and experimental settings in prose.
- Pages 4–5: reserved results and discussion, with two information tables, two figure slots, and interpretation space.
- Page 6: completed conclusion and references; remaining room accommodates final edits.

Three author/affiliation/email placeholders are included. For two authors, remove the third block and its preceding `\and`.

Results and the abstract's results/conclusion sentences have writing placeholders. Replace the `\resultspace` boxes with verified tables, figures and discussion. The explicit `\clearpage`/`\newpage` commands in the results area reserve the draft page budget; adjust/remove them during final typesetting and recheck six-page pagination after inserting results and actual author details. The abstract emphasizes information gained at the same probe count and explicitly reserves its results and conclusion sentences for verified findings. The body conclusion makes no unverified performance claims and can be updated with those findings.

A locally generated layout preview is stored at `output/pdf/eig_power_grid_conference.pdf` when available (excluded from Git).

The local `sir ode result/` folder holds supplementary benchmark outputs. Generated result collections and build artifacts are excluded from GitHub.

Result tables use zero-valued placeholders (including zero mean/SD for every method), explicitly labeled as unmeasured. Figures contain empty IEEE9/IEEE14 axes with no data series. Algorithm 1 summarizes the shared sequential design workflow and the Step-DAD refinement branch. Plot axes are drawn in LaTeX with PGFPlots; no external image files are required.

Random and Myopic use the mean and sample SD of evaluation-seed means.
They have no training seeds; shared copies must not count as extra replicates.
Trained methods instead average evaluation-seed means within each training,
then report the mean and sample SD across three training seeds.
The current IEEE9/IEEE14 conference records have seeds 1001--1003; the
five-seed target requires 1004 and 1005 before it can be described as complete.
`python tools/summarize_baseline_eig.py --run PATH` checks the five-seed target
and outputs Random/Myopic mean +/- SD. For explicitly partial historical
reporting, add `--allow-incomplete`; missing seeds and actual counts are retained.
This is a reporting tool, not a simulator or a full campaign compatibility audit.
