# TPEC conference

The six-method IEEE9/IEEE14 EIG draft is [eig_power_grid_conference.tex](eig_power_grid_conference.tex).
It uses `\documentclass[conference]{IEEEtran}` and the [official IEEE conference template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn), with default class margins, two-column layout and IEEE bibliography style.

Build from this folder with pdfLaTeX, BibTeX, then pdfLaTeX twice. The independent bibliography is `conference_refs.bib` in this folder. Upload the conference `.tex` and `.bib` files to Overleaf and select the conference source as the main document. The bibliography was initialized from the general document and can be edited independently. The general manuscript remains in `documents/`.

The draft compiles to six pages including references with default IEEEtran formatting:

- Pages 1–3: abstract, introduction, model, information objective, six methods, and experimental protocol.
- Pages 4–5: reserved results and discussion, with two information tables, two figure slots, and interpretation space.
- Page 6: completed conclusion and references; remaining room accommodates final edits.

Three author/affiliation/email placeholders are included. For two authors, remove the third block and its preceding `\and`.

Results and the abstract's results/conclusion sentences have writing placeholders. Replace the `\resultspace` boxes with verified tables, figures and discussion. The explicit `\clearpage`/`\newpage` commands in the results area reserve the draft page budget; adjust/remove them during final typesetting and recheck six-page pagination after inserting results and actual author details. The abstract emphasizes information gained at the same probe count and explicitly reserves its results and conclusion sentences for verified findings. The body conclusion makes no unverified performance claims and can be updated with those findings.

A locally generated layout preview is stored at `output/pdf/eig_power_grid_conference.pdf` when available (excluded from Git).

The local `sir ode result/` folder holds supplementary benchmark outputs. Generated result collections and build artifacts are excluded from GitHub.
