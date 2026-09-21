# TPEC conference

The six-method IEEE9/IEEE14 EIG draft is [eig_power_grid_conference.tex](eig_power_grid_conference.tex).
It uses `\documentclass[conference]{IEEEtran}` and the [official IEEE conference template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn), with default class margins, two-column layout and IEEE bibliography style.

Build from this folder with pdfLaTeX, BibTeX, then pdfLaTeX twice. The bibliography is shared at `../documents/objective_driven_boed_refs.bib`; keep both folders when uploading to Overleaf and select this conference source as the main document. The general manuscript remains in `documents/`.

The target is six pages including references. Final pagination is pending compilation with completed results and author details; the current source contains explicit placeholders.

The local `sir ode result/` folder holds supplementary benchmark outputs. Generated result collections and build artifacts are excluded from GitHub.
