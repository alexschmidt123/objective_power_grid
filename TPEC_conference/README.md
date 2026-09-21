# TPEC conference

The six-method IEEE9/IEEE14 EIG draft is [eig_power_grid_conference.tex](eig_power_grid_conference.tex).
It uses `\documentclass[conference]{IEEEtran}` and the [official IEEE conference template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn), with default class margins, two-column layout and IEEE bibliography style.

Build from this folder with pdfLaTeX, BibTeX, then pdfLaTeX twice. The independent bibliography is `conference_refs.bib` in this folder. Upload the conference `.tex` and `.bib` files to Overleaf and select the conference source as the main document. The bibliography was initialized from the general document and can be edited independently. The general manuscript remains in `documents/`.

The target is six pages including references. Final pagination is pending compilation with completed results and author details; the current source contains explicit placeholders.

The local `sir ode result/` folder holds supplementary benchmark outputs. Generated result collections and build artifacts are excluded from GitHub.
