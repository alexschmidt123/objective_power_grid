# MoE-sBOED controlled diagnosis

Status: initial_controls_complete; cause interpretation and decisive correction still required

## Root cause

unresolved; routing optimization contributes, objective saturation and order sensitivity observed; controls pending

## A–D controls

| Model | EIG | Status |
|---|---:|---|
| A | 10.66260 | evaluation_complete |
| B | 8.31631 | evaluation_complete |
| C | 8.24200 | evaluation_complete |
| D | 8.65004 | evaluation_complete |
| capacity | 5.54353 | evaluation_complete |

## Existing results

| Policy | EIG | MSE |
|---|---:|---:|
| DAD | 9.76665 | 0.07231 |
| original MoE | 7.32612 | 0.14492 |
| component MoE | 8.65004 | 0.14193 |
| slower MoE | 8.81907 | 0.11347 |

## Interpretation limits

A preserves the original DAD sum encoder but uses the common64/64tanh control head. The published DAD linear-emitter result is a separate baseline. Its16-dimensional sum output is zero padded, and unused padded weights are reported separately. Single attention sees all4 contexts; individual experts see193 features. Sparse D retains tiny intrinsic routing penalties. One-seed intervals describe paired evaluation Monte Carlo uncertainty, not training-seed variability.

## Correction and next decision

Do not claim sparse MoE advantage; select one modification only after complete controls and inference diagnostics; then decisive3training-seed comparison
