# IEEE30 published dynamic reference

`configs/ieee30_mocu.yaml` now transcribes the machine, excitation and governor parameters from Tables I–III of the authors' IEEE30 data sheet:
https://www.kios.ucy.ac.cy/testsystems/wp-content/uploads/2020/03/IEEE-30.pdf

Reference: P. Demetriou, M. Asprou, J. Quirós-Tortós, E. Kyriakides, “Dynamic IEEE Test Systems for Transient Analysis,” IEEE Systems Journal 11(4), 2108–2117, 2017. DOI: 10.1109/JSYST.2015.2444893. Paper: https://zenodo.org/records/1238086 . Author network files: https://www.kios.ucy.ac.cy/testsystems/index.php/ieee-30-bus-modified-test-system/ . The data sheet hash is recorded in the YAML.

The model has six GENROU machines: generators at original buses 1 and 2; synchronous condensers at 5, 8, 11 and 13. All six have IEEET1 excitation; only the two generators have BPA_GG governors. Original buses map to added machine terminals 31–36 in that order. The published modified network therefore has 36 electrical buses but only six rotating machines. Load buses are algebraic, with constant-impedance loads. The nominal frequency is 50 Hz, as in the paper's IEEE30 response plots.

H = [4.130,5.078,1.520,1.520,1.200,1.200] seconds, on machine rated powers [270,51.2,40,40,25,25] MVA. The YAML preserves the published reactances, time constants, saturation, exciter and governor fields on their stated bases. These are published typical benchmark parameters, not utility measurements.

## Uncertainty and units

The proposed study prior has eight independent coordinates: six machine inertias H and two generator droops R. +/-30% uniform bounds are explicitly study assumptions, not supplied by the paper. Condensers have no governor-response latent variable. Their excitation gains are not active-power droop gains.

For the existing electrical angular-frequency convention, the inertia conversion is M_i=2 H_i (S_i/S_base)/(2*pi*f_base). Omitting the machine/system power-base factor would be incorrect. The derived static generator response K_i=(S_i/S_base)/(R_i*f_base) is documented for units only; it does not replace the BPA_GG governor with instantaneous damping.

## Implementation status

This is a reference configuration; the current reduced CPU/CUDA swing backend does not implement GENROU, IEEET1, BPA_GG or the modified AC network initialization. It explicitly rejects this YAML before simulation. No new experiment should be launched with it until a compatible backend and network import have been implemented and validated. The existing six-machine lossless topology helpers are retained for historical reproducibility, but are not the published full dynamic model.

The observation, prior widths and control protocol are study choices. The copied 22 Hz/s safety threshold was removed; its replacement is pending physical justification. Other retained control/probe choices are labeled unvalidated proposals. New dataset paths prevent reuse of synthetic banks; automatic generation is disabled.

## Superseded audit

IEEE30 jobs 19687885–19687888 were cancelled on 2026-09-07 at the user's request. Their frozen source and partial audit bank use the old synthetic six-machine model and remain historical artifacts, not validation of this paper-based configuration. IEEE14 jobs continue with their own frozen source.
