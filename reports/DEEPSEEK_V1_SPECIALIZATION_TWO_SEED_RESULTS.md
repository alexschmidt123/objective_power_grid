# DeepSeekMoE-v1 contextual expert learning: two independent runs

Both runs use frozen source 042915bfe8312faa333b1b825b94766c00c68dbf, derived exclusively from v1 foundation 13681d7091189a4e68056e6a2864dc58f9d5700b.
One methodological change: observed-history context protection of routed-expert full-horizon gradients. Architecture, simulator, original 2000 training contrasts and original router remain unchanged.

| Machine | Training seed | Status | Updates | Official EIG ± MC SE | Confirmation EIG ± MC SE |
|---|---:|---|---:|---|---|
|LabPC / RTX4090|101|Complete|26335|11.04404166 ± 0.04281140|11.04982475 ± 0.04480642|
|UbuntuPC / RTX5080|102|Complete|20575|10.83415593 ± 0.04301131|10.81753690 ± 0.04379193|

## Seed 101

Charged training: 14397.158s; measured specialization subset inside allowance: 1467.958s.
Learned minus uniform-routing development difference: {"difference": 0.14470255312420638, "paired_MC_SE": 0.054029059945061735, "CI95": [0.038805595631885376, 0.2505995106165274], "N": 512}.
Full minus routed-masked development difference: {"difference": 0.2408626011525236, "paired_MC_SE": 0.06430946510360083, "CI95": [0.11481604954946595, 0.3669091527555812], "N": 512}.
Multiplicity-controlled context reversals: 0.
All audited useful-specialization criteria satisfied: False.
Online frozen inference latency: {"mean_ms": 0.45929708615100634, "median_ms": 0.4614449862856418, "p95_ms": 0.474054177175276, "decisions": 300, "frozen_forward_only": true}.
Checkpoint SHA256: b680081479f1ca0227f250265642cea0452ce57e110d26220746d7966b50edfd.

| Reference | Paired new minus reference | Paired MC SE | Approximate 95% CI |
|---|---:|---:|---|
|v1|+0.05497576|0.04543997|[-0.034086585569130384, 0.14403810159794309]|
|vsOED|-1.43680523|0.05665340|[-1.5478459040870414, -1.3257645576109809]|
|corrected_DAD|+0.76792879|0.05803476|[0.6541806692113336, 0.8816769206741916]|
|dense|-0.17435370|0.03982771|[-0.25241600260113123, -0.09629139713382051]|
|shared_only|-0.12693568|0.04851461|[-0.22202432512715825, -0.031847037733543124]|

## Seed 102

Charged training: 14396.959s; measured specialization subset inside allowance: 1124.428s.
Learned minus uniform-routing development difference: {"difference": -0.07167427291962492, "paired_MC_SE": 0.051039141182633196, "CI95": [-0.17171098963758596, 0.028362443798336148], "N": 512}.
Full minus routed-masked development difference: {"difference": -0.0554651427289089, "paired_MC_SE": 0.0599491440481196, "CI95": [-0.1729654650632233, 0.06203517960540551], "N": 512}.
Multiplicity-controlled context reversals: 0.
All audited useful-specialization criteria satisfied: False.
Online frozen inference latency: {"mean_ms": 0.4594594876592358, "median_ms": 0.45986048644408584, "p95_ms": 0.477365538245067, "decisions": 300, "frozen_forward_only": true}.
Checkpoint SHA256: 1256720d370f42ab7a069ffa87a20a565bcd3623d60480b377c9d02c9e79a65c.

| Reference | Paired new minus reference | Paired MC SE | Approximate 95% CI |
|---|---:|---:|---|
|v1|-0.15490997|0.05095405|[-0.254779907169235, -0.055040037014392015]|
|vsOED|-1.64669096|0.05551418|[-1.755498756752677, -1.5378831651577858]|
|corrected_DAD|+0.55804306|0.05810096|[0.44416518661683685, 0.6719209430562486]|
|dense|-0.38423943|0.04903831|[-0.4803545207583976, -0.2881243391889765]|
|shared_only|-0.33682141|0.04816151|[-0.4312179789030756, -0.24242484417005844]|

## Interpretation and limits

Trajectory Monte Carlo SE is conditional on each checkpoint and fixed contrast bank; it is not training-seed uncertainty. The two runs also use different GPUs and PyTorch versions.
Per-seed reports preserve raw paired scores, development controls, intervention data, competence matrices and costs. A larger EIG alone does not identify a causal MoE advantage.
Dense and shared-only references are historical teacher-free controls; their throughput and update counts differ. Frozen masking and uniform-routing interventions are not retrained controls.
Expert function differences or routing frequencies alone are not evidence of useful information specialization. The competence analysis uses full-horizon utilities averaged over joint-distribution histories with identical observed prefixes, not a hidden true theta as a fixed-history posterior.
Necessity of every individual routed expert is NOT established by a whole-routed-branch masking result. Do not claim that all seven are necessary without individual-expert evidence.
The router retains original v1 selected-gate pathwise learning. No new discrete-membership credit estimator or router teacher is introduced.
The bounded development pilot did not establish an EIG improvement; it passed only the predeclared correctness/mechanism gate. Pilot weights were discarded. No official score was used to tune the method.
Training-side extra backward work is recorded within each four-hour allowance. Reproduction, smoke, pilot, validation and evaluation computation are additional and must be read separately from the recorded readiness and cost files.
Historical reference targets: v1 10.98907; vsOED 12.48085 nats. No claim of improvement or useful specialization is warranted beyond the saved controlled comparisons.
