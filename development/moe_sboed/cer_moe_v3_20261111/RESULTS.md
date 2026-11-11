# CER-MoE-sBOED V3: specialization prerequisite not met

Status: STOPPED BEFORE SPECIALIST PREPARATION. The requested training and final
comparison are incomplete. There is no trained CER model and no performance claim.
Stopping follows the requirement not to manufacture four meaningful groups;
unproven novelty is not the stopping reason.

## Implemented and checked

Four complete feasible action heads share native history encoding and frozen
Bayesian information. One categorical expert supplies the entire action; no
coordinate mixture or shared action-head bypass exists. The critic supplies
action-conditioned remaining-horizon variational values. Exact categorical
actor-surrogate differentiation includes the exploration mixture and both gate
and proposal derivatives. All 30 stage physical/history tests, finite differences,
score-function identity, stochastic frequencies, explicit likelihood checks,
posterior importance identity at the prior and normalized native terminal-density
parity including extreme tails pass on LabPC. These are policy/value-unit checks,
not a claim that native end-to-end training has been validated.

## Group discovery evidence

Each valid audit used 192 training-replay prefixes at stages 3/12/24, split 96/96
calibration/validation, 9 complete candidate designs, 4096 exact-likelihood SNIS
posterior proposal particles, and two independent 32-suffix replications per
candidate. Whole remaining suffixes were resimulated after the proposed action;
no stored future observations or hidden true parameters entered decisions.

|Audit|Held-out group occupancy|Silhouette|Independent MC group agreement|Distinct calibrated preferred designs|Decision|
|---|---|---:|---:|---|---|
|Corrected fixed probes|85.4%,3.1%,9.4%,2.1%|0.504|94.8%|No: all prefer HEC|Fail|
|Fresh local probes|19.8%,67.7%,8.3%,4.2%|0.252|78.1%|Yes, on calibration|Fail|

The local audit held the support thresholds fixed. At stage 3 its validation
counts were 1/31/0/0; at stage 12, 9/20/3/0; at stage 24, 9/14/5/4. Two groups failed
the predeclared 10% minimum occupancy, and within-stage support failed at the
early sampled stage. This does not establish four stable nonchronological
specialist training distributions. Rare groups may exist; the data do not prove
their absence. The initial coarse-palette failure alone was insufficient, so one
fresh diagnostic used local offsets 0.25/0.75 around HEC, then stopped without
relaxing thresholds or adding further probes, training seeds or architectures.

Local posterior ESS met 32/4096 in 91.1% of
held-out replicas; minimum ESS was 1.05. Low-ESS
cases remain unresolved. Median paired within-replica MCSE for nonreference
proposals was 0.320 nats; RMS disagreement between independent
posterior/suffix replicas was 0.606 nats. No exact conditional-MI
error bound follows. Finite SNIS bias, missing posterior modes, the frozen
variational terminal score, limited probe coverage and the frozen HEC suffix
continuation all limit the inference. Preferred-design evidence is about probe
values, not learned expert competencies.

## Preserved error and computational costs

The first audit omitted the native density's tiny prior component in terminal
scoring. It was invalidated, kept intact, fixed and rerun after tail-parity
checks. Its results are excluded, but its compute is counted. This was a new
scorer error; no historical model or source was changed.

Each audit simulated 1,880,064 conditional future observations across 110,592
branched suffixes. The two valid audits used 3,760,128 observations; including the
invalidated audit, total expenditure was 5,640,192. Each audit additionally
recorded 1,126,597,632 exact signal evaluations, 113,258,496 analytic parameter
Jacobians and 5,308,992 frozen opportunity-critic rows. Those counts include
internal Bayesian model-query work and are not extra training episodes.

The fully instrumented local audit took 4.12 s for estimation and
14.27 s including
loading/hash verification, with 14.37 process CPU seconds.
Peak process RSS was 1.45 GiB; peak allocated
CUDA tensors were 761.6 MiB. Fourteen 1 second
samples measured peak GPU utilization 59%, device memory
1435 MiB and power 130.8 W on RTX 4090. Sampled GPU
utilization includes loading/hash periods and is not a per-kernel efficiency
claim. Raw telemetry remains on LabPC. Training seconds/update and trained-model
inference/evaluation time are unavailable because neither arm was trained.

## Requested assessments and outstanding work

Full versus trained uniform, full versus each frozen expert, useful conditional
expert strengths, full versus HEC-v2, and an ICML-level algorithmic contribution
are all unestablished. No 2000 trajectory/1M contrast evaluation bank was generated,
and no old EIG table was substituted. There are zero specialist-preparation or
main-training optimizer steps. Native 10000 episode training-batch integration,
two-arm training, frozen-expert evaluation and latency tests remain unexecuted.

CIAS/competitive-V3/HEC distinctions are documented in METHOD_AND_PROTOCOL.md,
but a value-dependent categorical gate is not by itself evidence of novelty,
retained competence, avoided collapse or improved pure EIG. Epsilon responsibility
prevents zero selection probability; it does not prove useful specialization.

All 346 selected historical source/checkpoint hashes matched before/after the
local audit. No original trees, running jobs or Ubuntu computation were changed.
Source and aggregate reports are synchronized via the dedicated GitHub branch;
raw histories, posterior particles, model files and telemetry remain outside Git.
VALIDATION_RESULTS.json contains exact aggregate values and evidence hashes.
