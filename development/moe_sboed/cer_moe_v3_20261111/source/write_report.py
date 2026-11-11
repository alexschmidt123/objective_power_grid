"""Publish aggregate evidence only; never promote weights or raw histories."""
import pathlib,json,hashlib,sys
import numpy as np

def main():
    root=pathlib.Path.home()/'Documents/objective_power_grid_temp'
    runs=root/'scratch_runs'
    fixed=runs/'cer_moe_sboed_v3_20261111_validated_values'
    local=runs/'cer_moe_sboed_v3_20261111_local_values'
    invalid=runs/'cer_moe_sboed_v3_20261111'
    destination=pathlib.Path(sys.argv[1]);destination.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    f=read(fixed/'evidence/GROUP_SUPPORT.json');l=read(local/'evidence/GROUP_SUPPORT.json')
    assert f['status']==l['status']=='FOUR_GROUPS_NOT_SUPPORTED'
    assert read(local/'evidence/IMPLEMENTATION_CHECKS.json')['status']=='PASS'
    assert read(invalid/'AUDIT_INVALIDATED.json')['status']=='INVALIDATED_IMPLEMENTATION_ERROR'
    z=np.load(local/'evidence/signatures.npz');v=z['values'];val=z['validation'];groups=z['groups']
    d=v-v[:,:,8:9,:]
    se=d[:,:,:8].std(-1,ddof=1)/np.sqrt(v.shape[-1])
    replicate_error=float(np.sqrt(np.mean((d[:,0,:,:].mean(-1)-d[:,1,:,:].mean(-1))**2)))
    validation_strengths=[d[val[groups[val]==k]].mean((0,1,3)).tolist() for k in range(4)]
    resources=[json.loads(line) for line in (local/'audit_resources.jsonl').read_text().splitlines()]
    gpu=np.array([[float(v) for v in row['GPU'].split(',')] for row in resources])
    hashes={str(p.relative_to(runs)):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in [fixed,local] for p in [folder/'evidence/GROUP_SUPPORT.json',folder/'evidence/IMPLEMENTATION_CHECKS.json',folder/'evidence/signatures.npz']}
    result=dict(status='STOPPED_BEFORE_SPECIALIST_PREPARATION_FOUR_GROUPS_UNSUPPORTED',
                task_complete=False,trained_policy_available=False,implementation_checks=read(local/'evidence/IMPLEMENTATION_CHECKS.json'),
                corrected_fixed_probe_audit=f,local_probe_audit=l,
                invalidated_first_audit=dict(reason='Missing normalized native terminal density prior component',
                                            observations_spent=read(invalid/'evidence/GROUP_SUPPORT.json')['simulator_observations'],excluded_from_scientific_results=True),
                approximation_error=dict(paired_candidate_MCSE_nats_quantiles_10_50_90=np.quantile(se,[.1,.5,.9]).tolist(),
                                         independent_posterior_and_suffix_replica_difference_RMS_nats=replicate_error,
                                         validation_group_mean_probe_minus_HEC_values=validation_strengths,
                                         finite_SNIS_bias_unbounded=True,frozen_variational_score_error_unbounded=True),
                local_resource_measurement=dict(samples=len(resources),interval_seconds=1,
                    GPU_percent_mean_sample=float(gpu[:,0].mean()),GPU_percent_max_sample=float(gpu[:,0].max()),
                    GPU_memory_MiB_max_sample=float(gpu[:,1].max()),GPU_power_W_max_sample=float(gpu[:,2].max()),
                    CPU_percent_one_core=100*l['process_CPU_seconds']/l['wall_seconds_including_loading_and_preservation_hashes'],
                    RSS_peak_bytes=l['process_peak_RSS_bytes'],GPU_tensor_peak_bytes=l['GPU_peak_allocated_bytes'],
                    wall_seconds=l['wall_seconds_including_loading_and_preservation_hashes'],
                    estimation_seconds=l['seconds'],process_CPU_seconds=l['process_CPU_seconds']),
                simulator_observations_total_including_invalidated=3*l['simulator_observations'],
                simulator_observations_valid_audits=f['simulator_observations']+l['simulator_observations'],
                conditional_suffix_trajectories_per_audit=192*9*2*32,
                group_discovery_histories_per_audit=192,calibration_histories_per_audit=96,validation_histories_per_audit=96,
                specialist_preparation_updates=0,full_arm_updates=0,uniform_arm_updates=0,
                full_native_training_batch_integration='NOT_EXECUTED',main_evaluation_bank_generated=False,
                inference_latency='NOT_MEASURED_FOR_AN_UNTRAINED_POLICY',
                assessment={name:'NOT_ESTABLISHED_NO_TRAINED_CER_POLICY' for name in ['beats_trained_uniform','beats_each_frozen_expert','useful_expert_competencies','beats_HEC_v2','ICML_contribution']},
                source_stage='complete_policy_and_value_audit; native training integration not implemented after failed prerequisite',
                original_source_checkpoint_hashes_verified=l['protected_files'],evidence_sha256=hashes)
    (destination/'VALIDATION_RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    doc=f'''# CER-MoE-sBOED V3: specialization prerequisite not met

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
|Corrected fixed probes|85.4%,3.1%,9.4%,2.1%|{f['silhouette']:.3f}|{f['independent_MC_assignment_agreement']:.1%}|No: all prefer HEC|Fail|
|Fresh local probes|19.8%,67.7%,8.3%,4.2%|{l['silhouette']:.3f}|{l['independent_MC_assignment_agreement']:.1%}|Yes, on calibration|Fail|

The local audit held the support thresholds fixed. At stage 3 its validation
counts were 1/31/0/0; at stage 12, 9/20/3/0; at stage 24, 9/14/5/4. Two groups failed
the predeclared 10% minimum occupancy, and within-stage support failed at the
early sampled stage. This does not establish four stable nonchronological
specialist training distributions. Rare groups may exist; the data do not prove
their absence. The initial coarse-palette failure alone was insufficient, so one
fresh diagnostic used local offsets 0.25/0.75 around HEC, then stopped without
relaxing thresholds or adding further probes, training seeds or architectures.

Local posterior ESS met 32/4096 in {l['posterior_ess_adequate_fraction']:.1%} of
held-out replicas; minimum ESS was {l['posterior_ess_quantiles'][0]:.2f}. Low-ESS
cases remain unresolved. Median paired within-replica MCSE for nonreference
proposals was {np.median(se):.3f} nats; RMS disagreement between independent
posterior/suffix replicas was {replicate_error:.3f} nats. No exact conditional-MI
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

The fully instrumented local audit took {l['seconds']:.2f} s for estimation and
{l['wall_seconds_including_loading_and_preservation_hashes']:.2f} s including
loading/hash verification, with {l['process_CPU_seconds']:.2f} process CPU seconds.
Peak process RSS was {l['process_peak_RSS_bytes']/2**30:.2f} GiB; peak allocated
CUDA tensors were {l['GPU_peak_allocated_bytes']/2**20:.1f} MiB. Fourteen 1 second
samples measured peak GPU utilization {gpu[:,0].max():.0f}%, device memory
{gpu[:,1].max():.0f} MiB and power {gpu[:,2].max():.1f} W on RTX 4090. Sampled GPU
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
'''
    (destination/'RESULTS.md').write_text(doc)
    print('CER_AGGREGATE_REPORT_WRITTEN',destination)

if __name__=='__main__':main()
