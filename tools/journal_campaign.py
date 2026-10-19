"""Frozen ten-method fixed-budget replication; root launchers, artifact-only reports."""
import argparse,csv,fcntl,hashlib,json,math,os,shutil,subprocess,sys,threading,time,traceback
from pathlib import Path
import numpy as np
import torch
import yaml
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'experiments/10152026/10152026_journal_five_seed_fixed_budget'
METHODS=('random','fixed','dad','rl_sboed','vsoed','scorebed','moe','resmoe','switchmoe','switchresmoe')
FAMILY=('moe','resmoe','switchmoe','switchresmoe')
RESIDUAL=('resmoe','switchresmoe')
NAMES=dict(zip(METHODS,('Random','Optimized Fixed','DAD','Simplified RL-sBOED','vsOED','ScoreBED','MoE-sBOED','ResMoE-sBOED','SwitchMoE-sBOED','SwitchResMoE-sBOED')))
PAIRS=(('resmoe','scorebed'),('resmoe','dad'),('resmoe','moe'),('switchmoe','moe'),('switchresmoe','resmoe'),('switchresmoe','switchmoe'))

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def textfile(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');tmp.write_text(text);os.replace(tmp,path)
def table(path,rows,fields):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    os.replace(tmp,path)
def mc(x):
    x=np.asarray(x,dtype=float);mean=float(x.mean());se=float(x.std(ddof=1)/len(x)**.5)
    return dict(mean=mean,se=se,low=mean-1.96*se,high=mean+1.96*se)
def paired_identity(rows):return [[r['truth'],r['measurement_noise_standard']] for r in rows]
def native_method(method):return 'sum_single' if method=='base' else method
def train_record(entry):return read(Path(entry['run'])/'models'/(entry['native_method']+'_training.json'))
def eval_rows(entry):return read(Path(entry['run'])/'eval'/(entry['native_method']+'_seed1001.json'))

def make_config(benchmark,cap):
    filename='location_finding_eig.yaml' if benchmark=='location' else 'sir_ode_research.yaml'
    raw=yaml.safe_load((ROOT/'configs'/filename).read_text());p=raw['research']['journal_protocol']
    raw['experiment']['methods']=list(METHODS)
    p.update(status='frozen_location_replication' if benchmark=='location' else 'SIR_development_pending_readiness',checkpoint_selection_mode='final_fixed_budget',checkpoint_selection='final_predeclared_fixed_budget',comparison='one_million_total_primary_episodes',training_seeds=[101,102,103,104,105],development_training_seeds=[101],training_device='cuda_if_available')
    p['method_stopping']={}
    p['stopping_rule'].update(minimum_episodes=cap,maximum_episodes=cap,validation_every_episodes=125000,maximum_training_seconds=7200 if benchmark=='location' else 86400)
    p['common_pathwise_settings'].update(hidden=64,contrasts=1024,moe_gradient_clip_mode='component',moe_router_warmup_fraction=.3,moe_router_finish_fraction=.95,strategy_routing_mode='learned_switch',strategy_initial_reroute_probability=.99)
    # Explicitly override the effective per-method clock, not only common defaults.
    for method in FAMILY:p['method_settings'][method]['router_schedule_episodes']=500000 if method in RESIDUAL else 1000000
    p['method_settings']['fixed']={**raw['research'].get('training_profiles',{}).get('fixed',{})}
    p['method_settings']['dad']={**raw['research'].get('training_profiles',{}).get('adaptive',{})}
    p['method_settings']['scorebed']=dict(scorebed_pretraining_episodes=692990,scorebed_batch_size=64)
    p['validation'].update(systems=512 if benchmark=='location' else 128,contrasts=1000000 if benchmark=='location' else 16384,batch_size=16,chunk=16384 if benchmark=='location' else 4096)
    if benchmark=='sir':
        raw['experiment']['step_number']=5
        # 320 + 7810*128 = 1M, avoiding an unreported short final native replay batch.
        p['native_vsoed_profile']['initial_replay']=320
        p['native_vsoed_profile']['budget_note']='Initial replay320 instead of draft256 permits exactly1M episodes with native new-trajectory batches128; ODE adapter, not published SDE reproduction'
    else:p['native_vsoed_profile']=dict(raw['research']['upstream_vsoed'])
    p['native_vsoed_profile']['updates']=(cap-p['native_vsoed_profile']['initial_replay'])//p['native_vsoed_profile']['new_trajectories']
    raw['research']['paper_protocol']=dict(raw['research'].get('paper_protocol',{}),nominal_training_trajectories=cap,training_budget_unit='primary_episodes',checkpoint_selection='final',regime='journal_1M_total_fixed_budget')
    settings=raw['research']['settings'];settings.update(eval_systems=2000 if benchmark=='location' else 2048,eval_contrasts=1000000 if benchmark=='location' else 16384,evaluation_population_systems=2000 if benchmark=='location' else 2048,evaluation_batch_size=16,evaluation_contrast_chunk=16384 if benchmark=='location' else 4096,strategy_diagnostics_every=1)
    # Historical RL simplifications remain unchanged, not silently upgraded to paper settings.
    settings.update(rl_batch_size=100,rl_replay_batch=4096,rl_replay_capacity=1000000,rl_min_buffer=100000,rl_gradient_steps=2)
    for method in FAMILY:
        merged=settings|p['common_pathwise_settings']|p['method_settings'][method]
        assert merged['router_schedule_episodes']==(500000 if method in RESIDUAL else 1000000)
        assert merged['moe_router_warmup_fraction']==.3 and merged['moe_router_finish_fraction']==.95
    return raw

def historical_entries():
    raw=read(ROOT/'experiments/10152026/10152026_journal_location_sir/audit/historical_development_reference.json');result={}
    for item in raw['items']:
        method={'moe_sboed':'moe','strategy_moe_sboed':'switchmoe'}.get(item['method'],item['method'])
        if method in METHODS and method not in RESIDUAL:
            result[method]=dict(method=method,native_method=item['method'],run=str(Path(item['run']).resolve()),seed=101,benchmark='location',status='candidate_reuse',provenance='historical_development')
    prior=ROOT/'experiments/10152026/10152026_location_residual_1m_total_v2/development_seed101'
    for method in ('base',*RESIDUAL):
        result[method]=dict(method=method,native_method=native_method(method),run=str(prior/method),seed=101,benchmark='location',status='candidate_reuse',provenance='verified_1M_total_development')
    return result

def audit_entry(entry,expected):
    from src.research.benchmarks import make_benchmark
    from src.research.moe_family import benchmark_signature,BasePolicy
    from src.research.journal_checkpoints import load_checkpoint
    from src.research.engine import rollout
    path=Path(entry['run']);cfg=read(path/'run_config.json');method=entry['method'];t=train_record(entry)
    b=make_benchmark(expected);old=make_benchmark(cfg['benchmark'])
    assert benchmark_signature(b)==benchmark_signature(old),'Benchmark physics/prior/noise mismatch'
    assert cfg['train_seed']==101 and cfg['horizon']==30
    if method!='random':
        assert t['generated_training_trajectories']==(500000 if method in (*RESIDUAL,'base') else 1000000)
        assert t['checkpoint_selection'].startswith('final'),'Historical model was selected using validation/test'
    if method=='switchmoe':
        assert t.get('routing_warmup_fraction')==.3 and t.get('routing_finish_fraction')==.95,'Historical SwitchMoE uses0/80 instead of matched30/95 routing schedule'
    if method=='moe':
        assert t['routing_warmup_fraction']==.3 and t['routing_finish_fraction']==.95
        assert t['gradient_clip_mode']=='component' and t['history_encoder_revision']==2
    if method in RESIDUAL:
        assert t['total_training_episodes']==1000000
        e=cfg['benchmark']['research']['journal_protocol']['method_settings'][method]
        assert e['router_schedule_episodes']==500000
        assert t['base_checkpoint']['sha256']==sha(Path(historical_entries()['base']['run'])/'models/sum_single.pt')
    if method=='scorebed':
        s=read(path/'models/scorebed_official_training.json')
        assert s['initialization_trajectories']+s['score_training_trajectories']+s['policy_training_trajectories']==1000000
        assert s['score_training_trajectories']==692990 and s['score_batch']==64
    if method=='vsoed':
        p=t['profile'];assert p['initial_replay']+p['updates']*p['new_trajectories']==1000000 and p['discount']==.9
    if method=='rl_sboed':assert t['gamma']==1 and t['gradient_steps_per_update']==2
    native=entry['native_method'];cp=path/'models'/(native+'.pt')
    if method=='base':
        assert t['generated_training_trajectories']==500000
        return dict(compatible=True,checkpoint_sha256=sha(cp),checks=['500k final internal base','compatible full encoder/action head','benchmark identity'])
    rows=eval_rows(entry);assert len(rows)==2000
    summary=next(x for x in read(path/'summary.json') if x['method']==native and x['eval_seed']==1001)
    assert summary['contrasts']==1000000
    rng=torch.Generator().manual_seed(1001);truth=b.sample((2000,),rng);noise=torch.randn(2000,30,1,generator=rng,dtype=torch.float64);bank=b.sample((1000000,),rng)
    assert np.array_equal(bank.numpy(),np.load(path/'eval/contrast_bank_seed1001.npy'))
    p=None if method=='random' else load_checkpoint(cp,b)
    if method in RESIDUAL:
        reference=BasePolicy.from_checkpoint(Path(historical_entries()['base']['run'])/'models/sum_single.pt',b,30).policy
        for key,value in reference.state_dict().items():torch.testing.assert_close(p.core.base.policy.state_dict()[key],value,rtol=0,atol=0)
    with torch.no_grad():r=rollout(b,p,30,16,1001,0,truth=truth[:16],noise=noise[:16],method='random' if method=='random' else 'rl_sboed' if method=='rl_sboed' else None)
    for stage,a in enumerate(r['actions']):
        target=torch.tensor([x['actions'][stage] for x in rows[:16]],dtype=torch.float64)
        torch.testing.assert_close(a,target,rtol=2e-6 if method in {'scorebed','vsoed','rl_sboed'} else 1e-10,atol=2e-6 if method in {'scorebed','vsoed','rl_sboed'} else 1e-10)
    return dict(compatible=True,checkpoint_sha256=None if method=='random' else sha(cp),cases_sha256=hashlib.sha256(json.dumps(paired_identity(rows),separators=(',',':')).encode()).hexdigest(),bank_sha256=sha(path/'eval/contrast_bank_seed1001.npy'),checks=['benchmark/prior/noise/horizon','episode count','final selection','native settings','same evaluation bank','first16 complete action sequences replayed'])

def prepare():
    assert not C.exists(),'Campaign exists; never overwrite'
    C.mkdir(parents=True);(C/'configs').mkdir();(C/'audit').mkdir();(C/'figures').mkdir()
    shutil.copy('/tmp/journal_multiseed_preflight.log',C/'audit/preflight.log')
    assert '50 passed' in (C/'audit/preflight.log').read_text()
    for benchmark in ('location','sir'):
        for cap in (500000,1000000):
            (C/'configs'/f'{benchmark}_{cap}.yaml').write_text(yaml.safe_dump(make_config(benchmark,cap),sort_keys=False))
    entries=historical_entries();audit=[];registry=[];expected=make_config('location',1000000)
    torch.set_num_threads(4)
    for method,entry in entries.items():
        try:
            evidence=audit_entry(entry,expected);entry.update(status='complete',audit=evidence)
            registry.append(entry);audit.append(dict(method=method,reusable=True,reason='Frozen protocol compatibility checks passed',**evidence))
        except Exception as exc:
            audit.append(dict(method=method,reusable=False,reason=str(exc),exception=type(exc).__name__,historical_run=entry['run']))
    write(C/'audit/seed101_compatibility.json',audit);write(C/'registry.json',registry)
    protocol=dict(status='prepared_pending_fixed_budget_ScoreBED_smoke',methods=list(METHODS),training_seeds=[101,102,103,104,105],training_episodes_per_method=1000000,base_episodes=500000,residual_episodes=500000,evaluation_seed=1001,reserved_final_seed=8831001,reserved_final_seed_used=False,checkpoint_selection='final_fixed_budget',validation='diagnostic_only',primary_pairs=PAIRS,statistical_rule='Paired training-seed mean differences; Student t95CI and Bonferroni simultaneous95CI over6contrasts; conclusions use simultaneous interval; trajectory MC intervals separately',training_seed101='Development-used seed, reused only if compatible;102–105 independent fixed-configuration replications; no best-seed selection',SIR_status='readiness/calibration pending; T5 only; no SIR multiseed launch',resource_safety=dict(location_phase_seconds=10800,sir_phase_seconds=90000,whole_campaign_seconds=604800),resume='Completed phase markers can be reused; partial native training is not silently resumed or counted as complete',git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(C/'protocol.json',protocol)
    md='''# Journal protocol: ten methods, location finding and SIR ODE

Exactly six baselines (Random, Fixed, DAD, simplified RL-sBOED, vsOED, ScoreBED) and four MoE variants (moe,resmoe,switchmoe,switchresmoe). No DAD-MLP baseline or tanh-architecture ablation. A compatible sum-encoder single policy is only an internal residual base.

## Budget, selection and provenance

Each trained method has exactly1000000 newly simulated primary training episodes per seed. Residual methods share an exact seed-specific500000-episode frozen base, then each trains500000 additional episodes. Count pretraining in each complete method; disclose shared physical computation. ScoreBED uses10014 initialization/standardization episodes,692990 score episodes and296996 policy episodes; one restart, batch64. vsOED location uses20000 replay initialization +980*1000 new episodes, native GMM-I discount0.9. Simplified RL uses1million fresh trajectories, L1024, gamma1, twoQ networks, two replay cycles; replay reuse is not a new trajectory. Fixed includes every training simulation. DAD/sPCE training L1024, native sum256/16 linear emitter. All published-method departures remain explicit. Baseline architectures/objectives/optimizers are not replaced by MoE conventions.

Final allocated-budget checkpoint only; no best-validation/test selection and no early plateau stop. Validation every125000 phase episodes is diagnostic and uses independent draws. Fixed-budget complete does not establish convergence. Safety-cap/failed/partial jobs are reported as incomplete. A completed phase may be reused after identity/budget checks; exact mid-phase resume is not claimed for native trainers lacking complete replay/auxiliary state.

## Location and matched routing

Two2D standard-normal sources, log(0.1+sum1/(0.0001+distance²)), Gaussian SD0.5, T30, bounds[-4,4]², unchanged simulator and action transformations. Four64/64tanh experts, sum encoder, component clipping, native regularizers. Matched MoE/SwitchMoE router clock1M; ResMoE/SwitchResMoE clock500k. Gamma1 through30%, cosine to0.001 at95%; gate sigmoid unchanged. Same optimizer settings within each switching pair. Historical SwitchMoE0/80 schedule is incompatible and must be replaced, never interpreted as switching-only evidence.

## Seeds, evaluation and statistics

Training seeds101–105. Seed101 compatibility is checked from saved artifacts, benchmark signatures, budget, native settings, final selection, bank identity and complete action-sequence replay for16 cases. No favorable-seed replacement. Random has no training replicates; one common development reference is not five independent random trainings. Held-out location2000 paired trajectories/L1000000, seed1001. Validation512/L1000000, seed771001. Reserved8831001 remains untouched through development; no final-test selection. The algorithms, source/config manifest and this analysis are frozen before remaining replication.

Principal contrasts: ResMoE-ScoreBED, ResMoE-DAD, ResMoE-MoE, SwitchMoE-MoE, SwitchResMoE-ResMoE, SwitchResMoE-SwitchMoE. Pair identical training-seed indices. Report individual checkpoint trajectoryMC uncertainty separately from SD/SE over independently trained policy means. Use paired Student t95% intervals across available complete seed differences and conservative Bonferroni simultaneous95% intervals over six contrasts; evidence labels use simultaneous intervals. Five runs are limited precision, not10000 independent training replicates. All1001 analyses are development-evaluation replication, not untouched confirmation.

## Computational accounting and mechanisms

Serialize training/evaluation on one device, sample GPU utilization/memory where available. Report primary episodes, contrast particles, stage likelihood work, gradient/replay updates, offline optimization/validation/evaluation seconds and online latency. Historical timings may have overlap and are flagged, not presented as controlled speed comparisons. Residual end-to-end curves include the base; reused computation is disclosed. Missing historical validation points remain missing. Mechanisms report E1–E4 utilization/stage, gradient exposure and final20% starvation, matched-history proposals/residual magnitudes and observation counterfactuals, requests versus actual changes/durations. Counterfactuals are not rollout EIG; no hand-assigned expert roles or guaranteed specialization.

## SIR T5 readiness, then development

Preserve config population500,I0=2,end100,gap0.1; beta=exp(log0.5+0.5Z1),gamma=exp(log0.1+0.5Z2), infected observations with Gaussian SD1, original carried adaptive-min512 RK4 and explicit likelihood. Parameters0.5/0.1 are log-normal medians. No location checkpoints. All four families/baselines require finite feasible rollout/likelihood/gradient checks. Native vsOED is an explicitODE adapter, not the paper stochasticSIR bank reproduction; initial replay320 (instead of unexecuted draft256) plus7810*128 fresh episodes totals exactly1M. ScoreBED pretraining is counted. Proposed common SIR scoring2048/L16384, chunk4096 requires nested-resolution checks; initial readiness checks on smoke/random policies cannot certify developed policies. SIR base500k and residual500k; same base within seed. One T5 development seed101 after location validation and readiness; T10 deferred. GPU batch64 pathwise timing predicts roughly15hours per million episodes, not convergence. Per-phase25hour execution safety cap: incomplete jobs remain explicit; no silent budget reduction. Full SIR five-seed replication is prepared only after development/readiness and benchmark-specific precision freeze.

## Reproduction

Rootrun.sh launches every training phase. tools/journal_campaign.py orchestrates sequential work with source guards and per-phase command JSON. CSV/JSON/figures are generated from artifacts by `python -m tools.journal_campaign report`. Source snapshots/commit hashes and atomic run registry are retained. No historical source manifest/result/checkpoint is overwritten. The source tree includes preexisting uncommitted work, whose frozen content hashes are authoritative in addition to the Git commit.
'''
    textfile(C/'journal_protocol.md',md)
    report();print(json.dumps(audit,indent=2),flush=True)

def seal():
    smoke=C/'readiness/scorebed_fixed_smoke';assert (smoke/'exit_code').read_text().strip()=='0'
    t=read(smoke/'models/scorebed_training.json');assert t['checkpoint_selection']=='final_predeclared_fixed_budget'
    assert t['selected_checkpoint']['episodes']==34 and t['generated_training_trajectories']==34
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/'run.sh',ROOT/'tools/journal_campaign.py',ROOT/'tools/report_moe_family.py',ROOT/'tools/residual_1m_campaign_v2.py']+list((C/'configs').glob('*.yaml'))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in paths}
    write(C/'source_manifest.json',dict(sha256=manifest))
    # Never overwrite the frozen protocol; seal it with an additional immutable record.
    path=C/'location_freeze.json';assert not path.exists()
    write(path,dict(protocol_sha256=sha(C/'protocol.json'),written_protocol_sha256=sha(C/'journal_protocol.md'),source_manifest_sha256=sha(C/'source_manifest.json'),source_hash_count=len(manifest),primary_pairs=PAIRS,evaluation_seed=1001,final_seed_used=False,settings_frozen=True))
    print('Location source/config/analysis seal complete',flush=True)

def guard():
    seal=read(C/'location_freeze.json')
    assert sha(C/'protocol.json')==seal['protocol_sha256'] and sha(C/'journal_protocol.md')==seal['written_protocol_sha256']
    for rel,digest in read(C/'source_manifest.json')['sha256'].items():
        if sha(ROOT/rel)!=digest:raise RuntimeError('Frozen source/config changed: '+rel)

def sample_gpu(path,stop):
    while not stop.is_set():
        try:
            result=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=5)
            with path.open('a') as f:f.write(result)
        except Exception:pass
        stop.wait(10)

def run_phase(benchmark,seed,method,base=None,timeout_seconds=None,evaluation_cases=None,evaluation_bank=None):
    registry=read(C/'registry.json');found=next((e for e in registry if e['benchmark']==benchmark and e['seed']==seed and e['method']==method),None)
    if found:return found
    guard();out=C/benchmark/f'seed{seed}'/method;out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():
        if (out/'exit_code').exists() and (out/'exit_code').read_text().strip()=='0':
            t=read(out/'models'/(native_method(method)+'_training.json'))
            expected=0 if method=='random' else 500000 if method in (*RESIDUAL,'base') else 1000000
            assert t['generated_training_trajectories']==expected
            if method!='random':assert t['checkpoint_selection']=='final_predeclared_fixed_budget'
            entry=dict(method=method,native_method=native_method(method),benchmark=benchmark,seed=seed,run=str(out),status='complete',provenance='recovered_completed_phase_after_coordinator_interruption')
            registry.append(entry);write(C/'registry.json',registry);return entry
        raise RuntimeError('Partial/unregistered phase exists; preserve and inspect before any resume: '+str(out))
    cap=500000 if method in (*RESIDUAL,'base') else 1000000
    config=C/'configs'/f'{benchmark}_{cap}.yaml'
    command=['bash','run.sh','--config',str(config),'--journal','--seed',str(seed),'--eval-seeds','1001','--eig-only','--source-check-manifest',str(C/'source_manifest.json'),'--output',str(out)]
    if method in FAMILY:command+=['--variant',method]
    else:command+=['--methods',native_method(method)]
    if base:command+=['--base-checkpoint',str(base)]
    if evaluation_cases or evaluation_bank:
        if not evaluation_cases or not evaluation_bank:raise ValueError('Both saved evaluation arrays are required')
        command+=['--evaluation-cases',str(evaluation_cases),'--evaluation-bank',str(evaluation_bank)]
    write(out.parent/(method+'_command.json'),dict(command=command,training_cap=cap))
    stop=threading.Event();watch=threading.Thread(target=sample_gpu,args=(out.parent/(method+'_gpu.csv'),stop),daemon=True)
    started=time.monotonic();watch.start();status='failed';failure=None
    try:
        with (out.parent/(method+'_launcher.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout_seconds if timeout_seconds is not None else (10800 if benchmark=='location' else 90000))
        name=native_method(method);t=read(out/'models'/(name+'_training.json'))
        if method!='random':
            assert t['generated_training_trajectories']==cap and t['checkpoint_selection']=='final_predeclared_fixed_budget'
            assert t['selected_checkpoint']['episodes']==cap
        else:assert t['generated_training_trajectories']==0
        if method in RESIDUAL:
            assert t['total_training_episodes']==1000000 and t['base_checkpoint']['sha256']==sha(base)
            from src.research.moe_family import load_policy,BasePolicy
            from src.research.benchmarks import make_benchmark
            b=make_benchmark(yaml.safe_load(config.read_text()));policy=load_policy(out/'models'/(name+'.pt'),b);reference=BasePolicy.from_checkpoint(base,b,policy.horizon).policy
            for key,value in reference.state_dict().items():torch.testing.assert_close(policy.core.base.policy.state_dict()[key],value,rtol=0,atol=0)
        status='complete'
    except Exception as exc:
        failure=dict(type=type(exc).__name__,message=str(exc),classification='computational_limit' if isinstance(exc,subprocess.TimeoutExpired) else 'requires_inspection',native_failure=read(out/'failure.json') if (out/'failure.json').exists() else None)
    finally:stop.set();watch.join(timeout=6)
    entry=dict(method=method,native_method=native_method(method),benchmark=benchmark,seed=seed,run=str(out),status=status,failure=failure,serialized_wall_seconds=time.monotonic()-started,provenance='new_fixed_configuration_training')
    if status=='complete':entry['checkpoint_sha256']=sha(out/'models'/(native_method(method)+'.pt')) if method!='random' else None
    registry.append(entry);write(C/'registry.json',registry);write(out.parent/(method+'_completion.json'),entry)
    print(benchmark,seed,method,status,flush=True);report();return entry

def run_location():
    guard()
    lock=(C/'device.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Correct the incompatible historical SwitchMoE before the independent-seed grid.
    for seed in (101,102,103,104,105):
        for method in METHODS:
            if method=='random' and seed!=101:continue
            if method in RESIDUAL:
                base=run_phase('location',seed,'base')
                if base['status']!='complete':
                    registry=read(C/'registry.json');registry.append(dict(method=method,native_method=method,benchmark='location',seed=seed,run='',status='failed_dependency',failure='Internal base failed'));write(C/'registry.json',registry);continue
                checkpoint=Path(base['run'])/'models/sum_single.pt'
                run_phase('location',seed,method,checkpoint)
            else:run_phase('location',seed,method)
    report();mechanisms('location');figures()
    registry=read(C/'registry.json');missing=[(method,seed) for method in METHODS for seed in ([101] if method=='random' else [101,102,103,104,105]) if not any(e['benchmark']=='location' and e['method']==method and e['seed']==seed and e['status']=='complete' for e in registry)]
    write(C/'location_completed.json',dict(completed=not missing,missing_or_failed=missing,final_seed_used=False,registry_sha256=sha(C/'registry.json')))
    if missing:raise RuntimeError('Location grid has failed/missing cells; reported, SIR launch held')
    # SIR readiness happens only after the complete location phase, never concurrently.
    sir_readiness()

def record_metrics(entry):
    method=entry['method'];seed=entry['seed'];benchmark=entry['benchmark']
    basic=dict(benchmark=benchmark,method=method,name=NAMES.get(method,'Internal residual base'),training_seed=seed,status=entry['status'],provenance=entry.get('provenance'),evaluation_seed=1001,phase='development_evaluation_replication')
    if entry['status']!='complete':return basic,None,[]
    run=Path(entry['run']);t=train_record(entry);rows=eval_rows(entry);s=next(v for v in read(run/'summary.json') if v['method']==entry['native_method'] and v['eval_seed']==1001)
    values=np.asarray([v['spce'] for v in rows]);score=mc(values)
    basic.update(eig=score['mean'],trajectory_mc_se=score['se'],trajectory_mc_low=score['low'],trajectory_mc_high=score['high'],training_episodes=0 if method=='random' else 500000 if method=='base' else 1000000,checkpoint_selection=t.get('checkpoint_selection'),convergence='budget_limited_or_unverified' if method!='random' else 'not_applicable',run=str(run))
    return basic,(t,s),rows

def report():
    registry=read(C/'registry.json');location=[];sir=[];eff=[];curves=[];case_rows={};method_scores={}
    for entry in registry:
        if entry['method']=='base':continue
        row,records,rows=record_metrics(entry)
        (location if entry['benchmark']=='location' else sir).append(row)
        if not records:continue
        t,s=records;method=entry['method'];seed=entry['seed'];benchmark=entry['benchmark'];h=30 if benchmark=='location' else 5
        method_scores[(benchmark,method,seed)]=row['eig'];case_rows[(benchmark,method,seed)]=rows
        phases=[t];base_time=base_episodes=0;base_eval_seconds=base_eval_work=0
        if method in RESIDUAL:
            base_entry=next(e for e in registry if e['benchmark']==benchmark and e['seed']==seed and e['method']=='base' and e['status']=='complete')
            base=train_record(base_entry);phases=[base,t];base_time=base['training_seconds'];base_episodes=500000
            base_summary=read(Path(base_entry['run'])/'summary.json')[0];base_eval_seconds=base_summary['evaluation_seconds'];base_eval_work=base_summary['forward_particle_stages']
        contrast=sum(p.get('contrast_training_trajectories',0) for p in phases)
        primary=0 if method=='random' else 1000000
        validation_seconds=sum(p.get('validation_seconds',p.get('diagnostic_validation_seconds',0.)) for p in phases)
        optimizer=sum(p['training_seconds'] for p in phases)
        historical=entry.get('provenance')=='historical_development'
        eff.append(dict(benchmark=benchmark,method=method,training_seed=seed,primary_episodes=primary,contrast_trajectories=contrast,truth_plus_contrast_trajectories=primary+contrast,particle_stage_work=h*(primary+contrast),likelihood_stage_evaluations=0 if method in {'vsoed','scorebed','random'} else h*(primary+contrast),base_episodes=base_episodes,base_training_seconds=base_time,optimization_seconds=optimizer,validation_seconds=validation_seconds,offline_seconds_including_validation=optimizer+validation_seconds,evaluation_seconds=s['evaluation_seconds'],base_diagnostic_evaluation_seconds=base_eval_seconds,base_diagnostic_evaluation_stage_work=base_eval_work,offline_total_with_all_diagnostics_seconds=optimizer+validation_seconds+base_eval_seconds+s['evaluation_seconds'],online_latency_definition='Batched policy/action wall time divided by trajectories and stages; excludes simulator, not standalone batch1 latency',online_decision_seconds=s['mean_decision_seconds'],validation_particle_stage_work=sum(v['validation']['forward_particle_stages'] for phase in phases for v in phase.get('journal_validation_history',[])) if all('journal_validation_history' in phase for phase in phases) else None,evaluation_particle_stage_work=s.get('forward_particle_stages'),gradient_updates=t.get('actual_updates',t.get('actor_optimizer_updates',t.get('gradient_updates'))),replay_updates=t.get('gradient_updates'),peak_gpu_memory_mib=gpu_peak(entry),timing_comparability='historical_overlap_not_controlled' if historical else 'serialized_campaign',base_computation_shared=method in RESIDUAL,run=entry['run']))
        offset_episode=offset_time=0
        for phase_index,p in enumerate(phases):
            for point in p.get('journal_validation_history',[]):
                curves.append(dict(benchmark=benchmark,method=method,training_seed=seed,phase='base' if len(phases)>1 and phase_index==0 else 'policy',episodes=offset_episode+point['episodes'],training_seconds=offset_time+point['training_seconds'],validation_eig=point['eig'],validation_mc_se=point['mc_standard_error']))
            offset_episode+=p.get('generated_training_trajectories',0);offset_time+=p['training_seconds']
    existing={(x['method'],x['training_seed']) for x in location}
    for method in METHODS:
        for seed in ([101] if method=='random' else [101,102,103,104,105]):
            if (method,seed) not in existing:location.append(dict(benchmark='location',method=method,name=NAMES[method],training_seed=seed,status='planned',evaluation_seed=1001,phase='development_evaluation_replication'))
    fields=['benchmark','method','name','training_seed','status','eig','trajectory_mc_se','trajectory_mc_low','trajectory_mc_high','training_episodes','checkpoint_selection','convergence','evaluation_seed','phase','provenance','run']
    table(C/'location_finding_results.csv',location,fields);table(C/'sir_t5_results.csv',sir or [dict(benchmark='sir',method=m,name=NAMES[m],training_seed=101,status='readiness_pending') for m in METHODS],fields)
    ef=['benchmark','method','training_seed','primary_episodes','contrast_trajectories','truth_plus_contrast_trajectories','particle_stage_work','likelihood_stage_evaluations','gradient_updates','replay_updates','base_episodes','base_training_seconds','optimization_seconds','validation_seconds','offline_seconds_including_validation','evaluation_seconds','base_diagnostic_evaluation_seconds','base_diagnostic_evaluation_stage_work','offline_total_with_all_diagnostics_seconds','online_latency_definition','online_decision_seconds','validation_particle_stage_work','evaluation_particle_stage_work','peak_gpu_memory_mib','timing_comparability','base_computation_shared','run']
    table(C/'training_efficiency.csv',eff,ef);table(C/'learning_curves.csv',curves,['benchmark','method','training_seed','phase','episodes','training_seconds','validation_eig','validation_mc_se'])
    comparisons=[]
    for left,right in PAIRS:
        differences=[]
        for seed in (101,102,103,104,105):
            a=case_rows.get(('location',left,seed));b=case_rows.get(('location',right,seed))
            if a is None or b is None:continue
            assert paired_identity(a)==paired_identity(b),'Unpaired evaluation cases'
            d=mc(np.asarray([r['spce'] for r in a])-np.asarray([r['spce'] for r in b]));differences.append(d['mean'])
            comparisons.append(dict(scope='trajectory_MC_conditional_checkpoint',left=left,right=right,training_seed=seed,n_training_pairs=1,difference=d['mean'],standard_error=d['se'],ci_low=d['low'],ci_high=d['high'],conclusion='conditional_only_not_training_seed_inference'))
        n=len(differences)
        if n:
            a=np.asarray(differences);mean=float(a.mean());sd=float(a.std(ddof=1)) if n>1 else None;se=sd/n**.5 if sd is not None else None
            ci=student_t.ppf(.975,n-1)*se if n>1 else None;sim=student_t.ppf(1-.05/(2*len(PAIRS)),n-1)*se if n>1 else None
            label='inconclusive' if sim is None or mean-sim<=0<=mean+sim else 'positive_supported' if mean-sim>0 else 'negative_supported'
            comparisons.append(dict(scope='paired_training_seed_means',left=left,right=right,n_training_pairs=n,difference=mean,across_seed_sd=sd,standard_error=se,ci_low=mean-ci if ci is not None else None,ci_high=mean+ci if ci is not None else None,simultaneous_ci_low=mean-sim if sim is not None else None,simultaneous_ci_high=mean+sim if sim is not None else None,conclusion=label,completion='complete5' if n==5 else 'interim_incomplete'))
    table(C/'location_finding_comparisons.csv',comparisons,['scope','left','right','training_seed','n_training_pairs','difference','across_seed_sd','standard_error','ci_low','ci_high','simultaneous_ci_low','simultaneous_ci_high','conclusion','completion'])
    aggregates=[]
    for method in METHODS:
        x=[v for (benchmark,m,seed),v in method_scores.items() if benchmark=='location' and m==method];n=len(x)
        aggregates.append(dict(method=method,name=NAMES[method],n_completed=n,eig_mean=float(np.mean(x)) if n else None,across_seed_sd=float(np.std(x,ddof=1)) if n>1 and method!='random' else None,training_seed_se=float(np.std(x,ddof=1)/n**.5) if n>1 and method!='random' else None,random_has_no_training_replicates=method=='random'))
    for a in aggregates:
        n=a['n_completed'];se=a['training_seed_se'];width=student_t.ppf(.975,n-1)*se if n>1 and se is not None else None
        a['training_seed_ci_low']=a['eig_mean']-width if width is not None else None
        a['training_seed_ci_high']=a['eig_mean']+width if width is not None else None
    write(C/'location_aggregates.json',aggregates)
    table(C/'location_finding_summary.csv',aggregates,['method','name','n_completed','eig_mean','across_seed_sd','training_seed_se','training_seed_ci_low','training_seed_ci_high','random_has_no_training_replicates'])
    textfile(C/'reproduce_figures.txt','python -m tools.journal_campaign report\nFigures are regenerated from registry.json, saved per-trajectory evaluations, learning_curves.csv and diagnostics/*.json.\n')
    textfile(C/'status.md','# Journal campaign status\n\nDevelopment evaluation1001 only; final8831001 untouched.\n\n'+''.join(f"- {x['name']}: {x['n_completed']} completed; mean EIG {x['eig_mean']}\n" for x in aggregates)+'\nSee CSVs for every completed, failed and planned seed. Historical SwitchMoE is excluded until its matched-schedule replacement completes.\n')
    if not (C/'sir_readiness.md').exists():textfile(C/'sir_readiness.md','# SIR T5\n\nReadiness and contrast-resolution calibration are pending location completion. Existing smoke tests are not full-budget results. No location checkpoint will be used.\n')
    if not (C/'moe_diagnostics.md').exists():textfile(C/'moe_diagnostics.md','# MoE diagnostics\n\nFull across-seed diagnostics run after location phases. Historical high rerouting and inactive experts are observations to test, not established persistent specialization.\n')

def gpu_peak(entry):
    path=Path(entry['run']).parent/(entry['method']+'_gpu.csv')
    if not path.exists():return None
    values=[]
    for line in path.read_text().splitlines():
        try:values.append(float(line.split(',')[-1]))
        except ValueError:pass
    return max(values) if values else None

def mechanisms(benchmark):
    from tools.report_moe_family import summarize_rows,counterfactuals
    from tools.residual_1m_campaign_v2 import training_diagnostics
    from src.research.journal_checkpoints import load_checkpoint
    from src.research.benchmarks import make_benchmark
    torch.set_num_threads(2);reports=[]
    for entry in read(C/'registry.json'):
        if entry['benchmark']!=benchmark or entry['method'] not in FAMILY or entry['status']!='complete':continue
        run=Path(entry['run']);rows=eval_rows(entry);native=entry['native_method'];cfg=read(run/'run_config.json');b=make_benchmark(cfg['benchmark']);p=load_checkpoint(run/'models'/(native+'.pt'),b)
        management=summarize_rows(rows) if rows[0]['routing'] and 'strategy' in rows[0]['routing'][0] else dict(note='Legacy per-step MoE has routing weights/actions; strategy state was not logged',selection_frequency=np.asarray([[r['weights'] for r in case['routing']] for case in rows]).argmax(-1).reshape(-1).tolist())
        if 'note' in management:
            a=np.asarray(management['selection_frequency']);management['selection_frequency']=[float((a==j).mean()) for j in range(4)]
        trace=run/'models'/(native+'_diagnostics.jsonl');budget=500000 if entry['method'] in RESIDUAL else 1000000
        training=training_diagnostics(trace,budget) if trace.exists() else dict(note='No per-update trace available')
        # The existing counterfactual helper supports native family policy objects.
        proposal=observation=None
        if hasattr(p,'core'):
            proposal,observation=counterfactuals(b,p,rows,8)
        item=dict(benchmark=benchmark,method=entry['method'],seed=entry['seed'],management=management,training=training,proposals=proposal,observation_counterfactuals=observation)
        write(C/'diagnostics'/f"{benchmark}_{entry['method']}_seed{entry['seed']}.json",item);reports.append(item)
    md='# MoE expert and switching diagnostics\n\nAll results are development-evaluation replication. First learned measurement excluded from expert statistics. Counterfactual proposals/observations are not rollout EIG or proof of specialization.\n\n| Method | Seed | E1–E4 hard usage | Reroute fraction |\n|---|---:|---|---:|\n'
    for r in reports:md+=f"| {NAMES[r['method']]} | {r['seed']} | {r['management'].get('selection_frequency')} | {r['management'].get('reroute_fraction_excluding_initial')} |\n"
    md+='\nPer-expert gradients/final20% starvation, router/gate spikes, durations, proposal differences/residual corrections and observation dependence are in diagnostics/*.json. Distinct actions and usage do not establish that learned routing improves EIG. No algorithm was changed in response to results.\n'
    textfile(C/('moe_diagnostics_'+benchmark+'.md'),md)
    combined='\n\n'.join(p.read_text() for p in [C/'moe_diagnostics_location.md',C/'moe_diagnostics_sir.md'] if p.exists())
    textfile(C/'moe_diagnostics.md',combined)

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'savefig.dpi':300,'pdf.fonttype':42})
    aggregate=read(C/'location_aggregates.json');fig,ax=plt.subplots(figsize=(9,4))
    xs=[];ys=[];errors=[];names=[]
    for a in aggregate:
        if a['eig_mean'] is None:continue
        xs.append(len(xs));ys.append(a['eig_mean']);names.append(a['name']);errors.append(0 if a['training_seed_se'] is None else student_t.ppf(.975,a['n_completed']-1)*a['training_seed_se'])
    ax.errorbar(xs,ys,yerr=errors,fmt='o',capsize=3);ax.set_xticks(xs,names,rotation=35,ha='right');ax.set_ylabel('Held-out EIG (nats)');ax.set_title('Development evaluation1001; across-training-seed95% t intervals where n≥2');fig.tight_layout()
    for ext in ('pdf','png'):fig.savefig(C/'figures'/('location_eig.'+ext))
    plt.close(fig)
    curves=list(csv.DictReader((C/'learning_curves.csv').open()))
    for axis,label in [('episodes','Cumulative primary training episodes'),('training_seconds','Cumulative optimization seconds')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for method in METHODS:
            points=[r for r in curves if r['benchmark']=='location' and r['method']==method and r['training_seed']=='102']
            if points:ax.plot([float(r[axis]) for r in points],[float(r['validation_eig']) for r in points],label=NAMES[method])
        ax.set_xlabel(label);ax.set_ylabel('Diagnostic validation EIG');ax.set_title('Representative seed102; available validation records only')
        if ax.lines:ax.legend(fontsize=7,ncol=2)
        fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(C/'figures'/('validation_vs_'+axis+'.'+ext))
        plt.close(fig)
    diagnostics=[read(p) for p in (C/'diagnostics').glob('location_*_seed102.json')] if (C/'diagnostics').exists() else []
    for r in diagnostics:
        u=r['management'].get('stage_selection_frequency')
        if u is not None:
            fig,ax=plt.subplots(figsize=(7,3));ax.stackplot(range(2,31),np.asarray(u).T,labels=['E1','E2','E3','E4']);ax.set(xlabel='Experiment stage',ylabel='Selection fraction',title=NAMES[r['method']]+' seed102');ax.legend(ncol=4);fig.tight_layout();fig.savefig(C/'figures'/(r['method']+'_stage_usage.pdf'));plt.close(fig)
        durations=r['management'].get('duration_by_expert',{})
        if durations:
            fig,axes=plt.subplots(1,2,figsize=(8,3));axes[0].plot(range(2,31),r['management']['stage_reroute_fraction'],label='Requests');axes[0].plot(range(2,31),r['management']['stage_actual_change_fraction'],label='Changes');axes[0].legend();axes[0].set_xlabel('Stage');axes[1].bar(['E1','E2','E3','E4'],[durations[e]['mean'] if durations.get(e) else 0 for e in ['E1','E2','E3','E4']]);axes[1].set_ylabel('Mean consecutive duration');fig.suptitle(NAMES[r['method']]+' seed102');fig.tight_layout();fig.savefig(C/'figures'/(r['method']+'_switching.pdf'));plt.close(fig)

def sir_readiness():
    guard();out=C/'sir_readiness';out.mkdir(exist_ok=True)
    tests=[sys.executable,'-m','pytest','-q','tools/tests/test_sir_offline.py','tools/tests/test_sir_residual_base.py','tools/tests/test_baseline_readiness.py','tools/tests/test_moe_family.py']
    with (out/'tests.log').open('w') as f:result=subprocess.run(tests,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:
        textfile(C/'sir_readiness.md','# SIR T5 readiness\n\nReadiness tests failed. No full SIR job launched. See sir_readiness/tests.log.\n');return
    from src.research.benchmarks import make_benchmark
    from src.research.journal_validation import CommonValidation
    from src.research.journal_checkpoints import load_checkpoint
    raw=yaml.safe_load((C/'configs/sir_1000000.yaml').read_text());b=make_benchmark(raw)
    cp=ROOT/'experiments/10152026/10152026_journal_location_sir/readiness/sir_base_smoke/models/sum_single.pt';p=load_checkpoint(cp,b)
    v=CommonValidation(raw,5,772001,16,65536,8,4096);scores={};records=[]
    for contrasts in (4096,16384,65536):
        a,cost=v(p,contrasts);scores[contrasts]=a;records.append(dict(contrasts=contrasts,eig=mc(a),cost=cost));np.save(out/f'initial_resolution_{contrasts}.npy',a)
    x=mc(scores[65536]-scores[16384]);write(out/'initial_resolution.json',dict(records=records,paired_difference=x,limitation='Initial/smoke policy only; repeat on completed developed policies before SIR precision freeze'))
    textfile(C/'sir_readiness.md','# SIR T5 readiness\n\nCPU/GPU finite gradients, originalODE/reference agreement, explicit likelihood, action constraints and four variants passed focused tests. NativeSIR vsOED/ScoreBED smoke adapters previously passed; explicitODE benchmark adaptation remains disclosed. Initial contrast-resolution results are saved but do not certify trained policies. Full development uses provisional2048/L16384; developed-policy resolution will be checked before final claims. Native vsOED initial replay320 +7810*128 episodes meets1M exactly. SIR-specific base500k is required; no location checkpoints. Expected pathwise cost~15 GPU hours per million at batch64. No SIR multi-seed launch in this task stage.\n')
    write(out/'ready_for_development.json',dict(correctness_tests_passed=True,initial_resolution_only=True,final_seed_used=False,expected_pathwise_hours_per_million=15.3))

def run_sir():
    assert read(C/'location_completed.json')['completed'] and (C/'sir_readiness/ready_for_development.json').exists()
    guard();lock=(C/'device.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for method in METHODS:
        base=None
        if method in RESIDUAL:
            e=run_phase('sir',101,'base')
            if e['status']!='complete':continue
            base=Path(e['run'])/'models/sum_single.pt'
        run_phase('sir',101,method,base)
    report();mechanisms('sir');sir_resolution_and_figures();write(C/'sir_development_completed.json',dict(completed=True,precision_freeze_pending=True,final_seed_used=False))

def sir_resolution_and_figures():
    from src.research.benchmarks import make_benchmark
    from src.research.journal_checkpoints import load_checkpoint
    from src.research.engine import rollout
    from src.research.sir_evaluation import contrast_scores
    raw=yaml.safe_load((C/'configs/sir_1000000.yaml').read_text());b=make_benchmark(raw)
    rng=torch.Generator().manual_seed(772001);truth=b.sample((128,),rng);noise=torch.randn(128,5,1,generator=rng,dtype=torch.float64);bank=b.sample((65536,),rng)
    resolution=[];examples=[]
    for entry in read(C/'registry.json'):
        if entry['benchmark']!='sir' or entry['method']=='base' or entry['status']!='complete':continue
        path=Path(entry['run']);method=entry['method'];p=None if method=='random' else load_checkpoint(path/'models'/(method+'.pt'),b)
        scores={L:[] for L in (4096,16384,65536)}
        for start in range(0,128,16):
            with torch.no_grad():r=rollout(b,p,5,16,772001+start*97451,0,truth=truth[start:start+16],noise=noise[start:start+16],method='random' if method=='random' else 'rl_sboed' if method=='rl_sboed' else None)
            for L in scores:
                info,_=contrast_scores(b,r,bank[:L],4096);scores[L].extend(info[:,-1].tolist())
        delta=mc(np.asarray(scores[65536])-np.asarray(scores[16384]));adequate=abs(delta['mean'])+1.96*delta['se']<=.05
        resolution.append(dict(method=method,scores={str(L):mc(v) for L,v in scores.items()},paired_65536_minus_16384=delta,passes_predeclared_point05_tolerance=adequate))
        examples.append((method,eval_rows(entry)[0]))
    write(C/'sir_readiness/developed_policy_resolution.json',dict(systems=128,seed=772001,records=resolution,note='Development sensitivity, not final-test EIG; fail means L16384 cannot be certified',all_completed_pass=bool(resolution) and all(x['passes_predeclared_point05_tolerance'] for x in resolution)))
    with (C/'sir_readiness.md').open('a') as f:f.write('\nDeveloped-policy nested-resolution checks are in sir_readiness/developed_policy_resolution.json. A failed tolerance blocks SIR precision freeze and multi-seed confirmation; results remain provisional.\n')
    if examples:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,1,figsize=(8,6))
        from scipy.integrate import solve_ivp
        theta=np.asarray(examples[0][1]['truth']);beta,gamma=theta
        def rhs(t,x):
            infection=beta*x[0]*x[1]/b.population;recovery=gamma*x[1]
            return [-infection,infection-recovery,recovery]
        grid=np.linspace(0,b.end,300);sol=solve_ivp(rhs,[0,b.end],[b.population-b.i0,b.i0,0],t_eval=grid,rtol=1e-9,atol=1e-10)
        axes[1].plot(grid,sol.y[1],color='black',label='Paired truth ODE (illustration)')
        for i,(method,row) in enumerate(examples):
            times=np.asarray(row['actions']).reshape(-1);ys=np.asarray(row['observations']).reshape(-1)
            axes[0].scatter(times,np.full(5,i),s=12);axes[1].scatter(times,ys,s=10,label=NAMES[method])
        axes[0].set_yticks(range(len(examples)),[NAMES[m] for m,_ in examples]);axes[0].set_xlabel('Selected observation time');axes[1].set_xlabel('Time');axes[1].set_ylabel('Infected count');axes[1].legend(fontsize=6,ncol=2);fig.tight_layout()
        for ext in ('pdf','png'):fig.savefig(C/'figures'/('sir_t5_designs.'+ext),dpi=300)
        plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','seal','location','report','sir-readiness','sir']);args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='seal':seal()
    elif args.action=='location':run_location()
    elif args.action=='report':report();figures()
    elif args.action=='sir-readiness':sir_readiness()
    else:run_sir()

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if C.exists():write(C/'coordinator_failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
