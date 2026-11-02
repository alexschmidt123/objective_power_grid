"""Five-seed conventional location baseline campaign, entered through run.sh.

Native method schedules are separate from whole-method resource ceilings.
Completed jobs are reusable; interrupted native phases are never silently reset.
"""
import argparse, csv, fcntl, hashlib, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
PY = Path.home() / 'miniconda3/envs/mocu_optimized/bin/python'
METHODS = ('dad', 'fixed', 'rl_sboed', 'vsoed', 'scorebed')
SEEDS = (101, 102, 103, 104, 105)
FINAL_SEED, VALIDATION_SEED = 73831031, 73831032

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(p)

def config(method, seconds):
    import yaml
    from src.research.original_protocol import dad_settings, vsoed_profile
    raw = yaml.safe_load((R/'configs/current_location.yaml').read_text())
    research = raw['research']
    # Discard all inherited pilot caps, method overrides, and legacy curricula.
    research.clear()
    settings = dict(updates=50000, hidden=64, threads=4, batch_size=2000,
        contrasts=2000, training_bound=4, primary_episode_budget=1,
        spce_trajectory_budget_unit='primary_episodes', spce_curriculum=1,
        learning_rate=5e-5, spce_adam_beta1=.8, spce_adam_beta2=.998,
        spce_lr_decay=.98, spce_lr_schedule_every=1000, spce_gradient_clip=None,
        first_design_revision=1, validate_every=100, validation_systems=64,
        validation_contrasts=1000000, eval_systems=2000, eval_contrasts=1000000,
        evaluation_population_systems=2000, evaluation_batch_size=16,
        evaluation_contrast_chunk=16384)
    research['enabled'] = True; research['settings'] = settings
    research['paper_protocol'] = dict(corrected_methods=True,
        checkpoint_selection='latest_budget_eligible_checkpoint',
        regime='native_algorithms_training_sufficiency_five_seed',
        adaptations='Common bounded [-4,4]^2 design box; executable log-signal likelihood; method-specific native schedules; resource ceilings do not establish convergence')
    profile = vsoed_profile('location_finding', 20000+10001*1000)
    base = dad_settings() | dict(spce_trajectory_budget_unit='primary_episodes',
        reference_optimizer_updates=50000, reference_training_episodes=100000000,
        reference_schedule_episodes=100000000)
    specific = dict(dad=base, fixed=base | dict(batch_size=64,contrasts=1024,
            learning_rate=5e-5,first_design_revision=2,spce_gradient_clip=5.,
            reference_training_episodes=3200000,reference_schedule_episodes=3200000),
        rl_sboed=dict(rl_original_reference=True, rl_native_contrasts=100000,
            rl_native_replay_capacity=1000000, reference_schedule_episodes=20001*100),
        vsoed=dict(reference_schedule_episodes=20000+10001*1000),
        scorebed=dict(scorebed_batch_size=1024, scorebed_restarts=1,
            scorebed_reference_sampler=True,
            score_schedule_reference_steps=190000, policy_schedule_reference_steps=10000,
            scorebed_declared_full_score_updates=190000,
            scorebed_declared_full_policy_updates=10000,
            scorebed_pretraining_episodes=177365*1024,
            reference_schedule_episodes=10014+(177365+9335)*1024,
            scorebed_compute_phase_plan=dict(hardware='NVIDIA GeForce RTX 4090',
                score_updates=177365,policy_updates=9335,phase_ratio=19,
                estimated_seconds=13679.133742374821,allowance_seconds=14400,
                reserve_seconds=720,native_batch_size=1024,
                original_score_scheduler_updates=190000,original_policy_scheduler_updates=10000,
                timing_based=True,selection_used_EIG=False,
                provenance='Archived seed101 hardware timing calibration b9776095457953ec54170e5b2410009eb040a4210514fec40419aca57f878c40')))
    p = dict(version='validation_convergence_v1',
        study_id='location_corrected_five_seed_training_sufficiency_20261031',
        training_seeds=list(SEEDS), development_training_seeds=list(SEEDS),
        evaluation_seeds=[FINAL_SEED], checkpoint_selection_mode='final_compute_budget',
        training_device='cuda_if_available', preserve_progress_checkpoints=True,
        validation=dict(seed=VALIDATION_SEED, systems=128, contrasts=1000000,
            batch_size=16, chunk=16384, device='cuda_if_available'),
        stopping_rule=dict(minimum_episodes=1, minimum_checkpoints=8,
            validation_every_episodes=1000000000, patience=6, minimum_improvement=.05,
            noise_multiplier=1.96, maximum_episodes=1000000000,
            maximum_training_seconds=seconds),
        method_settings=specific, common_pathwise_settings={},
        native_vsoed_profile=profile, compute_budget=dict(seconds=seconds,
            hardware='NVIDIA GeForce RTX 4090', checkpoint_seconds=1800,
            finalization_reserve_seconds=120, require_prelaunch_start=True),
        stopping='Native finite schedule or charged resource ceiling; no poor-score stopping. Plateau assessed separately, not assumed.',
        final_evaluation=dict(seed=FINAL_SEED, systems=2000, contrasts=1000000))
    research['journal_protocol'] = p
    raw['experiment']['methods'] = [method]
    raw.pop('moe_target', None); raw.pop('current_method_registry', None)
    raw.pop('baseline_protocol_provenance', None)
    raw['scope_note'] = 'New conventional baseline campaign; no MoE training or historical initialization'
    return raw

def acquire():
    files=[]
    try:
        for p in (R.parent/'objective_power_grid/.gpu0.lock',
                  Path.home()/'.cache/objective_power_grid/device_0.lock', R/'.gpu0.lock'):
            f=p.open('a+');files.append(f);fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if busy: raise BlockingIOError('GPU compute process(es): '+busy)
        return files
    except BaseException:
        for f in files:f.close()
        raise

def state(c, status, **kw):
    write(c/'status.json', dict(state=status, controller_pid=os.getpid(),
        updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), **kw))

def verify(c):
    manifest=json.loads((c/'source_manifest.json').read_text())['sha256']
    changed=[p for p,h in manifest.items() if not (R/p).exists() or sha(R/p)!=h]
    if changed:raise RuntimeError('Frozen source changed: '+','.join(changed[:10]))
    cfg=json.loads((c/'manifest.json').read_text())
    for p,h in cfg['config_sha256'].items():
        if sha(c/p)!=h:raise RuntimeError('Frozen configuration changed: '+p)

def prepare(c, seconds):
    import yaml
    if R.name!='objective_power_grid_temp':raise ValueError('TEMP only')
    if seconds!=14400:raise ValueError('User requires exactly a four-hour maximum allowance')
    c.relative_to(R/'scratch_runs');c.mkdir(parents=True,exist_ok=False)
    for d in ('configs','reports','logs','runs','evaluations','readiness'):(c/d).mkdir()
    hashes={}
    for m in ('random',)+METHODS:
        p=c/'configs'/f'{m}.yaml';p.write_text(yaml.safe_dump(config(m,seconds),sort_keys=False))
        hashes[str(p.relative_to(c))]=sha(p)
    write(c/'manifest.json',dict(status='PREPARING',benchmark='location_finding',horizon=30,
        seeds=list(SEEDS),methods=list(METHODS),random='One common untrained reference',
        per_method_seed_ceiling_seconds=seconds,maximum_training_seconds=25*seconds,
        final_seed=FINAL_SEED,validation_seed=VALIDATION_SEED,
        final_draws_access='Only after all training candidates are frozen',
        config_sha256=hashes,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    refs=R.parent/'objective_power_grid/experiments/_legacy_archive/labpc/stable_original/experiments/10182026/10182026_original_baseline_reproduction/upstream_references'
    shutil.copytree(refs,c/'readiness/upstream_references')
    (c/'reports/PROTOCOL.md').write_text(
        '# Corrected location-finding five-seed campaign\n\n'
        'T=30; two independent standard-normal 2D sources; log(.1 + sum 1/(1e-4+distance^2)); Gaussian observation sigma=.5; common [-4,4]^2 designs. Random retains truncated standard-normal designs.\n\n'
        'Fresh initialization for seeds101–105. Native DAD full-horizon sPCE, native SAC/REDQ RL, native variational actor-critic-posterior vsOED, native two-stage ScoreBED; no shared episode quota. Fixed is one nonadaptive schedule optimized by joint sPCE.\n\n'
        f'Whole-method ceiling: {seconds} seconds per seed/method, including preparation, pretraining, restarts and serialization. Native schedule completion can occur sooner. Budget exhaustion is COMPUTE_LIMITED, never convergence. No arbitrary tiny update target.\n\n'
        'DAD:50000 updates B2000,L2000. Fixed:50000 updates B64,L1024,matching the completed corrected seed101 recipe; this is a disclosed method-specific offline optimizer setting, not equal updates forced across all algorithms. RL:20001 collection cycles B100,L100000 and64 native SAC/REDQ updates per eligible cycle. vsOED:10001 native cycles plus20000 initialization. ScoreBED:P1 reference190000 score updates and10000 policy updates, native B1024,one policy arm per training seed; the hardware-timing-only four-hour adaptation freezes177365 score updates then9335 policy updates, preserves native scheduler horizons, and is COMPUTE_LIMITED. No expensive stage is uncharged or silently dropped. This single-restart adaptation is explicit and is not an all-restarts paper reproduction.\n\n'
        'Preserve common charged-time checkpoints at approximately0.5h,1h,2h,4h and thereafter every0.5h. Fair-compute comparisons and training-sufficiency comparisons are separate. Latest valid checkpoint, no final-EIG selection.\n\n'
        'Independent development validation:128 cases,L1M,seed73831032, only to report progression. Final2000 paired cases,L1M,seed73831031, generated only after checkpoints are frozen. Raw paired scores, within-policy MCSE, between-training-seed SD and mean SE reported separately.\n\n'
        'Training adequacy is not inferred from a time limit: report native completion, saturation, recent development improvements and their paired uncertainty. An evidently improving capped run is not an adequately trained scientific endpoint. No automatic extension, retuning, STABLE promotion or GitHub push.\n\n'
        'Persistent controller never restarts a partial native phase from zero. Completed phases are reusable; partial phases remain blocked pending exact-state recovery.\n')
    state(c,'PREPARING')

def freeze(c):
    from src.research.runner import freeze_source
    freeze_source(R,c)
    for p in (R/'tools/tests').glob('*.py'):
        # Test receipts record executed tests; source snapshots include harnesses.
        if p.name in ('test_location_five_seed.py',):shutil.copy2(p,c/'readiness'/p.name)
    m=json.loads((c/'manifest.json').read_text());m['status']='FROZEN';m['source_manifest_sha256']=sha(c/'source_manifest.json');write(c/'manifest.json',m)
    state(c,'READY')

def command(c, method, seed, out, smoke=False):
    cmd=['bash',str(R/'run.sh'),'--config',str(c/'configs'/f'{method}.yaml'),
         '--journal','--methods',method,'--seed',str(seed),'--train-only',
         '--output',str(out),'--source-check-manifest',str(c/'source_manifest.json')]
    if smoke:cmd.append('--smoke')
    return cmd

def readiness(c):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',BASELINE_CHECKOUT=str(R),BASELINE_AUDIT=str(c/'readiness'),PYTHONPATH=str(R))
    with (c/'logs/cpu_reference.log').open('w') as f:
        ret=subprocess.run([str(PY),'tools/baseline_reference/reference_tests.py'],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode:raise RuntimeError('Pinned native reference comparison failed; see logs/cpu_reference.log')
    tests=['test_dad_reference_budget','test_compute_budget','test_fixed_budget_selection','test_primary_budget','test_baseline_readiness','test_location_bounded','test_baseline_eig_reporting','test_location_five_seed']
    with (c/'logs/cpu_tests.log').open('w') as f:
        ret=subprocess.run([str(PY),'-m','pytest','-q',*[f'tools/tests/{t}.py' for t in tests]],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
    if ret.returncode:raise RuntimeError('CPU regression failed; see logs/cpu_tests.log')
    write(c/'readiness/cpu_receipt.json',dict(status='PASS',reference_tests_sha256=sha(c/'readiness/baseline_reference_tests.json'),tests=tests))

def smoke(c):
    verify(c);locks=acquire()
    try:
        for method in ('dad','fixed','rl_sboed','vsoed'):
            out=c/'readiness'/f'{method}_gpu_smoke'
            if (out/'exit_code').exists() and (out/'exit_code').read_text().strip()=='0':continue
            if out.exists():raise RuntimeError('Preserve failed smoke; no automatic retry '+str(out))
            env=dict(os.environ,BOED_TRAINING_START_MONOTONIC=str(time.monotonic()),PYTHONUNBUFFERED='1')
            with (c/'logs'/f'{method}_gpu_smoke.log').open('w') as f:
                ret=subprocess.run(command(c,method,101,out,True),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT)
            if ret.returncode:raise RuntimeError('GPU smoke failed: '+method)
        write(c/'readiness/gpu_receipt.json',dict(status='PASS',methods=['dad','fixed','rl_sboed','vsoed']))
    finally:
        for f in locks:f.close()

def results(c):
    rows=[]
    for seed in SEEDS:
        for method in METHODS:
            d=c/'runs'/f'seed{seed}'/method;p=d/'models'/f'{method}_training.json'
            t=json.loads(p.read_text()) if p.exists() else {}
            e=c/'evaluations'/f'{method}_seed{seed}.json';e=json.loads(e.read_text()) if e.exists() else {}
            rows.append(dict(method=method,training_seed=seed,
                status='EVALUATED' if e else 'TRAINED' if t and (d/'exit_code').exists() and (d/'exit_code').read_text().strip()=='0' else 'FAILED_OR_INTERRUPTED' if d.exists() else 'PENDING',
                optimizer_updates=t.get('actual_updates',t.get('actor_optimizer_updates',t.get('gradient_updates',t.get('actual_policy_updates','')))),
                training_trajectories=t.get('generated_training_trajectories',''),
                charged_training_seconds=t.get('compute_budget',{}).get('charged_seconds',''),
                native_schedule_complete=t.get('native_schedule_completed',''),
                convergence_status=t.get('training_status',''),eig_mean=e.get('eig_mean',''),eig_mcse=e.get('eig_mcse',''),
                checkpoint=str(d/'models'/f'{method}.pt'),checkpoint_sha256=t.get('checkpoint_sha256','')))
    p=c/'results.csv';tmp=p.with_suffix('.tmp')
    with tmp.open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    tmp.replace(p);return rows

def wait_with_hard_training_cap(proc,out,entry,seconds=14400):
    """Backstop only: native trainers normally stop with serialization reserve."""
    ledger=out/'models/compute_clock.json'
    while proc.poll() is None:
        cost=time.monotonic()-entry
        if ledger.exists():
            try:
                s=json.loads(ledger.read_text())
                if 'stopped_at' in s:return proc.wait()
                end=time.monotonic();excluded=sum(max(0.,min(end,b)-max(s['start'],a)) for a,b,_ in s['excluded'])
                if s.get('active_exclusion') is not None:excluded+=max(0.,end-s['active_exclusion'][0])
                cost=s['credit_seconds']+end-s['start']-excluded
            except (OSError,ValueError,KeyError):pass
        if cost>=seconds-2.:
            write(out/'hard_cap_guard.json',dict(charged_seconds_at_signal=cost,pid=proc.pid,status='SAFETY_STOP_INCOMPLETE',signal='SIGTERM',budget_seconds=seconds))
            os.killpg(proc.pid,signal.SIGTERM)
            try:return proc.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL);return proc.wait()
        time.sleep(.25)
    return proc.returncode

def import_seed101(c):
    """Copy independently verified completed artifacts; do not mutate archives."""
    from src.research.benchmarks import make_benchmark
    from src.research.journal_checkpoints import load_checkpoint
    import torch
    import yaml
    archive=R.parent/'objective_power_grid/experiments/_legacy_archive/labpc'
    a=archive/'objective_power_grid_dad_budget_fix_20261021/experiments/10212026/10212026_dad_primary_budget_correction/location/seed101'
    b=archive/'objective_power_grid_matched_compute_20261020/experiments/10202026/10202026_icml_matched_compute_protocol/benchmarks/location_seed101_matched_compute/location/seed101'
    receipts=[]
    # RL is rerun: its14399.27s ledger leaves insufficient startup/finalization margin
    # to certify the new strict prelaunch four-hour accounting without qualification.
    for method in ('dad','fixed','vsoed','scorebed'):
        old=(a if method in ('dad','fixed') else b)/method
        out=c/'runs/seed101'/method
        if out.exists():
            receipt=json.loads((out/'reuse_receipt.json').read_text())
            assert sha(out/'models'/f'{method}.pt')==receipt['checkpoint_sha256']
            receipts.append(receipt);continue
        t=json.loads((old/'models'/f'{method}_training.json').read_text())
        checkpoint=old/'models'/f'{method}.pt'
        assert sha(checkpoint)==t['checkpoint_sha256']
        clock=t['compute_budget'];assert clock['hardware']=='NVIDIA GeForce RTX 4090'
        assert 0<clock['charged_seconds']<14400 and clock['budget_seconds']==14400
        assert clock['checkpoints'][-1]['eligible']
        assert t['checkpoint_selection']=='latest_budget_eligible_checkpoint'
        newraw=yaml.safe_load((c/'configs'/f'{method}.yaml').read_text())
        newsettings=newraw['research']['settings']|newraw['research']['journal_protocol']['method_settings'][method]
        oldsettings=t['effective_training_settings']
        keys={'dad':['batch_size','contrasts','learning_rate','spce_adam_beta1','spce_adam_beta2','spce_lr_decay','spce_lr_schedule_every','first_design_revision','spce_gradient_clip','spce_curriculum'],
              'fixed':['batch_size','contrasts','learning_rate','spce_adam_beta1','spce_adam_beta2','spce_lr_decay','spce_lr_schedule_every','spce_curriculum'],
              'vsoed':[], 'scorebed':['scorebed_batch_size','scorebed_restarts','scorebed_reference_sampler','score_schedule_reference_steps','policy_schedule_reference_steps','scorebed_pretraining_episodes']}[method]
        for key in keys:assert oldsettings[key]==newsettings[key],(method,key,oldsettings[key],newsettings[key])
        for name in ('engine.py','benchmarks.py','policies.py','dad_location_policy.py','upstream_vsoed.py'):
            assert (old/'source/src/research'/name).read_bytes()==(R/'src/research'/name).read_bytes(),name
        if method=='vsoed':
            assert t['profile']==newraw['research']['journal_protocol']['native_vsoed_profile']
        # The archived trainer differs only by the new refusal of retired MoE methods.
        oldtrainer=(old/'source/src/research/training.py').read_text()
        newtrainer=(R/'src/research/training.py').read_text()
        guard="    if kind not in {'dad','fixed'}:\n        raise ValueError('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: use the validated MoE-sBOED-v1 native trainer')\n"
        if method in ('dad','fixed'):assert oldtrainer==newtrainer.replace(guard,'')
        out.mkdir(parents=True);(out/'models').mkdir()
        source_hashes={}
        for file in (old/'models').iterdir():
            # Keep final native weights and provenance, not every redundant milestone.
            if file.name in (f'{method}.pt',f'{method}_training.json',f'{method}_effective_settings.json','compute_clock.json') or (method=='scorebed' and file.name in ('scorebed_official_policy.npz','scorebed_official_training.json','scorebed_official_config.json','scorebed_official_score.msgpack','scorebed_normalization.npz')):
                dest=out/'models'/file.name;shutil.copy2(file,dest)
                assert sha(file)==sha(dest);source_hashes[file.name]=sha(dest)
        shutil.copy2(old/'run_config.json',out/'run_config.json')
        shutil.copy2(old/'source_manifest.json',out/'origin_source_manifest.json')
        t['reused_from']=str(old);t['source_revalidation']='CPU native reference and pinned-vendor tests; architecture/physics/optimizer settings checked; final common scoring pending'
        t['native_schedule_completed']=method!='scorebed'
        t['training_status']='NATIVE_FINITE_SCHEDULE_COMPLETE' if method!='scorebed' else 'COMPUTE_LIMITED_NATIVE_IMPLEMENTATION'
        write(out/'models'/f'{method}_training.json',t)
        with torch.device('cpu'):
            p=load_checkpoint(out/'models'/f'{method}.pt',make_benchmark(newraw));assert all(torch.isfinite(v).all() for v in p.state_dict().values())
        (out/'exit_code').write_text('0\n')
        row=dict(method=method,seed=101,source=str(old),checkpoint_sha256=sha(checkpoint),verified_copy_hashes=source_hashes,charged_seconds=clock['charged_seconds'],decision='REUSE_TRAINING_AND_REEVALUATE_ON_NEW_COMMON_BANK')
        write(out/'reuse_receipt.json',row);receipts.append(row)
    receipts.append(dict(method='rl_sboed',seed=101,decision='RERUN',reason='Conservative strict prelaunch cost accounting and finalization reserve; preserve archived14399.27s run unchanged'))
    write(c/'reports/SEED101_REUSE.json',receipts);results(c)

def run(c):
    verify(c)
    for p in ('cpu_receipt.json','gpu_receipt.json'):
        if json.loads((c/'readiness'/p).read_text())['status']!='PASS':raise RuntimeError('Readiness failed')
    control=(c/'controller.lock').open('a+');fcntl.flock(control,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for seed in SEEDS:
        for method in METHODS:
            verify(c);out=c/'runs'/f'seed{seed}'/method
            if (out/'exit_code').exists() and (out/'exit_code').read_text().strip()=='0':continue
            if out.exists():state(c,'BLOCKED_PARTIAL_RUN',method=method,seed=seed,path=str(out));return
            while True:
                try:locks=acquire();break
                except BlockingIOError as e:state(c,'WAITING_GPU',method=method,seed=seed,reason=str(e));time.sleep(30)
            try:
                log=c/'logs'/f'{method}_seed{seed}.log';out.parent.mkdir(parents=True,exist_ok=True)
                entry=time.monotonic()
                env=dict(os.environ,BOED_TRAINING_START_MONOTONIC=str(entry),PYTHONUNBUFFERED='1')
                with log.open('w') as f:
                    proc=subprocess.Popen(command(c,method,seed,out),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    state(c,'RUNNING',method=method,seed=seed,worker_pid=proc.pid,log=str(log),output=str(out))
                    code=wait_with_hard_training_cap(proc,out,entry)
                results(c)
                if code:
                    # Isolate a failed method, preserve evidence, continue other methods/seeds.
                    write(out/'controller_failure.json',dict(exit_code=code,requires_review=True,no_automatic_restart=True))
                    state(c,'METHOD_FAILED',method=method,seed=seed,exit_code=code)
            finally:
                for f in locks:f.close()
    rows=results(c)
    write(c/'training_freeze.json',dict(rows=rows,source_manifest_sha256=sha(c/'source_manifest.json'),all_training_finished=True))
    state(c,'TRAINING_FINISHED_STARTING_EVALUATION')
    locks=acquire()
    try:evaluate(c)
    finally:
        for f in locks:f.close()
    state(c,'FINISHED_WITH_LIMITATIONS' if any(r['status']=='FAILED_OR_INTERRUPTED' for r in rows) else 'COMPLETE')

def evaluate(c):
    import numpy as np, torch
    import yaml
    from src.research.benchmarks import make_benchmark
    from src.research.journal_validation import CommonValidation
    from src.research.journal_checkpoints import load_checkpoint
    if not (c/'training_freeze.json').exists():raise RuntimeError('Freeze candidates before opening final draws')
    verify(c);raw=yaml.safe_load((c/'configs/dad.yaml').read_text());torch.set_num_threads(4)
    validator=CommonValidation(raw,30,FINAL_SEED,2000,1000000,16,16384,device='cuda:0')
    write(c/'evaluations/paired_bank_identity.json',dict(seed=FINAL_SEED,sha256=validator.identity,systems=2000,contrasts=1000000))
    np.savez(c/'evaluations/final_cases.npz',truth=validator.truth.numpy(),noise=validator.noise.numpy())
    np.save(c/'evaluations/final_contrasts.npy',validator.bank.numpy())
    candidates=[('random',0,None)]+[(m,s,c/'runs'/f'seed{s}'/m/'models'/f'{m}.pt') for s in SEEDS for m in METHODS]
    for m,s,p in candidates:
        target=c/'evaluations'/f'{m}_seed{s}.json'
        if target.exists():continue
        if p is not None:
            d=p.parent.parent
            if not (d/'exit_code').exists() or (d/'exit_code').read_text().strip()!='0':continue
        state(c,'EVALUATING',method=m,seed=s)
        policy=None if p is None else load_checkpoint(p,make_benchmark(raw)).cpu().eval()
        # CommonValidation accepts a policy; Random uses the same rollout/scorer directly.
        if policy is None:
            from src.research.engine import rollout
            from src.research.location_evaluation import contrast_scores
            scores=[];start=time.monotonic()
            with torch.device('cuda:0'),torch.no_grad():
                b=make_benchmark(raw)
                for a in range(0,2000,16):
                    n=min(16,2000-a)
                    r=rollout(b,None,30,n,FINAL_SEED+a*97451,0,truth=validator.truth[a:a+n].cuda(),noise=validator.noise[a:a+n].cuda(),method='random')
                    x,_=contrast_scores(b,r,validator.bank,16384);scores.extend(x[:,-1].tolist())
            scores=np.array(scores);cost=dict(seconds=time.monotonic()-start)
        else:scores,cost=validator(policy)
        np.save(target.with_suffix('.npy'),scores)
        se=float(scores.std(ddof=1)/math.sqrt(len(scores)));mean=float(scores.mean())
        write(target,dict(method=m,seed=s,eig_mean=mean,eig_mcse=se,ci95=[mean-1.96*se,mean+1.96*se],paired_bank_sha256=validator.identity,checkpoint_sha256=None if p is None else sha(p),cost=cost))
        results(c)
    summary=[]
    for m in METHODS:
        paths=[c/'evaluations'/f'{m}_seed{s}.json' for s in SEEDS]
        vals=[json.loads(p.read_text())['eig_mean'] for p in paths if p.exists()]
        summary.append(dict(method=m,completed_seeds=len(vals),mean_over_training_seeds=float(np.mean(vals)) if vals else None,between_training_seed_sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None,between_training_seed_mean_se=float(np.std(vals,ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else None))
    paired=[]
    for s in SEEDS:
        for m in METHODS:
            a=c/'evaluations'/f'{m}_seed{s}.npy'
            if not a.exists():continue
            for ref in ('random','dad','rl_sboed','vsoed'):
                b=c/'evaluations'/f'{ref}_seed{0 if ref=="random" else s}.npy'
                if not b.exists() or ref==m:continue
                d=np.load(a)-np.load(b);se=float(d.std(ddof=1)/math.sqrt(len(d)));mean=float(d.mean())
                paired.append(dict(method=m,reference=ref,training_seed=s,mean_difference=mean,paired_mcse=se,ci95=[mean-1.96*se,mean+1.96*se]))
    write(c/'evaluations/five_seed_summary.json',summary);write(c/'evaluations/paired_comparisons.json',paired)
    (c/'reports/RESULTS.md').write_text('# Measured results\n\n'+json.dumps(summary,indent=2)+'\n\nMCSE and between-training-seed uncertainty are distinct. Resource exhaustion and native completion do not prove convergence.\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--location-baselines-five-seed',action='store_true')
    p.add_argument('--campaign',required=True);p.add_argument('--action',choices=['prepare','freeze','readiness','smoke','run','evaluate','import_seed101'],required=True)
    p.add_argument('--seconds',type=int,default=14400);a=p.parse_args();c=Path(a.campaign).resolve()
    if a.action=='prepare':prepare(c,a.seconds)
    else:globals()[a.action](c)

if __name__=='__main__':main()
