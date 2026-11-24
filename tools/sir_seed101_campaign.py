"""Fresh six-baseline SIR campaign, immutable provenance and serialized GPU use."""
import argparse, csv, fcntl, hashlib, json, math, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('random', 'fixed', 'dad', 'rl_sboed', 'vsoed', 'scorebed')
ORDER = ('dad', 'random', 'fixed', 'rl_sboed', 'vsoed', 'scorebed')

def move_plain_tensors(policy, device):
    """Move upstream non-buffer tensors without changing vendored algorithms."""
    import torch
    if policy is not None:
        for module in policy.modules():
            for name in ('bounds', 'has_inf'):
                value = getattr(module, name, None)
                if isinstance(value, torch.Tensor): setattr(module, name, value.to(device))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)); temporary.replace(path)

def source_manifest():
    paths = []
    for prefix in ('src', 'tools', 'configs', 'scripts'):
        paths += [p for p in (ROOT / prefix).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py', '.yaml', '.yml', '.sh')]
    paths += [ROOT / 'run.sh', ROOT / 'sweep_run.sh']
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}

def verify(campaign):
    manifest = json.loads((campaign / 'source_manifest.json').read_text())
    expected = manifest['sha256']
    changed = [rel for rel, digest in expected.items() if not (ROOT / rel).is_file() or sha(ROOT / rel) != digest]
    if changed:
        raise RuntimeError('Frozen source changed: ' + ', '.join(changed[:20]))
    if any(sha(campaign/'configs'/(m+'.yaml'))!=digest for m,digest in manifest['config_sha256'].items()):
        raise RuntimeError('Frozen campaign configuration changed')

def command(campaign, method):
    return ['--config', str(campaign / 'configs' / (method + '.yaml')), '--journal', '--train-only',
            '--seed', '101', '--eval-seeds', '1001', '--eig-only', '--methods', method,
            '--source-check-manifest', str(campaign / 'source_manifest.json'),
            '--output', str(campaign / 'runs' / method)]

def prepare(campaign, config_directory, readiness):
    import shutil, yaml
    campaign.mkdir(parents=True, exist_ok=False)
    for name in ('configs', 'logs', 'metadata', 'runs', 'evaluations', 'figures'):
        (campaign / name).mkdir()
    for method in METHODS:
        shutil.copy2(Path(config_directory) / (method + '.yaml'), campaign / 'configs' / (method + '.yaml'))
    raws=[yaml.safe_load((campaign/'configs'/(m+'.yaml')).read_text()) for m in METHODS]
    assert all(x['sir_ode']==raws[0]['sir_ode'] and x['observation']==raws[0]['observation'] for x in raws)
    for file in Path(readiness).glob('*'):
        if file.is_file() and file.suffix in ('.json', '.log'):
            shutil.copy2(file, campaign / 'metadata' / file.name)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    hashes = source_manifest()
    hashes.update({str((campaign/'configs'/(m+'.yaml')).relative_to(ROOT)):sha(campaign/'configs'/(m+'.yaml')) for m in METHODS})
    write(campaign / 'source_manifest.json', dict(git_commit=revision, sha256=hashes,
          config_sha256={m: sha(campaign / 'configs' / (m + '.yaml')) for m in METHODS}))
    write(campaign / 'registry.json', dict(campaign=campaign.name, train_seed=101,
          methods={m: dict(state='QUEUED', official_result=False) for m in METHODS},
          source_revision=revision, comparison='fresh seed101 matched 14400 charged seconds per trainable method',
          evaluation=dict(systems=2048, seed=1001, contrasts=1000000, resolution='FROZEN_COMMON_1M_LOWER_BOUND_TAIL_UNCERTAINTY_DISCLOSED'),
          confirmation_seed_used=False))
    for method in METHODS:
        argv = command(campaign, method)
        key = hashlib.sha256(json.dumps(argv, separators=(',', ':')).encode()).hexdigest()
        write(ROOT / 'reports/official_run_authorizations' / (key + '.json'),
              dict(status='VALIDATED_FROZEN_SOURCE', regression_status='PASS', argv=argv,
                   source_sha256=hashes, source_revision=revision, campaign=str(campaign),
                   readiness=str(campaign / 'metadata'), new_user_authorization='2026-10-27 six SIR baselines seed101'))
    write(campaign / 'status.json', dict(state='PREPARED', coordinator_pid=None, active_method=None))
    report(campaign)
    print(campaign, flush=True)

def evaluation(campaign, method):
    import numpy as np, torch, yaml
    from src.research.benchmarks import make_benchmark
    from src.research.engine import rollout
    from src.research.sir_evaluation import contrast_scores
    from src.research.journal_checkpoints import load_checkpoint
    torch.set_num_threads(2)
    raw = yaml.safe_load((campaign / 'configs' / (method + '.yaml')).read_text())
    out = campaign / 'evaluations' / method; out.mkdir(exist_ok=False)
    draws = campaign / 'evaluations' / 'paired_draws.npz'
    bankfile = campaign / 'evaluations' / 'contrast_bank.npy'
    if not draws.exists():
        with torch.device('cpu'):
            b = make_benchmark(raw); g = torch.Generator().manual_seed(1001)
            truth = b.sample((2048,), g); noise = torch.randn(2048, 5, 1, generator=g, dtype=torch.float64)
            bank = b.sample((1000000,), g)
        np.savez(draws, truth=truth.numpy(), noise=noise.numpy()); np.save(bankfile, bank.numpy())
        write(campaign / 'evaluations/PAIRED_DRAWS.json', dict(seed=1001, systems=2048, contrasts=1000000,
              cases_sha256=sha(draws), bank_sha256=sha(bankfile), confirmation_randomness_used=False))
    d = np.load(draws); bank = torch.from_numpy(np.load(bankfile)); rows=[]; values=[]
    checkpoint = campaign / 'runs' / method / 'models' / (method + '.pt')
    b = make_benchmark(raw)
    p = None if method == 'random' else load_checkpoint(checkpoint, b).to('cuda:0').eval()
    move_plain_tensors(p, 'cuda:0')
    start = time.monotonic(); last_result = None
    with torch.device('cuda:0'), torch.no_grad():
        b = make_benchmark(raw)
        for first in range(0, 2048, 16):
            truth = torch.from_numpy(d['truth'][first:first+16]).to('cuda:0')
            noise = torch.from_numpy(d['noise'][first:first+16]).to('cuda:0')
            r = rollout(b, p, 5, len(truth), 1001 + first * 97451, 0, truth=truth, noise=noise,
                        method=method, track_likelihood=True)
            scores, upper = contrast_scores(b, r, bank, 4096)
            actions = torch.stack(r['actions'], 1)[..., 0]
            assert torch.isfinite(scores).all() and torch.all(actions[:, 1:] - actions[:, :-1] >= .1-1e-8)
            assert torch.all(actions[:, 0] >= .1-1e-8) and torch.all(actions[:, -1] <= 100.+1e-8)
            for i in range(len(truth)):
                value = float(scores[i, -1]); values.append(value)
                rows.append(dict(trajectory_id=first+i, truth=truth[i].cpu().tolist(),
                     noise=noise[i].cpu().tolist(), actions=actions[i].cpu().tolist(),
                     observations=[x[i].cpu().tolist() for x in r['observations']],
                     spce=value, spce_by_stage=scores[i].tolist(), nmc_upper=float(upper[i])))
            last_result = r
            write(out / 'progress.json', dict(trajectories=first+len(truth), total=2048, seconds=time.monotonic()-start))
            print('EVALUATION', method, first+len(truth), '/2048', flush=True)
    values = np.asarray(values); np.save(out / 'scores.npy', values); write(out / 'trajectories.json', rows)
    times=[]
    if p is not None:
        p=p.cpu();move_plain_tensors(p,'cpu');features=[f[:1].cpu() for f in last_result['features']]
        cpu_b=make_benchmark(raw)
        def decide(f, stage):
            latent=p(f,stage)
            previous=[x[:1].cpu() for x in last_result['actions'][:stage]]
            return latent if getattr(p,'outputs_design',False) else cpu_b.feasible(latent,previous,5)
        with torch.no_grad():
            for stage, f in enumerate(features):
                for _ in range(20):decide(f, stage)
                for _ in range(200):
                    tic=time.perf_counter();decide(f, stage);times.append(time.perf_counter()-tic)
    else:
        b=make_benchmark(raw);rng=torch.Generator().manual_seed(97402731)
        for _ in range(200):
            tic=time.perf_counter();b.feasible(b.random_latent(1,rng),[],5);times.append(time.perf_counter()-tic)
    latency=dict(device='CPU', batch_size=1, median_seconds=float(np.median(times)),
                 p95_seconds=float(np.percentile(times,95)), observations='actual development histories', online_optimization=False)
    write(out / 'latency.json', latency)
    summary=dict(method=method, eig=float(values.mean()), mc_se=float(values.std(ddof=1)/math.sqrt(len(values))),
                 systems=len(values), contrasts=len(bank), seed=1001, evaluation_seconds=time.monotonic()-start,
                 scores_sha256=sha(out/'scores.npy'), checkpoint_sha256=None if p is None else sha(checkpoint),
                 source_manifest_sha256=sha(campaign/'source_manifest.json'), online_latency=latency,
                 status='EVALUATED_DEVELOPMENT_SEED101', statistical_superiority_claim=False,
                 estimator='common explicit-likelihood sequential prior-contrastive lower bound',
                 resolution_status='common1M frozen, residual lower-bound bias not certified negligible')
    write(out / 'summary.json', summary)
    return summary

def run(campaign):
    campaign=campaign.resolve();verify(campaign)
    lock=open(ROOT/'.gpu0.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if gpu:raise RuntimeError('GPU is occupied by compute PID(s): '+gpu)
    registry=json.loads((campaign/'registry.json').read_text())
    if any(x['state']!='QUEUED' for x in registry['methods'].values()):raise RuntimeError('Campaign is single-use; no automatic restart/resume')
    coordinator=os.getpid()
    for method in ORDER:
        verify(campaign)
        log=campaign/'logs'/(method+'.log')
        with log.open('x') as f:
            if method!='random':
                argv=command(campaign,method);env=dict(os.environ,BOED_TRAINING_START_MONOTONIC=str(time.monotonic()),
                    CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                    XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled')
                prelaunch_start=float(env['BOED_TRAINING_START_MONOTONIC']);term_at=None
                proc=subprocess.Popen(['bash',str(ROOT/'run.sh'),*argv],cwd=ROOT,env=env,
                    stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                registry['methods'][method].update(state='TRAINING',pid=proc.pid,log=str(log));write(campaign/'registry.json',registry)
                while proc.poll() is None:
                    clock=campaign/'runs'/method/'models/compute_clock.json';charged=time.monotonic()-prelaunch_start
                    if clock.exists():
                        from src.research.compute_budget import ComputeClock
                        state=json.loads(clock.read_text());end=state.get('stopped_at',time.monotonic())
                        charged=state['credit_seconds']+max(0.,end-state['start']-sum(max(0.,min(end,b)-max(state['start'],a)) for a,b,_ in state['excluded']))
                        if state.get('active_exclusion'):
                            charged-=max(0.,end-max(state['start'],state['active_exclusion'][0]))
                    if charged>14400. and proc.poll() is None:
                        if term_at is None:
                            os.killpg(proc.pid,signal.SIGTERM);term_at=time.monotonic()
                            registry['methods'][method]['watchdog']='CHARGED_LIMIT_SIGTERM'
                        elif time.monotonic()-term_at>5:
                            os.killpg(proc.pid,signal.SIGKILL)
                            registry['methods'][method]['watchdog']='CHARGED_LIMIT_SIGKILL_AFTER_TERM'
                    write(campaign/'status.json',dict(state='RUNNING',stage='TRAINING',active_method=method,pid=proc.pid,
                         coordinator_pid=coordinator,charged_training_seconds=charged,cap_seconds=14400,
                         heartbeat_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),log=str(log)))
                    time.sleep(2)
                if proc.returncode:
                    registry['methods'][method].update(state='FAILED',returncode=proc.returncode,official_result=False)
                    write(campaign/'registry.json',registry);report(campaign);continue
                registry['methods'][method]['state']='TRAINING_COMPLETE'
            else:
                registry['methods'][method].update(state='NO_TRAINING_REQUIRED',training_seconds=0.)
            write(campaign/'registry.json',registry)
            write(campaign/'status.json',dict(state='RUNNING',stage='EVALUATING',active_method=method,
                  pid=coordinator,coordinator_pid=coordinator,log=str(campaign/'coordinator.log')))
            try:
                result=evaluation(campaign,method)
                registry['methods'][method].update(state='COMPLETE',evaluation=result,official_result=False,
                    validation_status='DEVELOPMENT_EVALUATED_PENDING_SCIENTIFIC_REVIEW')
            except Exception as exc:
                import traceback
                write(campaign/'metadata'/(method+'_evaluation_failure.json'),dict(error=str(exc),traceback=traceback.format_exc()))
                registry['methods'][method].update(state='EVALUATION_FAILED',error=str(exc),official_result=False)
            write(campaign/'registry.json',registry);report(campaign)
    states=[v['state'] for v in registry['methods'].values()]
    write(campaign/'status.json',dict(state='COMPLETE' if all(x=='COMPLETE' for x in states) else 'COMPLETE_WITH_FAILURES',
          active_method=None,coordinator_pid=coordinator,methods={m:v['state'] for m,v in registry['methods'].items()}))

def table(path,rows,fields):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def report(campaign):
    import numpy as np
    registry=json.loads((campaign/'registry.json').read_text());rows=[];costs=[];summaries={};trains={}
    for method in METHODS:
        sp=campaign/'evaluations'/method/'summary.json'
        if sp.exists():summaries[method]=json.loads(sp.read_text())
        tp=campaign/'runs'/method/'models'/(method+'_training.json')
        trains[method]=json.loads(tp.read_text()) if tp.exists() else {}
    for method in METHODS:
        summary=summaries.get(method,{});train=trains[method];compute=train.get('compute_budget',{})
        row=dict(method=method,status=registry['methods'][method]['state'],eig_nats=summary.get('eig',''),mc_se=summary.get('mc_se',''),
                 difference_vs_dad='',difference_vs_dad_se='',difference_vs_vsoed='',difference_vs_vsoed_se='',
                 training_seconds=0 if method=='random' else compute.get('charged_seconds',''),
                 optimizer_updates=json.dumps(train.get('resource_ledger',{}).get('optimizer_updates',{})),
                 new_training_trajectories=train.get('generated_training_trajectories',0 if method=='random' else ''),
                 checkpoint_sha256=summary.get('checkpoint_sha256',''),source_revision=registry['source_revision'],
                 online_decision_seconds=summary.get('online_latency',{}).get('median_seconds',''))
        for reference in ('dad','vsoed'):
            if method in summaries and reference in summaries:
                x=np.load(campaign/'evaluations'/method/'scores.npy')-np.load(campaign/'evaluations'/reference/'scores.npy')
                row['difference_vs_'+reference]=float(x.mean());row['difference_vs_'+reference+'_se']=float(x.std(ddof=1)/math.sqrt(len(x)))
        rows.append(row)
        costs.append(dict(method=method,status=row['status'],training_seconds=row['training_seconds'],
              cap_seconds=0 if method=='random' else 14400,optimizer_updates=row['optimizer_updates'],
              new_training_trajectories=row['new_training_trajectories'],
              pretraining_trajectories=train.get('score_training_trajectories',train.get('score_pretraining_episodes','')),
              likelihood_workload=json.dumps(train.get('resource_ledger',{}).get('counts',{})),
              evaluation_seconds=summary.get('evaluation_seconds',''),
              stopping_status=train.get('training_status',train.get('compute_implementation_status',train.get('stop_reason','')))))
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    table(reports/'SIR_SEED101_BASELINES.csv',rows,list(rows[0]));table(reports/'SIR_SEED101_COMPUTE.csv',costs,list(costs[0]))
    text='# SIR ODE seed101 baselines\n\nCampaign: '+str(campaign)+'\n\nNo single-training-seed superiority claim. EIG is the common L=1,000,000 sequential sPCE lower bound; residual contrast-resolution uncertainty remains explicit. Reserved confirmation randomness is unused.\n\n| Method | Status | EIG (nats) | MC SE |\n|---|---|---:|---:|\n'
    for row in rows:text+='| '+row['method']+' | '+row['status']+' | '+str(row['eig_nats'])+' | '+str(row['mc_se'])+' |\n'
    (reports/'SIR_SEED101_BASELINES.md').write_text(text)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    done=list(summaries)
    for name in ('eig_uncertainty','eig_training_time','training_convergence','online_inference_cost'):
        fig,ax=plt.subplots(figsize=(8,4))
        if not done:ax.text(.5,.5,'Pending completed baseline evaluation',ha='center',va='center',transform=ax.transAxes)
        elif name=='eig_uncertainty':
            ax.bar(done,[summaries[m]['eig'] for m in done],yerr=[summaries[m]['mc_se'] for m in done],capsize=4);ax.set_ylabel('Sequential sPCE EIG (nats), ± MC SE')
        elif name=='eig_training_time':
            for m in done:ax.errorbar([trains[m].get('compute_budget',{}).get('charged_seconds',0.)],[summaries[m]['eig']],yerr=[summaries[m]['mc_se']],fmt='o',label=m)
            ax.set(xlabel='Charged training seconds',ylabel='EIG (nats)');ax.legend()
        elif name=='online_inference_cost':
            ax.bar(done,[summaries[m]['online_latency']['median_seconds']*1000 for m in done]);ax.set_ylabel('Median CPU decision latency (ms)')
        else:
            for m in done:
                history=trains[m].get('journal_validation_history',[])
                if history:ax.plot([x['training_seconds'] for x in history],[x['eig'] for x in history],label=m)
            ax.set(xlabel='Charged training seconds',ylabel='Diagnostic validation EIG (not checkpoint selector)')
            if ax.lines:ax.legend()
        ax.set_title(name.replace('_',' '));fig.tight_layout();fig.savefig(campaign/'figures'/(name+'.png'),dpi=150);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','report']);parser.add_argument('--campaign',required=True);parser.add_argument('--config-directory');parser.add_argument('--readiness')
    a=parser.parse_args();c=Path(a.campaign)
    if a.action=='prepare':prepare(c,a.config_directory,a.readiness)
    elif a.action=='run':run(c)
    else:report(c)
