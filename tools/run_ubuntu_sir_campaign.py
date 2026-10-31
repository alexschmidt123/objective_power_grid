"""Sequential, guarded Ubuntu SIR T=5 seed-101 development campaign."""
import csv,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

R=Path(__file__).resolve().parents[1]
C=R/'experiments/10162026/10162026_ubuntu_sir_t5_seed101'
FAMILY={'moe','resmoe','switchmoe','switchresmoe'}
# The base is an internal phase, never an eleventh main-table method.
TASKS=['random','base','resmoe','switchresmoe','moe','switchmoe','fixed','dad','rl_sboed','vsoed','scorebed']
START=time.monotonic()
MAX_WALL=72*3600

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def status(state,**kw):write(C/'status.json',dict(state=state,utc=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),**kw))
def guard():
    for rel,digest in read(C/'source_manifest.json')['sha256'].items():
        if not (R/rel).is_file() or sha(R/rel)!=digest:raise RuntimeError('Frozen code/config changed: '+rel)
    if time.monotonic()-START>MAX_WALL:raise RuntimeError('72h campaign safety cap reached')
    if __import__('shutil').disk_usage(R).free<20*1024**3:raise RuntimeError('20GiB free-disk safety floor')
def command(method,out,smoke=False):
    native='sum_single' if method=='base' else method
    config=C/('sir_500k.yaml' if method in {'base','resmoe','switchresmoe'} else 'sir_1m.yaml')
    cmd=['bash','run.sh','--config',str(config),'--journal','--seed','101','--eval-seeds','1001','--eig-only','--output',str(out)]
    cmd+=['--variant',native] if native in FAMILY else ['--methods',native]
    if method in {'resmoe','switchresmoe'}:
        base=C/('readiness/native_remaining/models/sum_single.pt' if smoke else 'runs/base_seed101/models/sum_single.pt')
        cmd+=['--base-checkpoint',str(base)]
    if smoke:cmd+=['--smoke']
    else:cmd+=['--source-check-manifest',str(C/'source_manifest.json')]
    return cmd
def execute(cmd,log,timeout):
    with Path(log).open('w') as f:
        p=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:rc=p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise RuntimeError('Wall safety cap exceeded; partial output preserved')
    if rc:raise RuntimeError('Root launcher failed: '+str(rc)+'; see '+str(log))
def wait_free():
    clear=0;t=time.monotonic()
    while clear<2:
        guard()
        jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
        clear=clear+1 if not jobs else 0
        if clear<2:
            status('waiting_for_free_GPU',gpu_jobs=jobs)
            if time.monotonic()-t>3600:raise RuntimeError('Foreign GPU contention exceeded 1h; no full job started')
            time.sleep(5)
def freeze():
    paths=[*R.glob('src/**/*.py'),*R.glob('tools/**/*.py'),*R.glob('configs/*.yaml'),R/'run.sh',C/'sir_1m.yaml',C/'sir_500k.yaml',C/'journal_protocol.md']
    write(C/'source_manifest.json',dict(sha256={str(p.relative_to(R)):sha(p) for p in paths},
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
        git_dirty=subprocess.check_output(['git','status','--short'],cwd=R,text=True)))
def readiness():
    remaining=C/'readiness/native_remaining'
    deadline=time.monotonic()+1800
    while not (remaining/'exit_code').exists():
        if time.monotonic()>deadline:raise RuntimeError('Native smoke deadline exceeded')
        status('readiness_native_smoke',full_training_started=False)
        time.sleep(10)
    if (remaining/'exit_code').read_text().strip()!='0':raise RuntimeError('Native smoke failed; '+str(remaining/'failure.json'))
    first=C/'readiness/native_smoke'
    for native in ['random','fixed','dad','rl_sboed']:
        assert len(read(first/'eval'/f'{native}_seed1001.json'))==2
    for native in ['vsoed','scorebed','moe','switchmoe','sum_single']:
        assert len(read(remaining/'eval'/f'{native}_seed1001.json'))==2
    assert read(R/'experiments/10162026/10162026_ubuntu_sir_execution/validation_equivalence.json')['status']=='passed'
    assert '35 passed' in Path('/tmp/objective_sir_fused_tests.log').read_text()
    jax_check=R/'experiments/10162026/10162026_ubuntu_sir_execution/jax_gpu_equivalence.json'
    while not jax_check.exists():
        if time.monotonic()>deadline:raise RuntimeError('ScoreBED GPU numerical check deadline exceeded')
        status('readiness_scorebed_GPU_check',full_training_started=False);time.sleep(5)
    assert read(jax_check)['status']=='passed'
    for method in ['resmoe','switchresmoe']:
        out=C/'readiness'/f'{method}_smoke'
        status('readiness_residual_smoke',method=method,full_training_started=False)
        cmd=command(method,out,smoke=True)
        write(C/'readiness'/f'{method}_command.json',cmd)
        execute(cmd,C/'readiness'/f'{method}_launcher.log',900)
        assert (out/'exit_code').read_text().strip()=='0'
    write(C/'readiness_complete.json',dict(status='passed',methods=10,
        finite_difference_and_reference_tests=35,reserved_final_seed_used=False,
        development_smokes_not_full_results=True))
def report(rows):
    planned=rows+[dict(method=m,seed=101,status='pending') for m in TASKS if m not in {x['method'] for x in rows}]
    write(C/'registry.json',planned)
    fields=['method','seed','status','eig','mc_se','additional_primary_episodes','total_primary_episodes','training_seconds','base_training_seconds','validation_seconds','run','error']
    with (C/'sir_t5_results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        w.writerows(x for x in planned if x['method']!='base')
    with (C/'phase_costs.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(planned)
def run(method):
    import numpy as np
    native='sum_single' if method=='base' else method
    cap=500000 if method in {'base','resmoe','switchresmoe'} else 1000000
    out=C/'runs'/f'{method}_seed101';marker=C/'completion'/f'{method}.json'
    if out.exists() or marker.exists():raise RuntimeError('Existing attempt preserved; no automatic overwrite/retry')
    guard();wait_free()
    base=C/'runs/base_seed101/models/sum_single.pt'
    base_hash=sha(base) if method in {'resmoe','switchresmoe'} else None
    cmd=command(method,out)
    write(C/'commands'/f'{method}.json',dict(command=cmd,cwd=str(R),seed=101))
    status('running',method=method,seed=101,run=str(out),full_training_started=True)
    entry=dict(method=method,seed=101,status='running',run=str(out));started=time.monotonic()
    try:
        execute(cmd,C/'logs'/f'{method}.log',min(90000,MAX_WALL-(time.monotonic()-START)))
        guard();tr=read(out/'models'/f'{native}_training.json')
        actual=tr['generated_training_trajectories']
        if method!='random':
            assert actual==cap and tr['selected_checkpoint']['episodes']==cap
            assert tr['checkpoint_selection']=='final_predeclared_fixed_budget'
        scores=np.asarray([r['spce'] for r in read(out/'eval'/f'{native}_seed1001.json')])
        assert len(scores)==2048 and np.isfinite(scores).all()
        cost=dict(additional_primary_episodes=actual,total_primary_episodes=actual,
            training_seconds=tr['training_seconds'],validation_seconds=tr.get('validation_seconds',0),
            contrast_training_trajectories=tr.get('contrast_training_trajectories'),
            forward_particle_stages=tr.get('forward_particle_stages'),base_training_seconds=0)
        if base_hash is not None:
            import torch
            assert sha(base)==base_hash
            checkpoint=torch.load(out/'models'/f'{native}.pt',map_location='cpu',weights_only=False)
            original=torch.load(base,map_location='cpu',weights_only=False)
            for key,value in original['state_dict'].items():
                assert torch.equal(value,checkpoint['state_dict']['core.base.policy.'+key]),key
            assert tr['total_training_episodes']==1000000
            assert checkpoint['base_provenance']['sha256']==base_hash
            cost.update(total_primary_episodes=1000000,base_checkpoint_sha256=base_hash,
                        base_training_seconds=tr['base_training_seconds'],frozen_base_verified=True)
        entry.update(status='complete',eig=float(scores.mean()),mc_se=float(scores.std(ddof=1)/len(scores)**.5),
                     evaluation_systems=2048,evaluation_contrasts=16384,evaluation_seed=1001,
                     development_only=True,**cost)
    except Exception as exc:
        entry.update(status='failed',error=str(exc),traceback=traceback.format_exc())
    entry['wall_seconds']=time.monotonic()-started
    write(marker,entry);return entry
def main():
    lock=(C/'campaign.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sub in ['runs','completion','logs','commands']:(C/sub).mkdir(exist_ok=True)
    readiness();freeze();rows=[];report(rows)
    for method in TASKS:
        guard()
        if method in {'resmoe','switchresmoe'} and not any(x['method']=='base' and x['status']=='complete' for x in rows):
            rows.append(dict(method=method,seed=101,status='blocked',error='Compatible base phase failed'))
        else:rows.append(run(method))
        report(rows)
    status('complete' if all(x['status']=='complete' for x in rows) else 'finished_with_failures',
           completed_main_methods=sum(x['status']=='complete' and x['method']!='base' for x in rows),
           final_test_seed_used=False,development_only=True)
if __name__=='__main__':
    try:main()
    except Exception as exc:status('stopped_requires_review',error=str(exc),traceback=traceback.format_exc());raise
