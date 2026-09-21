"""Bounded readiness checks; complete smoke/timing experiments use root run.sh."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
METHODS=['fixed','dad','rl_sboed','step_dad','myopic','random','moe_sboed']

def check_smoke(run):
    import numpy as np
    import torch
    from src.objectives.eig.continuous_eig import DurationPolicy
    from src.objectives.eig.continuous_redq import REDQPolicy
    from src.policies.direct_moe import DirectMoEPolicy
    done=json.loads((run/'completion.json').read_text())
    cfg=json.loads((run/'run_config.json').read_text())
    rows=json.loads((run/'rollouts.json').read_text())
    assert done['smoke_only'] and set(done['methods'])==set(METHODS)
    assert (run/'exit_code').read_text().strip()=='0'
    assert cfg['physical_config']['system']['name']=='ieee14'
    assert cfg['observation_kind']=='endpoint_rocof' and cfg['settings']['window']==3.5
    assert len(cfg['prior_lower'])==10 and not cfg['uses_probe_bank']
    assert len(rows)==14
    identities={}
    for row in rows:
        key=(row['evaluation_seed'],row['system'])
        if key in identities:assert identities[key]==row['true_MK']
        identities[key]=row['true_MK']
        assert len(row['true_MK'])==len(row['true_terminal_state'])==10
        assert np.isfinite(row['terminal_spce_nats'])
        ds=np.array(row['duration_sequence_s'])
        assert len(ds)==3 and np.all((ds>=.2)&(ds<=3)) and np.all(np.diff(ds)>=.01)
        if row['method']=='moe_sboed':
            assert len(row['routing_trace'])==3
            for trace in row['routing_trace']:
                w=np.array(trace['weights'])
                assert w.shape==(1,4) and np.all(w>=0) and np.allclose(w.sum(-1),1)
    for method in ('fixed','dad','rl_sboed','moe_sboed'):
        ck=torch.load(run/'models'/f'{method}.pth',map_location='cpu',weights_only=False)
        cls=DirectMoEPolicy(3,1) if method=='moe_sboed' else (REDQPolicy(3,1) if method=='rl_sboed' else DurationPolicy(3,1,fixed=method=='fixed'))
        cls.load_state_dict(ck['state_dict'],strict=True)
    return {'methods':METHODS,'finite_paired_records':len(rows),'checkpoint_loads':4,'passed':True}

def sensitivity(output):
    import numpy as np
    import torch
    from src.config import load_config
    from src.domains.swing.continuous_rocof import EndpointRocofObserver
    from src.domains.swing.continuous import ContinuousParticleBelief
    from src.objectives.eig.continuous_eig import OnlineEIG,DurationPolicy
    cfg=load_config(ROOT/'configs/ieee14_mocu.yaml')
    obs=EndpointRocofObserver(cfg,duration_bounds=(.2,3),injection_bus=1,amplitude=.05,window=3.5)
    lo=np.r_[cfg.swing['M_lower_nodes'],cfg.swing['K_lower_nodes']]
    hi=np.r_[cfg.swing['M_upper_nodes'],cfg.swing['K_upper_nodes']]
    rng=np.random.default_rng(940101);truth=rng.uniform(lo,hi,(16,1,10))
    alternatives=rng.uniform(lo,hi,(16,512,10));noise=rng.normal(size=(16,3,1))
    p=DurationPolicy(3,1,fixed=True)
    durations=[.3,.7,1.2];past=.2;logits=[]
    for k,d in enumerate(durations):
        lower=.2 if k==0 else durations[k-1]+.01+1e-10
        upper=3-(2-k)*(.01+1e-10)
        u=(d-lower)/(upper-lower);logits.append(float(np.log(u/(1-u))))
    with torch.no_grad():p.sequence.copy_(torch.tensor(logits))
    records=[];history=None
    for L in (128,256,512):
        engine=OnlineEIG(cfg,obs,horizon=3,sigma=.005,contrasts=L)
        theta=np.concatenate([truth,alternatives[:,:L]],1)
        state=obs.initial_state(16*(L+1)).reshape(theta.shape)
        with torch.no_grad():result=engine.rollout(p,np.random.default_rng(1),16,stochastic=False,initial=(theta,state),noise=noise)
        records.append({'contrasts':L,'mean_spce_nats':float(result['info'][:,-1].mean()),'ceiling_nats':float(np.log(L+1))})
        history=result
    particles=rng.uniform(lo,hi,(512,10));posterior=[]
    for count in (128,256,512):
        belief=ContinuousParticleBelief(obs,particles[:count],.005);ess=[]
        for stage in range(3):
            belief.update(history['actions'][stage][0],history['observations'][stage][0])
            ess.append(float(1/np.exp(2*belief.log_weights).sum()))
        posterior.append({'particles':count,'ess_by_stage':ess,
                          'posterior_mean':(np.exp(belief.log_weights)@belief.particles).tolist()})
    report={'scope':'Fixed predeclared durations; 16 paired systems; nested contrast sets and independent planner particles.',
        'contrast_sensitivity':records,'planner_particle_diagnostic_one_system':posterior,
        'adequacy_established':False,'note':'Numerical sensitivity diagnostic, not policy performance or publication-budget certification.'}
    (output/'sensitivity.json').write_text(json.dumps(report,indent=2))
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cpu-only',action='store_true')
    args=parser.parse_args()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    status={'state':'running','phase':'cpu' if args.cpu_only else 'gpu','checks':[]}
    target=output/('cpu_readiness.json' if args.cpu_only else 'readiness.json')
    def save():target.write_text(json.dumps(status,indent=2)+'\n')
    def execute(label,command,timeout=1800):
        save()
        with (output/(label+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
        status['checks'].append(label);save()
    start=time.monotonic()
    try:
        if args.cpu_only:
            execute('cpu_tests',[sys.executable,'-m','unittest','tools.tests.test_ieee14_cpu',
                'tools.tests.test_direct_moe','tools.tests.test_eig_collection',
                'tools.tests.test_continuous_eig','tools.tests.test_continuous_swing','-v'])
        else:
            import torch
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; cannot certify GPU readiness')
            execute('gpu_tests',[sys.executable,'-m','unittest','tools.tests.test_ieee14_gpu',
                'tools.tests.test_direct_moe_physics','tools.tests.test_endpoint_rocof.EndpointRocofTests',
                'tools.tests.test_continuous_swing','-v'])
            common=['bash','run.sh','--config','configs/ieee14_mocu.yaml','--objective','eig',
                '--method',','.join(METHODS),'--moe-training-mode','policy_pathwise','--T','3',
                '--N_obs','0','--observation-kind','endpoint_rocof','--window','3.5','--noise_sigma','.005',
                '--seed','101','--eval-seeds','900125']
            execute('seven_method_smoke',common+['--smoke','--output',str(output/'T3/run')])
            status['smoke']=check_smoke(output/'T3/run');status['checks'].append('smoke_artifact_checks');save()
            sensitivity(output);status['checks'].append('bounded_sensitivity_diagnostic');save()
            execute('timing',common+['--estimate-only','--updates','2000','--batch-size','32',
                '--contrasts','128','--planner-particles','128','--validation-systems','128',
                '--validate-every','100','--eval-systems','128','--learning-rate','.001',
                '--output',str(output/'timing')])
            status['timing']=json.loads((output/'timing/timing_estimate.json').read_text())
            status['resource_caveat']='RTX4090 timings cannot certify FASTER A100 runtime; no HPRC submission performed.'
        status['state']='passed'
        status['implementation_readiness']=not args.cpu_only
        status['publication_budget_adequacy']=False
        status['elapsed_seconds']=time.monotonic()-start
        status['source_hashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')}
        save()
        print('IEEE14_VALIDATION_PASSED',target,flush=True)
    except BaseException:
        status.update(state='failed',implementation_readiness=False,error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start)
        save();raise

if __name__=='__main__':main()
