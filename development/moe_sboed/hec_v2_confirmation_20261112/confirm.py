"""Predeclared independent fixed-V2 routing confirmation, no training."""
import argparse,gc,hashlib,json,os,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
WORK=ROOT/'development/hec_git_sync'
PREVIOUS=WORK/'development/moe_sboed/hec_v2_resolution_20261112'
sys.path.insert(0,str(PREVIOUS))
from run_resolution import sha,put,load_library,knockout_checks,BASE,DEPENDENCY
from resolution import nested_scores,summary,paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--run',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.run:parser.error('Explicit --run required')
    output=args.output.expanduser().resolve()
    if output.exists():raise FileExistsError('New isolated output required')
    if ROOT/'scratch_runs' not in output.parents:raise ValueError('Use LabPC TEMP/scratch_runs')
    output.mkdir(parents=True)
    cfg=dict(smoke=args.smoke,primary_cases=2 if args.smoke else 4096,
             secondary_cases=2 if args.smoke else 512,levels=[11,31,127] if args.smoke else [4000000,16000000],
             batch=16,case_seed=202611128001,contrast_seed=202611128002,
             primary='learned V2 minus frozen uniform V2 at16M,paired two-sided95% t interval;one primary test',
             secondary='four one-hot pathways and frozen vsOED,first512 fresh cases,Holm5 family at16M',
             resource_matching='external vsOED historical checkpoint not compute/initialization matched',
             cases_selection='independent new prior/noise draws;no outcome-based case selection',
             checkpoint_selection='none;existing V2 and baseline checkpoints locked before data generation',
             training_updates=0,wall_seconds=180 if args.smoke else 2400,
             source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    put(output/'CONFIG.json',cfg)
    sources=[*sorted(pathlib.Path(__file__).parent.glob('*.py')),*sorted(PREVIOUS.glob('*.py')),
             *sorted(DEPENDENCY.glob('*.py')),*sorted((BASE/'source').glob('*.py'))]
    inputs=[BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl',
            BASE/'inputs/vsoed.pt',BASE/'ARTIFACT_SHA256.json',BASE/'evaluation/diagnostic_candidate_histories.npz']
    protected={str(p):sha(p) for p in sources+inputs};put(output/'PRE_RUN_MANIFEST.json',protected)
    manifest=json.loads((BASE/'ARTIFACT_SHA256.json').read_text())['files']
    assert sha(BASE/'inputs/vsoed.pt')==manifest['inputs/vsoed.pt']
    os.environ['MOE_V2_RUN_ROOT']=str(BASE);sys.path.insert(0,str(BASE/'source'))
    from native_common import original
    original.setup()
    while True:
        try:lease=original.acquire();break
        except BlockingIOError:
            put(output/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU',pid=os.getpid()));time.sleep(30)
    start=time.monotonic();deadline=start+cfg['wall_seconds'];status='FAILED'
    try:
        lib=load_library();actor=lib.load_frozen('cuda')
        before={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        put(output/'KNOCKOUT_VALIDATION.json',knockout_checks(actor))
        rng=np.random.default_rng(cfg['case_seed']);n=cfg['primary_cases'];levels=cfg['levels']
        truth_array=rng.standard_normal((n,4));noise_array=rng.standard_normal((n,30))
        theta=torch.tensor(truth_array,device='cuda');noise=torch.tensor(noise_array,device='cuda')
        contrast_array=np.random.default_rng(cfg['contrast_seed']).standard_normal((levels[-1],4))
        bank_hash=hashlib.sha256(contrast_array.view(np.uint8)).hexdigest()
        contrasts=torch.tensor(contrast_array,device='cuda');del contrast_array;gc.collect()
        np.savez_compressed(output/'FRESH_CASES.npz',truth=truth_array,noise=noise_array)
        put(output/'BANK_PROVENANCE.json',dict(sha256=bank_hash,seed=cfg['contrast_seed'],levels=levels,
                                               large_bank_saved=False))
        scores={};methods={}
        for mode in ['original','uniform','expert1','expert2','expert3','expert4','vsoed']:
            count=n if mode in ['original','uniform'] else cfg['secondary_cases']
            tick=time.monotonic();cpu=time.process_time();model=actor
            if mode=='vsoed':
                import evaluate_v2
                model,_=evaluate_v2.actor_load('vsoed')
            values={level:[] for level in levels}
            for first in range(0,count,cfg['batch']):
                if time.monotonic()>deadline:raise TimeoutError('Fixed confirmatory budget reached')
                th,eps=theta[first:first+cfg['batch']],noise[first:first+cfg['batch']]
                actions,observations,_=lib.rollout(model,th,eps,'original' if mode=='vsoed' else mode,deadline=deadline)
                result=nested_scores(actions,observations,th,contrasts,levels,deadline=deadline)
                for level in levels:values[level].append(result[level].cpu().numpy())
                progress=dict(status='RUNNING',pid=os.getpid(),policy=mode,completed=min(first+cfg['batch'],count),
                              total=count,seconds=time.monotonic()-start)
                put(output/'STATUS.json',progress)
                if (first//cfg['batch'])%8==0 or first+cfg['batch']>=count:print(json.dumps(progress),flush=True)
            scores[mode]={level:np.concatenate(values[level]) for level in levels}
            methods[mode]={str(level):summary(scores[mode][level],level) for level in levels}
            methods[mode].update(wall_seconds=time.monotonic()-tick,process_CPU_seconds=time.process_time()-cpu,
                                 observations=count*30,negative_density_terms=count*30*levels[-1])
            for level in levels:np.save(output/f'{mode}_{level}_scores.npy',scores[mode][level])
            put(output/'METHODS_PARTIAL.json',methods)
            print('POLICY_COMPLETE',mode,json.dumps(methods[mode]),flush=True)
            if mode=='vsoed':del model;gc.collect();torch.cuda.empty_cache()
        primary={str(level):paired(scores['original'][level],scores['uniform'][level]) for level in levels}
        secondary={name:paired(scores['original'][levels[-1]][:cfg['secondary_cases']],scores[name][levels[-1]])
                   for name in ['expert1','expert2','expert3','expert4','vsoed']}
        previous=0.
        for rank,name in enumerate(sorted(secondary,key=lambda x:secondary[x]['p_two_sided'])):
            previous=max(previous,min(1.,(len(secondary)-rank)*secondary[name]['p_two_sided']))
            secondary[name]['Holm5_p']=previous
        assert all(torch.equal(v,actor.state_dict()[k].cpu()) for k,v in before.items())
        primary_success=primary[str(levels[-1])]['CI95'][0]>0
        result=dict(methods=methods,primary_learned_minus_uniform=primary,secondary_learned_minus=secondary,
                    primary_routing_superiority_established=bool(primary_success and not args.smoke),
                    secondary_all_positive_corrected=bool(all(v['mean']>0 and v['Holm5_p']<.05 for v in secondary.values()) and not args.smoke),
                    single_trained_lineage=True,finite_contrast_lower_bound=True,exact_EIG_claim=False,
                    no_model_or_hyperparameter_selection=True)
        put(output/'RESULTS.json',result);status='SMOKE_COMPLETE' if args.smoke else 'COMPLETE'
    except Exception as exc:
        import traceback
        traceback.print_exc();put(output/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
    finally:
        preserved=all(sha(pathlib.Path(p))==h for p,h in protected.items())
        completion=dict(status=status if preserved else 'PRESERVATION_FAILED',seconds=time.monotonic()-start,
                        preserved=preserved,training_updates=0)
        put(output/'COMPLETION.json',completion);put(output/'STATUS.json',completion)
        print('CONFIRMATION_END',json.dumps(completion),flush=True)
    if status not in ['COMPLETE','SMOKE_COMPLETE']:raise SystemExit(1)


if __name__=='__main__':main()
