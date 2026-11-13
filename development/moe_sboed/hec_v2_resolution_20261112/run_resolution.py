"""Bounded frozen-V2 resolution study; no optimizer or native training calls."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
import torch
from resolution import nested_scores,summary,paired,pathway_knockout

ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
BASE=ROOT/'scratch_runs/moe_hec_v2_icml_20261109'
WORK=ROOT/'development/hec_git_sync'
DEPENDENCY=WORK/'development/moe_sboed/hec_v2_counterfactual_routing_20261112'


def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()


def put(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def load_library():
    sys.path.insert(0,str(DEPENDENCY))
    spec=importlib.util.spec_from_file_location('frozen_v2_operations',DEPENDENCY/'experiment.py')
    lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
    return lib


@torch.no_grad()
def knockout_checks(actor):
    data=np.load(BASE/'evaluation/diagnostic_candidate_histories.npz')
    x=torch.tensor(data['X'][:8,[3,12,24]].copy(),device='cuda').reshape(-1,117)
    _,info=actor(x,collect=True);a0=actor.base(x);out={}
    for k in range(4):
        inputs=[info['history'],info['slots'],info['legacy_features'],info['responses'],a0]
        action,g=pathway_knockout(actor,*inputs,k)
        altered=[v.clone() for v in inputs]
        altered[1][:,k]+=100;altered[2][:,k]-=200;altered[3][:,k]+=300
        aa,gg=pathway_knockout(actor,*altered,k)
        assert torch.equal(action,aa) and torch.equal(g,gg)
        assert bool((g[:,k]==0).all()) and bool((action.abs()<=4).all())
        out[str(k+1)]=dict(removed_outputs_gate_error=float((g-gg).abs().max()),
                           removed_outputs_action_error=float((action-aa).abs().max()),
                           masked_weight_max=float(g[:,k].max()))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.run:parser.error('Explicit --run required; imports never launch computation')
    output=args.output.expanduser().resolve()
    if output.exists():raise FileExistsError('Preserve earlier work; require a new output')
    if ROOT/'scratch_runs' not in output.parents:raise ValueError('LabPC TEMP/scratch_runs output required')
    output.mkdir(parents=True)
    config=dict(smoke=args.smoke,trajectories=2 if args.smoke else 512,
                levels=[11,31,127] if args.smoke else [1000000,4000000,16000000],batch=16,
                extra_contrast_seed=202611127002,wall_seconds=180 if args.smoke else 1800,
                source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip(),
                training_updates=0,policies=['original','uniform'],
                interpretation='nested-bound resolution diagnostic; no final-test or architecture superiority claim',
                bank='original shared 1M contrasts followed by independent fixed-seed prior extensions',
                cases='first512 original development-bank cases; no outcome-based selection')
    put(output/'CONFIG.json',config)
    protected={str(p):sha(p) for p in [BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl',
               BASE/'evaluation/development_validation_bank.npz',BASE/'evaluation/diagnostic_candidate_histories.npz',
               *sorted((BASE/'source').glob('*.py')),*sorted(DEPENDENCY.glob('*.py')),
               *sorted(pathlib.Path(__file__).parent.glob('*.py'))]}
    put(output/'PRE_RUN_MANIFEST.json',protected)
    os.environ['MOE_V2_RUN_ROOT']=str(BASE);sys.path.insert(0,str(BASE/'source'))
    from native_common import original
    original.setup()
    while True:
        try:lease=original.acquire();break
        except BlockingIOError:
            put(output/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU',pid=os.getpid()));time.sleep(30)
    start=time.monotonic();deadline=start+config['wall_seconds'];status='FAILED'
    try:
        lib=load_library();actor=lib.load_frozen('cuda')
        frozen={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        put(output/'KNOCKOUT_VALIDATION.json',knockout_checks(actor))
        old=np.load(BASE/'evaluation/development_validation_bank.npz')
        n=config['trajectories'];levels=config['levels']
        theta=torch.tensor(old['truth'][:n],device='cuda');noise=torch.tensor(old['noise'][:n],device='cuda')
        prefix=old['contrasts'][:min(levels[0],len(old['contrasts']))]
        extra=np.random.default_rng(config['extra_contrast_seed']).standard_normal((levels[-1]-len(prefix),4))
        array=np.concatenate([prefix,extra]);bank_hash=hashlib.sha256(array.view(np.uint8)).hexdigest()
        contrasts=torch.tensor(array,device='cuda');del array,extra;gc.collect()
        put(output/'BANK_PROVENANCE.json',dict(sha256=bank_hash,seed=config['extra_contrast_seed'],
                                                levels=levels,stored_large_bank=False))
        scores={};methods={}
        for mode in config['policies']:
            policy_start=time.monotonic();values={level:[] for level in levels};saved_actions=[];saved_obs=[]
            for first in range(0,n,config['batch']):
                if time.monotonic()>deadline:raise TimeoutError('Fixed study budget reached')
                th,eps=theta[first:first+config['batch']],noise[first:first+config['batch']]
                actions,observations,_=lib.rollout(actor,th,eps,mode,deadline=deadline)
                score=nested_scores(actions,observations,th,contrasts,levels,deadline=deadline)
                for level in levels:values[level].append(score[level].cpu().numpy())
                saved_actions.append(actions.cpu());saved_obs.append(observations.cpu())
                put(output/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),policy=mode,
                                              completed=min(first+config['batch'],n),total=n,
                                              seconds=time.monotonic()-start))
                print(json.dumps(dict(policy=mode,cases=min(first+config['batch'],n),total=n,
                                      seconds=time.monotonic()-start)),flush=True)
            scores[mode]={level:np.concatenate(values[level]) for level in levels}
            methods[mode]={str(level):summary(scores[mode][level],level) for level in levels}
            methods[mode]['seconds']=time.monotonic()-policy_start
            torch.save(dict(actions=torch.cat(saved_actions),observations=torch.cat(saved_obs)),output/(mode+'_trajectories.pt'))
            for level in levels:np.save(output/f'{mode}_{level}_scores.npy',scores[mode][level])
            if not args.smoke:
                name='full' if mode=='original' else 'uniform'
                reference=np.load(ROOT/f'scratch_runs/moe_hec_v2_internal_ablation/{name}_scores.npy')[:n]
                error=float(np.max(np.abs(scores[mode][levels[0]]-reference)))
                assert error<1e-8,('Original 1M result reproduction failed',mode,error)
                methods[mode]['original_1M_max_error']=error
            put(output/'METHODS_PARTIAL.json',methods)
        comparisons={str(level):paired(scores['original'][level],scores['uniform'][level]) for level in levels}
        assert all(torch.equal(v,actor.state_dict()[k].cpu()) for k,v in frozen.items())
        result=dict(methods=methods,paired_learned_minus_uniform=comparisons,
                    contrast_bias_resolved=False,training_updates=0,
                    limitation='16M is still a finite lower bound; assess remaining ceiling fraction and paired stability')
        put(output/'RESULTS.json',result);status='SMOKE_COMPLETE' if args.smoke else 'COMPLETE'
    except Exception as exc:
        import traceback
        traceback.print_exc();put(output/'ERROR.json',dict(type=type(exc).__name__,message=str(exc)))
    finally:
        preserved=all(sha(pathlib.Path(p))==h for p,h in protected.items())
        completion=dict(status=status if preserved else 'PRESERVATION_FAILED',seconds=time.monotonic()-start,
                        preserved=preserved,training_updates=0)
        put(output/'COMPLETION.json',completion);put(output/'STATUS.json',completion)
        print('RESOLUTION_END',json.dumps(completion),flush=True)
    if status not in ['COMPLETE','SMOKE_COMPLETE']:raise SystemExit(1)


if __name__=='__main__':main()
