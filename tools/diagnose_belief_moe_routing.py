"""Read-only checkpoint diagnosis on fresh validation posterior continuations."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--landscape',action='store_true')
    parser.add_argument('--systems',type=int,default=32)
    parser.add_argument('--fantasies',type=int,default=16)
    args=parser.parse_args()
    run=Path(args.run).resolve(); output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(run/'source_snapshot'))
    import numpy as np
    import torch
    from src.config import load_config
    from src.objectives.eig.continuous_eig import OnlineEIG
    from src.domains.swing.continuous_rocof import EndpointRocofObserver
    from src.objectives.eig.continuous_belief_moe import (
        ContinuousBeliefMoE,ParticleContext,rollout,fantasy_initial)
    metadata=json.loads((run/'run_config.json').read_text()); setting=metadata['settings']
    cfg=load_config(str(run/'source_snapshot/configs/ieee9_eig.yaml'))
    cfg.raw=metadata['physical_config']
    observer=EndpointRocofObserver(cfg,duration_bounds=(setting['duration_min'],setting['duration_max']),
        injection_bus=setting['bus'],amplitude=setting['amplitude'],window=setting['window'])
    engine=OnlineEIG(cfg,observer,horizon=setting['T'],sigma=setting['noise_sigma'],
        contrasts=setting['contrasts'],min_separation=setting['min_duration_separation'])
    if metadata.get('moe_architecture','').startswith('specialist_belief_moe_'):
        from src.objectives.eig.continuous_specialist_moe import ContinuousSpecialistMoE
        policy=ContinuousSpecialistMoE(engine,setting['planner_particles'])
    else:
        policy=ContinuousBeliefMoE(engine,setting['planner_particles'])
    policy.load_state_dict(torch.load(run/'models/moe_sboed.pth',map_location='cpu',weights_only=False)['state_dict'])
    start=time.monotonic(); records=[]
    rng=np.random.default_rng(setting['seed']+940000)
    with torch.no_grad():
        result=rollout(engine,policy,rng,args.systems,stochastic=False)
        for stage,(belief,history,inputs) in enumerate(result['snapshots']):
            if args.landscape and stage>0:break
            count=args.fantasies; batch=args.systems*count
            repeated=ParticleContext(engine,np.repeat(belief.theta,count,axis=0),
                np.repeat(belief.states,count,axis=0),np.repeat(belief.logw,count,axis=0))
            past=([np.repeat(x,count,axis=0) for x in history[0]],
                  [np.repeat(x,count,axis=0) for x in history[1]])
            context=repeated.inputs(*past,stage)
            distribution=policy.distributions(context)
            prediction=policy.q_values(context,distribution.mean).numpy().reshape(args.systems,count,2)[:,0]
            initial=fantasy_initial(repeated,rng,engine.contrasts)
            noise=rng.normal(size=(batch,engine.horizon-stage,engine.n_obs))
            if args.landscape:
                landscape=[]
                from src.objectives.eig.continuous_eig import feasible_duration
                for z in [-4.,-2.,-1.,0.,1.,2.,4.]:
                    latent=torch.full((batch,),z)
                    branch=rollout(engine,policy,rng,batch,stochastic=False,initial=initial,
                        belief=repeated,history=past,stage_start=stage,noise=noise,
                        first_latent=latent,first_logp=torch.zeros(batch))
                    duration=feasible_duration(torch.sigmoid(latent.double()).numpy(),past[0],
                        engine.observer.bounds,engine.min_separation,engine.horizon)
                    row={'latent':z,'duration':float(duration.mean()),
                         'remaining_eig':float(branch['info'][:,-1].mean())}
                    landscape.append(row);print(json.dumps(row),flush=True)
                (output/'landscape.json').write_text(json.dumps({'run':str(run),
                    'validation_seed':setting['seed']+940000,'systems':args.systems,
                    'fantasies':count,'records':landscape,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
                continue
            values=[]
            for k in range(2):
                branch=rollout(engine,policy,rng,batch,stochastic=False,initial=initial,
                    belief=repeated,history=past,stage_start=stage,noise=noise,
                    first_latent=distribution.mean[:,k],first_logp=torch.zeros(batch))
                values.append(branch['info'][:,-1].reshape(args.systems,count))
            difference=values[0]-values[1]
            mean=difference.mean(1); se=difference.std(1,ddof=1)/np.sqrt(count)
            chosen=prediction.argmax(1)
            record={'stage':stage,'predicted_difference':(prediction[:,0]-prediction[:,1]).tolist(),
                'measured_difference':mean.tolist(),'paired_standard_error':se.tolist(),
                'expert0_clear_wins':int((mean>2*se).sum()),'expert1_clear_wins':int((mean<-2*se).sum()),
                'router_choice_counts':np.bincount(chosen,minlength=2).tolist(),
                'routing_disagrees_with_sample_mean':int((chosen!=(mean<0)).sum()),
                'clear_routing_errors':int((((chosen==0)&(mean<-2*se))|((chosen==1)&(mean>2*se))).sum()),
                'mean_remaining_eig':[float(v.mean()) for v in values],
                'note':'Finite-fantasy diagnostic, not a certified oracle or final-test result.'}
            records.append(record)
            (output/'routing_diagnosis.json').write_text(json.dumps({'run':str(run),
                'validation_seed':setting['seed']+940000,'systems':args.systems,'fantasies':count,
                'records':records,'elapsed_seconds':time.monotonic()-start},indent=2)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k not in
                ['predicted_difference','measured_difference','paired_standard_error']}),flush=True)


if __name__=='__main__':main()
