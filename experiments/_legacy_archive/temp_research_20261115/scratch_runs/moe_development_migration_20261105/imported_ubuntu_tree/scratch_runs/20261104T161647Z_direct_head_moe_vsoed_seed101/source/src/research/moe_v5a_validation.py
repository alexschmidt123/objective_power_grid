"""One fixed, bounded sampler validation; no held-out EIG or hyperparameter search."""
import time,math,json,copy
from pathlib import Path
import torch,numpy as np
from src.research.moe_v5_experiment import verified_reference,load_policy
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.moe_v5a_sampler import sample_prefix

def validate(reference,out,particles=512,moves=8,count=128,max_seconds=900):
    start=time.monotonic();torch.set_num_threads(2);ref,receipt=verified_reference(reference)
    model,cfg,_=load_policy(ref,101);model.eval();b=make_benchmark(cfg['benchmark'])
    with torch.no_grad():r=rollout(b,model,30,count,608101,0)
    records=[];ranks=[];predictive_ranks=[];replica_disagreements=[];failures=[]
    probes=torch.tensor([[-2.,-2.],[-2.,2.],[0.,0.],[2.,-2.],[2.,2.]],dtype=torch.float64)
    for i in range(count):
        if time.monotonic()-start>max_seconds:raise RuntimeError('Fixed sampler validation time allowance exceeded')
        stage=i%30;a=[x[i:i+1] for x in r['actions'][:stage]];y=[x[i:i+1] for x in r['observations'][:stage]]
        try:
            pools=[];stats=[]
            for rep in [0,1]:
                _,q,d,p=sample_prefix(b,a,y,min(128,particles),609000+i*1009+rep*1000003,particles,moves);pools.append(p);stats.append(d)
            p=torch.cat(pools);truth=r['truth'][i]
            ranks.append((p<truth).double().mean(0).tolist())
            action=r['actions'][stage][i:i+1]
            mean,_=b.forward(p[None],b.initial((1,len(p))),action[:,None].expand(1,len(p),2));mean=mean[0,:,0]
            # Exact Gaussian mixture predictive CDF at an independent next y.
            z=(r['observations'][stage][i,0]-mean)/b.sigma
            predictive_ranks.append(float((.5*(1+torch.erf(z/math.sqrt(2)))).mean()))
            predicted=[]
            for pool in pools:
                mu,_=b.forward(pool[:,None].expand(-1,len(probes),-1),None,probes[None].expand(len(pool),-1,-1))
                predicted.append(mu[:,:,0])
            delta=(predicted[0].mean(0)-predicted[1].mean(0)).abs()
            pooled_sd=torch.cat(predicted).std(0).clamp_min(.2)
            disagreement=float((delta/pooled_sd).mean());replica_disagreements.append(disagreement)
            records.append(dict(case=i,prefix=stage,replicas=stats,mean_standardized_predictive_replica_disagreement=disagreement))
        except (ValueError,RuntimeError) as exc:failures.append(dict(case=i,prefix=stage,error=str(exc)))
        if i%16==15:print(json.dumps(dict(phase='posterior_validation',completed=i+1,failures=len(failures),seconds=time.monotonic()-start)),flush=True)
    def ks(values):
        x=np.sort(np.asarray(values));n=len(x)
        return float(max(np.max(np.arange(1,n+1)/n-x),np.max(x-np.arange(n)/n))) if n else 1.
    coordinate_ks=[ks(np.array(ranks)[:,i]) for i in range(4)] if ranks else [1.]*4
    predictive_ks=ks(predictive_ranks)
    symmetry=[d['symmetry_label_fraction'] for row in records for d in row['replicas']]
    checks=dict(no_failed_prefixes=len(failures)==0,finite=all(math.isfinite(x) for x in coordinate_ks+[predictive_ks]),
        coordinate_rank_calibration=max(coordinate_ks)<=.20,predictive_rank_calibration=predictive_ks<=.20,
        replica_predictive_agreement=float(np.quantile(replica_disagreements,.9))<=.5 if replica_disagreements else False,
        source_swap_support=all(.30<=x<=.70 for x in symmetry))
    result=dict(status='PASS_EMPIRICAL_APPROXIMATE_SAMPLER' if all(checks.values()) else 'FAIL',checks=checks,
        thresholds=dict(rank_ks_maximum=.20,predictive_replica_disagreement_90pct_maximum=.5,source_label_fraction=[.30,.70],minimum_pool_unique=.5,minimum_mh_acceptance=.005,minimum_pre_resampling_ess=.79),
        particles=particles,moves=moves,independent_replica_pools=2,histories=count,stages='balanced0..29',
        coordinate_rank_ks=coordinate_ks,predictive_rank_ks=predictive_ks,failures=failures,records=records,
        exact_posterior_certified=False,independence_certified=False,
        guarantee='Empirical calibration only; finite correlated SMC and detached inference imply heuristic auxiliary and biased gradient, not certified conditional EIG bound',
        likelihood_particle_stages=b.calls,seconds=time.monotonic()-start)
    Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','failures']}),flush=True)
    return result
