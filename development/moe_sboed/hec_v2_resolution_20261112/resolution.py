"""V2 evaluation-resolution and all-consumer pathway masking; no training."""
import hashlib
import json
import math
import time
import numpy as np
import torch


def pathway_knockout(v2, history, slots, legacy, responses, a0, expert):
    """Remove one pathway's outputs from router AND fusion. Shared paths remain."""
    if expert not in range(4):
        raise ValueError('Exactly one expert index0..3 required')
    mask=slots.new_ones(1,4,1);mask[:,expert]=0
    legacy=legacy*mask;responses=responses*mask;slots=slots*mask
    raw=v2.coordinator(torch.cat([history,legacy.flatten(1)],-1))
    raw+=v2.response_gate(responses.flatten(1))
    gates=(10*(raw/10).tanh()).softmax(-1)*mask.squeeze(-1)
    gates=gates/gates.sum(-1,keepdim=True)
    pre=v2.fusion[0](torch.cat([history,(gates[...,None]*slots[...,:64]).flatten(1)],-1))
    pre+=v2.response_fusion((gates[...,None]*slots[...,64:]).flatten(1))
    delta=v2.fusion[2](v2.fusion[1](pre)).tanh()
    return a0+4*delta-a0*delta.abs(),gates


def signal(theta,action):
    distance=(theta-action[...,None,:]).square().sum(-1)
    return (.1+(distance+1e-4).reciprocal().sum(-1)).log()


@torch.no_grad()
def nested_scores(actions,observations,truth,contrasts,levels,chunk=16384,deadline=float('inf')):
    """Exact nested prior-sPCE; preserve all terms and include positive once."""
    if sorted(set(levels))!=list(levels) or not levels or levels[0]<1 or levels[-1]>len(contrasts):
        raise ValueError('Need increasing contrast counts within the complete bank')
    b,t,_=actions.shape
    if t!=30 or observations.shape!=(b,30):
        raise ValueError('Complete T30 histories required')
    actions=actions.double();observations=observations.double()
    true_mean=signal(truth.double().reshape(b,1,2,2),actions)
    positive=(-2*(observations-true_mean).square()).sum(-1)
    negative=torch.full_like(positive,-torch.inf)
    start=0;out={}
    for level in levels:
        for first in range(start,level,chunk):
            if time.monotonic()>deadline:
                raise TimeoutError('Fixed evaluation budget reached; partial output preserved')
            theta=contrasts[first:min(first+chunk,level)].double().reshape(1,-1,2,2)
            ll=torch.zeros(b,theta.shape[1],dtype=torch.float64,device=actions.device)
            for t in range(30):
                prediction=signal(theta,actions[:,t,None])
                ll+=-2*(observations[:,t,None]-prediction).square()
            negative=torch.logaddexp(negative,torch.logsumexp(ll,-1))
        out[level]=positive-torch.logaddexp(positive,negative)+math.log(level+1)
        start=level
    return out


def summary(values,contrasts):
    from scipy.stats import t
    x=np.asarray(values,float);n=len(x);mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(n))
    critical=float(t.ppf(.975,n-1));cap=math.log(contrasts+1)
    return dict(n=n,mean=mean,MCSE=se,CI95=[mean-critical*se,mean+critical*se],ceiling=cap,
                fraction_within_001=float(np.mean(cap-x<.01)),fraction_within_01=float(np.mean(cap-x<.1)),
                quantiles=np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist())


def paired(x,y):
    from scipy.stats import t
    v=np.asarray(x,float)-np.asarray(y,float);n=len(v);mean=float(v.mean())
    se=float(v.std(ddof=1)/math.sqrt(n));width=float(t.ppf(.975,n-1))*se
    return dict(n=n,mean=mean,MCSE=se,CI95=[mean-width,mean+width],
                p_two_sided=float(2*t.sf(abs(mean/se),n-1)) if se else float(mean==0))
