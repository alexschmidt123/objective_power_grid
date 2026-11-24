"""Router-independent offline learning-context discovery. No deployed network.

All gradient targets are adjoints of the TERMINAL native sPCE loss. The
conditional mean estimator uses only observed histories of a shared-only path.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

@contextmanager
def capture_contexts(model, stages):
    states={}; current=[-1]
    def pre(module,args): current[0]=int(args[1])
    def emitter(module,args):
        if current[0] in stages: states[current[0]]={'u':args[0]}
    hooks=[model.register_forward_pre_hook(pre),model.emitter.register_forward_pre_hook(emitter)]
    try: yield states
    finally:
        for h in hooks:h.remove()

def history_covariates(model,features,stage):
    """No truth, simulator states or routed-expert IDs. Used OFFLINE only."""
    z=model.encode_context(features,stage)
    slots=features[:,1:].reshape(len(features),30,4)[:,:stage,:3]
    summary=torch.cat([slots.mean(1),slots.square().mean(1).sqrt(),slots[:,-1]],-1)
    return torch.cat([z,z.square(),summary],-1)

def normalized(x): return F.normalize(x,dim=-1,eps=1e-30)

def fit_conditional(x,g,ridge=.01):
    mu=x.mean(0);scale=x.std(0).clamp_min(.01)
    a=(x-mu)/scale;a=torch.cat([torch.ones_like(a[:,:1]),a],-1)
    penalty=torch.eye(a.shape[1],device=a.device,dtype=a.dtype)*ridge;penalty[0,0]=0
    w=torch.linalg.solve(a.T@a/len(a)+penalty,a.T@g/len(a))
    return dict(mean=mu,scale=scale,weight=w)

def predict(fit,x):
    a=(x-fit['mean'])/fit['scale'];a=torch.cat([torch.ones_like(a[:,:1]),a],-1)
    return a@fit['weight']

def two_directions(v):
    """Deterministic two-component spherical clustering, no expert labels."""
    v=normalized(v); first=v[(v-v.mean(0)).square().sum(-1).argmax()]
    c=torch.stack([first,v[(v@first).argmin()]])
    for _ in range(50):
        labels=(v@c.T).argmax(-1)
        if len(labels.unique())<2: return None
        new=normalized(torch.stack([v[labels==k].mean(0) for k in range(2)]))
        if (new-c).abs().max()<1e-8:break
        c=new
    return c

def responsibilities(fit,centers,x,temperature=.25):
    return ((normalized(predict(fit,x))@centers.T)/temperature).softmax(-1)

def confidence(values):
    mean=values.mean();se=values.std(unbiased=True)/len(values)**.5
    return dict(mean=float(mean),SE=float(se),CI95=[float(mean-1.96*se),float(mean+1.96*se)])

def discovery_metrics(training,replicate,test):
    """Fit solely on training, independent replication, then untouched test.
    Each tuple contains observable x and per-trajectory unit terminal adjoint.
    Gates are fixed before data collection. Two contexts are a hypothesis, not
    a requirement imposed on seven experts. Failure yields no training labels.
    """
    x,g=training;xr,gr=replicate;xt,gt=test
    fit=fit_conditional(x,g); fit2=fit_conditional(xr,gr)
    centers=two_directions(predict(fit,x))
    if centers is None:return dict(PASS=False,reason='COLLAPSED_CONDITIONAL_DIRECTIONS'),None
    r=responsibilities(fit,centers,xt);rr=responsibilities(fit2,centers,xt)
    labels=r.argmax(-1);labels2=rr.argmax(-1)
    prior=responsibilities(fit,centers,x).mean(0)
    # Held-out improvement in squared prediction of gradient directions, versus
    # constant training mean. This cannot pass merely by predictable cluster IDs.
    difference=(gt-predict(fit,xt)).square().sum(-1)-(gt-g.mean(0)).square().sum(-1)
    prediction=confidence(difference)
    contexts=[];means=[]
    for k in range(2):
        halves=[gt[(labels==k)&(torch.arange(len(gt),device=gt.device)%2==h)] for h in range(2)]
        if min(map(len,halves))<2:return dict(PASS=False,reason='EMPTY_TEST_CONTEXT'),None
        mean=[a.mean(0) for a in halves];coherence=[float(a.norm()) for a in mean]
        agreement=float(normalized(mean[0])@normalized(mean[1]))
        snr=[float(a.mean(0).norm()/(a.var(0,unbiased=True).sum()/len(a)).sqrt().clamp_min(1e-30)) for a in halves]
        means.append(normalized(torch.stack(mean).mean(0)))
        contexts.append(dict(fraction=float((labels==k).double().mean()),split_direction_cosine=agreement,coherence=coherence,mean_gradient_SNR=snr))
    between=float(means[0]@means[1]);agreement=float((labels==labels2).double().mean())
    chances=sum(float((labels==k).double().mean()*(labels2==k).double().mean()) for k in range(2))
    kappa=(agreement-chances)/max(1e-12,1-chances)
    r32=responsibilities({k:v.float() for k,v in fit.items()},centers.float(),xt.float())
    precision_agreement=float((r.argmax(-1)==r32.argmax(-1)).double().mean())
    # No thresholds are adapted to observed EIG or grouping success.
    checks=dict(gradient_prediction=prediction['CI95'][1]<0,
        occupancy=all(c['fraction']>=.1 for c in contexts),
        split_reproducibility=all(c['split_direction_cosine']>=.8 for c in contexts),
        coherent_mean=all(min(c['coherence'])>=.1 for c in contexts),
        signal_to_noise=all(min(c['mean_gradient_SNR'])>=3 for c in contexts),
        differentiated=between<.8,assignment_replication=kappa>=.6,
        numerical_precision=precision_agreement>=.99)
    result=dict(PASS=all(checks.values()),checks=checks,contexts=contexts,
        between_context_cosine=between,assignment_agreement=agreement,assignment_kappa=kappa,
        gradient_prediction_vs_constant=prediction,precision_assignment_agreement=precision_agreement,
        float32_max_responsibility_error=float((r-r32.double()).abs().max()),prior=prior.tolist(),
        precision_scope='float32 vs float64 context fitting/inference arithmetic; trajectory-gradient precision tested separately')
    return result,dict(fit=fit,centers=centers)
