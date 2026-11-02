"""Offline-only SA-MoE objectives and conditional competence experiments."""
import math,time
import torch
from src.research.engine import rollout
from src.research.policies import history_features
from src.research.pimoe import AdaptiveBelief,rng
from src.research.sa_moe import competence_responsibilities


def conditional_rollout(b,policy,prefix_actions,prefix_y,posterior,forced_expert,seed,batch=16,contrasts=8192):
    """Conditional remaining-horizon sPCE; NOT an exact or unbiased EIG target.

    Complete continuation toT30 uses the current policy, no actor critic or IC.
    Finite contrast ceiling and conditional-SMC error are reported explicitly.
    """
    g=rng(seed);ids=torch.randint(len(posterior),(batch,contrasts+1),generator=g,device='cpu')
    theta=posterior[ids].to(next(policy.parameters()).device);truth=theta[:,0]
    actions=[a.to(theta).reshape(1,2).expand(batch,-1) for a in prefix_actions]
    ys=[y.to(theta).reshape(1,1).expand(batch,-1) for y in prefix_y]
    stage0=len(actions);state=b.initial((batch,contrasts+1));cumulative=theta.new_zeros(batch,contrasts+1)
    noise=torch.randn(batch,30-stage0,1,generator=g,dtype=torch.float64,device='cpu').to(theta)
    for t in range(stage0,30):
        f=history_features(b,actions,ys,batch,30)
        action=policy.forced_first(f,forced_expert) if t==stage0 and forced_expert is not None else policy(f,t)
        means,state=b.forward(theta,state,action[:,None].expand(-1,contrasts+1,-1))
        y=means[:,0]+b.sigma*noise[:,t-stage0]
        cumulative=cumulative+b.log_likelihood(y[:,None],means);actions.append(action);ys.append(y)
    info=cumulative[:,0]-torch.logsumexp(cumulative,1)+math.log(contrasts+1)
    return info


def posterior_from_history(b,actions,ys,seed,particles=2048,refined=8192):
    belief=AdaptiveBelief(b,particles,seed,moves=8)
    for action,y in zip(actions,ys):belief.update(action.cpu(),y.cpu().reshape(()))
    belief.refine(refined,steps=8)
    return belief


def measure_competence(b,model,contexts,seed,batch=32,contrasts=8192):
    mode=model.training;model.eval();rows=[]
    try:
        with torch.no_grad():
            with torch.device(next(model.parameters()).device):
                for i,ctx in enumerate(contexts):
                    values=torch.stack([conditional_rollout(b,model,ctx['actions'],ctx['observations'],ctx['posterior'],e,seed+i*101,batch,contrasts).cpu() for e in range(4)],1)
                    q=values.mean(0);se=values.std(0,unbiased=True)/math.sqrt(batch);ceiling=math.log(contrasts+1)
                    saturated=bool((values>ceiling-.1).double().mean()>.05)
                    log_r,r=competence_responsibilities(q[None],se[None],torch.tensor([saturated],device=q.device))
                    # Paired value SE supports only within-context contrasts.
                    paired=(values[:,:,None]-values[:,None,:]).std(0,unbiased=True)/math.sqrt(batch)
                    row=dict(stage=len(ctx['actions']),Q=q.tolist(),Q_SE=se.tolist(),paired_difference_SE=paired.tolist(),responsibilities=r[0].tolist(),log_responsibilities=log_r[0].tolist(),saturated=saturated,near_ceiling_fraction=float((values>ceiling-.1).double().mean()),conditional_rollout_replicates=batch,contrasts=contrasts,finite_contrast_lower_bound=True,context=i)
                    rows.append(row)
    finally:model.train(mode)
    return rows
