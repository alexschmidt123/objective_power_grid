"""Pathwise differentiation of the online swing simulator.

The forward model is unchanged. Backpropagation uses numerical simulator
Jacobians in incoming state and duration, not a REINFORCE gradient in actions.
M/K are sampled constants. This numerical derivative approximation is recorded
explicitly and tested against independent end-to-end finite differences.
"""
import numpy as np
import torch


class SwingStage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, observer, theta, state, duration):
        theta_np=theta.detach().cpu().numpy()
        state_np=state.detach().cpu().numpy()
        duration_np=duration.detach().cpu().numpy()
        result=observer.propagate(theta_np,state_np,duration_np)
        ctx.observer=observer
        ctx.arrays=(theta_np,state_np,duration_np)
        ctx.dtype,ctx.device=state.dtype,state.device
        return (torch.as_tensor(result.observations,dtype=state.dtype,device=state.device),
                torch.as_tensor(result.terminal_state,dtype=state.dtype,device=state.device))

    @staticmethod
    def backward(ctx, grad_obs, grad_final):
        theta,state,duration=ctx.arrays
        observer=ctx.observer
        go=grad_obs.detach().cpu().numpy() if grad_obs is not None else 0.
        gs=grad_final.detach().cpu().numpy() if grad_final is not None else 0.
        active=(observer.prepare_derivative(theta,state,duration)
                if hasattr(observer,'prepare_derivative') else None)
        def propagate(s,d):
            return (observer.propagate_derivative(theta,s,d,active) if active is not None
                    else observer.propagate(theta,s,d))
        def difference(sp,sm,dp,dm,denominator):
            plus=propagate(sp,dp)
            minus=propagate(sm,dm)
            return (np.sum((plus.observations-minus.observations)*go,axis=1)+
                    np.sum((plus.terminal_state-minus.terminal_state)*gs,axis=1))/denominator
        state_grad=None
        if ctx.needs_input_grad[2]:
            state_grad=np.empty_like(state)
            for j in range(state.shape[1]):
                h=1e-5*np.maximum(1.,np.abs(state[:,j]))
                plus,minus=state.copy(),state.copy()
                plus[:,j]+=h; minus[:,j]-=h
                state_grad[:,j]=difference(plus,minus,duration,duration,2*h)
            state_grad=torch.as_tensor(state_grad,dtype=ctx.dtype,device=ctx.device)
        duration_grad=None
        if ctx.needs_input_grad[3]:
            lo,hi=observer.bounds
            plus=np.minimum(duration+1e-5,hi)
            minus=np.maximum(duration-1e-5,lo)
            duration_grad=difference(state,state,plus,minus,plus-minus)
            duration_grad=torch.as_tensor(duration_grad,dtype=ctx.dtype,device=ctx.device)
        return None,None,state_grad,duration_grad


def feasible_duration_torch(unit,history,bounds,separation,horizon=None):
    """Piecewise differentiable interval-length map; keep history in order."""
    lo,hi=bounds
    result=[]
    gap=separation+1e-10
    if horizon is not None:
        lower = history[-1]+gap if history else torch.full_like(unit,lo)
        upper = hi-(horizon-len(history)-1)*gap
        if len(history)>=horizon or torch.any(lower>upper+1e-12):
            raise ValueError('No increasing duration remains')
        return lower+unit*torch.clamp(upper-lower,min=0.)
    for row,q in enumerate(unit.reshape(-1)):
        intervals=[(q.new_tensor(lo),q.new_tensor(hi))]
        for action in history:
            left,right=action[row]-gap,action[row]+gap
            remaining=[]
            for start,end in intervals:
                if right.item()<=start.item() or left.item()>=end.item():
                    remaining.append((start,end))
                else:
                    if left.item()>start.item():remaining.append((start,left))
                    if right.item()<end.item():remaining.append((right,end))
            intervals=remaining
        lengths=torch.stack([b-a for a,b in intervals])
        distance=q*lengths.sum()
        index=min(int(torch.searchsorted(lengths.cumsum(0).detach(),distance.detach(),right=True)),len(intervals)-1)
        result.append(intervals[index][0]+distance-lengths[:index].sum())
    return torch.stack(result)


def pathwise_rollout(engine,policy,rng,batch,*,initial=None,history=None,stage_start=0):
    theta,states=engine.sample(rng,batch) if initial is None else initial
    theta=torch.as_tensor(theta,dtype=torch.float64)
    states=torch.as_tensor(states,dtype=torch.float64)
    actions=[] if history is None else [torch.as_tensor(a,dtype=torch.float64) for a in history[0]]
    observations=[] if history is None else [torch.as_tensor(y,dtype=torch.float64) for y in history[1]]
    likelihood=torch.zeros(theta.shape[:2],dtype=torch.float64)
    particles=theta.shape[1]
    infos=[]
    for stage in range(stage_start,engine.horizon):
        feature=torch.zeros((batch,1+engine.horizon*(engine.n_obs+2)),dtype=torch.float64)
        feature[:,0]=stage/engine.horizon
        lo,hi=engine.observer.bounds
        for k,(action,obs) in enumerate(zip(actions,observations)):
            start=1+k*(engine.n_obs+2)
            feature[:,start]=(action-lo)/(hi-lo)
            feature[:,start+1:start+1+engine.n_obs]=obs/.05
            feature[:,start+1+engine.n_obs]=1.
        distribution,_=policy(feature.float(),stage)
        duration=feasible_duration_torch(torch.sigmoid(distribution.mean.double()),
            actions,engine.observer.bounds,engine.min_separation,engine.horizon)
        means,final=SwingStage.apply(engine.observer,theta.reshape(-1,theta.shape[-1]),
            states.reshape(-1,states.shape[-1]),duration.repeat_interleave(particles))
        means=means.reshape(batch,particles,engine.n_obs)
        states=final.reshape(states.shape)
        noise=torch.as_tensor(rng.normal(size=(batch,engine.n_obs)),dtype=torch.float64)
        y=means[:,0]+engine.sigma*noise
        likelihood=likelihood-.5*(((y[:,None]-means)/engine.sigma)**2).sum(-1)
        infos.append(likelihood[:,0]-torch.logsumexp(likelihood,dim=1)+np.log(particles))
        actions.append(duration); observations.append(y)
    return torch.stack(infos,dim=1),actions,observations


def improve_pathwise(engine,policy,optimizer,rng,batch,**kwargs):
    infos,_,_=pathwise_rollout(engine,policy,rng,batch,**kwargs)
    loss=-infos[:,-1].mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(),5.)
    optimizer.step()
    return -float(loss.detach())
