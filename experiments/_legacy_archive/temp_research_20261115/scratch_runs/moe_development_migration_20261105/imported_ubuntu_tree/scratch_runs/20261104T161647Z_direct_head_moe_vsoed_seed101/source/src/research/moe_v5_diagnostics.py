"""Observational V5 diagnostics; no change to the V4 loss or deployed policy."""
import copy, contextlib, hashlib, math, random, time
from dataclasses import dataclass, asdict
import numpy as np
import torch
from src.research.engine import rollout

@dataclass
class Work:
    trajectories:int=0
    negatives:int=0
    likelihood_evaluations:int=0
    simulator_particle_stages:int=0
    optimizer_updates:int=0
    posterior_samples:int=0
    max_trajectories:int=16
    max_negatives:int=32
    def rollout(self,batch,horizon,negatives):
        if negatives>self.max_negatives or self.trajectories+batch>self.max_trajectories:
            raise ValueError('Tiny-smoke resource allowance exceeded')
        self.trajectories+=batch;self.negatives=max(self.negatives,negatives)
        self.likelihood_evaluations+=batch*horizon*(negatives+1)
        self.simulator_particle_stages+=batch*horizon*(negatives+1)

def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

@contextlib.contextmanager
def preserve_rng():
    cpu=torch.get_rng_state();numpy=np.random.get_state();python=random.getstate()
    try:yield
    finally:torch.set_rng_state(cpu);np.random.set_state(numpy);random.setstate(python)

def prefix_scores(step_loglik):
    """Truth-inclusive fixed-prior sPCE. Signed increments are never clipped."""
    if step_loglik.ndim!=3 or step_loglik.shape[-1]<2 or not torch.isfinite(step_loglik).all():
        raise ValueError('Finite [batch,time,truth+fixed prior negatives] required')
    cumulative=step_loglik.cumsum(1)
    scores=cumulative[:,:,0]-torch.logsumexp(cumulative,-1)+math.log(cumulative.shape[-1])
    increments=scores-torch.cat((torch.zeros_like(scores[:,:1]),scores[:,:-1]),1)
    return dict(prefix=scores,terminal=scores[:,-1],increments=increments,
        truth_mass=cumulative.softmax(-1)[:,:,0],cumulative_loglik=cumulative,
        ceiling=math.log(cumulative.shape[-1]))

def captured_rollout(benchmark,model,horizon,batch,seed,negatives,work=None):
    """Instance-local likelihood capture; consumes no extra simulator/RNG calls."""
    if seed in {1001,8831001}:raise ValueError('Held-out/reserved seed forbidden in V5 diagnostics')
    if work is not None:work.rollout(batch,horizon,negatives)
    logs=[];routes=[];original=benchmark.log_likelihood;had_local='log_likelihood' in benchmark.__dict__
    def capture(y,mean):
        value=original(y,mean);logs.append(value);return value
    def route(module,args,value):
        d=module.last_dispatch
        if d is not None:routes.append(dict(indices=d['indices'].clone(),scores=d['scores'].clone()))
    hook=model.register_forward_hook(route);benchmark.log_likelihood=capture
    try:result=rollout(benchmark,model,horizon,batch,seed,negatives)
    finally:
        hook.remove()
        if had_local:benchmark.log_likelihood=original
        else:del benchmark.log_likelihood
    result['step_loglik']=torch.stack(logs,1);result['captured_routes']=routes
    return result

def gradient_norms(model,loss):
    names,parameters=zip(*model.named_parameters())
    gradients=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
    groups={k:0. for k in ['encoder','shared','router','head']+[f'routed.{i}' for i in range(7)]}
    inactive=[]
    for name,g in zip(names,gradients):
        if g is None:inactive.append(name);continue
        if not torch.isfinite(g).all():raise FloatingPointError('Nonfinite full-policy gradient')
        group='head' if name.startswith('emitter') else '.'.join(name.split('.')[:2]) if name.startswith('routed.') else name.split('.')[0]
        groups[group]=groups.get(group,0.)+float(g.square().sum())
    return {k:math.sqrt(v) for k,v in groups.items()},gradients,inactive

def functional_diagnostics(benchmark,model,result):
    """Action-level masks/similarity and observable-history sensitivity, not EIG."""
    p=copy.deepcopy(model).eval();contributions=[];sensitivity=[]
    with torch.no_grad():
        for stage,f in enumerate(result['features']):
            raw=p(f.detach(),stage);z=p.last_features;scores=p.last_dispatch['scores'];ids=p.last_dispatch['indices']
            shared=p.shared(z);routed=z.new_zeros(z.shape);by_expert=[]
            for e in range(7):
                selected=(ids==e).any(-1);value=p.routed[e](z)*scores[:,e,None]*selected[:,None]
                routed+=value;by_expert.append(value)
            full=benchmark.feasible(raw,[],model.horizon)
            removed=[(full-benchmark.feasible(p.emitter(z+shared+routed-v),[],model.horizon)).norm(dim=-1) for v in by_expert]
            routed_removed=(full-benchmark.feasible(p.emitter(z+shared),[],model.horizon)).norm(dim=-1)
            shared_removed=(full-benchmark.feasible(p.emitter(z+routed),[],model.horizon)).norm(dim=-1)
            pair=[]
            for i in range(7):
                for j in range(i+1,7):
                    # Same context, equal unit gate weight: functional difference through joint nonlinear head.
                    ai=benchmark.feasible(p.emitter(z+shared+p.routed[i](z)),[],model.horizon)
                    aj=benchmark.feasible(p.emitter(z+shared+p.routed[j](z)),[],model.horizon)
                    pair.append(float((ai-aj).norm(dim=-1).mean()))
            contributions.append(dict(stage=stage,expert_mask_action_difference=[float(x.mean()) for x in removed],
                routed_branch_action_difference=float(routed_removed.mean()),shared_branch_action_difference=float(shared_removed.mean()),
                mean_pairwise_unit_gate_action_difference=float(np.mean(pair))))
            if stage:
                changed=f.detach().clone();slots=changed[:,1:].reshape(len(f),model.horizon,4);slots[:,stage-1,2]+=.1
                altered=benchmark.feasible(p(changed,stage),[],model.horizon);newids=p.last_dispatch['indices']
                sensitivity.append(dict(stage=stage,observation_perturbation_in_encoded_units=.1,
                    mean_action_difference=float((full-altered).norm(dim=-1).mean()),
                    route_set_changed_fraction=float((ids.sort(-1).values!=newids.sort(-1).values).any(-1).double().mean())))
    return dict(action_contributions=contributions,observation_sensitivity=sensitivity,
        interpretation='Frozen functional interventions only; no full-horizon utility or causal specialization claim')

def diagnose(benchmark,model,batch=4,negatives=32,seed=596101):
    before=state_hash(model);rng=torch.get_rng_state().clone();started=time.monotonic();work=Work()
    with preserve_rng():
        p=copy.deepcopy(model).train()
        result=captured_rollout(benchmark,p,30,batch,seed,negatives,work)
        d=prefix_scores(result['step_loglik'])
        torch.testing.assert_close(d['prefix'],result['info'],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(d['increments'].sum(1),d['terminal'],rtol=1e-10,atol=1e-10)
        original=-result['info'][:,-1].mean()+p.balance_loss()
        instrumented=-d['terminal'].mean()+p.balance_loss()
        norms,g1,inactive=gradient_norms(p,original);_,g2,_=gradient_norms(p,instrumented)
        maxerr=0.
        for a,b in zip(g1,g2):
            assert (a is None)==(b is None)
            if a is not None:torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10);maxerr=max(maxerr,float((a-b).abs().max()))
        scores=d['prefix'].detach();ceiling=d['ceiling'];mass=d['truth_mass'].detach()
        stages=[]
        for stage,route in enumerate(result['captured_routes']):
            ids=route['indices'];prob=route['scores'];counts=torch.nn.functional.one_hot(ids,7).sum((0,1))
            stages.append(dict(stage=stage,selected_ids=ids.tolist(),selection_counts=counts.tolist(),
                mean_softmax=prob.mean(0).tolist(),router_entropy=float((-(prob*prob.clamp_min(1e-300).log()).sum(-1)).mean()),
                fraction_within_point01=float((ceiling-scores[:,stage]<=.01).double().mean()),
                fraction_within_point10=float((ceiling-scores[:,stage]<=.10).double().mean())))
        functional=functional_diagnostics(benchmark,p,result)
        final=ceiling-scores[:,-1];first=[]
        for row in scores:
            hits=(ceiling-row<=.1).nonzero().flatten();first.append(None if not len(hits) else int(hits[0]))
        output=dict(status='PASS',diagnostic_contrasts=negatives,actual_training_contrasts=2000,
            training_resolution_saturation='NOT_MEASURED',decision='inconclusive_at_training_resolution',
            smoke_near_ceiling_point01=float((final<=.01).double().mean()),smoke_near_ceiling_point10=float((final<=.1).double().mean()),
            ceiling=ceiling,training_ceiling=math.log(2001),first_smoke_saturation_stage=first,
            terminal_truth_mass=mass[:,-1].tolist(),terminal_negative_mass=(1-mass[:,-1]).tolist(),
            truth_likelihood_derivative_factor=(1-mass[:,-1]).tolist(),
            derivative_interpretation='1-q_truth is a likelihood partial derivative factor, not the complete policy gradient',
            gradient_norms=norms,inactive_parameter_gradients=inactive,gradient_parity_max_absolute_error=maxerr,
            stage_routing=stages,functional=functional,work=asdict(work),seconds=time.monotonic()-started,
            main_sPCE_mean=float(result['info'][:,-1].detach().mean()),not_a_performance_evaluation=True)
    assert state_hash(model)==before and torch.equal(torch.get_rng_state(),rng)
    output.update(original_state_hash=before,model_and_rng_unchanged=True)
    return output
