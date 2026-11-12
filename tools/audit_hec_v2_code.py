"""Read-only HEC-v2 CPU audit; no optimizer steps, new rollouts, or GPU jobs."""
import argparse
import copy
import hashlib
import inspect
import io
import json
import math
import os
import pathlib
import sys
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import t as student_t

ROOT = pathlib.Path.home() / 'Documents/objective_power_grid_temp'
BASE = ROOT / 'scratch_runs/moe_hec_v2_icml_20261109'
ABL = ROOT / 'scratch_runs/moe_hec_v2_internal_ablation'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def stats(x):
    x = np.asarray(x, dtype=float)
    mean = float(x.mean()); se = float(x.std(ddof=1)/math.sqrt(len(x)))
    width = float(student_t.ppf(.975, len(x)-1))*se
    p = float(2*student_t.sf(abs(mean/se), len(x)-1)) if se else float(mean == 0)
    return dict(n=len(x), mean=mean, MCSE=se, CI95=[mean-width,mean+width],p_two_sided=p)


def cka(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - x.mean(0); y = y - y.mean(0)
    denominator = np.linalg.norm(x.T @ x)*np.linalg.norm(y.T @ y)
    return float(np.square(x.T @ y).sum()/denominator) if denominator > 1e-20 else None


def normalized_gates(raw):
    return (10 * torch.tanh(raw / 10)).softmax(-1)


def head_parts(model, history, slots, dtype=None):
    """Exact affine decomposition; all tensors refer to the same hidden basis."""
    cv = lambda x: x if dtype is None else x.to(dtype=dtype)
    s, packets = cv(history), cv(slots)
    weight, bias = cv(model.fusion[0].weight), cv(model.fusion[0].bias)
    response = cv(model.response_fusion.weight)
    shared = F.linear(s, weight[:, :128], bias)
    first_w = weight[:, 128:].reshape(128,4,64)
    response_w = response.reshape(128,4,64)
    contributions = torch.einsum('bei,hei->beh', packets[..., :64], first_w)
    contributions += torch.einsum('bei,hei->beh', packets[..., 64:], response_w)
    return shared, contributions


def physical_head(model, shared, contributions, gates, a0):
    dtype = shared.dtype
    pre = shared + (gates[...,None] * contributions).sum(1)
    delta = F.linear(F.relu(pre), model.fusion[2].weight.to(dtype),
                     model.fusion[2].bias.to(dtype)).tanh()
    return a0.to(dtype) + 4*delta - a0.to(dtype)*delta.abs(), pre


def inventory(module):
    layers = []
    for name,m in module.named_modules():
        if not list(m.children()):
            layers.append(dict(name=name,definition=str(m),
                               parameters=sum(p.numel() for p in m.parameters()),
                               trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),
                               buffers=sum(b.numel() for b in m.buffers())))
    return dict(parameters=sum(p.numel() for p in module.parameters()),
                trainable=sum(p.numel() for p in module.parameters() if p.requires_grad),
                buffers=sum(b.numel() for b in module.buffers()),layers=layers)


def load_runtime():
    os.environ['MOE_V2_RUN_ROOT'] = str(BASE)
    sys.path.insert(0,str(BASE/'source'))
    from native_common import cloudpickle
    import restore_v1, evaluate_v2
    path = BASE/'arms/candidate/runtime.pkl'
    assert sha(path) == 'd38c09167c4936a6a0660d16079ecc3d5e1108c32bdbc840cc8851c1ef7266b1'
    old = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
    try:
        with path.open('rb') as f: state=cloudpickle.load(f)
    finally: torch.storage._load_from_bytes=old
    a=state['algorithm']
    path=BASE/'arms/candidate/actor.pt'
    assert sha(path)=='90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108'
    cp=torch.load(path,map_location='cpu',weights_only=True)['actor_state']
    assert all(torch.equal(v,a.actor_net.state_dict()[k]) for k,v in cp.items())
    assert int(a.update)==16001
    return a


def forward_checks(a, hist):
    model=a.actor_net.eval()
    stages=[0,3,12,24,29]
    x=torch.tensor(hist['X'][:8,stages].copy()).reshape(-1,117)
    stage=x[:,:30].argmax(-1)
    hooks={};handles=[]
    for name in ['history','commander','region','discriminate','sensitivity','sequential',
                 'adapter2','adapter4','coordinator','response_gate','response_fusion',
                 'fusion.0','fusion.1','fusion.2','opportunity']:
        module=dict(model.named_modules())[name]
        def hook(m,inp,out,name=name):
            hooks[name]=dict(input=[dict(shape=list(v.shape),dtype=str(v.dtype)) for v in inp],
                             output=dict(shape=list(out.shape),dtype=str(out.dtype)))
        handles.append(module.register_forward_hook(hook))
    with torch.no_grad(): action, info=model(x,collect=True);a0=model.base(x)
    for h in handles:h.remove()
    shared,u=head_parts(model,info['history'],info['slots'])
    manual,pre=physical_head(model,shared,u,info['gates'],a0)
    error=float((action-manual).abs().max())
    assert error<2e-5
    uniform=torch.full_like(info['gates'],.25)
    with torch.no_grad(): uniform_action, ui=model(x,routing_override=uniform,collect=True)
    umanual,upre=physical_head(model,shared,u,uniform,a0)
    assert float((uniform_action-umanual).abs().max())<2e-5
    assert torch.equal(info['slots'],ui['slots'])
    onehot=[]
    for k in range(4):
        gates=torch.zeros_like(uniform);gates[:,k]=1
        with torch.no_grad(): actual=model(x,routing_override=gates)
        predicted,_=physical_head(model,shared,u,gates,a0)
        torch.testing.assert_close(actual,predicted,atol=2e-5,rtol=1e-5)
        altered=info['slots'].clone()
        altered[:,torch.arange(4)!=k]+=100
        sp,uc=head_parts(model,info['history'],altered)
        altered_action,_=physical_head(model,sp,uc,gates,a0)
        torch.testing.assert_close(predicted,altered_action,rtol=0,atol=0)
        onehot.append(dict(expert=k+1,manual_error=float((actual-predicted).abs().max()),
                           unselected_packet_action_error=float((predicted-altered_action).abs().max()),
                           mean_action_change=float((actual-action).norm(dim=-1).mean())))
    # Counterfactual raw-logit perturbations keep all packets, history and base fixed.
    with torch.no_grad():
        raw=model.coordinator(torch.cat([info['history'],info['legacy_features'].flatten(1)],-1))
        raw+=model.response_gate(info['responses'].flatten(1))
        torch.testing.assert_close(normalized_gates(raw),info['gates'])
    sd,ud=head_parts(model,info['history'],info['slots'],torch.float64)
    sd,ud=sd.detach(),ud.detach()
    rawd=raw.double().detach().requires_grad_()
    actd,_=physical_head(model,sd,ud,normalized_gates(rawd),a0)
    jac=torch.stack([torch.autograd.grad(actd[:,i].sum(),rawd,retain_graph=True)[0]
                     for i in range(2)],1)
    direction=rawd.new_tensor([1.,-1.,.5,-.5]);direction/=direction.norm()
    eps=1e-5
    ap,_=physical_head(model,sd,ud,normalized_gates(rawd+eps*direction),a0)
    am,_=physical_head(model,sd,ud,normalized_gates(rawd-eps*direction),a0)
    fd=(ap-am)/(2*eps);analytic=torch.einsum('bij,j->bi',jac,direction)
    fd_error=float((fd-analytic).abs().max())
    assert fd_error<1e-6
    perturb={}
    for amount in [.1,1.]:
        gp=normalized_gates(raw+amount*direction.float())
        altered,pnew=physical_head(model,shared,u,gp,a0)
        weighted_new=gp[:,:,None]*info['slots']
        weighted_old=info['gates'][:,:,None]*info['slots']
        perturb[str(amount)]=dict(gate_RMS=float((gp-info['gates']).square().mean().sqrt()),
            weighted_packet_RMS=float((weighted_new-weighted_old).square().mean().sqrt()),
            hidden_RMS=float((pnew-pre).square().mean().sqrt()),
            action_RMS=float((altered-action).square().mean().sqrt()))
    # Dropping a packet after the router is NOT removal of that expert's
    # information from the coordinator. Test the remaining indirect path.
    removal_router_path=[]
    for k in range(4):
        def masked_gate(rr):
            gg=normalized_gates(rr).clone();gg[:,k]=0
            return gg/gg.sum(-1,keepdim=True)
        with torch.no_grad():
            original_masked=masked_gate(raw)
            legacy=info['legacy_features'].clone();response=info['responses'].clone()
            legacy[:,k]=0;response[:,k]=0
            modified_raw=model.coordinator(torch.cat([info['history'],legacy.flatten(1)],-1))
            modified_raw+=model.response_gate(response.flatten(1))
            modified_masked=masked_gate(modified_raw)
            before_action,_=physical_head(model,shared,u,original_masked,a0)
            after_action,_=physical_head(model,shared,u,modified_masked,a0)
        removal_router_path.append(dict(expert=k+1,disabled_weight_max=float(original_masked[:,k].abs().max()),
             remaining_gate_RMS=float((original_masked-modified_masked).square().mean().sqrt()),
             action_RMS_from_disabled_expert_router_input=float((before_action-after_action).square().mean().sqrt())))
    # A fixed positive gate can be absorbed exactly into the expert fusion columns.
    fixed=uniform.new_tensor([.1,.3,.2,.4]).expand_as(uniform)
    afixed,pfixed=physical_head(model,shared,u,fixed,a0)
    areparam,preparam=physical_head(model,shared,u*(4*fixed[:,:,None]),uniform,a0)
    compensation_error=float((afixed-areparam).abs().max())
    assert compensation_error<2e-5
    stage_metrics={}
    for t in stages:
        mask=stage==t
        weighted=(info['gates'][mask,:,None]*u[mask]).sum(1)
        stage_metrics[str(t)]=dict(gate_mean=info['gates'][mask].mean(0).tolist(),
            raw_squash_derivative=float((1-(raw[mask]/10).tanh().square()).mean()),
            gate_to_action_Jacobian_RMS=float(jac[mask].square().mean().sqrt()),
            shared_hidden_norm=float(shared[mask].norm(dim=-1).mean()),
            gated_expert_hidden_norm=float(weighted.norm(dim=-1).mean()),
            shared_to_expert_norm_ratio=float((shared[mask].norm(dim=-1)/weighted.norm(dim=-1).clamp_min(1e-12)).mean()),
            contribution_norms=u[mask].norm(dim=-1).mean(0).tolist(),
            learned_uniform_action_RMS=float((action[mask]-uniform_action[mask]).square().mean().sqrt()))
    # Only the intended expert packet changes when its command block is perturbed.
    command_effects=[];begin=0
    for k,width in enumerate([4,6,6,4]):
        commands=info['commands'].clone();commands[:,begin:begin+width]=(commands[:,begin:begin+width]+.05).clamp(-1,1)
        with torch.no_grad(): changed,ci=model(x,command_override=commands,routing_override=uniform,collect=True)
        changes=(ci['slots']-info['slots']).square().mean((0,2)).sqrt()
        assert float(changes[k])>0
        assert float(changes[torch.arange(4)!=k].max())==0
        command_effects.append(dict(expert=k+1,packet_RMS=changes.tolist(),action_RMS=float((changed-uniform_action).square().mean().sqrt())))
        begin+=width
    return dict(hooks=hooks,info_shapes={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in info.items()},
        manual_action_max_error=error,uniform_action_max_error=float((uniform_action-umanual).abs().max()),
        uniform_packets_unchanged=True,onehot=onehot,logit_perturbations=perturb,
        finite_difference_max_error=fd_error,static_gate_reparameterization_error=compensation_error,
        packet_removal_indirect_router_path=removal_router_path,
        stage_metrics=stage_metrics,command_block_isolation=command_effects,
        physical_bounds_verified=bool((action.abs()<=4).all()),
        single_history_all_expert_computation=True),x


def gradient_checks(a,x):
    model=a.actor_net
    model.zero_grad(set_to_none=True);a.critic_net.zero_grad(set_to_none=True)
    out=model(x,collect=True)[0]
    loss=-a.critic_net(torch.cat([x,out],-1)).mean()
    loss.backward()
    groups={}
    active=['history','commander','region','discriminate','sensitivity','sequential',
            'coordinator','adapter2','adapter4','response_gate','response_fusion','fusion']
    for name,module in model.named_children():
        pp=list(module.parameters());gg=[p.grad for p in pp if p.grad is not None]
        finite=all(bool(torch.isfinite(g).all()) for g in gg)
        norms=sum(float(g.square().sum()) for g in gg)**.5
        steps=[float(a.actor_optimizer.state[p]['step']) for p in pp if p in a.actor_optimizer.state]
        groups[name]=dict(parameter_elements=sum(p.numel() for p in pp),gradient_norm=norms,
                          finite=finite,parameters_with_gradient=len(gg),parameters=len(pp),
                          optimizer_step_min=min(steps) if steps else None,
                          optimizer_step_max=max(steps) if steps else None,
                          all_parameters_in_optimizer=all(any(p is q for g in a.actor_optimizer.param_groups for q in g['params']) for p in pp))
        if name in active:assert finite and norms>0
    initial=torch.load(BASE/'initial/candidate_actor.pt',map_location='cpu',weights_only=True)['actor_state']
    for name in active:
        groups[name]['initial_final_L2_change']=sum(float((p.detach()-initial[key]).square().sum())
            for key,p in model.state_dict().items() if key.startswith(name+'.'))**.5
        assert groups[name]['initial_final_L2_change']>0
    # Exercise the real checkpoint path and multiple chunks without optimizer steps.
    reference={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
    disposable=copy.deepcopy(model);disposable.chunk=7;disposable.zero_grad(set_to_none=True)
    out2=disposable(x);loss2=-a.critic_net(torch.cat([x,out2],-1)).mean();loss2.backward()
    numerator=sum(float((p.grad-reference[k]).square().sum()) for k,p in disposable.named_parameters() if k in reference)
    denominator=sum(float(g.square().sum()) for g in reference.values())
    checkpoint_relative_error=(numerator/max(denominator,1e-30))**.5
    assert checkpoint_relative_error<2e-3
    from migrate_v2 import phase
    phase(disposable,'B');disposable.zero_grad(set_to_none=True)
    b_loss=-a.critic_net(torch.cat([x,disposable(x)],-1)).mean();b_loss.backward()
    b_norm={name:sum(float(p.grad.square().sum()) for p in getattr(disposable,name).parameters() if p.grad is not None)**.5
            for name in active}
    assert b_norm['history']==0 and all(b_norm[name]>0 for name in ['region','discriminate','sensitivity','sequential','coordinator'])
    # Comparable gradients: isolated expert-induced native-loss sensitivity to the
    # SAME 128-d history embedding; detach all other contributions and direct path.
    with torch.no_grad():w,mu,sd=model.belief.params(x);a0=model.base(x)
    _,info=model.compute(x,w,mu,sd,a0,return_info=True)
    shared,u=head_parts(model,info['history'],info['slots'])
    g=info['gates'].detach();base=(g[:,:,None]*u.detach()).sum(1)
    vectors=[]
    for k in range(4):
        pre=shared.detach()+base+g[:,k,None]*(u[:,k]-u[:,k].detach())
        delta=model.fusion[2](F.relu(pre)).tanh();action=a0+4*delta-a0*delta.abs()
        ll=-a.critic_net(torch.cat([x,action],-1)).mean()
        grad=torch.autograd.grad(ll,info['history'],retain_graph=True)[0]
        vectors.append(grad.detach().flatten())
    matrix=[]
    for left in vectors:
        matrix.append([float(F.cosine_similarity(left,right,dim=0)) for right in vectors])
    return dict(native_actor_loss=float(loss.detach()),groups=groups,checkpoint_relative_gradient_error=checkpoint_relative_error,
                checkpoint_action_max_error=float((out.detach()-out2.detach()).abs().max()),phase_B_gradient_norms=b_norm,
                isolated_expert_gradient_history_cosine=matrix,
                isolated_expert_gradient_norms=[float(g.norm()) for g in vectors],
                gradient_comparison_scope='same-history embedding; expert contributions only, gates/direct shared path detached; not gradients across unequal parameter spaces',
                optimizer_steps_executed=0)


def native_checks(a,hist):
    from soft_policy import prefix_inputs
    from hec_policy import mean_signal,jacobian
    from native_reward import stateless_rewards
    from native_common import original
    from frozen_evaluator import contrast_scores
    model=a.actor_net
    ds=torch.tensor(hist['ds'][:4]).float();ys=torch.tensor(hist['ys'][:4]).float()
    theta=torch.tensor(hist['truth'][:4]).float()
    prefix=[]
    for t in [0,3,12,24,29]:
        stage=torch.full((4,),t,dtype=torch.long)
        x=prefix_inputs(ds,ys,stage)
        native=a.form_actor_input(t,ds[:,:t],ys[:,:t])
        torch.testing.assert_close(x,native,rtol=0,atol=0)
        altered_ds,altered_ys=ds.clone(),ys.clone();altered_ds[:,t:]+=123;altered_ys[:,t:]-=456
        altered=prefix_inputs(altered_ds,altered_ys,stage)
        assert torch.equal(x,altered)
        with torch.no_grad():assert torch.equal(model(x),model(altered))
        prefix.append(dict(stage=t,packing_max_error=float((x-native).abs().max()),future_invariance=True))
    source=original.SOURCE(multimodel=False,include_nuisance=False,include_goal=False)
    th=theta.double();actions=ds[:,3].double();target=mean_signal(th[:,None],actions[:,None])[:,0,0]
    actual,sigma=source.deterministic(th,actions)
    torch.testing.assert_close(target,actual[:,0],rtol=1e-12,atol=1e-12)
    observed=torch.tensor(hist['ys'][:4,3]).double()
    literal=torch.distributions.Normal(actual,sigma).log_prob(observed).flatten()
    native=source.loglikeli(3,observed,th,actions).flatten()
    # SOURCE returns a float32 sigma/normalization constant even for float64 means.
    torch.testing.assert_close(literal,native,rtol=1e-8,atol=1e-8)
    tt=th.clone().requires_grad_();vv=mean_signal(tt[:,None],actions[:,None])[:,0,0]
    autodiff=torch.autograd.grad(vv.sum(),tt)[0]
    analytic=jacobian(tt[:,None],actions)[:,0]
    torch.testing.assert_close(autodiff,analytic,rtol=1e-10,atol=1e-10)
    with torch.no_grad():
        reward=stateless_rewards(a,ds,ys,theta)
        terminal=a.post_approx.log_poi_post(29,ds,ys,theta)
        prior=a.post_approx.prior.log_poi_prior(theta)
        telescoping=float((reward.sum(-1)-(terminal-prior)).abs().max())
        assert telescoping<3e-5
    # Existing observations only: no model.model(), no new latent draws or rollouts.
    bank=np.load(BASE/'evaluation/development_validation_bank.npz')
    contrasts=torch.tensor(bank['contrasts'][:11])
    data=dict(actions=list(ds[:2].unbind(1)),observations=list(torch.tensor(hist['ys'][:2]).unbind(1)),
              truth=torch.tensor(hist['truth'][:2]))
    bench=type('Bench',(),dict(sigma=.5,calls=0))()
    score,_=contrast_scores(bench,data,contrasts,3,'cpu')
    exact=[]
    for row in range(2):
        hypotheses=torch.cat([data['truth'][row:row+1],contrasts])
        ll=torch.zeros(12,dtype=torch.float64)
        for t in range(30):
            action=data['actions'][t][row:row+1].double().expand(12,2)
            mean,sigma=source.deterministic(hypotheses,action)
            y=data['observations'][t][row]
            ll+=torch.distributions.Normal(mean[:,0],sigma).log_prob(y)
        exact.append(ll[0]-torch.logsumexp(ll,0)+math.log(12))
    scorer_error=float((score[:,-1]-torch.stack(exact)).abs().max())
    assert scorer_error<1e-8 and bench.calls==2*30*11
    assert a.discount==1 and not a.use_PCE and a.n_batch==10000 and a.n_newtraj==1000
    return dict(prefix_tests=prefix,mean_signal_max_error=float((target-actual[:,0]).abs().max()),
                native_loglikelihood_max_error=float((literal-native).abs().max()),
                analytic_jacobian_max_error=float((autodiff-analytic).abs().max()),
                variational_reward_telescoping_max_error=telescoping,literal_scorer_max_error=scorer_error,
                literal_negative_likelihood_terms=bench.calls,bank_shapes={k:list(bank[k].shape) for k in bank.files},
                discount=a.discount,use_PCE=a.use_PCE,actor_rows=a.n_batch*30,
                target_rate=a.lr_target,posterior_optimizer_groups=len(a.post_approx.poi_post_optimizers),
                critic_optimizer_step=float(next(iter(a.critic_optimizer.state.values()))['step']),
                posterior_stage0_optimizer_step=float(next(iter(a.post_approx.poi_post_optimizers[0][0].state.values()))['step']),
                simulator_observations_generated=0)


def specialization_checks(model,hist):
    from conditional_diagnostics import grouped_ci
    from parent_evaluation import holm
    results={}
    def matrix(features):
        return [[cka(features[:,i],features[:,j]) for j in range(4)] for i in range(4)]
    slots=hist['slots'][:,[3,12,24]].astype(float)
    results['packet_linear_CKA']=matrix(slots.reshape(-1,4,128))
    learned=np.concatenate([slots[...,:64],slots[...,76:]],-1)
    results['learned116_linear_CKA']=matrix(learned.reshape(-1,4,116))
    results['packet_CKA_by_stage']={str(t):matrix(hist['slots'][:,t]) for t in [3,12,24]}
    hx=torch.tensor(hist['X'][:,[3,12,24]].copy()).reshape(-1,117)
    hslots=torch.tensor(hist['slots'][:,[3,12,24]].copy()).reshape(-1,4,128)
    with torch.no_grad():
        history=model.history(hx);shared,u=head_parts(model,history,hslots)
        cosine=F.cosine_similarity(u[:,:,None],u[:,None],dim=-1).mean(0)
    results['consumed_hidden_contribution_cosine']=cosine.tolist()
    raw=np.load(BASE/'evaluation/competency_raw.npz');loss=raw['candidate_probe_losses']
    results['competency_array_shapes']={k:list(raw[k].shape) for k in raw.files}
    results['heldout_task_MSE_expert_rows_task_columns']=loss[:,:,256:].mean((2,3)).tolist()
    competency=json.loads((BASE/'evaluation/COMPETENCY_RESULTS.json').read_text())
    recomputed={}
    for k in range(4):
        rival=min([j for j in range(4) if j!=k],key=lambda j:loss[j,k,:256].mean())
        contrast=(loss[rival,k,256:]-loss[k,k,256:]).mean(1)
        recomputed[f'expert{k+1}']=stats(contrast)
        assert abs(recomputed[f'expert{k+1}']['mean']-competency['designated_advantages'][f'expert{k+1}']['mean'])<1e-12
    holm(recomputed,'Holm4_p');results['recomputed_designated_advantages']=recomputed
    results['existing_competency']=competency
    conditional=np.load(BASE/'evaluation/conditional_expert_native_raw.npz')
    cr=json.loads((BASE/'evaluation/CONDITIONAL_COMPETENCE.json').read_text())
    tests={}
    for j,t in enumerate([3,12,24]):
        for regime in [0,1]:
            for k in range(4):
                name=f'stage{t}_signal{regime}_expert{k+1}'
                test=grouped_ci(conditional['effects'][:,j,k].mean(-1),conditional['regimes'][:,j]==regime)
                tests[name]=test
                assert abs(test['mean']-cr['tests'][name]['mean'])<1e-12
    holm(tests,'Holm24_p')
    results['recomputed_conditional_native_tests']=tests
    # Separate exploratory interaction family on the existing case-clustered
    # native-return effects; not a new conditional-EIG measurement.
    interactions={}
    for j,t in enumerate([3,12,24]):
        for k in range(4):
            values=conditional['effects'][:,j,k].mean(-1)
            low=values[conditional['regimes'][:,j]==0];high=values[conditional['regimes'][:,j]==1]
            vlo=low.var(ddof=1)/len(low);vhi=high.var(ddof=1)/len(high)
            se=float(np.sqrt(vlo+vhi));difference=float(high.mean()-low.mean())
            df=float((vlo+vhi)**2/(vlo**2/(len(low)-1)+vhi**2/(len(high)-1)))
            width=float(student_t.ppf(.975,df))*se
            interactions[f'stage{t}_expert{k+1}']=dict(mean=difference,MCSE=se,
                CI95=[difference-width,difference+width],n_low=len(low),n_high=len(high),
                p_two_sided=float(2*student_t.sf(abs(difference/se),df)))
    holm(interactions,'Holm12_exploratory_p')
    results['exploratory_regime_interactions']=interactions
    results['conditional_significant_count']=sum(t['Holm24_p']<.05 for t in tests.values())
    results['conditional_metric']='terminal variational native return with frozen V1 posterior at14001, not exact or independently scored conditional EIG'
    results['global_gate_mean']=hist['gates'].astype(float).mean((0,1)).tolist()
    results['mean_within_stage_gate_SD']=hist['gates'].astype(float).std(0).mean(0).tolist()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path,required=True)
    args=parser.parse_args();output=args.output.expanduser().resolve()
    if output.exists():raise FileExistsError('Audit output must be new; preserve prior work')
    output.mkdir(parents=True)
    torch.set_num_threads(2);torch.manual_seed(202611126001);np.random.seed(202611126001 % (2**32))
    start=time.monotonic()
    files=list((BASE/'source').glob('*.py'))+[BASE/'arms/candidate/actor.pt',BASE/'arms/candidate/runtime.pkl',
        BASE/'initial/candidate_actor.pt',BASE/'CONFIG_LOCKED.json',BASE/'PRETRAIN_MANIFEST.json',
        BASE/'evaluation/diagnostic_candidate_histories.npz',BASE/'evaluation/competency_raw.npz',
        BASE/'evaluation/conditional_expert_native_raw.npz',BASE/'evaluation/development_validation_bank.npz']
    files+=list((BASE/'evaluation').glob('*.json'))+list((BASE/'arms/candidate').glob('*.json'))
    files+=list(ABL.glob('*.json'))+list(ABL.glob('*scores.npy'))
    protected={str(p):sha(p) for p in files}
    write(output/'INPUT_SHA256_BEFORE.json',protected)
    a=load_runtime();model=a.actor_net
    vendor=pathlib.Path(inspect.getfile(type(a))).parent
    for name in ['pg_vsoed.py','post_approx.py','models.py']:
        p=vendor/name;protected[str(p)]=sha(p)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    hist=dict(np.load(BASE/'evaluation/diagnostic_candidate_histories.npz'))
    assert hist['gates'].shape==(512,30,4)
    report=dict(actor_identity=dict(class_name=type(model).__name__,native_iteration=int(a.update),
        actor_sha=sha(BASE/'arms/candidate/actor.pt'),runtime_sha=sha(BASE/'arms/candidate/runtime.pkl')),
        module_inventory={k:inventory(v) for k,v in model.named_children()},
        critic_inventory=inventory(a.critic_net),source_vendor_path=str(vendor),
        actor_parameter_elements=sum(p.numel() for p in model.parameters()),
        actor_trainable_elements=sum(p.numel() for p in model.parameters() if p.requires_grad),
        posterior_bank={str(k):dict(parameters=sum(p.numel() for p in nets[0].parameters()),
                         first_layer=str(nets[0].feature_net[0])) for k,nets in a.post_approx.poi_post_nets.items()})
    print('AUDIT forward and intervention checks',flush=True)
    report['forward'],x=forward_checks(a,hist)
    write(output/'MEASURED_AUDIT.partial.json',report)
    print('AUDIT native gradient and checkpoint checks',flush=True)
    report['gradients']=gradient_checks(a,x)
    write(output/'MEASURED_AUDIT.partial.json',report)
    print('AUDIT native chronology, likelihood and estimator checks',flush=True)
    report['native']=native_checks(a,hist)
    write(output/'MEASURED_AUDIT.partial.json',report)
    print('AUDIT specialization and saved conditional tests',flush=True)
    report['specialization']=specialization_checks(model,hist)
    data=json.loads((ABL/'RESULTS.json').read_text());recomputed={}
    full=np.load(ABL/'full_scores.npy')
    for name in ['expert1','expert2','expert3','expert4','uniform']:
        recomputed[name]=stats(full-np.load(ABL/f'{name}_scores.npy'))
        assert abs(recomputed[name]['mean']-data['primary_comparisons'][name]['mean'])<1e-12
    from parent_evaluation import holm
    holm(recomputed,'Holm5_p');report['recomputed_EIG_comparisons']=recomputed
    report['saved_EIG_methods']=data['methods']
    report['saved_commander']=json.loads((BASE/'evaluation/COMMANDER_RESULTS.json').read_text())
    report['saved_specialization']=json.loads((BASE/'evaluation/SPECIALIZATION_RESULTS.json').read_text())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    after={p:sha(pathlib.Path(p)) for p in protected}
    assert protected==after
    report['completion']=dict(status='PASS',seconds=time.monotonic()-start,CPU_only=True,
        scientific_optimizer_steps=0,new_simulator_observations=0,model_state_unchanged=True,
        protected_file_count=len(protected),files_preserved=True)
    write(output/'INPUT_SHA256_BEFORE.json',protected);write(output/'INPUT_SHA256_AFTER.json',after)
    write(output/'MEASURED_AUDIT.json',report)
    write(output/'COMPLETION.json',report['completion'])
    print(json.dumps(report['completion']),flush=True)


if __name__=='__main__':main()
