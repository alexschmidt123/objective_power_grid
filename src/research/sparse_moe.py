"""sPCE-only sparse MoE-sBOED policy.

DSelect-k adaptation for four independently parameterized policy experts.
Training relaxes two selector bits; inference executes <=k unique experts.
No posterior, Q-value, information prediction, or branch-specific objectives.
"""
import torch
from torch import nn
from src.research.shared_history import MoEPolicy

class DSelectGate(nn.Module):
    def __init__(self,context_dim,k=1,gamma=1.):
        super().__init__()
        if k not in (1,2) or gamma<=0:raise ValueError('Require k=1/2 and positive gamma')
        self.k,self.gamma=k,float(gamma)
        self.bits=nn.Linear(context_dim,2*k)
        self.mixing=nn.Linear(context_dim,k)
        nn.init.normal_(self.bits.weight,std=.001);nn.init.uniform_(self.bits.bias,-.05,.05)
        nn.init.zeros_(self.mixing.weight);nn.init.zeros_(self.mixing.bias)
        codes=torch.tensor([[0,0],[0,1],[1,0],[1,1]],dtype=torch.float64)
        self.register_buffer('codes',codes)
    def forward(self,context,hard=False):
        logits=self.bits(context).reshape(-1,self.k,2)
        if hard:bits=(logits>=0).to(logits.dtype)
        else:
            t=(logits/self.gamma).clamp(-.5,.5)
            bits=-2*t.pow(3)+1.5*t+.5
        selectors=torch.where(self.codes[None,None].bool(),bits[:,:,None],1-bits[:,:,None]).prod(-1)
        mix=self.mixing(context).softmax(-1)
        weights=(selectors*mix[:,:,None]).sum(1)
        return weights,selectors,bits

class SparseMoESBOED(MoEPolicy):
    architecture='spce_only_shared_encoder_dselect_v1'
    kind='sparse_moe_prototype'
    def __init__(self,action_dim,obs_dim,horizon,hidden=64,k=1,dense=False,first_action=None,
                 history_encoder_revision=1,action_scale=1.,obs_scale=3.):
        if history_encoder_revision not in (1,2):raise ValueError('Unknown history encoder revision')
        if history_encoder_revision==2 and (action_dim,obs_dim)!=(2,1):
            raise ValueError('Sum-encoder correction is location-specific')
        if history_encoder_revision==2 and 3*hidden<16:raise ValueError('Sum encoder requires context capacity for16 features')
        super().__init__(action_dim,obs_dim,horizon,hidden,'moe_sboed',1.,first_action)
        self.kind='sparse_moe_prototype'
        self.dense=dense;self.k=k
        if not dense:self.router=DSelectGate(3*hidden+1,k).double()
        self.last_route=None
        self.history_encoder_revision=history_encoder_revision
        self.action_scale,self.obs_scale=action_scale,obs_scale
        if history_encoder_revision==2:
            # Restore the original location DAD set encoder; retain the four
            # existing expert heads and DSelect1 router, with fixed padding.
            del self.tokens;del self.queries;del self.empty
            self.sum_encoder=nn.Sequential(nn.Linear(3,256),nn.ReLU(),nn.Linear(256,16)).double()
            self.architecture='spce_only_sum_encoder_dselect_v1'
    def context(self,features):
        if self.history_encoder_revision==1:return super().context(features)
        slots=features[:,1:].reshape(len(features),self.horizon,4)
        raw=torch.cat((slots[...,:2]*self.action_scale,slots[...,2:3]*self.obs_scale),-1)
        z=(self.sum_encoder(raw)*(slots[...,3:4]>0)).sum(1)
        # Same 193-dimensional interface and stage coordinate as the original
        # experts. Padding introduces no learned adapter or extra information.
        z=torch.nn.functional.pad(z,(0,3*self.hidden-16))
        return torch.cat((z,features[:,:1]),-1)[:,None,:].expand(-1,4,-1)
    def gate(self,context,hard=False):
        if self.dense:
            w=self.router(context.mean(1)).softmax(-1)
            return w,None,None
        return self.router(context.mean(1),hard=hard)
    def dispatch(self,context,weights):
        if not torch.isfinite(weights).all():raise FloatingPointError('Nonfinite sparse gate')
        output=context.new_zeros(len(context),self.action_dim)
        for index,expert in enumerate(self.experts):
            rows=(weights[:,index]>0).nonzero().flatten()
            if len(rows):
                value=expert(context[rows,index])*weights[rows,index,None]
                output=output.index_add(0,rows,value)
        return output
    def forward(self,features,stage):
        if hasattr(self,'first_design') and stage==0:
            self.last_route=None
            return self.first_design[None].expand(len(features),-1)
        context=self.context(features)
        weights,selectors,bits=self.gate(context,hard=not self.training and not self.dense)
        self.last_route=dict(weights=weights,selectors=selectors,bits=bits)
        return self.dispatch(context,weights)
    def regularizers(self,collapse_threshold=.75):
        """Advisory mild anti-collapse and selector-hardening terms, separate from EIG.

        No equal-use constraint. Coefficients/schedules require validation.
        """
        if self.last_route is None:
            zero=next(self.parameters()).sum()*0
            return zero,zero
        w=self.last_route['weights'];load=(w.mean(0)-collapse_threshold).clamp_min(0).square().sum()
        q=self.last_route['selectors']
        entropy=w.sum()*0 if q is None else -(q*q.clamp_min(1e-15).log()).sum(-1).mean()
        return load,entropy

class MatchedDensePolicy(SparseMoESBOED):
    """DAD-style single policy head with the same shared history encoder.

    This is an architecture control; original DAD remains an external baseline.
    Count all trainable policy parameters, including the reference router.
    """
    def __init__(self,action_dim,obs_dim,horizon,hidden=64,first_action=None):
        super().__init__(action_dim,obs_dim,horizon,hidden,dense=True,first_action=first_action)
        target=sum(p.numel() for p in self.experts.parameters())+sum(p.numel() for p in self.router.parameters())
        inputs=4*(3*hidden+1)
        def count(w):return (inputs+1)*w+(w+1)*w+(w+1)*action_dim
        width=min(range(1,16*hidden+1),key=lambda w:abs(count(w)-target))
        del self.experts;del self.router
        self.head=nn.Sequential(nn.Linear(inputs,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,action_dim)).double()
        self.architecture='shared_history_parameter_matched_dense_policy_control'
    def forward(self,features,stage):
        if hasattr(self,'first_design') and stage==0:return self.first_design[None].expand(len(features),-1)
        return self.head(self.context(features).flatten(1))


def routing_gamma(fraction, warmup=0., finish=.8):
    """Preserve legacy homotopy by default; delayed cosine schedule is optional."""
    import math
    if not 0 <= warmup < finish <= 1:raise ValueError('Invalid routing schedule')
    if warmup==0. and finish==.8:
        return max(.001,(1-min(fraction/.8,1))**2)
    progress=min(max((fraction-warmup)/(finish-warmup),0.),1.)
    return .001+.999*.5*(1+math.cos(math.pi*progress))


def stabilize_gradients(policy, groups, mode='global', limit=5.):
    """Clip disjoint component groups without suppressing unrelated gradients."""
    if mode not in ('global','component'):raise ValueError('Unknown clipping mode')
    if mode=='global':
        return float(torch.nn.utils.clip_grad_norm_(policy.parameters(),limit,error_if_nonfinite=True))
    parameters=[v for g in groups.values() for v in g]
    if len({id(v) for v in parameters})!=len(parameters):raise ValueError('Gradient groups overlap')
    if {id(v) for v in parameters}!={id(v) for v in policy.parameters()}:raise ValueError('Gradient groups must cover the policy')
    squared=0.
    for group in groups.values():
        norm=float(torch.nn.utils.clip_grad_norm_(group,limit,error_if_nonfinite=True))
        squared+=norm*norm
    return squared**.5


def train_sparse_moe(b,h,s,seed,directory,trajectory_budget=None):
    import math,time,json
    from src.research.moe_architecture_controls import component_groups,grad_norm
    from src.research.engine import rollout
    torch.manual_seed(seed);start=time.perf_counter()
    budget=int(trajectory_budget or (s['updates']*s['batch_size']))
    revision=int(s.get('moe_history_encoder_revision',1))
    if revision==2 and b.name!='location_finding':raise ValueError('Sum correction is location-only')
    p=SparseMoESBOED(b.action_dim,b.obs_dim,h,s['hidden'],k=1,first_action=[0.]*b.action_dim,
                    history_encoder_revision=revision,action_scale=b.action_scale,obs_scale=b.obs_scale)
    opt=torch.optim.Adam(p.parameters(),lr=s['learning_rate'],betas=(s.get('spce_adam_beta1',.8),s.get('spce_adam_beta2',.998)))
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,s.get('spce_lr_decay',.98))
    records=[];count=0;step=0;contrasts=int(s['contrasts'])
    groups=component_groups(p);trace=directory/'moe_sboed_diagnostics.jsonl'
    while count<budget:
        n=min(s['batch_size'],budget-count);fraction=count/budget
        # Smooth homotopy to binary DSelect routing; terminal phase uses sparse
        # selectors. The sPCE objective remains the information criterion.
        p.router.gamma=routing_gamma(fraction,s.get('moe_router_warmup_fraction',0.),s.get('moe_router_finish_fraction',.8))
        penalties=[];routes=[]
        def capture(module,inputs,output):
            penalties.append(module.regularizers())
            if module.last_route is not None:routes.append(module.last_route['weights'].detach())
        handle=p.register_forward_hook(capture)
        try:r=rollout(b,p,h,n,seed+(step+1)*104729,contrasts)
        finally:handle.remove()
        eig=r['info'][:,-1].mean()
        load=sum(v[0] for v in penalties)/h;entropy=sum(v[1] for v in penalties)/h
        loss=-eig+1e-4*load+1e-4*fraction*entropy
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sparse MoE loss')
        opt.zero_grad();loss.backward()
        before={k:grad_norm(v) for k,v in groups.items()}
        norm=stabilize_gradients(p,groups,s.get('moe_gradient_clip_mode','global'),5.)
        after={k:grad_norm(v) for k,v in groups.items()}
        opt.step();step+=1;count+=n
        if (step-1)%s.get('spce_lr_schedule_every',1000)==0:scheduler.step()
        row=dict(update=step,primary_episodes=count,training_eig=float(eig.detach()),selector_entropy=float(entropy.detach()),collapse_penalty=float(load.detach()),gamma=p.router.gamma,gradient_norm=float(norm),gradient_before=before,gradient_after=after,clip_scale=(min(1.,5./(float(norm)+1e-6)) if s.get('moe_gradient_clip_mode','global')=='global' else None),gradient_clip_mode=s.get('moe_gradient_clip_mode','global'),component_clip_scales={k:min(1.,5./(v+1e-6)) for k,v in before.items()})
        scores=r['info'][:,-1].detach()
        row['training_score_distribution']=dict(quantiles=torch.quantile(scores,torch.tensor([0.,.1,.5,.9,.99,1.],device=scores.device,dtype=scores.dtype)).cpu().tolist(),fraction_within_01_of_ceiling=float((scores>math.log(contrasts+1)-.1).double().mean()),fraction_within_05_of_ceiling=float((scores>math.log(contrasts+1)-.5).double().mean()))
        if step%s['validate_every']==0 or count==budget:row['training_scores']=scores.cpu().tolist()
        if routes:
            w=torch.stack(routes);choices=torch.nn.functional.one_hot(w.argmax(-1),4).to(w.dtype)
            row.update(router_probabilities_by_stage=w.mean(1).cpu().tolist(),hard_choice_frequency_by_stage=choices.mean(1).cpu().tolist(),execution_utilization_by_stage=(w>0).double().mean(1).cpu().tolist(),router_probabilities=w.mean((0,1)).cpu().tolist(),hard_choice_frequency=choices.mean((0,1)).cpu().tolist(),execution_utilization=(w>0).double().mean((0,1)).cpu().tolist(),routing_entropy=float((-(w*w.clamp_min(1e-15).log()).sum(-1)).mean()))
        with trace.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if step%s['validate_every']==0 or count==budget:
            records.append(row);print('moe_sboed',{k:v for k,v in row.items() if not k.endswith('_by_stage')},flush=True)
    p.eval()
    torch.save(dict(kind='moe_sboed',architecture=p.architecture,horizon=h,hidden=s['hidden'],routing_k=1,gamma=p.router.gamma,state_dict=p.state_dict(),history_encoder_revision=revision,action_scale=b.action_scale,obs_scale=b.obs_scale),directory/'moe_sboed.pt')
    return p,dict(training='pathwise_sPCE_sparse_DSelect1',training_seconds=time.perf_counter()-start,generated_training_trajectories=count,contrast_training_trajectories=count*contrasts,training_forward_trajectories=count*(contrasts+1),training_trajectory_budget=budget,training_budget_unit='primary_episodes',history=records,policy_parameters=sum(v.numel() for v in p.parameters()),active_parameters_nonfirst=sum(v.numel() for n,v in p.named_parameters() if not n.startswith('experts.') and n!='first_design')+sum(v.numel() for v in p.experts[0].parameters()),history_encoder_revision=revision,effective_nonpadding_parameters=sum(v.numel() for v in p.parameters())-(4*s['hidden']+3)*(3*s['hidden']-16) if revision==2 else sum(v.numel() for v in p.parameters()),checkpoint_selection='final',diagnostic_trace=str(trace),routing_k=1,routing_warmup_fraction=s.get('moe_router_warmup_fraction',0.),routing_finish_fraction=s.get('moe_router_finish_fraction',.8),gradient_clip_mode=s.get('moe_gradient_clip_mode','global'),load_penalty_weight=1e-4,selector_entropy_weight_max=1e-4,hard_routing_at_evaluation=True,architecture=p.architecture)
