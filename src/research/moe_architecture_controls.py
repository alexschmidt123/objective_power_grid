"""Controlled existing encoder/head/routing ablations; no auxiliary objectives."""
import copy
import json
import math
import time
from pathlib import Path
import torch
from torch import nn
from src.research.sparse_moe import SparseMoESBOED,stabilize_gradients,routing_gamma
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores

CONTROL_METHODS = ['sum_single','attention_single','attention_dense_moe','attention_sparse_moe',
                   'attention_matched_dense','attention_matched_sparse','attention_uniform_moe']
EVALUATION_METHODS = ['existing_moe_hard','existing_moe_relaxed']

class ArchitectureControl(SparseMoESBOED):
    """Keep original attention context intact, including its four queries.

    Single-head controls consume the flattened full context: the dense control
    has access to all information available to the experts and router. The
    original DAD sum encoder has a native 16-dimensional output; fixed zero
    padding embeds it into the same head input without learned adapters.
    """
    kind = 'sparse_moe_prototype'
    def __init__(self, method, horizon, hidden=64,action_scale=1.,obs_scale=3.):
        if method not in CONTROL_METHODS: raise ValueError(method)
        dense = method != 'attention_sparse_moe'
        super().__init__(2,1,horizon,hidden,k=1,dense=dense,first_action=[0.,0.])
        self.control_method = method
        self.action_scale=action_scale;self.obs_scale=obs_scale
        self.routing_override = None
        self.architecture = 'existing_encoder_routing_ablation_v1:'+method
        single = method in {'sum_single','attention_single','attention_matched_dense','attention_matched_sparse'}
        if single:
            input_dim = 4*(3*hidden+1)
            width=hidden
            if 'matched' in method:
                # Match the exact reference's TOTAL trainable parameter count.
                reference=SparseMoESBOED(2,1,horizon,hidden,dense=method.endswith('dense'),first_action=[0.,0.])
                target=sum(p.numel() for p in reference.parameters())
                encoder=sum(p.numel() for name,p in self.named_parameters() if not name.startswith(('experts.','router.')))
                def count(w):return encoder+(input_dim+1)*w+(w+1)*w+(w+1)*2
                width=min(range(1,1025),key=lambda w:abs(count(w)-target))
                self.matched_target_parameters=target
            del self.experts;del self.router
            self.head=nn.Sequential(nn.Linear(input_dim,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,2)).double()
            self.head_width=width
            if method=='sum_single':
                del self.tokens;del self.queries;del self.empty
                self.sum_encoder=nn.Sequential(nn.Linear(3,256),nn.ReLU(),nn.Linear(256,16)).double()
        if method=='attention_uniform_moe':
            for parameter in self.router.parameters():parameter.requires_grad_(False)
        self.double()
    def forward(self,features,stage):
        if stage==0:
            self.last_route=None
            return self.first_design[None].expand(len(features),-1)
        if self.control_method=='sum_single':
            slots=features[:,1:].reshape(len(features),self.horizon,4)
            # Location action_scale=1, obs_scale=3, exactly as original DAD.
            raw=torch.cat((slots[:,:stage,:2]*self.action_scale,slots[:,:stage,2:3]*self.obs_scale),-1)
            z=self.sum_encoder(raw).sum(1)
            z=torch.nn.functional.pad(z,(0,4*(3*self.hidden+1)-16))
            self.last_route=None
            return self.head(z)
        if hasattr(self,'head'):
            self.last_route=None
            return self.head(self.context(features).flatten(1))
        context=self.context(features)
        hard=(not self.training) if self.routing_override is None else self.routing_override=='hard'
        weights,selectors,bits=self.gate(context,hard=hard and not self.dense)
        self.last_route=dict(weights=weights,selectors=selectors,bits=bits)
        return self.dispatch(context,weights)

class ExistingRoutingView(SparseMoESBOED):
    def __init__(self,saved,mode):
        super().__init__(2,1,saved['horizon'],saved['hidden'],k=saved['routing_k'],first_action=[0.,0.],history_encoder_revision=saved.get('history_encoder_revision',1),action_scale=saved.get('action_scale',1.),obs_scale=saved.get('obs_scale',3.))
        self.load_state_dict(saved['state_dict']);self.router.gamma=saved['gamma']
        self.routing_mode=mode;self.eval()
    def forward(self,features,stage):
        if stage==0:return self.first_design[None].expand(len(features),-1)
        context=self.context(features)
        weights,selectors,bits=self.gate(context,hard=self.routing_mode=='hard')
        self.last_route=dict(weights=weights,selectors=selectors,bits=bits)
        return self.dispatch(context,weights)

def load_existing(path,method,horizon,raw=None):
    import hashlib
    path=Path(path).resolve()
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if raw is not None:
        source=json.loads((path.parent.parent/'run_config.json').read_text())
        if source['benchmark']['location_finding']!=raw['location_finding'] or source['benchmark']['observation']!=raw['observation']:
            raise ValueError('Existing checkpoint benchmark/design bounds differ from evaluation configuration')
    if saved['horizon']!=horizon or saved['architecture'] not in {'spce_only_shared_encoder_dselect_v1','spce_only_sum_encoder_dselect_v1'}:
        raise ValueError('Expected original sparse MoE checkpoint with matching horizon')
    mode='hard' if method=='existing_moe_hard' else 'relaxed'
    p=ExistingRoutingView(saved,mode)
    return p,dict(training='evaluation_only_existing_checkpoint',training_seconds=0.,
                  source_checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  routing_mode=mode,gamma=saved['gamma'],not_an_independent_training=True)

def component_groups(p):
    groups=dict(encoder=[],router=[],head=[],first_design=[])
    for name,param in p.named_parameters():
        if not param.requires_grad:continue
        if name=='first_design':group='first_design'
        elif name.startswith('router.'):group='router'
        elif name.startswith('experts.'):group='expert_'+name.split('.')[1]
        elif name.startswith(('head.','emitter.')):group='head'
        else:group='encoder'
        groups.setdefault(group,[]).append(param)
    return {k:v for k,v in groups.items() if v}

def grad_norm(params):
    values=[p.grad.detach().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.

def validation(b,p,h,seed,systems=128,contrasts=65536):
    # CPU history execution, common streamed likelihood bank; no optimization.
    device=next(p.parameters()).device
    p.cpu().eval()
    reports=[]
    modes=['hard','relaxed'] if p.control_method=='attention_sparse_moe' else ['default']
    with torch.no_grad(),torch.device('cpu'):
        rng=torch.Generator().manual_seed(seed)
        truth=b.sample((systems,),rng)
        noise=torch.randn(systems,h,1,generator=rng,dtype=torch.float64)
        bank=b.sample((contrasts,),rng)
        for mode in modes:
            p.routing_override=None if mode=='default' else mode
            scores=[]
            for start in range(0,systems,16):
                r=rollout(b,p,h,min(16,systems-start),seed+start*97451,0,truth=truth[start:start+16],noise=noise[start:start+16])
                info,_=contrast_scores(b,r,bank,4096)
                scores.append(info[:,-1])
            x=torch.cat(scores)
            reports.append(dict(routing=mode,eig=float(x.mean()),mc_standard_error=float(x.std()/math.sqrt(len(x)))))
    p.routing_override=None;p.to(device).train()
    return dict(seed=seed,systems=systems,contrasts=contrasts,ceiling=math.log(contrasts+1),scores=reports)

def train_control(b,h,s,seed,directory,method,trajectory_budget=None,initial_checkpoint=None):
    torch.manual_seed(seed)
    if method=='dad':
        from src.research.policies import make_policy
        p=make_policy(b,h,'dad',s['hidden'],s).train()
        p.control_method='dad';p.dense=True;p.routing_override=None
        p.architecture='original_location_dad_sum_encoder_linear_emitter'
    elif method=='sum_single_193':
        from src.research.matched_sum_single import MatchedSumSingle193
        p=MatchedSumSingle193(h,s['hidden'],b.action_scale,b.obs_scale).train()
    else:p=ArchitectureControl(method,h,s['hidden'],b.action_scale,b.obs_scale).train()
    base_provenance={}
    if initial_checkpoint:
        if method!='sum_single':raise ValueError('Additional-training control requires original sum_single')
        from src.research.moe_family import BasePolicy
        loaded=BasePolicy.from_checkpoint(initial_checkpoint,b,h,finetune=True)
        p=loaded.policy.train();base_provenance=loaded.provenance
    groups=component_groups(p)
    budget=int(trajectory_budget or s['updates']*s['batch_size'])
    opt=torch.optim.Adam(p.parameters(),lr=s['learning_rate'],betas=(s.get('spce_adam_beta1',.8),s.get('spce_adam_beta2',.998)))
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,s.get('spce_lr_decay',.98))
    records=[];validation_records=[];count=step=0;L=int(s['contrasts']);started=time.perf_counter()
    validation_seconds=0.;next_validation=budget/4
    trace_path=directory/(method+'_diagnostics.jsonl')
    while count<budget:
        n=min(s['batch_size'],budget-count);fraction=count/budget
        if not p.dense:p.router.gamma=routing_gamma(fraction,s.get('moe_router_warmup_fraction',0.),s.get('moe_router_finish_fraction',.8))
        routes=[];penalties=[]
        def hook(module,inputs,output):
            if getattr(module,'last_route',None) is not None:
                routes.append(module.last_route['weights'].detach())
                penalties.append(module.regularizers())
        handle=p.register_forward_hook(hook)
        try:r=rollout(b,p,h,n,seed+(step+1)*104729,L)
        finally:handle.remove()
        eig=r['info'][:,-1].mean()
        zero=eig*0
        load=sum(v[0] for v in penalties)/h if penalties else zero
        entropy=sum(v[1] for v in penalties)/h if penalties else zero
        # Exactly the current sparse penalties. Dense regularizers are zero so
        # architecture controls share the sPCE objective, without BA/RL losses.
        sparse=method=='attention_sparse_moe'
        loss=-eig+(1e-4*load+1e-4*fraction*entropy if sparse else zero)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite control loss')
        opt.zero_grad();loss.backward()
        before={k:grad_norm(v) for k,v in groups.items()}
        norm=stabilize_gradients(p,groups,s.get('moe_gradient_clip_mode','global'),5.)
        after={k:grad_norm(v) for k,v in groups.items()}
        opt.step();step+=1;count+=n
        if (step-1)%s.get('spce_lr_schedule_every',1000)==0:scheduler.step()
        row=dict(update=step,primary_episodes=count,contrast_trajectories=count*L,
                 training_eig=float(eig.detach()),gradient_norm=float(norm),gradient_before=before,gradient_after=after,
                 clip_scale=(min(1.,5./(float(norm)+1e-6)) if s.get('moe_gradient_clip_mode','global')=='global' else None),gradient_clip_mode=s.get('moe_gradient_clip_mode','global'),gamma=None if p.dense else p.router.gamma,
                 collapse_penalty=float(load.detach()),selector_entropy=float(entropy.detach()))
        row['training_score_distribution']=dict(mean=float(r['info'][:,-1].mean()),quantiles=torch.quantile(r['info'][:,-1].detach(),torch.tensor([0.,.1,.5,.9,.99,1.],device=eig.device,dtype=eig.dtype)).cpu().tolist(),fraction_within_01_of_ceiling=float((r['info'][:,-1].detach()>math.log(L+1)-.1).double().mean()),fraction_within_05_of_ceiling=float((r['info'][:,-1].detach()>math.log(L+1)-.5).double().mean()))
        if step%s['validate_every']==0 or count==budget:row['training_scores']=r['info'][:,-1].detach().cpu().tolist()
        if routes:
            w=torch.stack(routes);hard=torch.nn.functional.one_hot(w.argmax(-1),4).to(w.dtype)
            row.update(router_probabilities_by_stage=w.mean(1).cpu().tolist(),
                       hard_choice_frequency_by_stage=hard.mean(1).cpu().tolist(),
                       execution_utilization_by_stage=(w>0).double().mean(1).cpu().tolist(),
                       router_probabilities=w.mean((0,1)).cpu().tolist(),
                       hard_choice_frequency=hard.mean((0,1)).cpu().tolist(),
                       execution_utilization=(w>0).double().mean((0,1)).cpu().tolist(),
                       routing_entropy_by_stage=(-(w*w.clamp_min(1e-15).log()).sum(-1)).mean(1).cpu().tolist(),
                       routing_entropy=float((-(w*w.clamp_min(1e-15).log()).sum(-1)).mean()))
        with trace_path.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if step%s['validate_every']==0 or count==budget:
            records.append(row);print(method,{k:v for k,v in row.items() if not k.endswith('_by_stage')},flush=True)
        if budget>=100000 and count>=next_validation and method!='sum_single_193' and not initial_checkpoint:
            st=time.perf_counter();val=validation(b,p,h,10000120)
            validation_seconds+=time.perf_counter()-st
            val.update(primary_episodes=count,validation_seconds=time.perf_counter()-st)
            validation_records.append(val);print(method,'held-out learning curve',val,flush=True)
            next_validation+=budget/4
    sparse=method=='attention_sparse_moe'
    p.eval()
    torch.save(dict(kind=method,architecture=p.architecture,method=method,horizon=h,hidden=s['hidden'],
                    gamma=None if p.dense else p.router.gamma,state_dict=p.state_dict()),directory/(method+'.pt'))
    return p,dict(training='pathwise_sPCE_architecture_control',base_pretraining=base_provenance,additional_training_episodes=count,total_training_episodes=count+base_provenance.get('training_episodes',0),base_training_seconds=base_provenance.get('training_seconds',0.),training_seconds=time.perf_counter()-started-validation_seconds,
                  diagnostic_validation_seconds=validation_seconds,generated_training_trajectories=count,
                  contrast_training_trajectories=count*L,training_forward_trajectories=count*(L+1),
                  history=records,validation_history=validation_records,validation_not_used_for_checkpoint_selection=True,
                  diagnostic_trace=str(trace_path),policy_parameters=sum(v.numel() for v in p.parameters()),
                  trainable_parameters=sum(v.numel() for v in p.parameters() if v.requires_grad),
                  active_parameter_definition='non-first stages: shared encoder+router+one selected expert for sparse; all modules for dense; excludes separate first-action',
                  active_parameters_nonfirst=(sum(v.numel() for name,v in p.named_parameters() if not name.startswith('experts.') and name!='first_design')+sum(v.numel() for v in p.experts[0].parameters()) if sparse else sum(v.numel() for name,v in p.named_parameters() if name!='first_design')),
                  gradient_clip_mode=s.get('moe_gradient_clip_mode','global'),
                  parameter_groups={k:sum(v.numel() for v in params) for k,params in groups.items()},
                  matched_target_parameters=getattr(p,'matched_target_parameters',None),
                  checkpoint_selection='final',architecture=p.architecture,
                  routing_algorithm='unchanged_DSelect1' if sparse else 'dense_softmax' if routes else 'none')
