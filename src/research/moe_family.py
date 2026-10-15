"""Canonical MoE-sBOED family. Legacy policy implementations remain unchanged.

The base always consumes its native representation and returns PRE-feasibility
logits. The simulator owns the single benchmark-specific feasibility mapping.
"""
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from src.research.sparse_moe import SparseMoESBOED, DSelectGate
from src.research.strategy_moe import StrategyMoEPolicy, StrategyManager, SwitchingGate, restore_strategy
from src.research.moe_architecture_controls import ArchitectureControl, ExistingRoutingView

NAMES = dict(moe='MoE-sBOED', resmoe='ResMoE-sBOED', switchmoe='SwitchMoE-sBOED', switchresmoe='SwitchResMoE-sBOED')
ALIASES = {'moe_sboed':'moe', 'strategy_moe_sboed':'switchmoe', 'Strategy-MoE':'switchmoe',
           'strategy_sum_moe_dselect1_v1':'switchmoe'}
EXPERT_IDS = ('E1','E2','E3','E4')
MODES = ('always_reroute','always_continue','learned_switch')
DSelect1Router = DSelectGate


def benchmark_signature(b):
    keys=('name','action_dim','obs_dim','sigma','action_scale','obs_scale','design_bounds',
          'population','i0','end','gap','steps','log_std','log_mean')
    result={}
    for k in keys:
        if hasattr(b,k):
            v=getattr(b,k)
            result[k]=v.detach().cpu().tolist() if torch.is_tensor(v) else v
    return result


def canonical_variant(value):
    value = ALIASES.get(value, value)
    value = {v:k for k,v in NAMES.items()}.get(value, value)
    if value not in NAMES: raise ValueError(f'Unknown variant: {value}')
    return value


def legacy_mode(mode):
    if mode == 'learned': mode = 'learned_switch'
    if mode not in MODES: raise ValueError(f'Unknown routing mode: {mode}')
    return 'learned' if mode == 'learned_switch' else mode


class PolicyExpert(nn.Sequential):
    def __init__(self, input_dim, action_dim):
        super().__init__(nn.Linear(input_dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,action_dim))
        self.double()


class ResidualExpert(PolicyExpert):
    def __init__(self,input_dim,action_dim,initial_std=.001):
        super().__init__(input_dim,action_dim)
        if initial_std <= 0: raise ValueError('Residual initialization must be small and nonidentical')
        nn.init.normal_(self[-1].weight,std=initial_std)
        nn.init.normal_(self[-1].bias,std=initial_std)


class HistoryEncoder(nn.Module):
    """Dimension-aware set encoder. Location legacy models use their exact core.

    SIR uses normalized time/count pairs, since their physical scales differ.
    Padding and stage retain the standard 193-dimensional expert interface.
    """
    def __init__(self,action_dim,obs_dim,horizon,action_scale=1.,obs_scale=1.,physical=False):
        super().__init__()
        self.action_dim,self.obs_dim,self.horizon=action_dim,obs_dim,horizon
        self.action_scale,self.obs_scale,self.physical=action_scale,obs_scale,physical
        self.sum_encoder=nn.Sequential(nn.Linear(action_dim+obs_dim,256),nn.ReLU(),nn.Linear(256,16)).double()

    def forward(self,features):
        slots=features[:,1:].reshape(len(features),self.horizon,self.action_dim+self.obs_dim+1)
        raw=slots[...,:-1]
        if self.physical:
            raw=torch.cat((raw[...,:self.action_dim]*self.action_scale,raw[...,self.action_dim:]*self.obs_scale),-1)
        z=(self.sum_encoder(raw)*slots[...,-1:]).sum(1)
        return torch.cat((torch.nn.functional.pad(z,(0,192-16)),features[:,:1]),-1)[:,None].expand(-1,4,-1)


class _GenericSumCore(SparseMoESBOED):
    def __init__(self,benchmark,horizon):
        super().__init__(benchmark.action_dim,benchmark.obs_dim,horizon,64,k=1,first_action=[0.]*benchmark.action_dim)
        del self.tokens,self.queries,self.empty
        self.set_encoder=HistoryEncoder(benchmark.action_dim,benchmark.obs_dim,horizon)
        self.history_encoder_revision=2
        self.architecture='sum_dselect1_dimension_aware_v1'
    @property
    def sum_encoder(self):return self.set_encoder.sum_encoder
    def context(self,features): return self.set_encoder(features)


class BasePolicy(nn.Module):
    """Complete native pretrained policy, never just its final action head."""
    def __init__(self,policy,provenance=None,finetune=False):
        super().__init__();self.policy=policy
        self.provenance=provenance or {'pretrained':False,'purpose':'shape-only test fixture'}
        self.finetune=bool(finetune)
        self.requires_grad_(self.finetune)
    def forward(self,features,stage): return self.policy(features,stage)

    @classmethod
    def from_checkpoint(cls,path,benchmark,horizon,finetune=False):
        path=Path(path).resolve();saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('kind')!='sum_single' or saved['horizon']!=horizon:
            raise ValueError('Base requires a compatible complete sum_single checkpoint')
        if benchmark.name!='location_finding':
            raise ValueError('No pretrained SIR DAD-MLP base registered; provide a validated benchmark-specific base')
        config=json.loads((path.parent.parent/'run_config.json').read_text())
        from src.research.benchmarks import make_benchmark
        source=make_benchmark(config['benchmark'])
        if source.design_bounds!=benchmark.design_bounds or source.sigma!=benchmark.sigma:
            raise ValueError('Base benchmark bounds/noise do not match')
        p=ArchitectureControl('sum_single',horizon,saved['hidden'],source.action_scale,source.obs_scale)
        p.load_state_dict(saved['state_dict'],strict=True)
        record=json.loads(path.with_name('sum_single_training.json').read_text())
        return cls(p,dict(pretrained=True,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            input_dim=p.head[0].in_features,benchmark=source.name,training_episodes=record['generated_training_trajectories'],
            training_seconds=record['training_seconds'],action_scale=source.action_scale,obs_scale=source.obs_scale),finetune)


class _ResidualCore(nn.Module):
    kind='sparse_moe_prototype'
    dispatch=SparseMoESBOED.dispatch
    regularizers=SparseMoESBOED.regularizers
    def __init__(self,base,benchmark,horizon,switching,mode,initial_probability,alpha,initial_std):
        super().__init__()
        self.base=base;self.action_dim=benchmark.action_dim;self.obs_dim=benchmark.obs_dim;self.horizon=horizon
        self.alpha=float(alpha)
        if not torch.isfinite(torch.tensor(alpha)) or alpha<0: raise ValueError('Invalid residual scale')
        self.router=DSelect1Router(193,1).double()
        self.experts=nn.ModuleList([ResidualExpert(193,self.action_dim,initial_std) for _ in range(4)])
        # Share the complete base encoder, without registering a second copy.
        self.extra_encoder=None if getattr(base.policy,'control_method',None)=='sum_single' else HistoryEncoder(self.action_dim,self.obs_dim,horizon)
        self.manager=StrategyManager(193,legacy_mode(mode),initial_probability) if switching else None
        if self.manager is not None:self.manager.available_strategies=EXPERT_IDS
        self.routing_override=None;self.last_route=None
        self.double()
    def context(self,features):
        p=self.base.policy
        if self.extra_encoder is not None:return self.extra_encoder(features)
        slots=features[:,1:].reshape(len(features),self.horizon,self.action_dim+self.obs_dim+1)
        raw=torch.cat((slots[...,:self.action_dim]*p.action_scale,slots[...,self.action_dim:-1]*p.obs_scale),-1)
        z=(p.sum_encoder(raw)*slots[...,-1:]).sum(1)
        return torch.cat((torch.nn.functional.pad(z,(0,192-16)),features[:,:1]),-1)[:,None].expand(-1,4,-1)
    def forward(self,features,stage):
        if stage==0:
            if self.manager is not None:self.manager.reset()
            self.last_route=None
            return self.base(features,stage)
        if stage==1 and self.manager is not None:self.manager.reset()
        base_action=self.base(features,stage)
        if self.alpha==0:
            self.last_route=None
            return base_action
        c=self.context(features)
        hard=not self.training if self.routing_override is None else self.routing_override=='hard'
        if self.manager is None:w,s,b=self.router(c.mean(1),hard=hard)
        else:w,s,b=self.manager(c.mean(1),self.router,hard=hard)
        self.last_route=dict(weights=w,selectors=s,bits=b)
        if self.manager is not None:self.last_route['strategy']=self.manager.last_diagnostics
        return base_action+self.alpha*self.dispatch(c,w)


class MoESBOEDPolicy(nn.Module):
    """Single public interface; original full-policy cores are reused verbatim."""
    kind='sparse_moe_prototype'
    def __init__(self,variant,benchmark,horizon,*,base=None,base_checkpoint=None,base_finetune=False,
                 routing_mode='learned_switch',initial_reroute_probability=.99,residual_scale=1.,residual_initial_std=.001,core=None):
        super().__init__()
        self.variant=canonical_variant(variant);self.canonical_name=NAMES[self.variant]
        self.horizon=horizon;self.benchmark_name=benchmark.name;self.benchmark_signature=benchmark_signature(benchmark)
        self.config=dict(routing_mode=routing_mode,initial_reroute_probability=initial_reroute_probability,
            residual_scale=residual_scale,residual_initial_std=residual_initial_std,base_finetune=base_finetune)
        switching=self.variant.startswith('switch');residual='resmoe' in self.variant
        if core is not None:self.core=core
        elif residual:
            if base is None:
                if base_checkpoint is None:raise ValueError('Residual variants require an explicit complete pretrained base')
                base=BasePolicy.from_checkpoint(base_checkpoint,benchmark,horizon,base_finetune)
            self.core=_ResidualCore(base,benchmark,horizon,switching,routing_mode,initial_reroute_probability,residual_scale,residual_initial_std)
        else:
            reference=SparseMoESBOED(2,1,horizon,64,k=1,first_action=[0.,0.],history_encoder_revision=2,
                action_scale=benchmark.action_scale,obs_scale=benchmark.obs_scale) if benchmark.name=='location_finding' else _GenericSumCore(benchmark,horizon)
            if switching:
                self.core=StrategyMoEPolicy(horizon,mode=legacy_mode(routing_mode),initial_reroute_probability=initial_reroute_probability,reference=reference)
                if benchmark.name!='location_finding':self.core.history_encoder=reference.set_encoder
                self.core.manager.available_strategies=EXPERT_IDS
            else:self.core=reference
        self.last_route=None;self._stats=None
        if self.manager is not None:self.manager.available_strategies=EXPERT_IDS
        self.double()
    @property
    def router(self):return self.core.router
    @property
    def experts(self):return self.core.experts
    @property
    def manager(self):return getattr(self.core,'manager',None)
    def context(self,features):return self.core.context(features)
    def regularizers(self):return self.core.regularizers()
    def set_routing(self,mode='learned_switch',relaxed=False):
        if self.manager is not None:self.manager.mode=legacy_mode(mode)
        self.core.routing_override='relaxed' if relaxed else 'hard'
        if isinstance(self.core,ExistingRoutingView):self.core.routing_mode='relaxed' if relaxed else 'hard'
        self.eval()
        return self
    def forward(self,features,stage):
        if stage==0:self._stats=None
        if isinstance(self.core,SparseMoESBOED) and not isinstance(self.core,ExistingRoutingView) and getattr(self.core,'routing_override',None) is not None and stage>0:
            c=self.core.context(features);w,s,b=self.core.gate(c,hard=self.core.routing_override=='hard')
            self.core.last_route=dict(weights=w,selectors=s,bits=b);action=self.core.dispatch(c,w)
        else:action=self.core(features,stage)
        self.last_route=self.core.last_route
        if self.last_route is not None and 'strategy' not in self.last_route:
            # Observation-only bookkeeping adds no parameters or routing changes.
            w=self.last_route['weights'];a=w.argmax(-1);initial=self._stats is None
            old=torch.full_like(a,-1) if initial else self._stats['active_strategy']
            change=torch.zeros_like(a,dtype=torch.bool) if initial else a!=old
            ones=torch.ones_like(a);true=ones.bool()
            duration=ones if initial else torch.where(change,ones,self._stats['consecutive_duration']+1)
            self._stats=dict(active_strategy=a,previous_strategy=old,routing_probabilities=w,
                proposal_probabilities=w,reroute_probability=ones.to(w.dtype),reroute_requested=true,
                actual_strategy_change=change,consecutive_duration=duration,
                strategy_changes=torch.zeros_like(a) if initial else self._stats['strategy_changes']+change.long(),
                reroute_requests=ones if initial else self._stats['reroute_requests']+1,
                initialized=torch.full_like(true,initial),router_evaluated=true,hard_state=torch.full_like(true,not self.training))
            self.last_route['strategy']=self._stats
        return action
    def checkpoint(self):
        return dict(kind='moe_sboed_family_v1',variant=self.variant,canonical_name=self.canonical_name,
            horizon=self.horizon,benchmark=self.benchmark_name,benchmark_signature=self.benchmark_signature,config=self.config,gamma=self.router.gamma,
            state_dict=self.state_dict(),base_provenance=getattr(getattr(self.core,'base',None),'provenance',None))


def load_policy(path,benchmark,*,mode='learned_switch',relaxed=False):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('benchmark_signature') and saved['benchmark_signature']!=benchmark_signature(benchmark):
        raise ValueError('Checkpoint benchmark physics/likelihood/feasibility do not match')
    if saved.get('kind')!='moe_sboed_family_v1':
        metadata=Path(path).resolve().parent.parent/'run_config.json'
        if metadata.exists():
            from src.research.benchmarks import make_benchmark
            source=make_benchmark(json.loads(metadata.read_text())['benchmark'])
            if benchmark_signature(source)!=benchmark_signature(benchmark):raise ValueError('Legacy checkpoint benchmark mismatch')
        elif benchmark.name!='location_finding':raise ValueError('Legacy location checkpoint cannot be loaded for another benchmark')
    if saved.get('kind')=='moe_sboed_family_v1':
        base=None
        if saved.get('base_provenance'):
            # Reconstruct the complete native base from checkpoint-contained weights;
            # no dependency on the original base file after training.
            if benchmark.name!='location_finding':raise ValueError('SIR base serialization requires a registered native base type')
            base=BasePolicy(ArchitectureControl('sum_single',saved['horizon'],64,benchmark.action_scale,benchmark.obs_scale),saved['base_provenance'],saved['config']['base_finetune'])
        p=MoESBOEDPolicy(saved['variant'],benchmark,saved['horizon'],base=base,**saved['config'])
        p.load_state_dict(saved['state_dict'],strict=True)
    elif saved.get('kind')=='strategy_moe_sboed':
        p=MoESBOEDPolicy('switchmoe',benchmark,saved['horizon'],core=restore_strategy(saved))
    elif saved.get('architecture')=='spce_only_sum_encoder_dselect_v1':
        p=MoESBOEDPolicy('moe',benchmark,saved['horizon'],core=ExistingRoutingView(saved,'hard'))
    else:raise ValueError('Unsupported checkpoint; historical attention models are not silently relabeled')
    p.router.gamma=saved['gamma']
    return p.set_routing(mode,relaxed)


def component_groups(policy):
    groups={}
    for name,p in policy.named_parameters():
        if '.experts.' in name:group='expert_'+name.split('.experts.')[1].split('.')[0]
        elif '.manager.switching_gate.' in name:group='switching_gate'
        elif '.router.' in name:group='router'
        elif '.base.' in name:group='base'
        elif name.endswith('first_design'):group='first_design'
        else:group='encoder'
        groups.setdefault(group,[]).append(p)
    return groups


def measured_parameters(policy,features,stage):
    """Count unique parameters actually touched by a BATCH-ONE hard forward.

    First-design Parameters are accessed directly; all other weights belong to
    modules with registered forward hooks. No architecture-based estimates.
    """
    if len(features)!=1:raise ValueError('Use one trajectory to measure active sparse parameters')
    touched=set();handles=[]
    for m in policy.modules():
        def hook(module,inputs,output):
            touched.update(id(p) for n,p in module.named_parameters(recurse=False) if n!='first_design')
        handles.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():policy(features,stage)
    finally:
        for h in handles:h.remove()
    if stage==0:touched.update(id(p) for n,p in policy.named_parameters() if n.endswith('first_design'))
    params=list(policy.parameters());pretrained=list(policy.core.base.parameters()) if hasattr(policy.core,'base') else []
    return dict(total=sum(p.numel() for p in params),active=sum(p.numel() for p in params if id(p) in touched),
        trainable=sum(p.numel() for p in params if p.requires_grad),pretrained=sum(p.numel() for p in pretrained),
        newly_initialized=sum(p.numel() for p in params if id(p) not in {id(v) for v in pretrained}))
