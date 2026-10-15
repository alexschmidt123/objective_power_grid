"""Persistent, observation-adaptive strategies; existing sPCE/physics unchanged."""
import math
import torch
from torch import nn
from src.research.sparse_moe import SparseMoESBOED, DSelectGate

DSelect1Router = DSelectGate
ROUTING_MODES = ('always_reroute', 'always_continue', 'learned')


class HistoryEncoder(nn.Module):
    """Reuse the corrected sum context, including its original normalization."""
    def __init__(self, reference):
        super().__init__()
        self.sum_encoder = reference.sum_encoder
        for name in ('horizon','hidden','action_scale','obs_scale','history_encoder_revision'):
            setattr(self,name,getattr(reference,name))

    def forward(self,features):
        return SparseMoESBOED.context(self,features)


class SwitchingGate(nn.Module):
    """Stage is the last history-context coordinate; previous state stays soft."""
    def __init__(self,context_dim,initial_reroute_probability=.99):
        super().__init__()
        if not 0 < initial_reroute_probability < 1:
            raise ValueError('Initial rerouting probability must be strictly between 0 and 1')
        self.logit=nn.Linear(context_dim+4,1).double()
        nn.init.zeros_(self.logit.weight)
        nn.init.constant_(self.logit.bias,math.log(initial_reroute_probability/(1-initial_reroute_probability)))

    def forward(self,context,previous_state):
        return self.logit(torch.cat((context,previous_state),-1)).sigmoid()


class StrategyManager(nn.Module):
    """Four neutral strategy slots; ephemeral state is reset at each rollout.

    Hard requests use b>=0.5 (no sampling/straight-through estimator). Relaxed
    decisions use q=(1-b)q_previous+b*p without detaching either occurrence of
    q_previous. Rerouting requests and actual expert changes are distinct.
    """
    available_strategies=('S0','S1','S2','S3')
    def __init__(self,context_dim,mode='learned',initial_reroute_probability=.99):
        super().__init__()
        if mode not in ROUTING_MODES:raise ValueError('Unknown strategy routing mode')
        self.mode=mode
        self.switching_gate=SwitchingGate(context_dim,initial_reroute_probability)
        self.reset()

    def reset(self):
        self.state=None
        self.active=None
        self.previous_active=None
        self.duration=None
        self.change_count=None
        self.request_count=None
        self.last_diagnostics=None

    def forward(self,context,router,hard=False):
        n=len(context);initial=self.state is None
        if not initial and len(self.state)!=n:
            raise ValueError('Batch changed without resetting strategy state')
        previous=self.state
        previous_active=torch.full((n,),-1,dtype=torch.long,device=context.device) if initial else self.active
        if initial:
            b=context.new_ones(n,1);requested=torch.ones(n,dtype=torch.bool,device=context.device)
        elif self.mode=='always_reroute':
            b=context.new_ones(n,1);requested=torch.ones(n,dtype=torch.bool,device=context.device)
        elif self.mode=='always_continue':
            b=context.new_zeros(n,1);requested=torch.zeros(n,dtype=torch.bool,device=context.device)
        else:
            b=self.switching_gate(context,previous)
            requested=b[:,0]>=.5
        if hard:
            # Do not invoke DSelect for continued rows. Carried p entries are
            # state probabilities, marked router_evaluated=False in the record.
            p=context.new_zeros(n,4) if initial else previous.clone()
            indices=requested.nonzero().flatten()
            if len(indices):
                fresh,_,_=router(context[indices],hard=True)
                p=p.index_copy(0,indices,fresh)
            q=p
            selectors=None;bits=None
        elif not initial and self.mode=='always_continue':
            p=previous;q=previous;selectors=None;bits=None
        else:
            p,selectors,bits=router(context,hard=False)
            q=p if initial or self.mode=='always_reroute' else (1-b)*previous+b*p
        active=q.argmax(-1)
        changed=torch.zeros(n,dtype=torch.bool,device=context.device) if initial else active!=previous_active
        duration=torch.ones(n,dtype=torch.long,device=context.device) if initial else torch.where(changed,1,self.duration+1)
        changes=torch.zeros_like(duration) if initial else self.change_count+changed.long()
        requests=requested.long() if initial else self.request_count+requested.long()
        self.previous_active=previous_active
        self.state=q;self.active=active;self.duration=duration;self.change_count=changes;self.request_count=requests
        self.last_diagnostics=dict(active_strategy=active,previous_strategy=previous_active,
            routing_probabilities=q,proposal_probabilities=p,reroute_probability=b[:,0],
            reroute_requested=requested,actual_strategy_change=changed,
            consecutive_duration=duration,strategy_changes=changes,reroute_requests=requests,
            initialized=torch.full_like(requested,initial),
            router_evaluated=requested if hard else torch.full_like(requested,not (not initial and self.mode=='always_continue')),
            hard_state=torch.full_like(requested,hard))
        return q,selectors,bits


class StrategyMoEPolicy(nn.Module):
    kind='sparse_moe_prototype'
    architecture='strategy_sum_moe_dselect1_v1'
    def __init__(self,horizon,hidden=64,mode='learned',initial_reroute_probability=.99,
                 action_scale=4.,obs_scale=3.,reference=None):
        super().__init__()
        base=reference if reference is not None else SparseMoESBOED(2,1,horizon,hidden,k=1,
            first_action=[0.,0.],history_encoder_revision=2,action_scale=action_scale,obs_scale=obs_scale)
        if base.history_encoder_revision!=2 or base.k!=1 or base.dense:
            raise ValueError('Strategy model requires corrected sum DSelect1 MoE')
        self.horizon,self.hidden=base.horizon,base.hidden
        self.action_dim=base.action_dim;self.obs_dim=base.obs_dim
        self.action_scale,self.obs_scale=base.action_scale,base.obs_scale
        self.history_encoder=HistoryEncoder(base)
        self.router=base.router;self.experts=base.experts;self.first_design=base.first_design
        self.manager=StrategyManager(3*self.hidden+1,mode,initial_reroute_probability)
        self.initial_reroute_probability=initial_reroute_probability
        self.routing_override=None;self.last_route=None
        self.double()

    def context(self,features):return self.history_encoder(features)
    dispatch=SparseMoESBOED.dispatch
    regularizers=SparseMoESBOED.regularizers

    def forward(self,features,stage):
        if stage==0:
            self.manager.reset();self.last_route=None
            return self.first_design[None].expand(len(features),-1)
        if stage==1:self.manager.reset()
        context=self.context(features)
        hard=not self.training if self.routing_override is None else self.routing_override=='hard'
        weights,selectors,bits=self.manager(context.mean(1),self.router,hard=hard)
        self.last_route=dict(weights=weights,selectors=selectors,bits=bits,strategy=self.manager.last_diagnostics)
        return self.dispatch(context,weights)


def component_groups(policy):
    groups=dict(encoder=list(policy.history_encoder.parameters()),router=list(policy.router.parameters()),
        switching_gate=list(policy.manager.switching_gate.parameters()),first_design=[policy.first_design])
    groups.update({f'expert_{j}':list(expert.parameters()) for j,expert in enumerate(policy.experts)})
    return groups


def restore_strategy(saved,mode=None,relaxed=False):
    p=StrategyMoEPolicy(saved['horizon'],saved['hidden'],mode or saved['routing_mode'],
        saved['initial_reroute_probability'],saved['action_scale'],saved['obs_scale'])
    p.load_state_dict(saved['state_dict']);p.router.gamma=saved['gamma'];p.eval()
    p.routing_override='relaxed' if relaxed else 'hard'
    return p
