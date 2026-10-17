import pathlib,shutil,json,hashlib,subprocess
import yaml
R=pathlib.Path('/home/grads/g/g.lin/Documents/objective_power_grid')
old=R/'experiments/10162026/10162026_location_root_cause_v4'
C=R/'experiments/10172026/10172026_moe_st_routing_dev_103_105';C.mkdir(parents=True,exist_ok=False)
S=C/'source'
def ignore(directory,names):
 return set(names)&{'experiments','data','models','.git','.venv_scorebed','__pycache__','.pytest_cache','AGENTS.md'}
shutil.copytree(old/'source',S,ignore=ignore)
for d in ['logs','scripts','reports','commands','completion']: (C/d).mkdir()
def edit(rel,a,b):
 p=S/rel;s=p.read_text();assert s.count(a)==1,(rel,a[:80],s.count(a));p.write_text(s.replace(a,b))
edit('src/research/sparse_moe.py','class SparseMoESBOED(MoEPolicy):', '''    def straight_through(self,context):
        """Hard DSelect-1 forward, native relaxed-selector surrogate backward.

        Return relaxed weights separately for unchanged routing regularization.
        Native >=0 logit tie convention is retained exactly. No extra RNG draws.
        """
        if self.k!=1:raise ValueError('Straight-through mode requires DSelect-1')
        logits=self.bits(context).reshape(-1,1,2)
        t=(logits/self.gamma).clamp(-.5,.5)
        bits=-2*t.pow(3)+1.5*t+.5
        selectors=torch.where(self.codes[None,None].bool(),bits[:,:,None],1-bits[:,:,None]).prod(-1)
        mix=self.mixing(context).softmax(-1)
        soft=(selectors*mix[:,:,None]).sum(1)
        hard_bits=(logits>=0).to(logits.dtype)
        hard_selectors=torch.where(self.codes[None,None].bool(),hard_bits[:,:,None],1-hard_bits[:,:,None]).prod(-1)
        hard=(hard_selectors*mix[:,:,None]).sum(1)
        weights=hard.detach()+(soft-soft.detach())
        return weights,soft,selectors,bits

class SparseMoESBOED(MoEPolicy):''')
edit('src/research/sparse_moe.py',"w=self.last_route['weights'];load=", "w=self.last_route.get('regularization_weights',self.last_route['weights']);load=")
edit('src/research/moe_family.py','residual_scale=1.,residual_initial_std=.001,core=None):',"residual_scale=1.,residual_initial_std=.001,core=None,training_routing_mode='relaxed'):")
edit('src/research/moe_family.py','        switching=self.variant.startswith(\'switch\');residual=', '''        if training_routing_mode not in ('relaxed','straight_through'):
            raise ValueError('Unknown MoE training routing mode')
        if training_routing_mode!='relaxed':
            if self.variant!='moe':raise ValueError('Straight-through experiment is restricted to moe')
            self.config['training_routing_mode']=training_routing_mode
        switching=self.variant.startswith('switch');residual=''')
edit('src/research/moe_family.py',"        if isinstance(self.core,SparseMoESBOED) and not isinstance(self.core,ExistingRoutingView) and getattr(self.core,'routing_override',None) is not None and stage>0:", '''        if stage>0 and self.training and self.config.get('training_routing_mode','relaxed')=='straight_through':
            c=self.core.context(features)
            w,soft,s,b=self.router.straight_through(c.mean(1))
            self.core.last_route=dict(weights=w,regularization_weights=soft,selectors=s,bits=b)
            # Compute ALL expert proposals for router surrogate gradients. Do not
            # use sparse dispatch here: it removes unselected expert contributions
            # to the routing derivative. Only the hard-selected action is emitted.
            proposals=torch.stack([expert(c[:,j]) for j,expert in enumerate(self.experts)],1)
            action=(proposals*w[:,:,None]).sum(1)
        elif isinstance(self.core,SparseMoESBOED) and not isinstance(self.core,ExistingRoutingView) and getattr(self.core,'routing_override',None) is not None and stage>0:''')
edit('src/research/family_training.py',"        residual_scale=s.get('residual_scale',1.),residual_initial_std=s.get('residual_initial_std',.001))", "        residual_scale=s.get('residual_scale',1.),residual_initial_std=s.get('residual_initial_std',.001),\n        training_routing_mode=s.get('moe_training_routing_mode','relaxed'))")
edit('src/research/family_training.py','        penalties=[];routes=[];strategies=[]','        penalties=[];routes=[];strategies=[];soft_routes=[]')
edit('src/research/family_training.py',"                routes.append(module.last_route['weights'].detach())", "                routes.append(module.last_route['weights'].detach())\n                if 'regularization_weights' in module.last_route:soft_routes.append(module.last_route['regularization_weights'].detach())")
edit('src/research/family_training.py',"        if strategies and (step%int(s.get('strategy_diagnostics_every',1))==0 or count==budget):", "        if soft_routes:\n            sw=torch.stack(soft_routes)\n            row.update(relaxed_router_probabilities=sw.mean((0,1)).cpu().tolist(),relaxed_router_probabilities_by_stage=sw.mean(1).cpu().tolist(),training_routing_mode='straight_through',training_expert_proposal_execution='all_four_for_router_surrogate',execution_utilization_interpretation='nonzero forward weights; all four proposals computed for backward')\n        if strategies and (step%int(s.get('strategy_diagnostics_every',1))==0 or count==budget):")
# Configurable checkpoint observation only; default v4 behavior unchanged.
edit('src/research/moe_diagnostic_logging.py',"        if self.start==0:self.targets=[64,128,192,256]", "        if self.start==0:self.targets=[64,128,192,256]\n        if 'diagnostic_checkpoint_episodes' in settings:self.targets=sorted(set(map(int,settings['diagnostic_checkpoint_episodes'])))")
cfg=yaml.safe_load((old/'original.yaml').read_text());jp=cfg['research']['journal_protocol']
jp.update(status='controlled_straight_through_development',development_training_seeds=[103,105],diagnostic_only=True)
jp['common_pathwise_settings'].update(diagnostic_late_logging=True,diagnostic_start_episodes=0,diagnostic_checkpoint_episodes=[125000,250000,500000,750000,775000,800000,812544,825024,832000,850000,875000,883200,900000,925000,950000,975000,1000000])
jp['method_settings']['moe']['moe_training_routing_mode']='straight_through'
(S/'configs/moe_st_development.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
(C/'original.yaml').write_bytes((old/'original.yaml').read_bytes())
changed=['src/research/sparse_moe.py','src/research/moe_family.py','src/research/family_training.py','src/research/moe_diagnostic_logging.py']
(C/'implementation_changes.json').write_text(json.dumps(dict(changed=changed,original_source=str(old/'source'),isolated_source=str(S),new_named_variant=False,regularization='unchanged native relaxed load and selector entropy',gradient_surrogate='biased straight-through routing derivative; same sPCE estimator'),indent=2)+'\n')
(C/'status.json').write_text(json.dumps(dict(state='prepared_pending_correctness',seeds=[103,105]),indent=2)+'\n')
print('PREPARED',C)
