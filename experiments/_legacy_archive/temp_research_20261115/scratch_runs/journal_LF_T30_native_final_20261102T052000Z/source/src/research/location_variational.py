"""DAD's sequential Variational baseline, with bounded-design/history adapters.

Pinned upstream function bodies are executed unchanged. Only the fixed-design
module output is mapped into the common feasible box. No SMC design search.
"""
import ast
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import torch


@lru_cache(None)
def load_upstream(bounds=None):
    import pyro
    import pyro.distributions as dist
    from pyro.infer.autoguide import AutoDiagonalNormal
    from pyro.infer.util import torch_item
    from pyro.util import torch_isnan
    from tqdm import trange
    directory=Path(__file__).parent/'vendor'/'dad'/'variational'
    provenance=json.loads((directory/'PROVENANCE.json').read_text())
    for name,digest in provenance['sha256'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Modified upstream DAD source: '+name)
    namespace=dict(__name__='dad_variational_adapter',torch=torch,nn=torch.nn,
        pyro=pyro,dist=dist,AutoDiagonalNormal=AutoDiagonalNormal,
        torch_item=torch_item,torch_isnan=torch_isnan,
        trange=lambda *a,**kw:trange(*a,**kw,disable=True))
    # These dependency modules have no CLI/logging imports or import side effects.
    for name in ['neural/modules.py','oed/primitives.py','oed/design.py','contrastive/mi.py']:
        exec(compile((directory/name).read_text(),str(directory/name),'exec'),namespace)
    tree=ast.parse((directory/'location_finding.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HiddenObjects')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(directory/'location_finding.py'),'exec'),namespace)
    if bounds is not None:
        original=namespace['BatchDesignBaseline'];lo,hi=bounds
        class BoundedDesign(original):
            def forward(self,*pairs):
                return lo+(hi-lo)*super().forward(*pairs).sigmoid()
        namespace['BatchDesignBaseline']=BoundedDesign
    tree=ast.parse((directory/'location_finding_variational.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)
               and n.name in {'fit_posterior_vi','optimise_design'}]
    exec(compile(ast.Module(body=functions,type_ignores=[]),
                 str(directory/'location_finding_variational.py'),'exec'),namespace)
    return namespace


class LocationVariationalMyopic:
    """History-only VI followed by single-experiment PCE optimization."""
    def __init__(self,benchmark,batch,horizon,seed,settings):
        if benchmark.name!='location_finding':raise ValueError('Location-only upstream adapter')
        self.b,self.batch,self.horizon,self.seed,self.settings=benchmark,batch,horizon,seed,settings
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.bounds=tuple(benchmark.design_bounds) if benchmark.design_bounds is not None else None
        self.upstream=load_upstream(self.bounds)
        self.diagnostics=[]

    @torch.enable_grad()
    def select(self,actions,observations):
        import pyro
        if len(actions)!=len(observations):raise ValueError('Mismatched history')
        stage=len(actions);designs=[];means=[];scales=[];vi_calls=0;oed_calls=0
        if self.settings['myopic_vi_num_steps']>=1000:print(f'Myopic VI/PCE planning stage {stage+1}/{self.horizon}, batch={self.batch}',flush=True)
        for episode in range(self.batch):
            if stage==0:
                # Upstream hard-coded first design: prior mean + 0.01, not truth.
                designs.append(torch.full((2,),.01,dtype=torch.float64));continue
            devices=[torch.device(self.device).index or 0] if self.device.startswith('cuda') else []
            dtype=torch.get_default_dtype()
            original_model=self.upstream['HiddenObjects']
            benchmark=self.b
            class CountedModel(original_model):
                def forward_map(inner,xi,theta):
                    benchmark.calls+=theta.numel()//4
                    return super().forward_map(xi,theta)
            try:
                self.upstream['HiddenObjects']=CountedModel
                with torch.random.fork_rng(devices=devices),pyro.get_param_store().scope(),torch.device(self.device):
                    torch.set_default_dtype(torch.float32)
                    torch.manual_seed((self.seed+1000003*episode+104729*stage)%(2**63-1))
                    xis=[a[episode:episode+1].detach().to(device=self.device,dtype=torch.float32) for a in actions]
                    ys=[y[episode].detach().to(device=self.device,dtype=torch.float32) for y in observations]
                    before=self.b.calls
                    mean,scale=self.upstream['fit_posterior_vi'](xis,ys,
                        prior_loc=torch.zeros(2,2),prior_covmat=torch.eye(2),noise_scale=self.b.sigma,
                        num_sources=2,p=2,num_steps=self.settings['myopic_vi_num_steps'],
                        initial_lr=self.settings['myopic_vi_lr'],gamma=self.settings['myopic_vi_gamma'],
                        momentum=self.settings['myopic_vi_momentum'],device=self.device)
                    vi_calls+=self.b.calls-before
                    # Clear inference parameters before creating the separate design optimizer.
                    pyro.clear_param_store()
                    before=self.b.calls
                    design=self.upstream['optimise_design'](mean,scale,self.b.sigma,p=2,num_sources=2,
                        device=self.device,num_outer_samples=self.settings['myopic_oed_outer_samples'],
                        num_inner_samples=self.settings['myopic_oed_inner_samples'],
                        num_steps=self.settings['myopic_oed_num_steps'],initial_lr=self.settings['myopic_oed_lr'],
                        momentum=self.settings['myopic_oed_momentum'],gamma=self.settings['myopic_oed_gamma'])
                    oed_calls+=self.b.calls-before
                    if not all(torch.isfinite(x).all() for x in [mean,scale,design]):
                        raise FloatingPointError('Nonfinite DAD Variational inference/design')
                    designs.append(design.reshape(2).detach().cpu().double())
                    means.append(mean.detach().cpu().flatten().tolist())
                    scales.append(scale.detach().cpu().flatten().tolist())
            finally:
                torch.set_default_dtype(dtype)
                self.upstream['HiddenObjects']=original_model
        design=torch.stack(designs)
        if self.bounds is None:latent=design
        else:
            lo,hi=self.bounds
            latent=torch.logit(((design-lo)/(hi-lo)).clamp(1e-12,1-1e-12))
        # Counters distinguish inference calls from the expensive one-step PCE bank.
        self.diagnostics.append(dict(stage=stage,implementation='DAD_Variational_Myopic',
            objective='one_step_PCE_under_AutoDiagonalNormal_posterior',truth_excluded=True,
            episodes=[dict(posterior_family='AutoDiagonalNormal',history_pairs=stage,
                posterior_mean=None if stage==0 else means[i],
                posterior_scale=None if stage==0 else scales[i],
                vi_steps=0 if stage==0 else self.settings['myopic_vi_num_steps'],
                design_steps=0 if stage==0 else self.settings['myopic_oed_num_steps']) for i in range(self.batch)],
            estimated_one_step_eig=[None]*self.batch,
            posterior=dict(family='AutoDiagonalNormal',history_pairs=stage,
                mean_by_episode=means,scale_by_episode=scales,truth_excluded=True,
                independent_episode_updates=True,vi_forward_particle_stages=vi_calls,
                oed_forward_particle_stages=oed_calls)))
        return latent
