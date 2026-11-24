"""Information-calibrated PIMoE: same normalized regional predictive architecture.

New losses use posterior-region conditional observations and independent reference
information targets. Posterior mixture mass is never an information-based weight.
"""
import contextlib,json,math,time,os
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from src.research.pimoe import PredictiveMoE,DTYPE

@lru_cache(maxsize=16)
def quadrature_constants(nodes,device):
    x,w=np.polynomial.hermite.hermgauss(nodes)
    return torch.as_tensor(x,dtype=DTYPE,device=device),torch.as_tensor(w/math.sqrt(math.pi),dtype=DTYPE,device=device)

class InformationCalibratedPredictive(PredictiveMoE):
    def forward(self,means,anchor_response,actions,weights,stage,sigma,bounds):
        # Same network/input/output class; stabilize only conditional normalization.
        A,P=means.shape;device=next(self.parameters()).device
        means=means.to(device);anchor_response=anchor_response.to(device);actions=actions.to(device);weights=weights.to(device)
        center=(means*weights).sum(-1);variance=((means-center[:,None]).square()*weights).sum(-1);spread=(variance+sigma**2).sqrt()
        anchored=(anchor_response-anchor_response.mean(0))/(anchor_response.std(0,unbiased=False)+sigma)
        normal=(means-center[:,None])/spread[:,None]
        phi=torch.cat([normal[...,None],anchored[None].expand(A,-1,-1),torch.full((A,P,1),stage/30,device=device,dtype=DTYPE)],-1)
        log_membership=self.router(phi).log_softmax(-1)
        log_weighted=log_membership+weights.clamp_min(torch.finfo(DTYPE).tiny).log()[None,:,None]
        log_mass=torch.logsumexp(log_weighted,1);mass=log_mass.softmax(-1)
        conditional=log_weighted.softmax(1)
        region=(conditional*means[:,:,None]).sum(1);var=(conditional*(means[:,:,None]-region[:,None,:]).square()).sum(1)
        sd=(var+sigma**2).sqrt();skew=(conditional*((means[:,:,None]-region[:,None,:])/sd[:,None,:]).pow(3)).sum(1);global_skew=(weights*((means-center[:,None])/spread[:,None]).pow(3)).sum(-1)
        ww=[];mm=[];ss=[];component=[]
        for e,head in enumerate(self.experts):
            f=torch.stack([region[:,e]/3,sd[:,e].log(),skew[:,e],mass[:,e],center/3,spread.log(),global_skew,actions[:,0]/bounds,actions[:,1]/bounds,torch.full_like(center,stage/30)],-1)
            w,m,scale=head(f,region[:,e],sd[:,e],sigma);component.append(w);ww.append(mass[:,e,None]*w);mm.append(m);ss.append(scale)
        return dict(weights=torch.cat(ww,-1),means=torch.cat(mm,-1),scales=torch.cat(ss,-1),membership=log_membership.exp(),log_membership=log_membership,mass=mass,region=region,regional_variance=var,posterior_region_weights=conditional,expert_component_weights=torch.stack(component,1))

    @classmethod
    def entropy(cls,result,nodes=128):
        # Numerical integration refinement, not a new entropy/EIG definition.
        x,w=quadrature_constants(nodes,str(result['means'].device));A,K=result['means'].shape
        y=(result['means'][...,None]+math.sqrt(2)*result['scales'][...,None]*x).reshape(A,-1)
        logp=PredictiveMoE.log_prob(result,y).reshape(A,K,nodes)
        return -(logp*w*result['weights'][...,None]).sum((-1,-2))

def conditional_log_prob(z,y,experts=4,components=3):
    A,S=y.shape
    mass=z['mass']
    weight=z['expert_component_weights'] if 'expert_component_weights' in z else z['weights'].reshape(A,experts,components)/mass[:,:,None].clamp_min(1e-300)
    means=z['means'].reshape(A,experts,components);scales=z['scales'].reshape(A,experts,components)
    u=(y[:,:,None,None]-means[:,None])/scales[:,None]
    logp=-.5*u.square()-scales[:,None].log()-.5*math.log(2*math.pi)+weight[:,None].clamp_min(1e-300).log()
    return torch.logsumexp(logp,-1)

def information_calibrated_loss(model,item,cfg):
    z=model(**item['features']);device=z['means'].device;y=item['y'].to(device);indices=item['sample_indices'].to(device)
    log_membership=z['log_membership'].gather(1,indices[...,None].expand(-1,-1,4))
    normalized_membership=log_membership.softmax(1)
    logqe=conditional_log_prob(z,y);ess=normalized_membership.square().sum(1).reciprocal()
    regional=-(normalized_membership*logqe).sum(1)
    reliability=(ess/(ess+cfg['conditional_ESS_shrinkage'])).detach()
    conditional=(regional*reliability).sum()/reliability.sum().clamp_min(1e-12)
    mix=-model.log_prob(z,y).mean();predictive=(1-cfg['conditional_loss_fraction'])*mix+cfg['conditional_loss_fraction']*conditional
    estimate=model.entropy(z)-.5*math.log(2*math.pi*math.e*item['features']['sigma']**2)
    reference=item['reference'].to(device);se=item['reference_se'].to(device)
    precision=1/(1+(se/cfg['information_reference_SE_scale']).square())
    information=(precision*(estimate-reference).square()).mean()
    difference=reference[:,None]-reference[None,:];combined=(se[:,None].square()+se[None,:].square()).sqrt();resolved=torch.triu(difference.abs()>2*combined,diagonal=1)
    ranking=torch.nn.functional.softplus(-difference.sign()*(estimate[:,None]-estimate[None,:]))[resolved].mean() if resolved.any() else estimate.sum()*0
    loss=predictive+cfg['loss_info_weight']*information+cfg['loss_rank_weight']*ranking
    return loss,dict(loss=float(loss.detach()),mixture_NLL=float(mix.detach()),conditional_NLL=float(conditional.detach()),information_loss=float(information.detach()),ranking_loss=float(ranking.detach()),resolved_pairs=int(resolved.sum()),conditional_effective_samples=ess.detach().mean(0).cpu().tolist(),expert_masses=z['mass'].detach().mean(0).cpu().tolist()),z

class ExperimentalLedger:
    """Accumulates measured experimental runtimes across attempts; no clock reset.

    Source authoring is separately excluded. This is not a FLOP/active-GPU meter.
    """
    def __init__(self,path):self.path=Path(path)
    def read(self):return json.loads(self.path.read_text())
    def remaining(self):
        x=self.read();return x['experimental_runtime_allowance_seconds']-x['charged_experimental_seconds']
    @contextlib.contextmanager
    def charge(self,label):
        if self.remaining()<=0:raise RuntimeError('Original-reference experimental allowance exhausted')
        begin=time.perf_counter()
        try:yield
        finally:
            elapsed=time.perf_counter()-begin;x=self.read();x['charged_experimental_seconds']+=elapsed;x['runtime_intervals'].append(dict(phase=label,seconds=elapsed));tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));os.replace(tmp,self.path)
    def record_optimization(self,elapsed):
        x=self.read();x['charged_optimization_seconds']+=elapsed;tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));os.replace(tmp,self.path)

def load_ic_checkpoint(path,device='cpu'):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('model')!='PIMoE-sBOED-IC':raise ValueError('Not a frozen IC checkpoint')
    if saved.get('training_numerics_revision')!=2:raise ValueError('Rejected provisional checkpoint: conditional-ESS normalization defect; corrected training required')
    model=InformationCalibratedPredictive(saved['experts'],saved['width'],saved['components']);model.load_state_dict(saved['state_dict'],strict=True);return model.to(device).eval()
