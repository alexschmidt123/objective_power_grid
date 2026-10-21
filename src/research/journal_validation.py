"""Shared held-out scoring and noise-aware plateau decisions (not an EIG estimator)."""
import copy,hashlib,io,json,math,time
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
import torch

class JournalStop(Exception):
    """Controlled stop requested after a complete native optimizer iteration."""

@dataclass(frozen=True)
class StoppingRule:
    minimum_episodes:int=500000
    minimum_checkpoints:int=8
    validation_every_episodes:int=125000
    patience:int=6
    minimum_improvement:float=.01
    noise_multiplier:float=1.96
    maximum_episodes:int=10000000
    maximum_training_seconds:float=7200.
    def __post_init__(self):
        for key in ['minimum_episodes','minimum_checkpoints','validation_every_episodes','patience','maximum_episodes']:
            if getattr(self,key)<1:raise ValueError('Stopping counts must be positive')
        if self.maximum_episodes<self.minimum_episodes or self.maximum_training_seconds<=0 or self.minimum_improvement<0 or self.noise_multiplier<0:raise ValueError('Invalid stopping rule')

class Plateau:
    def __init__(self,rule):self.rule=rule;self.records=[];self.best=None;self.anchor=None;self.stale=0
    def update(self,scores,episodes,seconds):
        x=np.asarray(scores,dtype=float)
        if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Finite paired validation scores required')
        if self.records and episodes<=self.records[-1]['episodes']:raise ValueError('Validation checkpoints must advance training')
        if self.anchor is not None and x.shape!=self.anchor.shape:raise ValueError('Validation population changed')
        mean=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)))
        selected=self.best is None or mean>self.best
        if selected:self.best=mean
        delta=None;threshold=None
        if self.anchor is None:self.anchor=x.copy();self.stale=0
        else:
            d=x-self.anchor;delta=float(d.mean());threshold=max(self.rule.minimum_improvement,self.rule.noise_multiplier*float(d.std(ddof=1)/math.sqrt(len(d))))
            if delta>threshold:self.anchor=x.copy();self.stale=0
            else:self.stale+=1
        eligible=episodes>=self.rule.minimum_episodes and len(self.records)+1>=self.rule.minimum_checkpoints
        converged=eligible and self.stale>=self.rule.patience
        capped=episodes>=self.rule.maximum_episodes or seconds>=self.rule.maximum_training_seconds
        status='converged' if converged else 'budget_limited' if capped else 'training'
        row=dict(checkpoint=len(self.records)+1,episodes=int(episodes),training_seconds=float(seconds),eig=mean,mc_standard_error=se,
            paired_improvement_from_anchor=delta,improvement_threshold=threshold,patience_used=self.stale,selected=selected,status=status)
        self.records.append(row);return row

def apply_selection_mode(row,episodes,rule,selection_mode):
    """Shared by native Torch trainers and the isolated JAX validation bridge."""
    if selection_mode not in {'best_validation','final_fixed_budget','final_compute_budget'}:raise ValueError('Unknown checkpoint selection mode')
    if selection_mode=='final_fixed_budget':
        row['selected']=episodes==rule.maximum_episodes
        row['status']='fixed_budget_complete' if row['selected'] else 'budget_limited' if row['training_seconds']>=rule.maximum_training_seconds else 'training'
    return row

class CommonValidation:
    """Uses existing streamed explicit-likelihood scorers and fixed paired draws."""
    def __init__(self,raw,horizon,seed,systems,contrasts,batch=16,chunk=16384,device='cpu'):
        if seed==8831001:raise ValueError('Reserved final evaluation seed cannot be used for validation')
        from src.research.benchmarks import make_benchmark
        self.raw=copy.deepcopy(raw);self.horizon=horizon;self.seed=seed;self.systems=systems;self.contrasts=contrasts;self.batch=batch;self.chunk=chunk
        self.device=str(device)
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('CUDA validation requested without a CUDA device')
        with torch.device('cpu'):
            b=make_benchmark(raw);g=torch.Generator().manual_seed(seed)
            self.truth=b.sample((systems,),g);self.noise=torch.randn(systems,horizon,b.obs_dim,generator=g,dtype=torch.float64);self.bank=b.sample((contrasts,),g)
        self.identity=hashlib.sha256(self.truth.numpy().tobytes()+self.noise.numpy().tobytes()+self.bank.numpy().tobytes()).hexdigest()
    @torch.no_grad()
    def __call__(self,policy,contrasts=None):
        from src.research.benchmarks import make_benchmark
        from src.research.engine import rollout
        from src.research.location_evaluation import contrast_scores as location_scores
        from src.research.sir_evaluation import contrast_scores as sir_scores
        started=time.perf_counter()
        with torch.device(self.device):
            buffer=io.BytesIO();torch.save(policy,buffer);buffer.seek(0)
            p=torch.load(buffer,map_location=self.device,weights_only=False).eval();b=make_benchmark(self.raw)
            if hasattr(p,'set_routing'):p.set_routing(getattr(getattr(p,'manager',None),'mode','learned_switch'),False)
            if hasattr(p,'stochastic'):p.stochastic=False
            bank=self.bank[:contrasts] if contrasts is not None else self.bank
            values=[];calls=b.calls
            scorer=sir_scores if b.name=='sir_ode' else location_scores
            for start in range(0,self.systems,self.batch):
                n=min(self.batch,self.systems-start)
                r=rollout(b,p,self.horizon,n,self.seed+start*97451,0,truth=self.truth[start:start+n].to(self.device),noise=self.noise[start:start+n].to(self.device),method='rl_sboed' if getattr(p,'kind',None)=='rl_sboed' else None)
                info,_=scorer(b,r,bank,self.chunk);values.extend(info[:,-1].tolist())
            return np.asarray(values),dict(seed=self.seed,systems=self.systems,contrasts=len(bank),paired_draws_sha256=self.identity,
                seconds=time.perf_counter()-started,forward_particle_stages=b.calls-calls,device=self.device,mode='native_stochastic_policy' if getattr(p,'kind',None)=='rl_sboed' else 'hard_deployed_policy',secondary_parameter_inference=False)

class ValidationObserver:
    def __init__(self,validator,rule,directory,method,selection_mode="best_validation",compute_clock=None,compute_checkpoint_seconds=900.):
        if selection_mode not in {"best_validation","final_fixed_budget","final_compute_budget"}:raise ValueError("Unknown checkpoint selection mode")
        self.selection_mode=selection_mode
        self.compute_clock=compute_clock;self.compute_checkpoint_seconds=float(compute_checkpoint_seconds);self.next_compute_check=self.compute_checkpoint_seconds;self.native_unit_bound=0.;self.last_compute_spent=0. if compute_clock is None else compute_clock._read()["credit_seconds"]
        if selection_mode=="final_compute_budget" and compute_clock is None:raise ValueError("Matched-compute selection requires an initialized whole-method clock")
        self.validator=validator;self.rule=rule;self.directory=Path(directory);self.method=method;self.plateau=Plateau(rule)
        self.started=time.perf_counter();self.validation_seconds=0.;self.next_check=rule.validation_every_episodes;self.best_state=None;self.best_gamma=None
    def check(self,policy,episodes,update,optimizer=None,scheduler=None,force=False):
        seconds=time.perf_counter()-self.started-self.validation_seconds
        compute_stop=False
        if self.compute_clock is not None:
            if torch.cuda.is_available():torch.cuda.synchronize()
            spent=self.compute_clock.spent();seconds=spent;self.native_unit_bound=max(self.native_unit_bound,spent-self.last_compute_spent);self.last_compute_spent=spent
            compute_stop=not self.compute_clock.can_start(self.native_unit_bound)
            force=force or compute_stop
            if not(force or spent>=self.next_compute_check):return False
        elif not(force or episodes>=self.next_check or episodes>=self.rule.maximum_episodes or seconds>=self.rule.maximum_training_seconds):return False
        # Validation owns a separate simulator/RNG. Preserve all native training RNGs.
        cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            if self.compute_clock is None:scores,cost=self.validator(policy)
            else:
                with self.compute_clock.exclude("validation"):scores,cost=self.validator(policy)
        finally:
            torch.set_rng_state(cpu_rng)
            if gpu_rng is not None:torch.cuda.set_rng_state_all(gpu_rng)
        self.validation_seconds+=cost['seconds'];row=self.plateau.update(scores,episodes,seconds);row.update(update=int(update),validation=cost)
        apply_selection_mode(row,episodes,self.rule,self.selection_mode)
        if self.compute_clock is not None:
            mark=dict(eligible=self.compute_clock.can_start());row.update(compute_checkpoint=mark)
            row['selected']=mark['eligible']
            row['status']=(('compute_budget_complete' if mark['eligible'] else 'compute_budget_exceeded') if compute_stop else 'planned_schedule_complete_before_compute_cap') if force else 'training'
            self.next_compute_check=self.compute_clock.spent()+self.compute_checkpoint_seconds
        self.next_check=episodes+self.rule.validation_every_episodes
        self.directory.mkdir(parents=True,exist_ok=True)
        # Eligibility is finalized after native checkpoint serialization below.
        state={k:v.detach().cpu().clone() for k,v in policy.state_dict().items()}
        gamma=getattr(getattr(policy,'router',None),'gamma',None)
        resume=dict(policy_state=state,optimizer_state=None if optimizer is None else optimizer.state_dict(),scheduler_state=None if scheduler is None else scheduler.state_dict(),episodes=episodes,update=update,rng_cpu=cpu_rng,rng_cuda=gpu_rng,gamma=gamma,rule=asdict(self.rule),selection=row)
        torch.save(resume,self.directory/(self.method+'_journal_last.pt'))
        if getattr(self.validator,'preserve_progress_checkpoints',False) or self.compute_clock is not None:
            target=self.directory/(self.method+f'_progress_{update}.pt');tmp=target.with_suffix('.partial');torch.save(resume,tmp);tmp.replace(target)
        np.save(self.directory/(self.method+f'_validation_scores_{update}.npy'),scores)
        if self.compute_clock is not None:
            mark=self.compute_clock.checkpoint(str(update));row['compute_checkpoint']=mark;row['selected']=mark['eligible']
            if compute_stop:row['status']='compute_budget_complete' if mark['eligible'] else 'compute_budget_exceeded'
            (self.directory/(self.method+f'_progress_{update}.cost.json')).write_text(json.dumps(mark))
        if row['selected']:
            self.best_state=state;self.best_gamma=gamma
            if self.compute_clock is None:torch.save(resume,self.directory/(self.method+'_journal_best.pt'))
            else:
                import os
                best=self.directory/(self.method+'_journal_best.pt');temp=best.with_suffix('.link')
                os.link(self.directory/(self.method+f'_progress_{update}.pt'),temp);os.replace(temp,best)
        with (self.directory/(self.method+'_validation.jsonl')).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(self.method,'journal validation',row,flush=True)
        return row['status']!='training'
    def allow_next_unit(self,policy,episodes,update,optimizer=None,scheduler=None):
        if self.compute_clock is None:return True
        if self.compute_clock.can_start(self.native_unit_bound):return True
        if self.plateau.records and self.plateau.records[-1]['episodes']==episodes:
            self.plateau.records[-1]['status']='compute_budget_complete';return False
        self.check(policy,episodes,update,optimizer,scheduler,force=True);return False
    def finish(self,policy):
        if self.best_state is None:raise RuntimeError('No validation-selected checkpoint')
        policy.load_state_dict(self.best_state,strict=True)
        if self.best_gamma is not None:policy.router.gamma=self.best_gamma
        return dict(compute_budget=None if self.compute_clock is None else self.compute_clock.report(),checkpoint_selection='latest_budget_eligible_checkpoint' if self.compute_clock is not None else 'final_predeclared_fixed_budget' if self.selection_mode=='final_fixed_budget' else 'best_predeclared_validation_EIG',convergence_status=self.plateau.records[-1]['status'],
            journal_validation_history=self.plateau.records,validation_seconds=self.validation_seconds,
            optimization_seconds=time.perf_counter()-self.started-self.validation_seconds,stopping_rule=asdict(self.rule),
            selected_checkpoint=next(r for r in reversed(self.plateau.records) if r['selected']),validation_seed=self.validator.seed)
