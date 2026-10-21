"""Explicit units; context-local opt-in accounting without changing model outputs."""
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
_ACTIVE=ContextVar("research_resource_ledger",default=None)
@dataclass
class ResourceLedger:
    phase: str
    counts: dict=field(default_factory=dict)
    def add(self,unit,n):
        if not isinstance(n,int) or n<0: raise ValueError("Nonnegative integer work counts required")
        self.counts[unit]=self.counts.get(unit,0)+n
    def record_paths(self,batch,contrasts,stages,likelihood=True):
        for v in (batch,contrasts,stages):
            if not isinstance(v,int) or v<0: raise ValueError("Invalid path dimensions")
        self.add("fresh_complete_episodes",batch)
        self.add("simulated_observation_stages",batch*stages)
        self.add("true_parameter_stage_propagations",batch*stages)
        self.add("contrast_parameter_stage_propagations",batch*contrasts*stages)
        if likelihood:self.add("scalar_conditional_log_density_evaluations",batch*(contrasts+1)*stages)
    def export(self):
        return {"schema":1,"phase":self.phase,"counts":dict(self.counts),"ode_rhs_evaluations":None,"solver_backward_work":None,"coverage":"shared engine only; native trainer adapters require separate counters","counting_basis":"executed tensor dimensions; not FLOPs or NLE-equivalent proxy"}
@contextmanager
def accounting(ledger):
    token=_ACTIVE.set(ledger)
    try: yield ledger
    finally: _ACTIVE.reset(token)
def active_ledger():return _ACTIVE.get()

def with_training_ledger(record,horizon):
    """Native-counter reconciliation. Unknown solver/score work is never zero-filled."""
    n=record.get("generated_training_trajectories")
    c=record.get("contrast_training_trajectories",0)
    if not isinstance(n,int) or n<0 or not isinstance(c,int) or c<0:raise ValueError("Invalid native episode counters")
    phases=["initialization_trajectories","score_training_trajectories","policy_training_trajectories"]
    if all(k in record for k in phases) and sum(record[k] for k in phases)!=n:raise ValueError("Native phase accounting mismatch")
    updates={}
    for unit,keys in {"policy":["actor_updates","actor_optimizer_updates","actual_policy_updates","actual_updates"],"critic":["critic_updates","critic_optimizer_updates"],"posterior":["posterior_optimizer_updates"],"score":["actual_score_updates"],"temperature":["temperature_updates"]}.items():
        updates[unit]=next((record[k] for k in keys if k in record),None)
    score="ScoreBED" in record.get("training","")
    vs="vsOED" in record.get("training","")
    replay=record.get("gradient_updates");batch=record.get("replay_batch")
    record["resource_ledger"]={"schema":2,"scope":"current training phase; base cost separately attributed","counts":{"fresh_complete_episodes":n,"true_parameter_stage_propagations":n*horizon,"contrast_parameter_stage_propagations":c*horizon,"scalar_conditional_log_density_evaluations":None if score else 0 if vs else (n+c)*horizon},"optimizer_updates":updates,"replay_transition_samples":replay*batch if replay is not None and batch is not None else None,"offline_seconds":record.get("training_seconds"),"validation_seconds":record.get("validation_seconds"),"base_training_seconds":record.get("base_training_seconds"),"ode_rhs_evaluations":None,"solver_backward_work":None,"gpu_occupancy_seconds":None,"nle_equivalent_proxy":None,"basis":"derived from native executed counters, not direct hardware measurement","limitations":["score-target and backward recomputation not counted","Unknown fields are not zero; walltime is not GPU occupancy"]}
    ledger=record["resource_ledger"]
    ledger["likelihood_gradient_work"]={"scalar_stage_terms":record.get("score_training_trajectories",0)*horizon if score and "score_training_trajectories" in record else None,"basis":"model-derived conditional-score target terms; excludes compiler/autodiff recomputation"}
    if score and "score_training_trajectories" in record:
        ledger["nle_equivalent_proxy"]={"value":record["score_training_trajectories"]*horizon,"unit":"conditional Gaussian stage terms differentiated for score targets","basis":"model estimate, not equal-cost hardware work"}
    if vs:ledger["replay_work"]={"minibatch_trajectories":record.get("replay_batch_size"),"optimizer_steps":updates,"note":"Posterior steps may include stage-specific nets; stored data are not fresh episodes"}
    return record

def progress_ledger(directory,method,episodes,contrasts,horizon,updates,seconds,**components):
    import json,os
    from pathlib import Path
    record=dict(training=method,generated_training_trajectories=int(episodes),contrast_training_trajectories=int(contrasts),actual_updates=int(updates),training_seconds=float(seconds),**components)
    data=with_training_ledger(record,horizon)["resource_ledger"]
    path=Path(directory)/(method+"_resource_progress.json");temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,indent=2));os.replace(temp,path)
