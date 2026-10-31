"""Matched exclusive-device wall time; framework-neutral, including JAX workers.

The budget is a resource ceiling, not a convergence criterion or FLOP estimate.
Only explicitly excluded evaluation intervals are removed from charged time.
"""
import contextlib,json,os,time
from pathlib import Path

class ComputeBudgetError(RuntimeError):pass
class ComputeClock:
    def __init__(self,path,seconds,*,credit_seconds=0.,hardware="UNVERIFIED",now=time.monotonic,start=None,finalization_reserve_seconds=None):
        self.path=Path(path);self.now=now
        if seconds<=0 or credit_seconds<0 or credit_seconds>seconds:raise ValueError("Invalid complete-method compute budget")
        if not self.path.exists():
            self.path.parent.mkdir(parents=True,exist_ok=True)
            reserve=0. if finalization_reserve_seconds is None else float(finalization_reserve_seconds)
            if reserve<0 or reserve>=seconds:raise ValueError('Finalization reserve must fit the complete-method quota')
            self._write(dict(version=2,start=self.now() if start is None else start,budget_seconds=float(seconds),credit_seconds=float(credit_seconds),hardware=hardware,finalization_reserve_seconds=reserve,excluded=[],counts={},components={},checkpoints=[]))
        state=self._read()
        if state["budget_seconds"]!=float(seconds) or state["credit_seconds"]!=float(credit_seconds) or state["hardware"]!=hardware:raise ComputeBudgetError("Compute budget identity changed")
        if finalization_reserve_seconds is not None and float(finalization_reserve_seconds)!=state.get('finalization_reserve_seconds',0.):raise ComputeBudgetError('Finalization reserve identity changed')
    def _read(self):return json.loads(self.path.read_text())
    def _write(self,state):
        temporary=self.path.with_name(self.path.name+f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state,indent=2,allow_nan=False));os.replace(temporary,self.path)
    def spent(self):
        s=self._read();end=s.get("stopped_at",self.now());intervals=list(s['excluded'])
        if s.get('active_exclusion') is not None:
            a,reason=s['active_exclusion'];intervals.append([a,end,reason])
        excluded=sum(max(0.,min(end,b)-max(s["start"],a)) for a,b,_ in intervals)
        return s["credit_seconds"]+max(0.,end-s["start"]-excluded)
    @contextlib.contextmanager
    def exclude(self,reason):
        if reason not in {"validation","heldout_evaluation","online_latency_measurement"}:raise ComputeBudgetError("Training phases cannot be excluded")
        a=self.now();s=self._read()
        if s.get('active_exclusion') is not None:raise ComputeBudgetError('Overlapping active exclusions are forbidden')
        s['active_exclusion']=[a,reason];self._write(s)
        try:yield
        finally:
            b=self.now();s=self._read()
            if any(a<old_b and b>old_a for old_a,old_b,_ in s["excluded"]):raise ComputeBudgetError("Overlapping exclusions would double-discount time")
            if s.get('active_exclusion')!=[a,reason]:raise ComputeBudgetError('Active evaluation exclusion identity changed')
            s.pop('active_exclusion',None);s["excluded"].append([a,b,reason]);self._write(s)
    def count(self,unit,n,component="training"):
        if not isinstance(n,int) or n<0:raise ValueError("Work counts must be nonnegative integers")
        s=self._read();s["counts"][unit]=s["counts"].get(unit,0)+n
        c=s["components"].setdefault(component,{});c[unit]=c.get(unit,0)+n;self._write(s)
    def can_start(self,next_unit_bound=0.):
        if next_unit_bound<0:raise ValueError("Negative native unit bound")
        s=self._read()
        return "stopped_at" not in s and self.spent()+next_unit_bound+s.get('finalization_reserve_seconds',0.)<s["budget_seconds"]
    def checkpoint(self,identifier,charged_seconds=None):
        s=self._read();cost=self.spent() if charged_seconds is None else float(charged_seconds)
        if cost<0 or cost>self.spent():raise ValueError("Invalid checkpoint cost timestamp")
        row=dict(identifier=identifier,charged_seconds=cost,eligible=cost<=s["budget_seconds"])
        s["checkpoints"].append(row);self._write(s);return row
    def selected(self):
        rows=[r for r in self._read()["checkpoints"] if r["eligible"]]
        if not rows:raise ComputeBudgetError("No budget-eligible checkpoint")
        return max(rows,key=lambda x:x["charged_seconds"])
    def stop(self):
        s=self._read()
        if "stopped_at" not in s:s["stopped_at"]=self.now();self._write(s)
        return self.report()
    def report(self):
        s=self._read();s.update(charged_seconds=self.spent(),fraction_of_allocated_budget=self.spent()/s["budget_seconds"],unused_seconds=max(0.,s["budget_seconds"]-self.spent()),unit="exclusive-device allocated training wall seconds",scope="includes setup, compilation, initialization, all training phases, replay and checkpoint overhead; excludes logged evaluation intervals",not_equal_to_FLOPs=True)
        return s

def score_phase_plan(seconds,initialization_seconds,score_update_seconds,policy_update_seconds,score_per_policy=19,reserve_seconds=0.):
    """Freeze native full-minibatch phase counts from timing, never from EIG."""
    if min(score_update_seconds,policy_update_seconds)<=0 or score_per_policy<1:raise ValueError("Measured phase timings required")
    remaining=seconds-initialization_seconds-reserve_seconds
    policies=int(remaining//(score_per_policy*score_update_seconds+policy_update_seconds))
    if policies<1:raise ComputeBudgetError("No complete native score-pretraining plus policy phase fits")
    return dict(score_updates=score_per_policy*policies,policy_updates=policies,estimated_seconds=initialization_seconds+policies*(score_per_policy*score_update_seconds+policy_update_seconds),phase_ratio=score_per_policy,timing_based=True)

def native_update_plan(seconds,unit_seconds,reference_updates,*,initialization_seconds=0.,finalization_seconds=60.,safety_factor=1.25):
    """Time-derived safety maximum; never changes the native scheduler horizon."""
    import math
    if unit_seconds<=0 or reference_updates<1 or int(reference_updates)!=reference_updates or min(initialization_seconds,finalization_seconds)<0 or safety_factor<1:
        raise ValueError('Positive measured native timing and finite reference schedule required')
    available=seconds-initialization_seconds-finalization_seconds
    maximum=min(int(reference_updates),max(0,int(math.floor(available/(safety_factor*unit_seconds)))))
    if maximum<1:raise ComputeBudgetError('No complete native optimizer/collection unit fits the charged allowance')
    return dict(reference_updates=int(reference_updates),maximum_updates=maximum,safe_max_updates=maximum,measured_unit_seconds=float(unit_seconds),initialization_seconds=float(initialization_seconds),finalization_reserve_seconds=float(finalization_seconds),safety_factor=float(safety_factor),estimated_charged_seconds=initialization_seconds+maximum*safety_factor*unit_seconds+finalization_seconds,training_status='NATIVE_FINITE_SCHEDULE' if maximum==reference_updates else 'COMPUTE_LIMITED_NATIVE_IMPLEMENTATION',selection_basis='measured_native_unit_wall_time_without_EIG')

def native_score_phase_plan(seconds,initialization_seconds,score_update_seconds,policy_update_seconds,*,score_reference_steps=50000,policy_reference_steps=500,finalization_seconds=60.,safety_factor=1.25):
    """Predeclared equal fractional truncation of native sequential phases.

    For the pinned dynamical-system reference this is50,000 score updates then
    500 policy updates. A measured compute-limited plan shortens both phases by
    the same fraction, while retaining their original optimizer/LR horizons.
    It is explicitly a compute-limited adaptation, not full native convergence.
    """
    import math
    if min(score_update_seconds,policy_update_seconds)<=0 or min(initialization_seconds,finalization_seconds)<0 or safety_factor<1:
        raise ValueError('Measured nonnegative setup and positive native phase timings required')
    if any(int(n)!=n or n<1 for n in (score_reference_steps,policy_reference_steps)):
        raise ValueError('Native finite score and policy schedules are required')
    score_reference_steps=int(score_reference_steps);policy_reference_steps=int(policy_reference_steps)
    if score_reference_steps%policy_reference_steps:raise ValueError('Exact native phase ratio is required for equal-fraction truncation')
    ratio=score_reference_steps//policy_reference_steps
    available=seconds-initialization_seconds-finalization_seconds
    cycles=min(policy_reference_steps,max(0,int(math.floor(available/(safety_factor*(ratio*score_update_seconds+policy_update_seconds))))))
    if cycles<1:raise ComputeBudgetError('Initialization plus one complete native-ratio score/policy block does not fit')
    score=ratio*cycles
    return dict(score_updates=score,policy_updates=cycles,score_reference_steps=score_reference_steps,policy_reference_steps=policy_reference_steps,phase_ratio=ratio,measured_initialization_seconds=float(initialization_seconds),measured_score_update_seconds=float(score_update_seconds),measured_policy_update_seconds=float(policy_update_seconds),safety_factor=float(safety_factor),finalization_reserve_seconds=float(finalization_seconds),estimated_charged_seconds=initialization_seconds+safety_factor*(score*score_update_seconds+cycles*policy_update_seconds)+finalization_seconds,training_status='NATIVE_FINITE_SCHEDULE' if cycles==policy_reference_steps else 'COMPUTE_LIMITED_NATIVE_IMPLEMENTATION',selection_basis='native_phase_ratio_and_measured_wall_time_without_EIG',scheduler_horizons_preserved=True)

def validate_native_score_phase_plan(plan,score_reference_steps,policy_reference_steps,initialization,batch,trajectory_guard):
    """Validate a frozen phase truncation independently in the isolated worker."""
    if batch!=1024:raise ValueError('Native SIR ScoreBED uses full1024-sample minibatches')
    a,b=plan.get('score_updates'),plan.get('policy_updates')
    if any(not isinstance(n,int) or isinstance(n,bool) or n<1 for n in (a,b)):raise ValueError('Positive complete native phase counts required')
    if a>score_reference_steps or b>policy_reference_steps:raise ValueError('Compute truncation cannot extend a native finite schedule')
    if a*policy_reference_steps!=b*score_reference_steps:raise ValueError('Compute truncation must preserve the declared native phase proportion')
    if plan.get('score_reference_steps')!=score_reference_steps or plan.get('policy_reference_steps')!=policy_reference_steps:raise ValueError('Native scheduler reference identity changed')
    total=initialization+(a+b)*batch
    if total>trajectory_guard:raise ValueError('Episode safety guard truncates the frozen compute plan')
    return dict(initialization=initialization,score=a*batch,policy=b*batch,total=total,compute_phase_plan=plan,training_budget_unit='charged_training_seconds',phase_ratio=f'{score_reference_steps}:{policy_reference_steps}')


def effective_episode_limit(safety_episodes,reference_episodes):
    """Schedules remain native; a separate explicit guard may shorten them."""
    if min(safety_episodes,reference_episodes)<=0:raise ValueError('Positive explicit schedule and safety guard required')
    return min(int(safety_episodes),int(reference_episodes))

def classify_stop(status,episodes,reference_episodes,safety_episodes):
    if status in {'compute_budget_complete','compute_budget_exceeded'}:return 'compute_budget_exhaustion'
    if episodes>=reference_episodes:return 'reference_schedule_completion'
    if episodes>=safety_episodes:return 'episode_safety_cap_exhaustion'
    return 'incomplete_or_interrupted'


def primary_reference_schedule(settings,safety_cap):
    """Explicit optimizer/episode dimensions for matched-compute pathwise methods."""
    updates=settings['reference_optimizer_updates'];batch=settings['batch_size']
    episodes=settings['reference_training_episodes']
    if any(not isinstance(x,int) or x<=0 for x in [updates,batch,episodes,safety_cap]):raise ValueError('Positive integer reference updates, batch, episodes and safety guard required')
    if episodes!=updates*batch:raise ValueError('Reference training episodes must equal optimizer updates times native outer batch')
    if settings.get('reference_schedule_episodes',episodes)!=episodes:raise ValueError('Conflicting reference episode fields')
    if settings.get('spce_trajectory_budget_unit')!='primary_episodes':raise ValueError('Matched-compute trajectory cap must explicitly count primary episodes; contrasts are separate work')
    return min(episodes,safety_cap)

def spce_update_plan(settings,trajectory_budget):
    import math
    unit=settings.get('spce_trajectory_budget_unit','primary_episodes' if settings.get('primary_episode_budget',0) else 'forward_including_contrasts')
    if unit not in {'primary_episodes','forward_including_contrasts'}:raise ValueError('Unknown sPCE trajectory budget unit')
    if unit=='primary_episodes':
        updates=math.ceil(trajectory_budget/settings['batch_size'])
        if 'reference_optimizer_updates' in settings:updates=min(updates,settings['reference_optimizer_updates'])
    else:
        updates=min(settings['updates'],math.ceil(trajectory_budget/((settings['contrasts']+1)*settings['batch_size'])))
    return updates,unit
