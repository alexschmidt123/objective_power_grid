"""Matched exclusive-device wall time; framework-neutral, including JAX workers.

The budget is a resource ceiling, not a convergence criterion or FLOP estimate.
Only explicitly excluded evaluation intervals are removed from charged time.
"""
import contextlib,json,os,time
from pathlib import Path

class ComputeBudgetError(RuntimeError):pass
class ComputeClock:
    def __init__(self,path,seconds,*,credit_seconds=0.,hardware="UNVERIFIED",now=time.monotonic,start=None):
        self.path=Path(path);self.now=now
        if seconds<=0 or credit_seconds<0 or credit_seconds>seconds:raise ValueError("Invalid complete-method compute budget")
        if not self.path.exists():
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self._write(dict(version=1,start=self.now() if start is None else start,budget_seconds=float(seconds),credit_seconds=float(credit_seconds),hardware=hardware,excluded=[],counts={},components={},checkpoints=[]))
        state=self._read()
        if state["budget_seconds"]!=float(seconds) or state["credit_seconds"]!=float(credit_seconds) or state["hardware"]!=hardware:raise ComputeBudgetError("Compute budget identity changed")
    def _read(self):return json.loads(self.path.read_text())
    def _write(self,state):
        temporary=self.path.with_name(self.path.name+f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(state,indent=2,allow_nan=False));os.replace(temporary,self.path)
    def spent(self):
        s=self._read();end=s.get("stopped_at",self.now());excluded=sum(max(0.,min(end,b)-max(s["start"],a)) for a,b,_ in s["excluded"])
        return s["credit_seconds"]+max(0.,end-s["start"]-excluded)
    @contextlib.contextmanager
    def exclude(self,reason):
        if reason not in {"validation","heldout_evaluation","online_latency_measurement"}:raise ComputeBudgetError("Training phases cannot be excluded")
        a=self.now()
        try:yield
        finally:
            b=self.now();s=self._read()
            if any(a<old_b and b>old_a for old_a,old_b,_ in s["excluded"]):raise ComputeBudgetError("Overlapping exclusions would double-discount time")
            s["excluded"].append([a,b,reason]);self._write(s)
    def count(self,unit,n,component="training"):
        if not isinstance(n,int) or n<0:raise ValueError("Work counts must be nonnegative integers")
        s=self._read();s["counts"][unit]=s["counts"].get(unit,0)+n
        c=s["components"].setdefault(component,{});c[unit]=c.get(unit,0)+n;self._write(s)
    def can_start(self,next_unit_bound=0.):
        if next_unit_bound<0:raise ValueError("Negative native unit bound")
        s=self._read()
        return "stopped_at" not in s and self.spent()+next_unit_bound<s["budget_seconds"]
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
