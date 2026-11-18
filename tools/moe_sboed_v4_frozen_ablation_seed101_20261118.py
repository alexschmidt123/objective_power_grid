"""Explicit frozen V4 gate overrides, seed101, identical checkpoint in all six modes."""
from pathlib import Path
import argparse,fcntl,importlib.util,json,os,sys,time
M=Path.home()/'Documents/objective_power_grid'
O=M/'experiments/location_finding/moe_sboed_v4_frozen_ablation_seed101_20261118'
R=M/'reports/moe_sboed_v4_frozen_ablation_seed101_20261118'
spec=importlib.util.spec_from_file_location('frozen_ablation_previous',M/'tools/moe_sboed_v4_allmethods_20261117.py')
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
import torch,numpy as np
from torch import nn
MODES=['learned','uniform','e1','e2','e3','e4']

class GateOverride(nn.Module):
    def __init__(self,actor,mode):super().__init__();self.actor=actor;self.mode=mode
    def forward(self,x):
        if self.mode=='learned':return self.actor(x)
        shape=x.shape[:-1];flat=b.p.V4Actor.forward.__globals__['observed_prefix'](x).reshape(-1,117)
        c=self.actor.query(flat);z=torch.stack([expert(c)[0] for expert in self.actor.experts],1)
        gates=flat.new_full((len(flat),4),.25) if self.mode=='uniform' else flat.new_zeros((len(flat),4))
        if self.mode!='uniform':gates[:,int(self.mode[-1])-1]=1
        assert torch.equal(gates.sum(-1),torch.ones_like(gates[:,0]))
        action,_=self.actor.decode(flat,z,gates)
        return action.reshape(*shape,2)

def put(path,data):b.put(path,data)
def report():
    rows=[];values={}
    for mode in MODES:
        path=O/mode/'RESULTS.json'
        if path.exists():
            d=b.read(path);rows.append(d);values[mode]=np.load(O/mode/(mode+'_scores.npy'))
    comparisons={}
    if 'learned' in values:
        for mode in MODES[1:]:
            if mode in values:comparisons[mode]=b.engine.summaries(values['learned']-values[mode])
    put(R/'RESULTS.json',dict(status='COMPLETE' if len(rows)==6 else 'PENDING',seed=101,rows=rows,full_minus_ablation=comparisons,
      scope='Frozen gate interventions on one learned V4 checkpoint;zero training updates. Uniform is not the independently trained uniform competitor.'))
    text='# V4 frozen routing ablations: seed101\n\nAll six modes use the SAME terminal learned checkpoint. No retraining,reinitialization,feature rescaling,head replacement or physical-coordinate averaging. Shared query,normalization,head and weak correction are unchanged. Original canonical2000-case/1M-contrast/T30 scorer.\n\n|Mode|Weights|EIG nats|MCSE|Reproduction error versus prior vector|\n|---|---|---:|---:|---:|\n'
    for d in rows:text+=f"|{d['mode']}|{d['weights']}|{d['mean']:.5f}|{d['MCSE']:.5f}|{d.get('max_previous_score_error','new uniform intervention')}|\n"
    text+='\nThese are inference-only ablations,not separately trained standalone policies. The four trained seed101 one-hot policies and independently trained uniform policy are preserved as different experiments. Case MCSE is not training-seed variability.\n'
    (R/'RESULTS.md').write_text(text)

@torch.no_grad()
def run():
    O.mkdir(parents=True,exist_ok=False);R.mkdir(parents=True,exist_ok=False)
    handle=(O/'.controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Wait for the existing SIR readiness controller rather than racing its between-job gaps.
    sir=M/'experiments/sir_ode/baseline_readiness_20261118/STATUS.json'
    while sir.exists() and b.read(sir)['status']=='RUNNING_READINESS':
        put(O/'STATUS.json',dict(status='WAITING_FOR_EXISTING_SIR_READINESS',training=False));time.sleep(20)
    while b.subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        put(O/'STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU_JOB',training=False));time.sleep(20)
    b.lock_protocol();b.p.setup();cp=b.checkpoint(101,'learned');original=b.p.sha(cp)
    put(R/'PROTOCOL.json',dict(seed=101,checkpoint=str(cp),checkpoint_sha256=original,evaluation=b.engine.LOCK,
      modes=dict(learned='Unchanged learned weights',uniform=[.25]*4,**{f'e{k+1}':[int(j==k) for j in range(4)] for k in range(4)}),
      training_updates=0,all_modes_same_checkpoint=True,feature_rescaling=False,
      source_sha256=b.p.sha(M/'tools/moe_sboed_v4_frozen_ablation_seed101_20261118.py')))
    with b.p.lease(),b.engine.Resources(O):
        state=b.p.load_parent('cuda');actor=b.actor_from_template(state['algorithm'].actor_net,101,'learned','cuda');del state
        frozen=b.p.n.digest(actor.state_dict());old=b.read(M/'reports/moe_sboed_v4_allmethods_20261117/RESULTS.json')
        data=np.load(b.p.n.BASE/'evaluation/diagnostic_candidate_histories.npz')['X'][:8]
        x=torch.tensor(data.copy(),device='cuda').reshape(-1,117)
        z,_,learned=actor.packets(x);tests=[]
        for mode in MODES:
            model=GateOverride(actor,mode).eval();actual=model(x)
            if mode=='learned':expected=actor(x)
            else:
                weights=x.new_full((len(x),4),.25) if mode=='uniform' else x.new_zeros((len(x),4))
                if mode!='uniform':weights[:,int(mode[-1])-1]=1
                expected,_=actor.decode(x,z,weights)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            tests.append(dict(mode=mode,explicit_original_decoder_error=float((actual-expected).abs().max())))
        put(R/'EXPLICIT_GATE_AUDIT.json',dict(status='PASS',seed=101,histories=8,stages=30,tests=tests))
        for mode in MODES:
            out=O/mode;out.mkdir();put(O/'STATUS.json',dict(status='EVALUATING',seed=101,mode=mode,training=False))
            scores,d=b.engine.evaluate(GateOverride(actor,mode).eval(),out,mode,time.monotonic()+3500,collect=False)
            weights='learned' if mode=='learned' else [.25]*4 if mode=='uniform' else [int(k==int(mode[-1])-1) for k in range(4)]
            d.update(status='COMPLETE',mode=mode,weights=weights,seed=101,training_updates=0,checkpoint_sha256=original)
            if mode!='uniform':
                name='moe-sboed' if mode=='learned' else 'moe-sboed-'+mode
                prior=next(row for row in old['records'] if row['method']==name and row['seed']==101)
                error=float(np.abs(scores-np.load(prior['scores'])).max());assert error<1e-9
                d['max_previous_score_error']=error
            assert b.p.sha(cp)==original and b.p.n.digest(actor.state_dict())==frozen
            put(out/'RESULTS.json',d);report();print(json.dumps(dict(mode=mode,mean=d['mean'],MCSE=d['MCSE'])),flush=True)
        put(O/'STATUS.json',dict(status='COMPLETE',seed=101,training_updates=0))

if __name__=='__main__':run()
