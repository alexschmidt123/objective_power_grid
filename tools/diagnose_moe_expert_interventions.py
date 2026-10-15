"""One-decision expert interventions on fixed paired development cases.

This is a frozen diagnostic, not a fifth policy variant. Future decisions use
normal learned routing. Different actions create different downstream histories.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from src.research.moe_family import load_policy,EXPERT_IDS
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores

class FrozenExpertIntervention(nn.Module):
    kind='sparse_moe_prototype'
    def __init__(self,policy,stage,expert):
        super().__init__();self.policy=policy;self.stage=stage;self.expert=expert
    def forward(self,features,stage):
        if stage!=self.stage:return self.policy(features,stage)
        def replace(router,inputs,output):
            w,selectors,bits=output
            w=torch.zeros_like(w);w[:,self.expert]=1.
            selectors=w[:,None,:]
            bits=router.codes[self.expert][None,None].expand(len(w),router.k,-1)
            return w,selectors,bits
        manager=self.policy.manager;old_mode=manager.mode if manager is not None else None
        if manager is not None:manager.mode='always_reroute'
        handle=self.policy.router.register_forward_hook(replace)
        try:return self.policy(features,stage)
        finally:
            handle.remove()
            if manager is not None:manager.mode=old_mode


def statistics(x):
    x=np.asarray(x);se=float(x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else None
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,standard_error=se,
        conditional_95_interval=[float(x.mean()-1.96*se),float(x.mean()+1.96*se)] if se is not None else None)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--rows',required=True);ap.add_argument('--bank',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--cases',type=int,default=64);ap.add_argument('--stages',default='5,15,25')
    args=ap.parse_args();torch.set_num_threads(1);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    run=Path(args.run);cfg=json.loads((run/'run_config.json').read_text());b=make_benchmark(cfg['benchmark'])
    if b.name!='location_finding':raise ValueError('Current diagnostic is location-only')
    rows=json.loads(Path(args.rows).read_text())[:args.cases]
    truth=torch.tensor([r['truth'] for r in rows],dtype=torch.float64);noise=torch.tensor([r['measurement_noise_standard'] for r in rows],dtype=torch.float64)
    bank=torch.from_numpy(np.load(args.bank));baseline=np.array([r['spce'] for r in rows])
    result=dict(interpretation='One routed-decision interventions; development subset, not primary full evaluation',
        cases=len(rows),contrastives=len(bank),training_updates=0,first_cases_rule='First N existing paired trajectories, no outcome selection',
        stages=[int(s) for s in args.stages.split(',')],experts=list(EXPERT_IDS),results=[],
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
        limitations=['Interventions can induce out-of-distribution histories','Local intervention effects do not establish general expert specialization','Single trained seed'])
    started=time.perf_counter()
    for measurement in result['stages']:
        stage=measurement-1
        if not 1<=stage<cfg['horizon']:raise ValueError('Choose a post-initial measurement within horizon')
        for j in range(4):
            p=load_policy(args.checkpoint,b);policy=FrozenExpertIntervention(p,stage,j).eval();scores=[]
            same_errors=[]
            for start in range(0,len(rows),16):
                n=min(16,len(rows)-start)
                r=rollout(b,policy,cfg['horizon'],n,1001+start*97451,0,truth=truth[start:start+n],noise=noise[start:start+n])
                info,_=contrast_scores(b,r,bank,cfg['settings']['evaluation_contrast_chunk']);scores.extend(info[:,-1].tolist())
                for i in range(n):
                    route=rows[start+i]['routing'][stage-1]
                    selected=route.get('selected_expert')
                    if selected is None:selected=int(np.argmax(route['weights']))
                    if selected==j:
                        expected=torch.tensor(rows[start+i]['actions'],dtype=torch.float64)
                        actual=torch.stack([a[i] for a in r['actions']])
                        error=float((actual-expected).abs().max());same_errors.append(error)
                        assert error<1e-9,'Forcing the actually selected expert changed the unperturbed rollout'
            scores=np.asarray(scores);difference=scores-baseline
            selected=np.array([r['routing'][stage-1].get('selected_expert') if r['routing'][stage-1].get('selected_expert') is not None else int(np.argmax(r['routing'][stage-1]['weights'])) for r in rows])
            item=dict(measurement=measurement,expert=EXPERT_IDS[j],eig=statistics(scores),difference=statistics(difference),
                difference_on_other_experts_histories=statistics(difference[selected!=j]),
                same_expert_replay_cases=len(same_errors),same_expert_max_action_error=max(same_errors,default=None),scores=scores.tolist())
            result['results'].append(item)
            print(measurement,EXPERT_IDS[j],item['difference'],flush=True)
            (out/'partial.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    result['seconds']=time.perf_counter()-started
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'exit_code').write_text('0\n')

if __name__=='__main__':main()
