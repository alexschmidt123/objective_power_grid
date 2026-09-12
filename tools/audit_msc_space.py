#!/usr/bin/env python3
"""Development audit of MSC probe value, adaptivity and receding two-step planning."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse,json,time,hashlib
from dataclasses import asdict
import numpy as np
import torch
from src.config import load_config_for_run
from src.objectives.mocu.context import build_context_from_config
from src.objectives.msc.objective import validate_msc_support
from tools.audits.space import AuditBudget,SpacePlanner,summarize_runs,paired_interval

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--config',required=True);p.add_argument('--output',required=True)
 p.add_argument('--horizons',default='3');p.add_argument('--seeds',default='101,202,303')
 p.add_argument('--eval-systems',type=int,default=16);p.add_argument('--inner',type=int,default=16)
 p.add_argument('--outer',type=int,default=8);p.add_argument('--first-candidates',type=int,default=54)
 a=p.parse_args();hs=list(map(int,a.horizons.split(',')));seeds=list(map(int,a.seeds.split(',')))
 if not hs or min(hs)<2 or a.eval_systems<4:p.error('Need horizons >=2 and at least four systems')
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 cfg=load_config_for_run(a.config,ROOT,step_number=max(hs))
 cfg.raw.setdefault('experiment',{})['experiment_type']='msc_based';cfg.validate_msc()
 cfg.raw['observation'].update(cfg.raw['observation']['msc_based'])
 cfg.raw['experiment']['allow_trivial_fixed']=True
 ctx=build_context_from_config(cfg,project_root=ROOT,out_dir=out,smoke=False,experiment_type='msc_based')
 if max(hs)>ctx.n_actions:p.error('Horizon exceeds action catalog')
 budget=AuditBudget(inner=a.inner,outer=a.outer,first_candidates=a.first_candidates,calibration=256,fixed_restarts=8,histories_per_batch=4,actions_per_batch=3)
 planner=SpacePlanner(ctx.centres_support,ctx.U_support,ctx.u_grid,sigma=ctx.sigma_y,alpha=ctx.alpha,penalty=ctx.undercontrol_penalty,device='cuda' if torch.cuda.is_available() else 'cpu',budget=budget,objective='msc')
 systems=ctx.validation_systems[:a.eval_systems]
 assert len(systems)==a.eval_systems
 centres=np.stack([s['obs_clean'] for s in systems]);required=np.array([s['u_req'] for s in systems])
 prior=float(planner.risk(planner.prior())[0][0])
 doc={'schema':'msc_space_development_v1','status':'running','objective':'msc','posterior_coverage':cfg.raw['control']['posterior_coverage'],'horizons':hs,'seeds':seeds,'n_validation_systems':len(systems),'validation_indices':list(range(len(systems))),'budget':asdict(budget),'n_actions':planner.A,'n_support':planner.P,'prior_msc':prior,'structure':planner.structure_summary(),'primary_metric':'posterior selected MSC = control; no MOCU loss in ranking','safety_metric':'bank threshold proxy only, not fresh physical safety simulation','scope':'Development diagnostic on previously used training-validation bank; not independent confirmation; no final test systems used. Two-step lookahead and greedy Fixed are approximate; null gaps do not prove absence of opportunity.','planning':'all actions at second step; independently resampled second-stage values; first candidates limited by configured budget','support_hash':hashlib.sha256(ctx.centres_support.tobytes()+ctx.U_support.tobytes()).hexdigest(),'results':{}}
 (out/'audit_summary.json').write_text(json.dumps(doc,indent=2)+'\n')
 t=time.perf_counter()
 for h in hs:
  fixed,score=planner.fixed_sequence(h,104729+h);runs=[]
  print(f'[msc-audit] T={h} fixed={fixed} calibration_msc={score}',flush=True)
  for seed in seeds:
   r=planner.evaluate(centres,required,h,fixed,seed);runs.append(r)
   np.savez_compressed(out/f'paired_T{h}_seed{seed}.npz',**{f'{m}_{k}':v for m,row in r.items() for k,v in row.items() if isinstance(v,np.ndarray)})
   summary=summarize_runs(runs,prior,budget)
   for key,b,m in [('probe_gain','no_probe','random'),('design_gain','random','fixed')]:
    summary[key]=paired_interval(np.stack([x[b]['loss']-x[m]['loss'] for x in runs]),bootstrap=budget.bootstrap)
   for m in r:
    summary['methods'][m]['mean_msc_by_stage']=np.mean([x[m]['stage_scores'] for x in runs],axis=(0,1)).tolist()
   summary['fixed_sequence']=fixed;summary['completed_seeds']=len(runs)
   doc['results'][str(h)]=summary;doc['elapsed_seconds']=time.perf_counter()-t
   (out/'audit_summary.json').write_text(json.dumps(doc,indent=2)+'\n')
 doc.update(status='complete',elapsed_seconds=time.perf_counter()-t)
 (out/'audit_summary.json').write_text(json.dumps(doc,indent=2)+'\n')
 print('AUDIT_SUMMARY='+str(out/'audit_summary.json'),flush=True)
if __name__=='__main__':main()
