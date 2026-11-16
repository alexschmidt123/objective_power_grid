"""CPU-only, evidence-derived campaign reporting; never selects checkpoints."""
import json,math
from pathlib import Path
import numpy as np

def update_reports(C,ready):
 from src.research.sir_t3_campaign import table,write
 rows=table(C,ready);text='# T3 training costs\n\nFour-hour maximum per optimized/trainable method. Profiling is separate and its discarded weights are never reused.\n\n|Method|Actual optimizer updates|Fresh trajectories|Charged seconds|Native schedule complete|Status|\n|---|---:|---:|---:|---|---|\n'
 for r in rows:text+=f"|{r['method']}|{r['optimizer_updates']}|{r['training_trajectories']}|{r['charged_training_seconds']}|{r['native_schedule_complete']}|{r['status']}|\n"
 text+='\nReadiness measurements:\n\n'+json.dumps(ready,indent=2)+'\n';(C/'reports/SIR_T3_TRAINING_COSTS.md').write_text(text)
 convergence={};curves={};markdown='# T3 learning curves\n\nFixed development cases:128 trajectories,65536 contrasts. Checkpoints are time-selected, never selected by favorable EIG. MC SE does not establish training-seed uncertainty.\n\n|Method|Updates|Charged seconds|Development EIG|MC SE|\n|---|---:|---:|---:|---:|\n'
 for method in ['fixed','dad','rl_sboed','vsoed','scorebed']:
  p=C/'logs'/f'{method}_learning_curve.json'
  if not p.exists():convergence[method]=dict(status='INVALID_OR_INCOMPLETE',reason='No eligible primary learning curve yet; readiness may be blocked');continue
  curve=json.loads(p.read_text());curves[method]=curve
  for x in curve:markdown+=f"|{method}|{x['update']}|{x['charged_seconds']:.3f}|{x['development']['mean']:.6f}|{x['development']['MC_SE']:.6f}|\n"
  equivalence=[]
  for a,b in zip(curve[-3:-1],curve[-2:]):
   pa=C/'evaluations/development'/f"{method}_update{a['update']}_L65536.npy";pb=C/'evaluations/development'/f"{method}_update{b['update']}_L65536.npy"
   if not(pa.exists() and pb.exists()):continue
   d=np.load(pb)-np.load(pa);mu=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d)));equivalence.append(dict(mean_change=mu,paired_MC_SE=se,upper_absolute_change=abs(mu)+1.96*se,interval_seconds=b['charged_seconds']-a['charged_seconds']))
  plateau=len(curve)>=3 and len(equivalence)==2 and all(x['upper_absolute_change']<=.05 and x['interval_seconds']>=1500 for x in equivalence)
  convergence[method]=dict(status='CONVERGED_OR_PLATEAU_SUPPORTED' if plateau else 'COMPUTE_LIMITED',evidence=equivalence,reason='Predeclared last-two paired change bounds <=.05 nats over >=1500s each; one training seed' if plateau else 'No supported plateau under predeclared equivalence assessment; finite schedule completion is not proof of convergence')
  complete=C/'manifests'/f'{method}_complete.json'
  if complete.exists():x=json.loads(complete.read_text());x['scientific_status']=convergence[method]['status'];write(complete,x)
 (C/'reports/SIR_T3_LEARNING_CURVES.md').write_text(markdown)
 write(C/'manifests/convergence_assessment.json',convergence);(C/'reports/SIR_T3_CONVERGENCE_ASSESSMENT.md').write_text('# T3 convergence assessment\n\n'+json.dumps(convergence,indent=2)+'\nNo paper-level convergence claim follows from completing a finite schedule.\n')
 if curves:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(7,4))
  for method,curve in curves.items():ax.errorbar([x['charged_seconds']/3600 for x in curve],[x['development']['mean'] for x in curve],yerr=[1.96*x['development']['MC_SE'] for x in curve],marker='o',label=method)
  ax.set(xlabel='Charged training hours',ylabel='Development terminal sPCE (nats)');ax.legend();fig.tight_layout();fig.savefig(C/'reports/sir_t3_learning_curves.png',dpi=160);plt.close(fig)
 table(C,ready)
