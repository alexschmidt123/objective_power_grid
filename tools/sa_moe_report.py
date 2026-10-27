"""Regenerate SA-MoE reports/figures from actual arrays; never hardcoded EIG."""
if __name__ == '__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: archived exploratory result generation')
import argparse,json,csv,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);a=p.parse_args();C=Path(a.campaign);reports=C/'reports';fig=C/'figures';fig.mkdir(exist_ok=True)
 read=lambda path:json.loads(path.read_text())
 summary=[]
 for name in ['sa_moe','single']:
  f=C/'costs'/f'{name}_final.json'
  if f.exists():summary.append(dict(model=name,**read(f)))
 (reports/'training_costs.json').write_text(json.dumps(summary,indent=2))
 dev=reports/'development_comparison.json'
 if dev.exists():
  x=read(dev);rows=[]
  for mode in ['learned','uniform','single']:
   vals=np.concatenate([row['trajectory_scores'] for row in x['rows'] if row['mode']==mode]);rows.append(dict(model=mode,mean=float(vals.mean()),MC_SE=float(vals.std(ddof=1)/np.sqrt(len(vals))),trajectories=len(vals),contrasts=16384))
  with (reports/'development_results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  stage=x['expert_by_stage'];plt.figure(figsize=(6,3.5));plt.stackplot([r['stage'] for r in stage],np.array([r['hard_frequency'] for r in stage]).T,labels=[f'Expert {i+1}' for i in range(4)]);plt.xlabel('Stage');plt.ylabel('Selection frequency');plt.legend(loc='upper right');plt.tight_layout()
  for ext in ['png','pdf']:plt.savefig(fig/f'utilization.{ext}',dpi=220)
  plt.close()
 heldout=reports/'heldout_expert_competence.json'
 if heldout.exists():
  x=read(heldout);contexts=x['contexts'];Q=np.array([row['competence']['Q'] for row in contexts]);SE=np.array([row['competence']['Q_SE'] for row in contexts]);plt.figure(figsize=(6,4));plt.imshow(Q,aspect='auto',cmap='viridis');plt.xticks(range(4),[f'Expert {i+1}' for i in range(4)]);plt.yticks(range(len(contexts)),[f"Held-out stage {r['stage']}" for r in contexts]);plt.colorbar(label='Remaining-information lower bound (nats)')
  for i in range(len(contexts)):
   for e in range(4):plt.text(e,i,f'{Q[i,e]:.2f}\n±{SE[i,e]:.2f}',ha='center',va='center',color='white',fontsize=8)
  plt.tight_layout()
  for ext in ['png','pdf']:plt.savefig(fig/f'competence_heatmap.{ext}',dpi=220)
  plt.close();resolved=[r for r in contexts if r['resolved_winner']];distinct=len(set(r['highest_estimated_expert'] for r in resolved))
  (reports/'SA_MOE_EXPERT_SPECIALIZATION.md').write_text('# Measured held-out specialization\n\n'+json.dumps(contexts,indent=2)+f'\n\nResolved winning experts in {len(resolved)} contexts: {distinct} distinct experts. At least2 distinct resolved winners are necessary but insufficient for a robust specialization claim. Only4held-out contexts, one training seed; finitecontrast andSMCbias remain. Learned/uniform fulltrajectory development comparisons saved separately.\n')
 results=[]
 for name in ['main','single']:
  path=C/'evaluations'/name/'result.json'
  if path.exists():results.append(read(path))
 (reports/'SA_MOE_SEED101_RESULTS.md').write_text('# Saved seed101 status\n\n'+('No completed official evaluation. Training/development are not official EIG results.' if not results else json.dumps(results,indent=2))+'\n\nOne training seed only. MCSE is trajectory-evaluation uncertainty, not training-seed SD. New training is additional-compute; no matched-to-IC claim.\n')
 if len(results)==2:
  d=np.load(C/'evaluations/main/scores.npy')-np.load(C/'evaluations/single/scores.npy');(reports/'paired_single_comparison.json').write_text(json.dumps(dict(mean=float(d.mean()),paired_MC_SE=float(d.std(ddof=1)/np.sqrt(len(d))),coupling='Canonical same truth/noise/contrast bank; adaptive policies differ'),indent=2))
 print(json.dumps(dict(training_costs=len(summary),official_results=len(results),reports=str(reports))))
if __name__=='__main__':main()
