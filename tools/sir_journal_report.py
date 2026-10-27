"""Six-row journal reports from immutable metadata and actual outputs only."""
import csv,json,hashlib,math
from pathlib import Path
import numpy as np
METHODS=['random','fixed','dad','rl_sboed','vsoed','scorebed']
LABELS=dict(random='Random',fixed='Optimized Fixed',dad='Corrected DAD',rl_sboed='Corrected RL-sBOED',vsoed='Corrected vsOED',scorebed='Corrected ScoreBED')
def read(p):return json.loads(Path(p).read_text())
def stats(x):
 x=np.asarray(x);mean=float(x.mean());se=float(x.std(ddof=1)/len(x)**.5);return mean,se,mean-1.96*se,mean+1.96*se
def generate(C):
 C=Path(C);P=C/'reports';P.mkdir(exist_ok=True);plan=read(C/'plan.json');rows=[];scores={};prov={};reg=read(C/'registry.json')
 for m in METHODS:
  entry=next((x for x in reg if x['method']==m and not x['probe']),None)
  t={};cp=None;source=plan['source_commit'];status='MISSING';cost=entry.get('GPU_locked_wall_seconds') if entry else None
  if m=='random':status='VALID_NO_TRAINING';t=dict(generated_training_trajectories=0,training_seconds=0)
  elif m=='fixed':
   reuse=read(C/'metadata/fixed_reuse.json');cp=Path(reuse['checkpoint']);t=read(reuse['training_json']);status='VALID_BUT_REEVALUATION_REQUIRED';source=read(Path(reuse['run'])/'source_manifest.json').get('git_commit');cost=t.get('training_seconds')
  elif entry:
   status='VALID_LABPC_COMPLETE' if entry['status']=='COMPLETE' else entry['status'];cp=C/f'runs/{m}/models/{m}.pt';tp=cp.parent/f'{m}_training.json'
   if tp.exists():t=read(tp)
  else:
   probe=next((x for x in reg if x['method']==m and x['probe']),None)
   if probe and probe['status']!='COMPLETE':status='READINESS_FAILED_INCOMPLETE'
  row=dict(method=LABELS[m],method_id=m,training_seed=101,status=status,fresh_training_episodes=t.get('generated_training_trajectories'),optimizer_updates=t.get('actual_policy_updates',t.get('gradient_updates',t.get('selected_update',t.get('selected_checkpoint',{}).get('update')))),training_seconds=t.get('training_seconds'),pretraining_seconds=t.get('score_pretraining_seconds'),GPU_locked_phase_seconds=cost,peak_GPU_MiB=entry.get('peak_device_memory_MiB') if entry else None,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest() if cp and cp.exists() else None,source_revision=source,config_sha256=plan['config_sha256'],EIG=None,MC_SE=None,CI95_low=None,CI95_high=None,contrasts=None,paired_minus_DAD=None,paired_SE_DAD=None,paired_CI95_DAD_low=None,paired_CI95_DAD_high=None,paired_minus_vsOED=None,paired_SE_vsOED=None,paired_CI95_vsOED_low=None,paired_CI95_vsOED_high=None,latency_mean_seconds=None,latency_median_seconds=None,latency_p95_seconds=None,evaluation_status='PENDING')
  result=C/f'evaluations/development_{m}.json';arr=C/f'evaluations/development_{m}.npz'
  if result.exists() and arr.exists():
   d=read(result);a=np.load(arr);scores[m]=(a['scores'],str(a['identity']));row.update(EIG=d['EIG'],MC_SE=d['MC_SE'],CI95_low=d['CI95'][0],CI95_high=d['CI95'][1],contrasts=d['metadata']['contrasts'],latency_mean_seconds=d['latency_seconds']['mean'],latency_median_seconds=d['latency_seconds']['median'],latency_p95_seconds=d['latency_seconds']['p95'],evaluation_status=d['precision_status'])
  prov[m]=dict(checkpoint=str(cp) if cp else None,training_record=t,phase_record=entry,source_revision=source,original_training_config=str(Path(read(C/'metadata/fixed_reuse.json')['run'])/'run_config.json') if m=='fixed' else str(C/'configs/full.yaml'))
  rows.append(row)
 for row in rows:
  m=row['method_id']
  for reference,suffix in [('dad','DAD'),('vsoed','vsOED')]:
   if m in scores and reference in scores:
    assert scores[m][1]==scores[reference][1],'Invalid trajectory pairing'
    mean,se,lo,hi=stats(scores[m][0]-scores[reference][0]);row.update({f'paired_minus_{suffix}':mean,f'paired_SE_{suffix}':se,f'paired_CI95_{suffix}_low':lo,f'paired_CI95_{suffix}_high':hi})
 for filename,fields in [('SIR_SEED101_BASELINES.csv',list(rows[0])),('SIR_SEED101_COMPUTE.csv',['method','fresh_training_episodes','optimizer_updates','training_seconds','pretraining_seconds','GPU_locked_phase_seconds','peak_GPU_MiB'])]:
  with (P/filename).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(rows)
 (P/'SIR_BASELINE_PROVENANCE.json').write_text(json.dumps(dict(plan=plan,methods=prov,completed_evaluations=len(scores),source_manifest=read(C/'source_manifest.json')),indent=2)+'\n')
 lines=['# LabPC corrected SIR seed101 baselines','',f'Completed common evaluations: {len(scores)}/6. Missing numerical cells are pending, not zero. Fixed reuses its verified LabPC1M checkpoint. Other-machine checkpoints are never primary LabPC results.','', '| Method | Status | Episodes | EIG ± MC SE | Evaluation |','|---|---|---:|---:|---|']
 for r in rows:lines.append(f"|{r['method']}|{r['status']}|{r['fresh_training_episodes'] if r['fresh_training_episodes'] is not None else 'pending'}|{format(r['EIG'],'.6f')+' ± '+format(r['MC_SE'],'.6f') if r['EIG'] is not None else 'pending'}|{r['evaluation_status']}|")
 lines+=['','95% normal trajectory-MC CIs and valid paired case-wise differences are in CSV. MCSE is conditional on checkpoint and contrast bank, not training-seed variation. Native objectives and compute differ;1Mepisodes is matched data only. Seed101 is development evidence. Confirmation/replications are separate. No posterior accuracy is claimed.']
 (P/'SIR_SEED101_BASELINES.md').write_text('\n'.join(lines)+'\n')
 resolution=C/'evaluations/evaluator_freeze.json';(P/'SIR_EIG_RESOLUTION_RESULTS.md').write_text('# EIG resolution results\n\n'+('```json\n'+resolution.read_text()+'\n```\n' if resolution.exists() else 'PENDING new predeclared calibration. Historical16384-contrast estimates were inadequate;1Mtail remains unresolved.\n'))
 conv=['# Seed101 convergence diagnostics','','Checkpoint choice is final1M, never the maximum validation EIG. Missing curves or still-rising curves do not establish convergence. Native Score pretraining has no deployable policy; its3policy updates under1Mallocation are budget-limited evidence. Fixed legacy curve uses a different held-out contrast resolution and is not silently pooled.']
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(7,4));present=[r for r in rows if r['EIG'] is not None]
 if present:ax.errorbar([r['method'] for r in present],[r['EIG'] for r in present],yerr=[1.96*r['MC_SE'] for r in present],fmt='o',capsize=3)
 else:ax.text(.5,.5,'Common EIG evaluation pending',transform=ax.transAxes,ha='center')
 ax.set_ylabel('Terminal sPCE (nats),95% MC CI');ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(P/'SIR_EIG_COMPARISON.pdf');plt.close(fig)
 for filename,key,ylabel in [('SIR_TRAINING_COST.pdf','training_seconds','Offline training seconds'),('SIR_INFERENCE_COST.pdf','latency_mean_seconds','Mean decision seconds')]:
  fig,ax=plt.subplots(figsize=(7,4));present=[r for r in rows if r[key] is not None]
  if present:ax.bar([r['method'] for r in present],[r[key] for r in present])
  else:ax.text(.5,.5,'Measurements pending',transform=ax.transAxes,ha='center')
  ax.set_ylabel(ylabel);ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(P/filename);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4));has=False
 for m in ['dad','rl_sboed','vsoed','scorebed']:
  h=prov[m]['training_record'].get('journal_validation_history',[])
  if h:
   ax.errorbar([x['episodes'] for x in h],[x['eig'] for x in h],yerr=[1.96*x['mc_standard_error'] for x in h],label=LABELS[m],marker='o',capsize=2);has=True
   conv.append(f"\n{LABELS[m]}: {len(h)} validation points; last status `{h[-1]['status']}`. Last EIG {h[-1]['eig']:.6f} ± {h[-1]['mc_standard_error']:.6f}. No convergence certification from final loss alone.")
 if has:ax.legend(fontsize=8)
 else:ax.text(.5,.5,'Common learning curves pending',transform=ax.transAxes,ha='center')
 ax.set_xlabel('Fresh primary training episodes');ax.set_ylabel('Development terminal sPCE,95% MC CI');fig.tight_layout();fig.savefig(P/'SIR_EIG_VS_EPISODES.pdf');plt.close(fig)
 (P/'SIR_SEED101_CONVERGENCE.md').write_text('\n'.join(conv)+'\n')
