"""Strict ten-method reporting; historical scores never replace four new baselines."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
METHODS=['random','fixed','dad','rl_sboed','vsoed','scorebed','moe','resmoe','switchmoe','switchresmoe'];NEW=['dad','rl_sboed','vsoed','scorebed']
TRAIN_SHA='16632ae4b00703ffb1cc4d9b6a956e4a953f00c7'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def table(p,rows):
 with Path(p).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stat(x):return float(x.mean()),float(x.std(ddof=1)/len(x)**.5)
def summarize(root,historical,mirror,status):
 out=root/'reports';out.mkdir(exist_ok=True);historical=Path(historical);mirror=Path(mirror);oldproof=read(historical/'SIR_SEED101_PROVENANCE.json');oldcsv={r['method']:r for r in csv.DictReader((historical/'SIR_SEED101_ALL_BASELINES.csv').open())};oldarr=np.load(historical/'SIR_seed101_paired_scores.npz');scores={m:oldarr[m] for m in METHODS if m not in NEW};records={};prov={};costs={};validation=[]
 H=Path.home()/'Documents';cases=H/'objective_power_grid_e1_sir_verified_20261019/experiments/10192026/10192026_e1_sir_verification/evaluations/cases.npz';a=np.load(cases)
 for m in METHODS:
  if m not in NEW:
   row=oldcsv[m];p=oldproof['methods'][m];records[m]=dict(status='VALID_COMPLETED',reason='Validated frozen E1 method retained; native settings unchanged for this method',EIG=float(row['EIG']),MC_SE=float(row['MC_SE']),training_episodes=int(row['training_episodes']),optimizer_updates=row['optimizer_updates'],training_wall_seconds=float(row['training_wall_seconds']),base_seconds=float(row['additional_base_seconds']),checkpoint_source=row['checkpoint_source'],checkpoint_sha256=row['checkpoint_sha256'],source_commit=row['source_commit'],config_sha256=row['config_sha256'],source_manifest_sha256=row['source_manifest_sha256']);prov[m]=p;costs[m]=dict(new_training_seconds=0,new_primary_episodes=0,readiness_wall_seconds=0);continue
  run=mirror/'runs'/m;completion=(run/'exit_code').exists() and (run/'exit_code').read_text().strip()=='0'
  records[m]=dict(status='INCOMPLETE',reason='Strict replacement pending; historical configuration is INVALID for the requested strict protocol',EIG=None,MC_SE=None,training_episodes=None,optimizer_updates=None,training_wall_seconds=None,base_seconds=0,checkpoint_source=str(run),checkpoint_sha256=None,source_commit=TRAIN_SHA,config_sha256=None,source_manifest_sha256=None);costs[m]=dict(new_training_seconds=None,new_primary_episodes=None,readiness_wall_seconds=None)
  if not completion:continue
  t=read(run/'models'/f'{m}_training.json');cfg=read(run/'run_config.json');manifest=read(run/'source_manifest.json');assert manifest['git_commit']==TRAIN_SHA
  for rel,digest in manifest['sha256'].items():assert sha(run/'source'/rel)==digest,(m,rel)
  cp=run/'models'/f'{m}.pt';assert sha(cp)==t['checkpoint_sha256'];assert t['generated_training_trajectories']==1000000 and t['selected_checkpoint']['episodes']==1000000 and t['checkpoint_selection']=='final_predeclared_fixed_budget'
  if m=='dad':assert t['batch_size']==2000 and t['initial_learning_rate']==5e-5 and t['effective_training_settings']['contrasts']==2000
  if m=='rl_sboed':assert t['contrasts']==100000 and t['replay_capacity']==1000000
  if m=='scorebed':assert t['score_batch']==1024 and t['potential_hidden_dim']==256 and t['actual_policy_updates']==3
  if m=='vsoed':
   profile=cfg['benchmark']['research']['journal_protocol']['native_vsoed_profile'];assert profile['new_trajectories']==1000 and profile['replay_batch']==10000 and profile['transition']==10000
  assert cfg['paired_evaluation_files']['cases']['sha256']==oldproof['bank_manifest']['sha256']['cases.npz'];assert cfg['paired_evaluation_files']['contrasts']['sha256']==oldproof['bank_manifest']['sha256']['contrasts_1m.npy']
  rows=read(run/'eval'/f'{m}_seed1001.json');assert len(rows)==2048;np.testing.assert_array_equal([x['truth'] for x in rows],a['truth']);np.testing.assert_array_equal([x['measurement_noise_standard'] for x in rows],a['noise']);scores[m]=np.asarray([r['spce'] for r in rows]);assert np.isfinite(scores[m]).all();eig,se=stat(scores[m])
  updates=t.get('native_optimizer_updates',t.get('actual_updates',t.get('gradient_updates')))
  if m=='rl_sboed':updates={k:t[k] for k in ['actor_updates','critic_updates','temperature_updates']}
  if m=='scorebed':updates=dict(score=t['actual_score_updates'],policy=t['actual_policy_updates'])
  records[m].update(status='VALID_COMPLETED',reason='New strict1Mcheckpoint and common1Mevaluation fully verified',EIG=eig,MC_SE=se,training_episodes=1000000,optimizer_updates=json.dumps(updates,sort_keys=True),training_wall_seconds=t['training_seconds'],checkpoint_sha256=sha(cp),config_sha256=sha(run/'run_config.json'),source_manifest_sha256=sha(run/'source_manifest.json'))
  prov[m]=dict(run=str(run),source_commit=manifest['git_commit'],manifest_sha256=sha(run/'source_manifest.json'),source_files_verified=len(manifest['sha256']),training=t,config=cfg,checkpoint_sha256=sha(cp));costs[m].update(new_training_seconds=t['training_seconds'],new_primary_episodes=1000000);validation.append(m)
 result=[];compute=[]
 for m in METHODS:
  r=records[m];dd=sdd=dv=sdv=None
  if m in scores and 'dad' in scores:dd,sdd=stat(scores[m]-scores['dad'])
  if m in scores and 'vsoed' in scores:dv,sdv=stat(scores[m]-scores['vsoed'])
  result.append(dict(method=m,status=r['status'],EIG=r['EIG'],MC_SE=r['MC_SE'],paired_minus_DAD=dd,paired_SE_vs_DAD=sdd,paired_minus_vsOED=dv,paired_SE_vs_vsOED=sdv,training_episodes=r['training_episodes'],optimizer_updates=r['optimizer_updates'],training_wall_seconds=r['training_wall_seconds'],additional_base_seconds=r['base_seconds'],checkpoint_source=r['checkpoint_source'],checkpoint_sha256=r['checkpoint_sha256'],source_commit=r['source_commit'],config_sha256=r['config_sha256'],source_manifest_sha256=r['source_manifest_sha256'],evaluation_status='COMMON_1M_FINITE_BOUND_PROVISIONAL_RESOLUTION' if m in scores else 'NOT_YET_EVALUATED_STRICT_CHECKPOINT',reason=r['reason']))
  compute.append(dict(method=m,status=r['status'],main_primary_episodes=r['training_episodes'],native_optimizer_updates=r['optimizer_updates'],main_training_seconds=r['training_wall_seconds'],shared_base_seconds=r['base_seconds'],new_training_seconds=costs[m]['new_training_seconds'],new_primary_episodes=costs[m]['new_primary_episodes'],extra_readiness_wall_seconds=costs[m]['readiness_wall_seconds'],prior_E1_training_seconds=oldcsv[m]['training_wall_seconds'],policy_initialization_and_score_pretraining='Included inside1Mmain budget; explicit native record in provenance' if m=='scorebed' else 'Residual500kbase included' if m in ['resmoe','switchresmoe'] else 'None inherited',fairness='Equal primary episodes, not equal wall time/updates/likelihood work; new probes extra'))
 table(out/'SIR_SEED101_ALL_BASELINES.csv',result);table(out/'SIR_SEED101_COMPUTE.csv',compute);np.savez(out/'SIR_SEED101_PAIRED_SCORES.npz',**scores);completed=len(scores)
 proof=dict(protocol='STRICT_RESTORATION_EXPLICIT_USER_CHOICE',training_commit=TRAIN_SHA,completed_methods=completed,total_methods=10,new_strict_completed=validation,reused=[m for m in METHODS if m not in NEW],records=prov,status_snapshot=status,historical_protocol='Separate, not substituted for four new strict methods',evaluation=dict(N=2048,L=1000000,seed=1001,common_cases_sha256=oldproof['bank_manifest']['sha256']['cases.npz'],common_bank_sha256=oldproof['bank_manifest']['sha256']['contrasts_1m.npy'],resolution='Frozen E1finite-bound configuration;16k inadequate;full10-method convergence beyond1Munverified'),reserved8831001_used=False,source='LabPC aggregation only')
 (out/'SIR_SEED101_PROVENANCE.json').write_text(json.dumps(proof,indent=2))
 text=[f'# Strict SIR ODE seed101: {completed}/10 completed','',
 'This table follows the user-selected later strict restoration. Six compatible methods are reused; DAD,RL-sBOED,vsOEDandScoreBEDare fresh strict-protocol runs. Their old E1scores are historical only and are NEVER substituted when a strict result is missing.','',
 '|Method|Status|EIG +/- MC SE|','|---|---|---:|']
 for r in result:text.append(f"|{r['method']}|{r['status']}|"+(f"{r['EIG']:.8f} +/- {r['MC_SE']:.8f}" if r['EIG'] is not None else 'Unavailable')+'|')
 text+=['','The CSV includes paired differences and SE against strict DAD/vsOEDwhen available. Missing reference policies leave those comparisons blank. Main training must reach exactly1Mprimary episodes and use the final-budget checkpoint before a row is marked complete. Resource-capped/failed runs remain incomplete; partial checkpoints are not promoted.','',
 'Physics:T5on-the-fly adaptive RK4 SIR,population500,I0=2,original log-normal rates,Gaussian infected-count noise sigma1,horizon100,minspacing.1. The shared residual base remains internal:each residual method includes500kbase+500kcontinuation episodes and its base cost. No extraMoE/FHtraining or new seeds are launched.','',
 'Strict settings:DAD B2000/L2000/lr5e-5;native REDQ100kcontrasts/replay1M;vsOED1000fresh episodes/update,replay10000,980updates and original transition10000;ScoreBEDbatch1024,native GRU policy,corrected256potential head,100014initialization+897736score+2250policy episodes(3policy updates). No schedule rescaling or convergence claim is made.','',
 'Evaluator:common2048development truth/noise draws and1Mcontrast bank,seed1001;every complete row verifies identity hashes. The existing nested study establishes16kcompression.1Mwas resource-frozen by E1scope;partial1M/4Mresults do not prove full10-method numerical convergence at.02/.01targets. EIG comparisons remain finite-bound/development and resolution-qualified. Reserved8831001is untouched. MCSE is conditional on checkpoint/shared bank,not training-seed variability.','',
 'Native algorithms have different optimizer counts and likelihood/simulator work;equal1Mprimary episodes do not imply equal compute. Extra native-size probes and historical superseded experiments are separately disclosed. Source/config/checkpoint hashes and native counters are retained in provenance.','',
 f'Current queue snapshot: `{json.dumps(status,sort_keys=True)}`. The queue and reporting process continue independently of the location audit. Generated tables update as verified strict phases complete; Git history preserves prior statuses. No main merge.','']
 (out/'SIR_SEED101_ALL_BASELINES.md').write_text('\n'.join(text));return completed

def main():
 p=argparse.ArgumentParser();p.add_argument('--historical',required=True);p.add_argument('--mirror',required=True);p.add_argument('--status',required=True);a=p.parse_args();R=Path(__file__).resolve().parents[1];print('STRICT_COMPLETED',summarize(R,a.historical,a.mirror,read(a.status)))
if __name__=='__main__':main()
