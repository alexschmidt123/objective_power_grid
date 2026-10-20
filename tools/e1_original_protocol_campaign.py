"""Reference-protocol correction using existing journal campaign/launchers."""
import argparse,csv,hashlib,json,subprocess,sys,copy,time
from pathlib import Path
import yaml
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.research.original_protocol import score_allocation,vsoed_profile,dad_settings
from tools import journal_campaign as j
C=R/'experiments/10192026/10192026_e1_original_baseline_restoration'
MASTER=Path.home()/'Documents/objective_power_grid'
def dump(name,x):j.write(C/name,x)
def table(name,rows):j.table(C/'reports'/name,rows,list(rows[0]))
def prepare():
 for d in ['metadata','reports','configs','logs','evaluations','frozen_source','figures','plotting']:(C/d).mkdir(parents=True,exist_ok=True)
 allocations=[]
 for benchmark,label,config in [('location_finding','location','e1_location_original_methods.yaml'),('sir_ode','sir','e1_sir_native.yaml')]:
  raw=yaml.safe_load((R/'configs'/config).read_text());p=raw['research']['journal_protocol'];p['study_id']='E1_original_protocol_budget_only_20261019';p['native_vsoed_profile']=vsoed_profile(benchmark);p['checkpoint_selection_mode']='final_fixed_budget';p['training_seeds']=[101,102,103,104,105];p['training_device']='cuda_if_available';p['common_pathwise_settings'].pop('learning_rate',None)
  p['method_settings']['dad'].update(dad_settings());p['method_settings']['rl_sboed'].update(rl_original_reference=True,rl_native_contrasts=100000,rl_native_replay_capacity=1000000)
  allocation=score_allocation(1000000,benchmark);allocations.append(dict(benchmark=benchmark,**allocation));p['method_settings']['scorebed'].update(scorebed_pretraining_episodes=allocation['score'],scorebed_batch_size=1024,scorebed_reference_sampler=True,scorebed_restarts=1)
  p['stopping_rule'].update(minimum_episodes=1000000,maximum_episodes=1000000,maximum_training_seconds=172800);p['validation']['device']='cpu';p['final_evaluation'].update(systems=2048 if label=='sir' else 2000,contrasts=1000000,seed=1002)
  raw['research']['settings'].update(eval_systems=2048 if label=='sir' else 2000,eval_contrasts=1000000,evaluation_population_systems=2048 if label=='sir' else 2000,evaluation_contrast_chunk=4096)
  raw['research']['paper_protocol']['nominal_training_trajectories']=1000000
  (C/'configs'/f'{label}_1000000.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
  if label=='sir':
   half=copy.deepcopy(raw);half['research']['journal_protocol']['stopping_rule'].update(minimum_episodes=500000,maximum_episodes=500000);half['research']['paper_protocol']['nominal_training_trajectories']=500000
   (C/'configs/sir_500000.yaml').write_text(yaml.safe_dump(half,sort_keys=False))
  (R/'configs'/f'e1_{label}_published_budget_only.yaml').write_text(yaml.safe_dump(raw,sort_keys=False))
 table('scorebed_allocation_calculation.csv',allocations)
 dump('registry.json',[])
 refs=json.loads((C/'references/upstream_reference_versions.json').read_text());dump('metadata/pinned_upstream_commits.json',{k:{a:v for a,v in x.items() if a!='files'} for k,x in refs.items()})
 entries=json.loads((MASTER/'experiments/10182026/10182026_e1_four_system/registry.json').read_text())
 inventory=[];matrix=[];decisions=[];vsrows=[]
 for e in entries:
  if e['method'] not in ['dad','rl_sboed','vsoed','scorebed']:continue
  run=Path(e['training_run']);m=e['method'];t=json.loads((run/'models'/f'{m}_training.json').read_text());s=t.get('effective_training_settings',{});components=[]
  if m=='dad':components=[('batch_size',2000,t.get('batch_size')),('contrasts',2000,s.get('contrasts')),('learning_rate',5e-5,t.get('initial_learning_rate'))]
  if m=='rl_sboed':components=[('contrasts',100000,t.get('contrasts')),('replay_capacity',1000000,t.get('replay_capacity')),('discount',.9,t.get('gamma'))]
  if m=='scorebed':components=[('batch_size',1024,t.get('score_batch')),('score_primary_episodes',940486,t.get('score_training_trajectories')),('policy_primary_episodes',49500,t.get('policy_training_trajectories'))]
  if m=='vsoed':
   profile=t.get('profile',{});expected=vsoed_profile('location_finding');components=[(k,v,profile.get(k)) for k,v in expected.items() if k not in ['n_incre','posterior_coordinate_reference','log_every']]
   # Old bounded trainer defaults to horizon increments when profile lacks n_incre.
   cp=run/'models/vsoed.pt'
   import torch
   ck=torch.load(cp,map_location='cpu',weights_only=False);count=len(ck.get('posterior_states',{}));components.append(('n_incre',30,profile.get('n_incre',count or None)))
  bad=False
  for component,expected,actual in components:
   ok=actual==expected;bad|=not ok;row=dict(benchmark='location',method=m,component=component,original_setting=str(expected),executed_setting=str(actual),difference='none' if ok else 'executed differs',classification='EXACT_ORIGINAL' if ok else 'UNVERIFIED' if actual is None else 'INCORRECT_METHOD_MODIFICATION',source_reference='pinned upstream README/experiments/source.py; references/pinned_upstream_commits',action='reuse' if ok else 'verify/retrain');matrix.append(row)
   if m=='vsoed':vsrows.append(dict(seed=e['seed'],**row))
  cp=run/'models'/f'{m}.pt';decisions.append(dict(benchmark='location',method=m,seed=e['seed'],run=str(run),checkpoint_sha256=j.sha(cp),decision='RETRAIN' if bad else 'REUSE',reason='strict original hyperparameters differ' if bad else 'actual frozen native profile matches',source_manifest_sha256=e.get('source_manifest_sha256','')))
  inventory.append(dict(benchmark='location',method=m,seed=e['seed'],run=str(run),training=t,source_manifest=e.get('source_manifest_sha256')))
 sir=MASTER/'experiments/10182026/10182026_e1_sir_t5_seed101'
 for m in ['dad','rl_sboed','vsoed','scorebed']:
  run=sir/'sir/seed101'/m
  if not run.exists():run=sir/'worker_ubuntu/sir/seed101'/m
  if m=='scorebed':run=Path.home()/'Documents/objective_power_grid_e1_sir_verified_20261019/experiments/10192026/10192026_sir_e1_corrected_implementations/worker_ubuntu/sir/seed101/scorebed'
  t=json.loads((run/'models'/f'{m}_training.json').read_text());s=t.get('effective_training_settings',{});profile=t.get('profile',{})
  if m=='vsoed':components=[(k,v,profile.get(k)) for k,v in vsoed_profile('sir_ode').items() if k not in ['log_every','posterior_coordinate_reference']]+[('posterior_log_coordinate_limits','[-6,4];sigma<=0.5','old Z bounds[-6,6];sigma<=1')]
  elif m=='dad':components=[('batch_size',2000,t.get('batch_size')),('contrasts',2000,s.get('contrasts')),('learning_rate',5e-5,t.get('initial_learning_rate'))]
  elif m=='rl_sboed':components=[('contrasts',100000,t.get('contrasts')),('replay_capacity',1000000,t.get('replay_capacity')),('discount',.9,t.get('gamma'))]
  else:components=[('batch_size',1024,t.get('score_batch')),('score_primary_episodes',897736,t.get('score_training_trajectories')),('policy_primary_episodes',2250,t.get('policy_training_trajectories')),('sampler','native dynamical mixture first-coordinate marginal + chronological map','sequential uniform')]
  for component,expected,actual in components:
   row=dict(benchmark='sir',method=m,component=component,original_setting=str(expected),executed_setting=str(actual),difference='none' if expected==actual else 'executed differs',classification='EXACT_ORIGINAL' if expected==actual else 'INCORRECT_METHOD_MODIFICATION',source_reference='pinned experiments/sir.py; ScoreBED paper Table9',action='reuse component' if expected==actual else 'correct/retrain');matrix.append(row)
   if m=='vsoed':vsrows.append(dict(seed=101,**row))
  decisions.append(dict(benchmark='sir',method=m,seed=101,run=str(run),checkpoint_sha256=j.sha(run/'models'/f'{m}.pt'),decision='RETRAIN',reason='strict original hyperparameters/phase settings differ',source_manifest_sha256='preserved in original run'))
  inventory.append(dict(benchmark='sir',method=m,seed=101,run=str(run),training=t))
 table('original_baseline_fidelity_matrix.csv',matrix);table('vsoed_reference_configuration_comparison.csv',vsrows);table('baseline_reuse_or_retrain_decisions.csv',decisions);dump('metadata/actual_frozen_training_inventory.json',inventory)
 (C/'reports/baseline_budget_definition.md').write_text('B_train counts newly generated complete T-stage primary trajectories exactly once. Include initialization, normalization and fresh pretraining/policy episodes. Replay is not a new episode; contrast prior draws are not primary simulations. Separate validation/evaluation and likelihood/ODE work. Final-budget checkpoint, no best-EIG selection. Preserve fixed original initialization and batches; only final partial minibatches counted explicitly.\n')
 (C/'reports/published_training_protocols.md').write_text('DAD: ae-foster/dad README Location Finding command: batch2000, L2000, Adam5e-5,beta(.8,.998),gamma.98 every1000 updates, sum256/16 shared empty-history emitter. SIR transfers this original architecture/settings, with normalized time/count adapter. RL: csiro-mlai/RL-BOED README source command: L100000,discount.9,tau.001,pi1e-4,Q3e-4,replay1M,two critics, native batch4096 and64updates/100fresh episodes. Earlier10M replay description was incorrect. vsOED: pinned experiments/source.py +README incremental command; experiments/sir.py +README terminal GMM command. Preserve20k initialization,1k new,10k replay,1M capacity,5critic/5posterior steps,per-update exponential/exploration schedules. SIR transition10000 remains10000 even when E1 ends at980updates; not scaled to force a target transition. SIR trains in native log(beta),log(gamma) coordinates with bounds[-6,4],maxsigma.5 and the original training prior-term convention; no bounds are copied into standardized coordinates. ScoreBED P1: paper AppendixD.7 Tables8/9, location190000:10000, dynamics199500:500, batches1024. P1 no restart selection. SIR schedule transferred from dynamics, not a published SIR reproduction. SIR uses the published double-pendulum dynamics profile: native IqbalGRUNet with 256/256 encoder,64embedding,two64 GRU cells,256/256 emitter,stochastic Gaussian; native constant1e-4 Adam(.9,.999), clip1, positional conservative Transformer, score weight.02/clip10000; fixed100k normalization+4scoreinit+10policyinit. The native mixture sampler is retained, projected from its first-coordinate marginal and passed through the chronological map. Scalar-potential head256 remains a disclosed necessary dimension adapter. Native Transformer/loss/Adam/EMA and learning-rate formulas retained. SIR only3policy updates under1M: budget-truncated, not converged. No E2 training.\n')
 (C/'reports/e1_corrected_baselines_report.md').write_text('IN PROGRESS. Prior code-equivalence checks did not imply original hyperparameter equivalence. Actual protocol departures are recorded component-wise. Location vsOED reuse is conditional on30posterior increments and all reference-profile checks. All old results preserved. Baseline replacement results pending; no fabricated numerical rows. New source is isolated from running frozen MoE.\n')
 for name in ['corrected_e1_run_registry.csv','corrected_location_e1_results.csv','corrected_sir_e1_results.csv']:
  (C/'reports'/name).write_text('benchmark,method,seed,status,eig,mc_se,run\n')
 print('Prepared exact profiles and actual-run compatibility decisions',C)
def seal():
 assert not (C/'location_freeze.json').exists()
 raw={'status':'frozen_original_method_budget_only','budget':1000000,'seeds':[101,102,103,104,105],'development_eval_seed':1001,'contrast_samples':1000000,'E2_not_executed':True};dump('protocol.json',raw);(C/'journal_protocol.md').write_text((C/'reports/published_training_protocols.md').read_text()+'\n'+(C/'reports/baseline_budget_definition.md').read_text())
 paths=[*R.glob('src/**/*.py'),*R.glob('tools/**/*.py'),R/'run.sh',R/'sweep_run.sh',*C.glob('configs/*.yaml')]
 dump('source_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),sha256={str(p.relative_to(R)):j.sha(p) for p in paths}))
 import shutil
 for p in paths:
  target=C/'frozen_source'/p.relative_to(R);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 dump('location_freeze.json',dict(protocol_sha256=j.sha(C/'protocol.json'),written_protocol_sha256=j.sha(C/'journal_protocol.md')))
 print('Frozen',C)
def run(benchmark,method,seeds=(101,102,103,104,105)):
 import fcntl
 lock=(C/"metadata/device.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 j.C=C;j.guard()
 for seed in seeds:
  e=j.run_phase(benchmark,seed,method,timeout_seconds=259200,evaluation_cases=C/'evaluations'/f'{benchmark}_cases.npz',evaluation_bank=C/'evaluations'/f'{benchmark}_contrasts.npy')
  if e['status']!='complete':raise RuntimeError('Preserved failed replacement; stop affected cohort')
def run_probed(benchmark,method,seeds):
 probe=C/'reports'/f'cost_probe_{benchmark}_{method}.json'
 if not probe.exists():
  subprocess.run([sys.executable,'tools/e1_reference_cost_probe.py','--benchmark',benchmark,'--method',method],cwd=R,check=True)
 proof=json.loads(probe.read_text());assert proof.get('exit_code')==0,proof
 dump(f'metadata/{benchmark}_{method}_native_size_gate.json',dict(status='PASS',proof=str(probe),proof_sha256=j.sha(probe),acceptance='Published algorithm/hyperparameters; exact budget scaling; no EIG threshold'))
 run(benchmark,method,seeds)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','seal','run','run-probed']);p.add_argument('--benchmark',choices=['location','sir']);p.add_argument('--method',choices=['dad','rl_sboed','vsoed','scorebed']);p.add_argument('--seeds',default='101,102,103,104,105');a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='seal':seal()
 elif a.action=='run-probed':run_probed(a.benchmark,a.method,[int(x) for x in a.seeds.split(',')])
 else:run(a.benchmark,a.method,[int(x) for x in a.seeds.split(',')])
