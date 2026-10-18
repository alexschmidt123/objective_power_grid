"""Artifact-only E1 closure using existing journal analysis and common scorer."""
from pathlib import Path
import json,hashlib,sys,subprocess,datetime,secrets,tarfile,io,os
import numpy as np
import torch,yaml
import tools.journal_campaign as j
R=Path.home()/'Documents/objective_power_grid'
W=Path(__file__).resolve().parents[1]
C=R/'experiments/10182026/10182026_e1_four_system'
A=R/'experiments/10182026/10182026_e1_original_methods_v2'
S=R/'experiments/10182026/10182026_e1_st_switching_corrected'

def analysis():
 # The public registry has46 evaluation entries. Five reused internal bases
 # are supplied only to the existing cost/learning-curve analysis helper.
 j.C=C;j.ROOT=W;j.NAMES['rl_sboed']='Native RL-sBOED (resource-adapted)'
 original=j.read
 def read_with_internal_bases(path):
  data=original(path)
  if Path(path)==C/'registry.json':
   bases=[dict(method='base',native_method='sum_single',benchmark='location',seed=seed,status='complete',run=str(Path(original(S/'metadata/base_checkpoints.json')[str(seed)]['original_path']).parent.parent),provenance='shared_internal500k_base_not_main_method') for seed in range(101,106)]
   return data+bases
  return data
 j.read=read_with_internal_bases
 try:j.report();j.figures()
 finally:j.read=original

def prepare():
 assert not C.exists(),'Never overwrite a frozen campaign'
 assert j.read(S/'metadata/master_import_verified.json')['status']=='PASS'
 for d in ['configs','metadata','reports','figures','plotting','logs','evaluations','location','frozen_source']:(C/d).mkdir(parents=True,exist_ok=True)
 cfg=C/'configs/location_1000000.yaml';cfg.write_bytes((A/'configs/location_1000000.yaml').read_bytes())
 entries=[]
 for e in j.read(A/'metadata/reused_execution_entries.json'):
  if e['method'] in ['random','fixed','vsoed','moe']:entries.append(dict(method=e['method'],native_method=e['native_method'],seed=e['seed'],training_run=e['run'],evaluation_run=e['run']))
 for method in ['dad','rl_sboed','scorebed']:
  for seed in range(101,106):
   tr=A/('worker_location' if seed in [102,104] else 'location')/f'seed{seed}'/method
   ev=A/'worker_portable'/f'seed{seed}'/method if seed in [102,104] else tr
   entries.append(dict(method=method,native_method=method,seed=seed,training_run=str(tr),evaluation_run=str(ev)))
 for seed in range(101,106):
  tr=R/'experiments/10172026/10172026_location_routing_scorebed_correction/location'/f'seed{seed}'/'resmoe'
  entries.append(dict(method='resmoe',native_method='resmoe',seed=seed,training_run=str(tr),evaluation_run=str(tr)))
 for method in ['switchmoe','switchresmoe']:
  for seed in range(101,106):entries.append(dict(method=method,native_method=method,seed=seed,training_run=str(S/'location'/f'seed{seed}'/method),evaluation_run=str(S/'portable_outputs/hard'/f'seed{seed}'/method)))
 assert len(entries)==46 and len({(e['method'],e['seed']) for e in entries})==46
 from src.research.benchmarks import make_benchmark
 from src.research.moe_family import benchmark_signature
 from src.research.journal_checkpoints import load_checkpoint
 b=make_benchmark(yaml.safe_load(cfg.read_text()));bank=j.read(A/'evaluations/development_bank/manifest.json');audits=[];base_hashes={}
 for e in entries:
  m,n,s=e['method'],e['native_method'],e['seed'];tr=Path(e['training_run']);ev=Path(e['evaluation_run']);t=j.read(tr/'models'/(n+'_training.json'));rc=j.read(tr/'run_config.json')
  assert rc['horizon']==30 and benchmark_signature(b)==benchmark_signature(make_benchmark(rc['benchmark']))
  if m!='random':
   assert t.get('total_training_episodes',t['generated_training_trajectories'])==1000000,(m,s,'budget')
   assert t['checkpoint_selection'].startswith('final')
   assert (tr/'exit_code').read_text().strip()=='0'
   e['checkpoint']=str((tr/'models'/(n+'.pt')).resolve());e['checkpoint_sha256']=j.sha(e['checkpoint'])
   p=load_checkpoint(e['checkpoint'],b);assert p.horizon==30
   if m in j.FAMILY:
    assert p.config.get('training_routing_mode')=='straight_through',(m,s,'not ST')
    if m in ['switchmoe','switchresmoe']:assert torch.load(e['checkpoint'],map_location='cpu',weights_only=False)['switching_training_revision']==2
   if m in j.RESIDUAL:
    assert t['generated_training_trajectories']==500000 and t['base_checkpoint']['training_episodes']==500000
    base_hashes[(m,s)]=t['base_checkpoint']['sha256']
    reference=S/'dependencies'/f'base_seed{s}'/'models/sum_single.pt';saved=torch.load(reference,map_location='cpu',weights_only=False)
    assert j.sha(reference)==base_hashes[(m,s)]
    for k,v in saved['state_dict'].items():torch.testing.assert_close(p.core.base.policy.state_dict()[k],v,atol=0,rtol=0)
  manifest=j.read(tr/'source_manifest.json');bad=[name for name,h in manifest['sha256'].items() if j.sha(tr/'source'/name)!=h];assert not bad,(m,s,bad)
  rows=j.read(ev/'eval'/(n+'_seed1001.json'));assert len(rows)==2000 and all(np.isfinite(x['spce']) for x in rows)
  identity=hashlib.sha256(json.dumps([[x['truth'],x['measurement_noise_standard']] for x in rows],separators=(',',':')).encode()).hexdigest();assert identity==bank['case_semantic_sha256']
  assert j.sha(ev/'eval/contrast_bank_seed1001.npy')==bank['bank_file_sha256']
  summary=next(x for x in j.read(ev/'summary.json') if x['method']==n and x['eval_seed']==1001);assert summary['contrasts']==1000000
  view=C/'location'/f'seed{s}'/m;view.mkdir(parents=True)
  for label,src in [('models',tr/'models'),('eval',ev/'eval'),('run_config.json',tr/'run_config.json'),('summary.json',ev/'summary.json')]: (view/label).symlink_to(src.resolve(),target_is_directory=src.is_dir())
  e.update(benchmark='location',status='complete',run=str(view),provenance='verified_native_baseline_or_selected_ST_cohort',source_commit=manifest['git_commit'],source_manifest_sha256=j.sha(tr/'source_manifest.json'))
  j.write(view/'artifact_links.json',e);audits.append(dict(method=m,seed=s,status='PASS',eig=float(np.mean([x['spce'] for x in rows])),cases_sha256=identity,bank_sha256=bank['bank_file_sha256'],source_snapshot_verified=True,checkpoint_load_verified=m!='random'))
 for s in range(101,106):assert base_hashes['resmoe',s]==base_hashes['switchresmoe',s]
 j.write(C/'registry.json',entries);j.write(C/'reports/location_46_policy_audit.json',dict(status='PASS',evaluations=46,entries=audits))
 protocol=dict(methods=list(j.METHODS),systems=['location_T30','sir_T5','ieee9_T3','ieee14_T3'],training_seeds=list(range(101,106)),primary_episodes=1000000,selection='final_budget',moe_training='straight_through',residual_allocation=[500000,500000],development_seed=1001,consumed_confirmation_seed=8831001,location_status='development_verified_confirmation_pending',other_systems_status='readiness_pending',scope='E1 only; no ablations, E2, E3 or other horizons',analysis='paired seed differences; t95 with4df; simultaneous Bonferroni95 over six declared family contrasts; trajectory MC intervals separate')
 j.write(C/'protocol.json',protocol);(C/'journal_protocol.md').write_text('# Four-system E1\n\n'+json.dumps(protocol,indent=2)+'\n\nSelected procedures and checkpoints are frozen before fresh confirmation. Original and resource-adapted baseline fidelity evidence is retained in the source campaigns. No guarantee of a method ranking. Random is one untrained reference, not five training replicates.\n')
 analysis()
 (C/'plotting/regenerate.txt').write_text(f'cd {W}\npython -m tools.e1_location_close report\n')
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive','HEAD'],cwd=W))) as archive:archive.extractall(C/'frozen_source')
 print('Verified and aggregated46 development policies',flush=True)

def freeze():
 assert not (C/'confirmation_gate.json').exists()
 with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive','HEAD'],cwd=W))) as archive:archive.extractall(C/'frozen_source')
 assert j.read(C/'reports/location_46_policy_audit.json')['status']=='PASS'
 entries=j.read(C/'registry.json');seed=int(sys.argv[3]) if len(sys.argv)>3 else 1002
 assert seed>=1002 and seed<8831001
 # Sequential evaluation RNGs are audited by actual output artifacts.
 scanned=0
 for worktree in (Path.home()/'Documents').glob('objective_power_grid*'):
  base=worktree/'experiments'
  if not base.exists():continue
  for p in base.rglob('*'):
   if p.is_file():
    scanned+=1
    if p.name.endswith('_seed'+str(seed)+'.json') or p.name=='contrast_bank_seed'+str(seed)+'.npy':
     raise RuntimeError('Evaluation seed already has saved output: '+str(p))
 bankdir=C/'evaluations/confirmation_bank';bankdir.mkdir()
 rng=torch.Generator().manual_seed(seed);truth=torch.randn(2000,4,generator=rng,dtype=torch.float64);noise=torch.randn(2000,30,1,generator=rng,dtype=torch.float64);contrasts=torch.randn(1000000,4,generator=rng,dtype=torch.float64)
 np.savez(bankdir/'cases.npz',truth=truth.numpy(),noise=noise.numpy());np.save(bankdir/'contrasts.npy',contrasts.numpy())
 paths=list((W/'src').rglob('*.py'))+[W/'run.sh',W/'sweep_run.sh',W/'tools/e1_location_close.py',W/'tools/journal_campaign.py']
 source={str(p.relative_to(W)):j.sha(p) for p in paths};j.write(C/'source_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=W,text=True).strip(),sha256=source))
 policies=[{k:e.get(k) for k in ['method','native_method','seed','checkpoint','checkpoint_sha256']}|dict(source_run=e['training_run']) for e in entries]
 sealed=[C/'protocol.json',C/'journal_protocol.md',C/'registry.json',C/'reports/location_46_policy_audit.json',C/'source_manifest.json',C/'configs/location_1000000.yaml',C/'location_finding_summary.csv',C/'location_finding_comparisons.csv',C/'training_efficiency.csv',bankdir/'cases.npz',bankdir/'contrasts.npy']
 gate=dict(status='frozen_ready_for_confirmation',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),evaluation_seed=seed,horizon=30,systems=2000,contrasts=1000000,cases=str(bankdir/'cases.npz'),bank=str(bankdir/'contrasts.npy'),policies=policies,sealed_files={str(p):j.sha(p) for p in sealed},source_sha256=source,seed_provenance=dict(source='Sequential evaluation seed after unchanged checkpoint/analysis freeze; audited saved historical outputs',historical_manifest_files_scanned=scanned),claim_limit='Five training replicates; confirmation trajectories are not new training replicates')
 j.write(C/'confirmation_gate.json',gate);print('Frozen fresh confirmation seed',seed,flush=True)

def confirm():
 g=j.read(C/'confirmation_gate.json');reg=[]
 if not (C/'metadata/confirmation_started.json').exists():j.write(C/'metadata/confirmation_started.json',dict(gate_sha256=j.sha(C/'confirmation_gate.json'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 for e in g['policies']:
  m,s=e['method'],e['seed'];out=W/'experiments'/C.parent.name/C.name/'confirmation'/f'seed{s}'/m
  command=['bash','run.sh','--config',str(C/'configs/location_1000000.yaml'),'--native-evaluate-from',e['source_run'] if m=='random' else e['checkpoint'],'--e1-confirmation-gate',str(C/'confirmation_gate.json'),'--methods',m,'--seed',str(s),'--eval-seeds',str(g['evaluation_seed']),'--output',str(out)]
  if not out.exists():
   j.write(C/'metadata'/f'confirmation_{m}_{s}_command.json',dict(cwd=str(W),command=command))
   with (C/'logs'/f'confirmation_{m}_{s}.log').open('w') as log:subprocess.run(command,cwd=W,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
  assert (out/'exit_code').read_text().strip()=='0'
  rows=j.read(out/'eval'/f"{m}_seed{g['evaluation_seed']}.json");assert len(rows)==2000
  reg.append(e|dict(run=str(out),status='complete',eig=float(np.mean([x['spce'] for x in rows]))));j.write(C/'metadata/confirmation_registry.json',reg)
  print('CONFIRMATION',m,s,'complete',flush=True)
 j.table(C/'reports/location_confirmation_per_seed.csv',reg,['method','seed','eig','status','run'])
 j.write(C/'reports/location_confirmation_complete.json',dict(status='complete',policies=len(reg),evaluation_seed=g['evaluation_seed'],no_training_or_tuning=True))
if __name__=='__main__':
 if len(sys.argv)>2:C=Path(sys.argv[2]).resolve()
 if sys.argv[1]=='prepare':prepare()
 elif sys.argv[1]=='freeze':freeze()
 elif sys.argv[1]=='confirm':confirm()
 elif sys.argv[1]=='report':analysis()
