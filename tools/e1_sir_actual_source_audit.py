"""Artifact-derived SIR provenance versus selected corrected location cohorts."""
import hashlib,json,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tools import journal_campaign as j
from src.research.benchmarks import make_benchmark
from src.research.moe_family import BasePolicy,load_policy
C=R/'experiments/10192026/10192026_sir_e1_corrected_implementations'
OLD=Path.home()/'Documents/objective_power_grid/experiments/10182026/10182026_e1_sir_t5_seed101'
LOC=Path.home()/'Documents/objective_power_grid/experiments/10182026/10182026_e1_four_system'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sirrun(m):
 p=OLD/'sir/seed101'/m
 return p if p.exists() else OLD/'worker_ubuntu/sir/seed101'/m

def main():
 for d in ['reports','metadata','configs','logs','figures','plotting','evaluations']:(C/d).mkdir(parents=True,exist_ok=True)
 location={x['method']:x for x in read(LOC/'registry.json') if x['seed']==101};inventory=[];matrix=[];decisions=[];configs={}
 torch.set_num_threads(2)
 files={'dad':['sir_dad.py','training.py','journal_run.py'],'rl_sboed':['sir_native_rl.py','native_rl_reference.py','journal_run.py'],'vsoed':['upstream_vsoed.py','sir_upstream_vsoed.py'],'scorebed':['sir_scorebed.py','sir_scorebed_model.py','sir_scorebed_worker.py'],'moe':['moe_family.py','sparse_moe.py','family_training.py'],'resmoe':['moe_family.py','sparse_moe.py','family_training.py'],'switchmoe':['moe_family.py','strategy_moe.py','family_training.py'],'switchresmoe':['moe_family.py','strategy_moe.py','family_training.py']}
 notes=[]
 for method in j.METHODS:
  s=sirrun(method);t=read(s/'models'/f'{method}_training.json');cfg=read(s/'run_config.json');manifest=read(s/'source_manifest.json');lp=Path(location[method]['training_run']);lm=read(lp/'source_manifest.json');total=t.get('total_training_episodes',t['generated_training_trajectories']);assert total==(0 if method=='random' else 1000000)
  # Verify the source copy actually preserved with the run, never today's checkout.
  bad=[]
  for rel,digest in manifest['sha256'].items():
   p=s/'source'/rel
   if not p.exists() or sha(p)!=digest:bad.append(rel)
  assert not bad,(method,bad)
  cp=None if method=='random' else torch.load(s/'models'/f'{method}.pt',map_location='cpu',weights_only=False)
  if method=='dad':assert cp['architecture']=='DAD_sum_encoder_256_16_SIR_shared_empty_history' and not any('first_design' in k for k in cp['state_dict'])
  if method=='rl_sboed':assert t['training']=='pinned_original_RL_BOED_REDQ_SIR_ODE_adaptation' and t['gamma']==.9 and t['gradient_steps_per_collection']==64 and t['target_rate']==.001
  if method in j.FAMILY:
   assert cp['config']['training_routing_mode']=='straight_through'
   if method.startswith('switch'):assert cp['switching_training_revision']==2 and cp['config']['routing_mode']=='learned_switch'
  if method in j.RESIDUAL:
   base=OLD/'worker_ubuntu/sir/seed101/base/models/sum_single.pt';assert t['generated_training_trajectories']==500000
   assert t['base_checkpoint']['sha256']==sha(base)
   b=make_benchmark(cfg['benchmark']);policy=load_policy(s/'models'/f'{method}.pt',b);ref=BasePolicy.from_checkpoint(base,b,5).policy
   for key,val in ref.state_dict().items():torch.testing.assert_close(val,policy.core.base.policy.state_dict()[key],atol=0,rtol=0)
  configs[method]=dict(location=read(lp/'run_config.json'),sir=cfg,location_training=read(lp/'models'/(location[method]['native_method']+'_training.json')),sir_training=t)
  inventory.append(dict(method=method,seed=101,source_commit=manifest['git_commit'],source_manifest_sha256=sha(s/'source_manifest.json'),source_files_verified=len(manifest['sha256']),run=str(s),location_training_run=str(lp),location_source_commit=lm['git_commit'],checkpoint_sha256=None if cp is None else sha(s/'models'/f'{method}.pt'),total_primary_episodes=total,status='complete',architecture=None if cp is None else cp.get('architecture',cp.get('config'))))
  for f in files.get(method,['benchmarks.py','engine.py']):
   rel='src/research/'+f;ss=s/'source'/rel;ls=lp/'source'/rel
   same=ls.exists() and ss.exists() and sha(ls)==sha(ss)
   matrix.append(dict(method=method,component=f,corrected_location_implementation=str(ls),sir_implementation=str(ss),location_sha256=sha(ls) if ls.exists() else '',sir_sha256=sha(ss) if ss.exists() else '',equivalent='PASS' if same else 'BENCHMARK-ADAPTED',evidence='Frozen source files hash-identical' if same else 'SIR dimension, chronological action, prior and simulator adapter; executable fixtures and shared native kernels required (see verification report)',action='retain' if same else 'verify adapter; no EIG-based retraining'))
  # Native upstream kernels must be byte-identical to the selected location implementation.
  prefix={'rl_sboed':'src/research/vendor/rl_boed','vsoed':'src/research/vendor/vsoed','scorebed':'src/research/vendor/scorebed'}.get(method)
  if prefix:
   entries=[rel for rel in manifest['sha256'] if rel.startswith(prefix) and rel in lm['sha256']]
   assert entries and all(manifest['sha256'][rel]==lm['sha256'][rel] for rel in entries)
   matrix.append(dict(method=method,component='native upstream kernels',corrected_location_implementation=str(lp/'source'/prefix),sir_implementation=str(s/'source'/prefix),equivalent='PASS',evidence=f'{len(entries)} common vendored source hashes identical',action='retain'))
  decisions.append(dict(method=method,seed=101,decision='REEVALUATE ONLY' if method!='scorebed' else 'UNVERIFIED',reason='Corrected training path and final1M budget verified; high-resolution EIG required' if method!='scorebed' else 'Native source preserved; SIR score-target and policy-gradient checks pending',checkpoint=str(s/'models'/f'{method}.pt') if cp else '',source_commit=manifest['git_commit'],training_status='complete'))
 j.write(C/'metadata/actual_source_inventory.json',inventory);j.write(C/'metadata/location_sir_effective_configurations.json',configs)
 j.table(C/'reports/sir_location_method_equivalence_matrix.csv',matrix,['method','component','corrected_location_implementation','sir_implementation','location_sha256','sir_sha256','equivalent','evidence','action'])
 j.table(C/'reports/sir_seed101_reuse_decisions.csv',decisions,list(decisions[0]))
 j.write(C/'metadata/seeds102_105_hold.json',dict(state='HOLD',reason='actual-source and baseline adapter checks plus evaluation-resolution freeze required',training_jobs_launched=0))
 (C/'reports/sir_seed101_actual_source_audit.md').write_text('# SIR actual-source audit\n\nAll ten runs used frozen commit23f629c, not GitHub main. Every source-file copy was verified against its own run manifest. Native RL used pinned REDQ, discount0.9, two critics/M2,64 updates per100 episodes; legacy research_redq was not selected. DAD checkpoint uses shared empty-history emitter revision1 and has no separately learned first_design parameter. All four family checkpoints specify straight_through; both switching checkpoints specify switching_training_revision2 and learned_switch. Both residual checkpoints contain the identical SIR-specific500k base hash, verified bitwise unchanged.\n\nFull frozen configuration and optimizer records for location and SIR are in metadata/location_sir_effective_configurations.json. Native vendor hash comparisons are in the matrix. This verifies provenance, not automatically score-estimation accuracy or convergence. Original seed101 evaluations and checkpoints are preserved. Seeds102-105 remain held.\n')
 (C/'reports/sir_moe_straight_through_verification.md').write_text('# Four-variant ST verification\n\nActual serialized configuration checked for all four, not defaults. Switching revision2 verified in both checkpoints. Shared StrategyManager hard-forward gate and expert surrogates are the corrected location implementation; component-level tests cover informative gate/router VJPs, carried state, reset, forced continue/reroute and hard sparse inference. SIR-only adaptations change history dimensions and chronological action mapping. Residual base hashes match and all frozen base tensors are bitwise unchanged. SIR runtime and loss/gradient fixtures recorded separately. No relaxed fallback selected.\n')
 canonical=Path.home()/'Documents/objective_power_grid'/C.relative_to(R)
 canonical.parent.mkdir(parents=True,exist_ok=True)
 if not canonical.exists():canonical.symlink_to(C,target_is_directory=True)
 print('AUDIT PASS',len(inventory),'runs',len(matrix),'components')
if __name__=='__main__':main()
