"""Gated SIR E1 replication, reusing existing journal phases and root launchers."""
if __name__ == '__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: historical campaign is archived')
import argparse,copy,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np,torch,yaml
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from tools import journal_campaign as j
from tools.e1_sir_resolution import nested
C=R/'experiments/10192026/10192026_sir_e1_five_seed_verified'
MASTER=Path.home()/'Documents/objective_power_grid'
PREVIOUS=MASTER/'experiments/10182026/10182026_e1_sir_t5_seed101'
VERIFY=Path.home()/'Documents/objective_power_grid_e1_sir_verified_20261019/experiments/10192026/10192026_e1_sir_verification'
CORRECT=Path.home()/'Documents/objective_power_grid_e1_sir_verified_20261019/experiments/10192026/10192026_sir_e1_corrected_implementations'
TASKS={'labpc':['random','fixed','dad','rl_sboed','vsoed','switchmoe'],'ubuntu-pc':['base','resmoe','switchresmoe','moe','scorebed']}
PAIRS=[('moe','dad'),('switchmoe','dad'),('moe','switchmoe'),('resmoe','fixed'),('switchresmoe','resmoe'),('resmoe','scorebed'),('resmoe','dad'),('resmoe','moe'),('switchresmoe','switchmoe')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def status(state,**kw):j.write(C/'metadata/master_status.json',dict(state=state,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
def oldrun(m):
 p=PREVIOUS/'sir/seed101'/m
 return p if p.exists() else PREVIOUS/'worker_ubuntu/sir/seed101'/m

def prepare():
 assert not C.exists()
 for d in ['metadata','reports','configs','logs','evaluations','frozen_source','figures','plotting','seed101_sources']:(C/d).mkdir(parents=True,exist_ok=True)
 raw=yaml.safe_load((R/'configs/e1_sir_native.yaml').read_text())
 from src.research.benchmarks import make_benchmark
 b=make_benchmark(raw);prefix=np.load(VERIFY/'evaluations/contrasts_nested.npy');g=torch.Generator().manual_seed(1001)
 b.sample((2048,),g);torch.randn(2048,5,1,generator=g,dtype=torch.float64);b.sample((131072,),g);b.sample((1000000-131072,),g)
 full=np.concatenate([prefix,b.sample((3000000,),g).numpy()]);np.save(C/'evaluations/contrasts_4m.npy',full)
 shutil.copy2(VERIFY/'evaluations/cases.npz',C/'evaluations/cases.npz')
 for m in j.METHODS:
  p=oldrun(m);shutil.copy2(p/'eval'/f'{m}_seed1001.json',C/'seed101_sources'/f'{m}.json')
 j.write(C/'metadata/bank_manifest.json',dict(seed=1001,systems=2048,sha256={p.name:sha(p) for p in (C/'evaluations').iterdir()},prefix_1m_sha256=sha(VERIFY/'evaluations/contrasts_nested.npy')))
 j.write(C/'metadata/precision_rule.json',dict(primary_candidate=1000000,upgrade_candidate=4000000,mean_resolution_target=.02,paired_resolution_target=.01,coarse_check='Full2048 nested16k/65k/262k/1M',upgrade_check='Common128 cases at1M/4M, corrected ScoreBED included',maximum_4m_shift=.15,maximum_paired_shift=.05,claim_rule='Do not claim true-EIG superiority for an effect comparable to the measured contrast-resolution shift; report finite-bound resolution limits and across-seed intervals',resource_cap=4000000,criteria='Method-agnostic numerical changes, never ranking',training='102-105 HOLD until actual source, corrected ScoreBED101 completion, and precision freeze pass'))
 j.write(C/'metadata/assignment_manifest.json',[dict(benchmark='sir',seed=s,method=m,host=next(h for h,ms in TASKS.items() if m in ms),status='reuse_after_reevaluation' if s==101 else 'held') for m in j.METHODS for s in ([101] if m=='random' else range(101,106))])
 j.write(C/'registry.json',[]);status('prepared_hold_implementation_and_precision_gates');print(C)

def worker_resolution():
 # Only after the corrected full-budget ScoreBED development run has completed.
 entry=j.read(CORRECT/'metadata/score101_complete.json');assert entry['status']=='complete';p=Path(entry['run']);t=j.read(p/'models/scorebed_training.json');assert t['generated_training_trajectories']==1000000 and t['sir_score_adapter_revision']==2
 rows=j.read(p/'eval/scorebed_seed1001.json');j.write(C/'seed101_sources/scorebed.json',rows)
 bank=torch.from_numpy(np.load(C/'evaluations/contrasts_4m.npy'));raw=yaml.safe_load((R/'configs/e1_sir_native.yaml').read_text());from src.research.benchmarks import make_benchmark
 torch.set_num_threads(2);records=[]
 for method in j.METHODS:
  out=C/'evaluations'/f'{method}_N128_1m_4m.npz'
  if out.exists():a=np.load(out);scores={k:a[str(k)] for k in [1000000,4000000]}
  else:
   cases=j.read(C/'seed101_sources'/f'{method}.json')[:128];scores={1000000:[],4000000:[]}
   for start in range(0,128,16):
    batch=cases[start:start+16];truth=torch.tensor([x['truth'] for x in batch],dtype=torch.float64);actions=torch.tensor([x['actions'] for x in batch],dtype=torch.float64).unbind(1);obs=torch.tensor([x['observations'] for x in batch],dtype=torch.float64).unbind(1)
    ans=nested(make_benchmark(raw),truth,actions,obs,bank,[1000000,4000000]);[scores[k].extend(ans[k]['scores']) for k in scores];print(method,start+len(batch),flush=True)
   scores={k:np.asarray(v) for k,v in scores.items()};np.savez(out,**{str(k):v for k,v in scores.items()})
  diff=scores[4000000]-scores[1000000];records.append(dict(method=method,systems=128,shift=float(diff.mean()),mc_se=float(diff.std(ddof=1)/128**.5),eig_1m=float(scores[1000000].mean()),eig_4m=float(scores[4000000].mean())))
  j.table(C/'reports/sir_1m_4m_resolution.csv',records,list(records[0]))
 pairs=[]
 for first,second in PAIRS:
  a=np.load(C/'evaluations'/f'{first}_N128_1m_4m.npz');b=np.load(C/'evaluations'/f'{second}_N128_1m_4m.npz');d=(a['4000000']-b['4000000'])-(a['1000000']-b['1000000']);pairs.append(dict(first=first,second=second,paired_shift=float(d.mean()),mc_se=float(d.std(ddof=1)/128**.5)))
 j.table(C/'reports/sir_1m_4m_pairs.csv',pairs,list(pairs[0]));j.write(C/'metadata/worker_precision_complete.json',dict(status='complete',maximum_mean_shift=max(abs(x['shift']) for x in records),maximum_paired_shift=max(abs(x['paired_shift']) for x in pairs),corrected_scorebed=True))

def freeze():
 assert (VERIFY/'metadata/resolution_N2048_complete.json').exists()
 proof=dict(status='E1_FIXED_1M',basis='Explicit user scope: 4M deferred to E2; disclose finite-contrast resolution limits')
 assert (C/'metadata/corrected_score101_import_verified.json').exists()
 #4M was the predeclared upgrade if the initial1M resolution target was not established.
 L=1000000;raw=yaml.safe_load((R/'configs/e1_sir_native.yaml').read_text());raw['research']['settings']['eval_contrasts']=L
 protocol=raw['research']['journal_protocol'];protocol.update(status='frozen_SIR_E1_native_ST_corrected_score',evaluation_seeds=[1002]);protocol['final_evaluation'].update(contrasts=L,systems=2048,seed=1002,resolution_status='1M common finite sPCE; residual contrast sensitivity disclosed; 4M deferred to E2')
 for cap in [500000,1000000]:
  cfg=copy.deepcopy(raw);cfg['research']['journal_protocol']['stopping_rule'].update(minimum_episodes=cap,maximum_episodes=cap);cfg['research']['paper_protocol']['nominal_training_trajectories']=cap;(C/'configs'/f'sir_{cap}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 j.write(C/'protocol.json',raw)
 (C/'journal_protocol.md').write_text('SIR E1: ten methods, training101-105; Random untrained once. Exactly1M total primary episodes, residuals500k base+500k residual; same benchmark-specific frozen base per seed. Corrected original baselines; source/optimizer/physics retained from verified cohort; ScoreBED native scalar potential compatibility head256 revision2 only correction. All four MoE ST, switch revision2. Final-budget checkpoints. Validation1003 diagnostic only. Development1001,2048 paired cases/common1M contrast bank. Original16k results and diagnostic histories preserved.1M is the user-declared E1 resolution; 4M is deferred to E2; residual sensitivity and MC uncertainty must accompany small-effect claims. Confirmation1002 reserved for SIR until complete cohort and analysis freeze. Primary paired training-seed t intervals df4 and Bonferroni simultaneous intervals over9 contrasts, not pooled trajectories. No tuning after freeze. Training-phase safety cap24h; partial methods labeled budget-limited/incomplete, never substituted. Two-PC assignment prevents GPU contention. No E2/E3/grid/horizon extension.\n')
 j.write(C/'metadata/analysis_plan.json',dict(pairs=PAIRS,training_seeds=[101,102,103,104,105],confidence=.95,multiple_comparison='Bonferroni over9 predeclared paired seed-level contrasts',random_training_replicates=False,trajectory_mc_distinct_from_training_seed_uncertainty=True))
 source=[*R.glob('src/**/*.py'),*R.glob('tools/**/*.py'),*R.glob('configs/*.yaml'),R/'run.sh',R/'sweep_run.sh',*C.glob('configs/*.yaml')]
 j.write(C/'source_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),sha256={str(p.relative_to(R)):sha(p) for p in source}))
 for p in source:
  dst=C/'frozen_source'/p.relative_to(R);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
 j.write(C/'location_freeze.json',dict(protocol_sha256=sha(C/'protocol.json'),written_protocol_sha256=sha(C/'journal_protocol.md')))
 j.write(C/'metadata/replication_gate.json',dict(status='PASS',bank_manifest_sha256=sha(C/'metadata/bank_manifest.json'),contrasts=L,systems=2048,precision=proof,scientific_source='native algorithms plus verified SIR adapters; original seed101 source checked',score101_correction_complete=True));status('frozen_ready_for_replication')

def run(host):
 lock=(C/'metadata'/f'{host}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 gate=j.read(C/'metadata/replication_gate.json');assert gate['status']=='PASS' and gate['bank_manifest_sha256']==sha(C/'metadata/bank_manifest.json')
 bank_manifest=j.read(C/'metadata/bank_manifest.json');assert all(sha(C/'evaluations'/p)==v for p,v in bank_manifest['sha256'].items())
 j.C=C;j.NAMES['rl_sboed']='Native RL-sBOED';j.guard()
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 for seed in [102,103,104,105]:
  base=None
  for method in TASKS[host]:
   if method=='random':continue
   if shutil.disk_usage(R).free<4*1024**3:raise RuntimeError('4GiB disk floor; preserve outputs and stop')
   if method in j.RESIDUAL and base is None:
    j.write(C/'metadata'/f'{method}_{seed}_dependency_failed.json',dict(status='blocked',reason='base failed'));continue
   j.write(C/'metadata'/f'{host}_status.json',dict(state='running',seed=seed,method=method,pid=os.getpid()))
   entry=j.run_phase('sir',seed,method,base=base if method in j.RESIDUAL else None,timeout_seconds=180000,evaluation_cases=C/'evaluations/cases.npz',evaluation_bank=C/'evaluations/contrasts_1m.npy')
   if method=='base' and entry['status']=='complete':base=Path(entry['run'])/'models/sum_single.pt'
 j.write(C/'metadata'/f'{host}_status.json',dict(state='finished_review_registry'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker-resolution','freeze','run']);p.add_argument('--host',choices=list(TASKS));a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='worker-resolution':worker_resolution()
 elif a.action=='freeze':freeze()
 else:run(a.host)
