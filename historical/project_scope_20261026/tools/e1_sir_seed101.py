"""E1 SIR seed101 assignment using the existing journal campaign and root launcher."""
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import journal_campaign as j
DATE='10182026'
REL=Path('experiments')/DATE/(DATE+'_e1_sir_t5_seed101')
C=ROOT/REL
TASKS={'labpc':['random','fixed','dad','rl_sboed','vsoed','switchmoe'], 'ubuntu-pc':['base','resmoe','switchresmoe','moe','scorebed']}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
 assert datetime.now(ZoneInfo('America/New_York')).strftime('%m%d%Y')==DATE
 assert not C.exists(),'Never overwrite a campaign'
 for d in ['configs','metadata','reports','logs','evaluations/development_bank','frozen_source','figures','plotting']:(C/d).mkdir(parents=True,exist_ok=True)
 raw=yaml.safe_load((ROOT/'configs/e1_sir_native.yaml').read_text())
 for cap in [1000000,500000]:
  cfg=copy.deepcopy(raw);p=cfg['research']['journal_protocol'];p['stopping_rule'].update(minimum_episodes=cap,maximum_episodes=cap)
  cfg['research']['paper_protocol']['nominal_training_trajectories']=cap
  cfg['research']['settings'].update(primary_episode_budget=1,training_bound=1)
  (C/'configs'/f'sir_{cap}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
 from src.research.benchmarks import make_benchmark
 b=make_benchmark(raw);g=torch.Generator().manual_seed(1001)
 truth=b.sample((2048,),g);noise=torch.randn(2048,5,1,dtype=torch.float64,generator=g)
 contrasts=b.sample((131072,),g)
 bank=C/'evaluations/development_bank'
 np.savez(bank/'cases.npz',truth=truth.numpy(),noise=noise.numpy())
 np.save(bank/'contrasts.npy',contrasts[:16384].numpy());np.save(bank/'contrasts_resolution.npy',contrasts.numpy())
 j.write(bank/'manifest.json',dict(seed=1001,systems=2048,contrasts=16384,resolution_prefixes=[16384,32768,65536,131072],sha256={p.name:sha(p) for p in bank.iterdir() if p.is_file()},note='Saved common draws; resolution bank diagnostic only, not checkpoint selection'))
 j.write(C/'registry.json',[])
 j.write(C/'protocol.json',raw)
 (C/'journal_protocol.md').write_text('E1 SIR ODE T5: ten methods, seeds101-105. Seed101 first; 102-105 held for estimator precision freeze. One million total primary episodes, residuals500k base+500k residual, shared frozen base per seed. Final-budget checkpoint selection; validation1003 diagnostic only, development1001 saved paired draws; confirmation1002 reserved until protocol freeze. Native algorithms resource-adapted; all four MoE use straight-through. Trusted on-the-fly RK4, at least512 steps per stage, no path bank. 24h per training-phase safety cap; incomplete runs reported, never substituted. Approximately3.5h/million pathwise episodes before evaluation on measured worker GPU; native methods have different costs.\n')
 assignments=[dict(benchmark='sir',method=m,seed=s,host=next(h for h,t in TASKS.items() if m in t),status='assigned' if s==101 else 'held_precision_gate') for m in j.METHODS for s in ([101] if m=='random' else range(101,106))]
 assert len(assignments)==46
 j.write(C/'metadata/assignments.json',assignments)
 j.write(C/'metadata/base_assignment.json',dict(method='base',seed=101,host='ubuntu-pc',episodes=500000,shared_by=['resmoe','switchresmoe']))
 source=[*ROOT.glob('src/**/*.py'),*ROOT.glob('tools/**/*.py'),*ROOT.glob('configs/*.yaml'),ROOT/'run.sh',ROOT/'sweep_run.sh',*C.glob('configs/*.yaml')]
 manifest={str(p.relative_to(ROOT)):sha(p) for p in source}
 j.write(C/'source_manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),sha256=manifest))
 for p in source:
  out=C/'frozen_source'/p.relative_to(ROOT);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,out)
 j.write(C/'location_freeze.json',dict(protocol_sha256=sha(C/'protocol.json'),written_protocol_sha256=sha(C/'journal_protocol.md'),note='Existing journal guard seal reused for SIR'))
 j.write(C/'metadata/readiness.json',dict(status='passed',torch_focused_tests=18,checkpoint_replay='all9 trained methods exact actions in saved smoke replay',jax_gpu=json.loads((ROOT/'experiments/10182026/10182026_e1_sir_native_readiness/jax_gpu_equivalence.json').read_text()),smokes_are_not_full_results=True))
 print(C,flush=True)
def run(host):
 import fcntl
 lock=(C/'metadata'/f'{host}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied'
 j.C=C;j.NAMES['rl_sboed']='Native RL-sBOED'
 j.guard()
 j.write(C/'metadata'/f'{host}_environment.json',dict(host=host,torch=torch.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True),source_sha256=sha(C/'source_manifest.json')))
 base=None
 for method in TASKS[host]:
  if method in j.RESIDUAL and base is None:
   j.write(C/'metadata'/f'{method}_blocked.json',dict(status='blocked',reason='shared base failed'));continue
  j.write(C/'metadata'/f'{host}_status.json',dict(state='running',method=method,seed=101,pid=os.getpid()))
  entry=j.run_phase('sir',101,method,base=base if method in j.RESIDUAL else None,timeout_seconds=180000,evaluation_cases=C/'evaluations/development_bank/cases.npz',evaluation_bank=C/'evaluations/development_bank/contrasts.npy')
  if method=='base' and entry['status']=='complete':base=Path(entry['run'])/'models/sum_single.pt'
 j.write(C/'metadata'/f'{host}_status.json',dict(state='finished_review_registry',seed=101))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--host',choices=list(TASKS));a=p.parse_args()
 if a.action=='prepare':prepare()
 else:run(a.host)
