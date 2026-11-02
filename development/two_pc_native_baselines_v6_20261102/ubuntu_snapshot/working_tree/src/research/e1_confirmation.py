"""Sealed E1 evaluation only; reuse the common scorer and explicit saved bank."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def verify_gate(gate_path,method,seed,source,root):
 g=json.loads(Path(gate_path).read_text())
 if g['status']!='frozen_ready_for_confirmation':raise ValueError('Unfrozen E1 confirmation')
 if g['evaluation_seed'] in [1001,8831001]:raise ValueError('Consumed evaluation seed')
 for p,digest in g['sealed_files'].items():
  if sha(p)!=digest:raise ValueError('Sealed file changed: '+p)
 for p,digest in g['source_sha256'].items():
  if sha(Path(root)/p)!=digest:raise ValueError('Frozen source changed: '+p)
 e=next(x for x in g['policies'] if x['method']==method and x['seed']==seed)
 if Path(e['source_run']).resolve()!=Path(source).resolve():raise ValueError('Unregistered policy run')
 if method!='random' and sha(e['checkpoint'])!=e['checkpoint_sha256']:raise ValueError('Checkpoint changed')
 return g,e

def main(args,raw,root):
 from src.research.runner import DEFAULTS,evaluate,write_json,freeze_source
 from src.research.benchmarks import make_benchmark
 from src.research.moe_family import benchmark_signature
 from src.research.journal_checkpoints import load_checkpoint
 method=args.methods
 if not method or ',' in method or args.variant:raise ValueError('Exactly one registered method')
 path=Path(args.native_evaluate_from).resolve();source=path if method=='random' else path.parent.parent
 g,e=verify_gate(args.e1_confirmation_gate,method,args.seed,source,root)
 if [int(x) for x in args.eval_seeds.split(',')]!=[g['evaluation_seed']]:raise ValueError('Wrong confirmation seed')
 if args.evaluation_relaxed_routing or args.smoke:raise ValueError('No diagnostic interventions in confirmation')
 cfg=json.loads((source/'run_config.json').read_text());b=make_benchmark(raw)
 if benchmark_signature(b)!=benchmark_signature(make_benchmark(cfg['benchmark'])):raise ValueError('Benchmark changed')
 if cfg['horizon']!=g['horizon']:raise ValueError('Horizon changed')
 training=json.loads((source/'models'/(e['native_method']+'_training.json')).read_text())
 if method!='random':
  if training.get('total_training_episodes',training['generated_training_trajectories'])!=1000000:raise ValueError('Budget mismatch')
  if not training['checkpoint_selection'].startswith('final'):raise ValueError('Checkpoint selection mismatch')
 p=None if method=='random' else load_checkpoint(e['checkpoint'],b).cpu().eval()
 cases=np.load(g['cases']);bank=np.load(g['bank'])
 draws=dict(truth=torch.from_numpy(cases['truth']).double(),noise=torch.from_numpy(cases['noise']).double(),contrasts=torch.from_numpy(bank).double())
 settings=DEFAULTS|raw['research'].get('settings',{})
 settings.update(eval_systems=g['systems'],eval_contrasts=g['contrasts'],evaluation_population_systems=g['systems'],evaluation_batch_size=16,evaluation_contrast_chunk=16384,threads=2)
 out=Path(args.output).resolve();rel=out.relative_to(root/'experiments')
 if len(rel.parts)<2 or not rel.parts[1].startswith(rel.parts[0]+'_'):raise ValueError('Dated campaign required')
 out.mkdir(parents=True,exist_ok=False)
 for d in ['models','logs','eval']:(out/d).mkdir()
 torch.set_num_threads(2);freeze_source(root,out)
 write_json(out/'run_config.json',dict(evaluation_only=True,confirmation=True,benchmark=raw,horizon=g['horizon'],train_seed=args.seed,eval_seed=g['evaluation_seed'],source_training_run=str(source),source_checkpoint_sha256=e.get('checkpoint_sha256'),gate_sha256=sha(args.e1_confirmation_gate),cases_sha256=sha(g['cases']),bank_sha256=sha(g['bank']),new_training_episodes=0))
 start=time.perf_counter()
 try:
  rows,summary=evaluate(b,p,method,g['horizon'],settings,g['evaluation_seed'],training,out,paper=True,infer_parameters=False,evaluation_draws=draws)
  write_json(out/'eval'/f"{method}_seed{g['evaluation_seed']}.json",rows)
  np.save(out/'eval/scores.npy',np.asarray([x['spce'] for x in rows]));write_json(out/'summary.json',[summary]);write_json(out/'completion.json',dict(status='complete',evaluation_only=True,new_training_episodes=0,seconds=time.perf_counter()-start))
  (out/'exit_code').write_text('0\n')
 except BaseException:
  (out/'exit_code').write_text('1\n');raise
