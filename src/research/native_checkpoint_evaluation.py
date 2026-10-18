"""Evaluation-only checkpoint path with portable saved cases; common scorer."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch

def main(args,raw,root):
 from src.research.runner import DEFAULTS,evaluate,freeze_source,write_json
 from src.research.benchmarks import make_benchmark
 from src.research.moe_family import benchmark_signature
 from src.research.journal_checkpoints import load_checkpoint
 if not args.evaluation_cases or not args.evaluation_bank or not args.output:raise ValueError('Provide frozen cases, contrast bank and a new dated output')
 seeds=[int(x) for x in (args.eval_seeds or '').split(',') if x]
 if seeds!=[1001]:raise ValueError('This development reevaluation accepts only seed1001; confirmation requires a separately frozen protocol')
 path=Path(args.native_evaluate_from).resolve();source=path.parent.parent
 cfg=json.loads((source/'run_config.json').read_text());method=args.methods
 if method not in {'dad','rl_sboed','vsoed','scorebed','fixed','moe','resmoe','switchmoe','switchresmoe'} or path.stem!=method:raise ValueError('Select exactly the matching native method')
 training=json.loads((source/'models'/(method+'_training.json')).read_text())
 total=training.get('total_training_episodes',training['generated_training_trajectories'])
 if total!=1000000 or not training['checkpoint_selection'].startswith('final'):raise ValueError('Source must have a complete1M final-budget policy')
 if (source/'exit_code').exists() and (source/'exit_code').read_text().strip()!='0':raise ValueError('Failed source run cannot be certified')
 b=make_benchmark(raw);old=make_benchmark(cfg['benchmark']);h=cfg['horizon']
 if benchmark_signature(b)!=benchmark_signature(old) or h!=raw['experiment']['step_number']:raise ValueError('Evaluation benchmark differs from training physics')
 cases_path=Path(args.evaluation_cases);bank_path=Path(args.evaluation_bank);cases=np.load(cases_path);bank=np.load(bank_path)
 draws=dict(truth=torch.from_numpy(cases['truth']).double(),noise=torch.from_numpy(cases['noise']).double(),contrasts=torch.from_numpy(bank).double())
 settings=DEFAULTS|raw['research'].get('settings',{})
 settings.update(eval_systems=len(draws['truth']),eval_contrasts=len(bank),evaluation_population_systems=len(draws['truth']),evaluation_batch_size=16,evaluation_contrast_chunk=16384,threads=2)
 if not args.smoke and (len(draws['truth'])!=2000 or len(bank)!=1000000):raise ValueError('Full location development reevaluation requires2000/L1M')
 if b.name!='location_finding':raise ValueError('Portable SIR evaluation requires its separate readiness gate')
 torch.set_num_threads(2);p=load_checkpoint(path,b).cpu().eval()
 if getattr(args,'evaluation_relaxed_routing',False):
  if method not in {'moe','resmoe','switchmoe','switchresmoe'}:raise ValueError('Relaxed routing is a family-only diagnostic')
  p.set_routing(relaxed=True)
 before=hashlib.sha256(path.read_bytes()).hexdigest()
 out=Path(args.output).resolve();rel=out.relative_to(root/'experiments')
 if len(rel.parts)<2 or len(rel.parts[0])!=8 or not rel.parts[1].startswith(rel.parts[0]+'_'):raise ValueError('Use the dated campaign hierarchy')
 out.mkdir(parents=True,exist_ok=False)
 for name in ['models','logs','eval']:(out/name).mkdir()
 metadata=copy.deepcopy(cfg);metadata.update(evaluation_only=True,settings=settings,eval_seeds=[1001],workability_only=args.smoke,source_training_run=str(source),source_checkpoint_sha256=before,new_training_episodes=0,routing_diagnostic=('relaxed' if getattr(args,'evaluation_relaxed_routing',False) else 'hard'),evaluation_cases_sha256=hashlib.sha256(cases_path.read_bytes()).hexdigest(),evaluation_bank_sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest())
 write_json(out/'run_config.json',metadata);freeze_source(root,out)
 # Reference existing trained parameters and records; do not duplicate weights.
 for file in (source/'models').iterdir():
  if file.is_file():(out/'models'/file.name).symlink_to(file.resolve())
 started=time.perf_counter()
 try:
  rows,summary=evaluate(b,p,method,h,settings,1001,training,out,paper=True,infer_parameters=False,evaluation_draws=draws)
  assert before==hashlib.sha256(path.read_bytes()).hexdigest()
  write_json(out/'eval'/(method+'_seed1001.json'),rows);write_json(out/'summary.json',[summary])
  write_json(out/'evaluation_provenance.json',dict(training_updates=0,new_training_episodes=0,source_checkpoint=str(path),source_checkpoint_sha256=before,cases_sha256=metadata['evaluation_cases_sha256'],bank_sha256=metadata['evaluation_bank_sha256'],estimator='unchanged runner.evaluate and location_evaluation.contrast_scores',evaluation_seconds=time.perf_counter()-started,workability_only=args.smoke))
  (out/'exit_code').write_text('0\n');print('RESULT_DIR='+str(out),flush=True)
 except BaseException:
  (out/'exit_code').write_text('1\n');raise
