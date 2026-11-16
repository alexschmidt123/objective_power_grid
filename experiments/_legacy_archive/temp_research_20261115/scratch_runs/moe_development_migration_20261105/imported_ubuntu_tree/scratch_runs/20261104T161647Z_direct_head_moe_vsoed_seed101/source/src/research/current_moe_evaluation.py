"""Explicit frozen current-v1 evaluation dispatch. Not run by scope cleanup."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,yaml
from src.research.project_scope import load_current_policy,validate_current_benchmark
from src.research.journal_validation import CommonValidation

def main():
 p=argparse.ArgumentParser();p.add_argument('--current-moe-evaluate',action='store_true');p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--systems',type=int,required=True);p.add_argument('--contrasts',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
 cfg=yaml.safe_load(Path(a.config).read_text());raw=cfg.get('benchmark',cfg);benchmark=raw['system']['name']
 from src.research.benchmarks import make_benchmark
 validate_current_benchmark(make_benchmark(raw));model=load_current_policy(a.checkpoint,benchmark)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 val=CommonValidation(raw,model.horizon,a.seed,a.systems,a.contrasts,device=a.device);scores,meta=val(model)
 np.save(out/'scores.npy',scores);(out/'result.json').write_text(json.dumps(dict(method='MoE-sBOED-v1',EIG=float(scores.mean()),MC_SE=float(scores.std(ddof=1)/len(scores)**.5),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),metadata=meta),indent=2))
if __name__=='__main__':main()
