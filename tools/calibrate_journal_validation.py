"""Read-only nested-contrast resolution diagnostic, independent of final draws."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch,yaml
from src.research.benchmarks import make_benchmark
from src.research.journal_validation import CommonValidation
from src.research.journal_checkpoints import load_checkpoint

def stats(x):
 x=np.asarray(x);se=float(x.std(ddof=1)/len(x)**.5)
 return dict(mean=float(x.mean()),mc_standard_error=se,conditional_95_interval=[float(x.mean()-1.96*se),float(x.mean()+1.96*se)])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--checkpoint',action='append',required=True);ap.add_argument('--systems',type=int,default=64);ap.add_argument('--contrasts',default='65536,262144,1000000,4000000');ap.add_argument('--output',required=True);args=ap.parse_args()
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);raw=yaml.safe_load(Path(args.config).read_text());b=make_benchmark(raw);h=raw['experiment']['step_number'];L=sorted(set(map(int,args.contrasts.split(','))))
 cfg=raw['research']['journal_protocol']['validation'];v=CommonValidation(raw,h,cfg['seed'],args.systems,max(L),cfg['batch_size'],cfg['chunk']);records=[];values={}
 for index,path in enumerate(args.checkpoint):
  p=load_checkpoint(path,b);scores={};rows=[]
  for contrast in L:
   a,cost=v(p,contrast);scores[contrast]=a;rows.append(dict(contrasts=contrast,eig=stats(a),ceiling=float(np.log(contrast+1)),fraction_within_point1_of_ceiling=float(np.mean(a>np.log(contrast+1)-.1)),cost=cost))
   np.save(out/(f'{index}_'+Path(path).stem+f'_{contrast}_scores.npy'),a)
  sensitivity={f'{L[i]}_minus_{L[i-1]}':stats(scores[L[i]]-scores[L[i-1]]) for i in range(1,len(L))}
  records.append(dict(checkpoint=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),scores=rows,resolution_differences=sensitivity));values[str(path)]=scores
  print(json.dumps(records[-1],indent=2),flush=True)
 paired_policy_differences={f'{a}_minus_{b}':{str(contrast):stats(values[a][contrast]-values[b][contrast]) for contrast in L} for i,a in enumerate(values) for b in list(values)[:i]}
 report=dict(paired_policy_differences=paired_policy_differences,phase='development_resolution_diagnostic',seed=cfg['seed'],systems=args.systems,draws_sha256=v.identity,final_seed_used=False,records=records,
 note='Nested finite-contrast sensitivity on fixed paired validation histories, not a new trained result or true-EIG certificate. Repeat on developed selected policies before freezing.')
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');(out/'exit_code').write_text('0\n')
if __name__=='__main__':main()
