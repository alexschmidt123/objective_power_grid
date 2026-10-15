"""Frozen-policy per-trajectory sPCE resolution and pathwise sensitivity probe.

No optimizer updates. Truth/noise and nested contrastive samples are paired.
Gradients are trajectory-specific sensitivities to post-initial action logits;
base parameters remain frozen while their input derivatives remain available.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout,generator
from src.research.moe_family import load_policy,BasePolicy


def describe(a):
 a=np.asarray(a)
 return dict(mean=float(a.mean()),quantiles_p0_p10_p50_p90_p99_p100=np.quantile(a,[0,.1,.5,.9,.99,1]).tolist())


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--cases',type=int,default=64);ap.add_argument('--contrasts',default='1024,4096');ap.add_argument('--checkpoint-type',choices=['family','dad_mlp'],default='family');args=ap.parse_args();torch.set_num_threads(1)
 out=Path(args.output);out.mkdir(parents=True,exist_ok=False);cfg=json.loads((Path(args.run)/'run_config.json').read_text());b=make_benchmark(cfg['benchmark']);counts=[int(v) for v in args.contrasts.split(',')]
 assert b.name=='location_finding' and cfg['horizon']==30
 before=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest();result=dict(optimizer_updates=0,cases=args.cases,seed=7100104,
  paired='Same truths and noise, per-trajectory nested contrast samples; same frozen relaxed training policy',
  gradient_definition='L2 norm of d(sum_i sPCE_i)/d(post-initial pre-bounded action logits); includes downstream history effects; independent trajectories',policy_type=args.checkpoint_type,
  checkpoint_sha256=before,counts={},limitation='Fresh checkpoint diagnostic, not a training performance or held-out EIG result')
 device='cuda:0' if torch.cuda.is_available() else 'cpu'
 with torch.device(device):
  rng=generator(7100104);truth=b.sample((args.cases,),rng);particles=b.sample((args.cases,max(counts)),rng)
  noise=torch.randn(args.cases,30,1,generator=generator(7100105),dtype=torch.float64)
  if args.checkpoint_type=='dad_mlp':p=BasePolicy.from_checkpoint(args.checkpoint,b,30,finetune=True).policy.train()
  else:
   p=load_policy(args.checkpoint,b,relaxed=True).train();p.core.routing_override=None
  reference=None
  for L in counts:
   if device.startswith('cuda'):torch.cuda.synchronize()
   started=time.perf_counter()
   r=rollout(b,p,30,args.cases,7100104,L,truth=truth,noise=noise,prior_particles=particles[:,:L])
   actions=torch.stack(r['actions']).detach()
   error=0. if reference is None else float((actions-reference).abs().max())
   if reference is None:reference=actions
   assert error<1e-10,'Changing only contrast samples changed the paired policy histories'
   scores=r['info'][:,-1];logits=[v for v in r['latents'][1:] if v.requires_grad]
   derivatives=torch.autograd.grad(scores.sum(),logits,allow_unused=True)
   norm=torch.stack([torch.zeros(args.cases,device=device) if g is None else g.detach().square().sum(-1) for g in derivatives]).sum(0).sqrt().cpu().numpy()
   values=scores.detach().cpu().numpy();ceiling=math.log(L+1)
   if device.startswith('cuda'):torch.cuda.synchronize()
   result['counts'][str(L)]=dict(ceiling=ceiling,score_distribution=describe(values),
    fraction_within_01=float(np.mean(values>ceiling-.1)),fraction_within_001=float(np.mean(values>ceiling-.01)),
    gradient_distribution=describe(norm),fraction_gradient_below_1e_6=float(np.mean(norm<1e-6)),
    fraction_gradient_below_1e_3=float(np.mean(norm<1e-3)),gradient_norms=norm.tolist(),scores=values.tolist(),
    max_action_difference_from_other_L=error,seconds=time.perf_counter()-started,
    likelihood_trajectories=args.cases*(L+1),likelihood_particle_stages=args.cases*(L+1)*30)
   print(L,{k:v for k,v in result['counts'][str(L)].items() if k not in ['gradient_norms','scores']},flush=True)
 assert before==hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
 (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'exit_code').write_text('0\n')
if __name__=='__main__':main()
