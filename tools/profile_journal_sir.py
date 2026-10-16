"""Bounded SIR runtime/gradient diagnostic; never trains or selects a policy."""
import argparse,json,time
from pathlib import Path
import torch,yaml
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.sir_dad import SIRSumMLP

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--larger-batches',action='store_true');args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 raw=yaml.safe_load(Path('configs/sir_ode_research.yaml').read_text());records=[];torch.set_num_threads(2)
 cases=[('cuda:0',64,1024,5),('cuda:0',128,1024,5),('cuda:0',256,1024,5)] if args.larger_batches else [('cpu',4,8,5),('cuda:0',4,8,5),('cuda:0',16,1024,5),('cuda:0',16,1024,10)]
 for device,n,L,h in cases:
  if device.startswith('cuda') and not torch.cuda.is_available():continue
  b=make_benchmark(raw)
  with torch.device(device):
   torch.manual_seed(201);p=SIRSumMLP(h)
   if device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   start=time.perf_counter();r=rollout(b,p,h,n,201901,L);forward=time.perf_counter()-start
   loss=-r['info'][:,-1].mean();loss.backward()
   if device.startswith('cuda'):torch.cuda.synchronize()
   seconds=time.perf_counter()-start
   assert torch.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in p.parameters() if x.grad is not None)
   row=dict(device=device,batch=n,contrasts=L,horizon=h,forward_backward_seconds=seconds,forward_seconds_unsynchronized=forward,particle_stages=b.calls,
    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
    extrapolated_one_million_episode_optimization_hours=seconds*1000000/n/3600,parameter_update=False,workability_only=True,solver='original_adaptive_min512_RK4')
   records.append(row);print(json.dumps(row),flush=True)
   (out/'cost.json').write_text(json.dumps(records,indent=2)+'\n')
 (out/'exit_code').write_text('0\n')
if __name__=='__main__':main()
