
"""Verify exact v1 source and immutable checkpoint against saved canonical scores."""
import argparse,hashlib,json,subprocess,time,sys
from pathlib import Path
import numpy as np,torch,yaml
from src.research.deepseek_sboed import load_deepseek
from src.research.deepseek_sboed_experiment import write,sha
from src.research.journal_validation import CommonValidation
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
def main():
 R=Path.cwd();C=R/'experiments/10262026/10262026_v1_specialization_development';ref=R/'reference_artifacts/v1'
 cfg=yaml.safe_load((ref/'seed101.yaml').read_text());expected=json.loads((ref/'official_seed101_result.json').read_text())
 assert sha(ref/'selected.pt')==expected['checkpoint_sha256']
 files=['src/research/deepseek_sboed.py','src/research/deepseek_sboed_experiment.py','src/research/engine.py','src/research/journal_validation.py','src/research/dad_location_policy.py','src/research/benchmarks.py']
 provenance={}
 for name in files:
  blob=subprocess.check_output(['git','show','13681d7091189a4e68056e6a2864dc58f9d5700b:'+name])
  assert (R/name).read_bytes()==blob,name
  provenance[name]=hashlib.sha256(blob).hexdigest()
 bankroot=Path(str(cfg['canonical_evaluation_directory']).replace('/home/grads/g/g.lin',str(Path.home())))
 assert sha(bankroot/'location_cases.npz')==expected['cases_sha256']
 assert sha(bankroot/'location_contrasts.npy')==expected['contrast_sha256']
 from tools.journal_campaign import exclusive_device_lock
 lock=exclusive_device_lock(wait=True);torch.set_num_threads(2);start=time.perf_counter()
 m=load_deepseek(ref/'selected.pt','cuda');b=make_benchmark(cfg['benchmark'])
 assert m.counts()['total_parameters']==8434 and m.counts()['activated_parameters_per_context']==6898
 with torch.no_grad(),torch.device('cuda'):r=rollout(b,m,30,16,97663000,0)
 maximum=0.
 with torch.no_grad():
  for t,f in enumerate(r['features']):
   z=m.encode_context(f,t);p=m.router(z).softmax(-1);w,i=p.topk(3,-1);route=z*0
   for e in range(7):
    mask=(i==e);weight=(w*mask).sum(-1)
    route+=weight[:,None]*m.routed[e](z)
   manual=m.emitter(z+m.shared(z)+route);actual=m(f,t)
   maximum=max(maximum,float((manual-actual).abs().max()))
 assert maximum<1e-12
 val=CommonValidation(cfg['benchmark'],30,1001,2000,1000000,batch=16,chunk=16384,device='cuda:0')
 cases=np.load(bankroot/'location_cases.npz');val.truth=torch.as_tensor(cases['truth']);val.noise=torch.as_tensor(cases['noise']);val.bank=torch.as_tensor(np.load(bankroot/'location_contrasts.npy'))
 val.identity=hashlib.sha256(val.truth.numpy().tobytes()+val.noise.numpy().tobytes()+val.bank.numpy().tobytes()).hexdigest()
 assert val.identity==expected['metadata']['paired_draws_sha256']
 values,meta=val(m);saved=np.load(ref/'official_seed1001_scores.npy')
 error=float(np.max(abs(values-saved)));np.save(C/'readiness/v1_reproduced_seed1001_scores.npy',values)
 result=dict(PASS=bool(error<1e-8),source_commit='13681d7091189a4e68056e6a2864dc58f9d5700b',checkpoint_sha256=sha(ref/'selected.pt'),architecture_source_sha256=provenance,config_sha256=sha(ref/'seed101.yaml'),EIG=float(values.mean()),MC_SE=float(values.std(ddof=1)/len(values)**.5),saved_EIG=expected['EIG'],maximum_per_trajectory_error=error,maximum_manual_inference_error=maximum,metadata=meta,wall_seconds=time.perf_counter()-start,hardware=torch.cuda.get_device_name(0),original_checkpoint_unchanged=sha(ref/'selected.pt')==expected['checkpoint_sha256'],optimization_updates=0)
 write(C/'readiness/v1_reproduction.json',result);print(json.dumps(result),flush=True)
 assert result['PASS'],'Frozen v1 checkpoint failed numeric reproduction'
 lock.close()
if __name__=='__main__':main()
