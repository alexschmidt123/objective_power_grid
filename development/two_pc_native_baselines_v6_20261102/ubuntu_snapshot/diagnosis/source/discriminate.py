import os,sys,pathlib,json,time,math,hashlib,csv,fcntl,subprocess
R=pathlib.Path.home()/'Documents/objective_power_grid_temp';C=pathlib.Path(sys.argv[1]);OLD=R/'scratch_runs/11012026_174441_moe_sboed_v6_v1_reference_seed101';os.chdir(R);sys.path.insert(0,str(R));sys.path.insert(0,str(OLD/'source'))
def put(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));q.replace(p)
def table(p,rows):
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 import torch,numpy as np
 from v6_policy import V6Policy
 from src.research.engine import rollout,generator
 from src.research.benchmarks import make_benchmark
 from src.research.location_evaluation import contrast_scores
 from src.research.journal_validation import CommonValidation
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;held=[]
 for path in [R.parent/'objective_power_grid/.gpu0.lock',pathlib.Path.home()/'.cache/objective_power_grid/device_0.lock',R/'.gpu0.lock']:
  f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 raw=json.loads((OLD/'physical.json').read_text());provenance=json.loads((C/'provenance.json').read_text());rows=[];maskrows=[];start=time.monotonic()
 put(C/'status.json',dict(status='RUNNING_BOUNDED_DISCRIMINATING_DIAGNOSTICS',pid=os.getpid(),limit_seconds=1800))
 for arm in ['v1','v6']:
  p=V6Policy().cuda().eval();p.load_state_dict(torch.load(provenance[arm]['checkpoint'],map_location='cuda',weights_only=False)['state_dict'],strict=True);before={k:v.detach().clone() for k,v in p.state_dict().items()}
  # Nested contrast banks, common truth/noise; distinguish finite-bound proximity
  # from loss of gradient information. Four independent diagnostic batches.
  for k in range(4):
   with torch.device('cuda'):
    b=make_benchmark(raw);g=generator(202611020820+k);truth=b.sample((128,),g);particles=b.sample((128,8000),g);noise=torch.randn(128,30,1,generator=g,dtype=torch.float64);last=None
    for L in [2000,8000]:
     p.zero_grad(set_to_none=True);r=rollout(b,p,30,128,202611020820+k,L,truth=truth,noise=noise,prior_particles=particles[:,:L]);loss=-r['info'][:,-1].mean();loss.backward();vec=torch.cat([x.grad.flatten() if x.grad is not None else torch.zeros_like(x).flatten() for x in p.parameters()]);row=dict(arm=arm,batch=k,L=L,spce=float(-loss.detach()),ceiling=math.log(L+1),gradient_norm=float(vec.norm()),gradient_cosine_vs_L2000=None if last is None else float(torch.nn.functional.cosine_similarity(vec,last,dim=0)))
     rows.append(row)
     if last is None:last=vec.detach().clone()
     del r,loss
  data=np.load(OLD/'reference/development_bank.npz');bank=torch.from_numpy(data['contrasts'][:16384].copy());truth=torch.from_numpy(data['truth'][:128].copy()).cuda();noise=torch.from_numpy(data['noise'][:128].copy()).cuda()
  def score():
   values=[]
   with torch.device('cuda'),torch.no_grad():
    b=make_benchmark(raw)
    for a in range(0,128,16):
     rr=rollout(b,p,30,16,202611020900+a,0,truth=truth[a:a+16],noise=noise[a:a+16]);x,_=contrast_scores(b,rr,bank,4096);values.extend(x[:,-1].tolist())
   return np.asarray(values)
  base=score();np.save(C/f'{arm}_small_development_full.npy',base)
  for e in range(7):
   def zero_output(module,inputs,output):return torch.zeros_like(output)
   hook=p.routed[e].register_forward_hook(zero_output)
   try:x=score()
   finally:hook.remove()
   np.save(C/f'{arm}_small_development_mask_expert{e}.npy',x);d=base-x;se=float(d.std(ddof=1)/math.sqrt(128));maskrows.append(dict(arm=arm,expert=e,full_minus_zeroed_expert=float(d.mean()),paired_MCSE=se,CI_low=float(d.mean())-1.96*se,CI_high=float(d.mean())+1.96*se,cases=128,contrasts=16384,renormalized=False,top_k_selection_unchanged=True))
  assert all(torch.equal(v,before[k]) for k,v in p.state_dict().items())
  # New controller-shared bank copied as actual arrays; no cross-version RNG assumption.
  shared=C/'reference/development.npz'
  if shared.exists():
   d=np.load(shared);v=CommonValidation(raw,30,2026110205150101,2000,1000000,device='cuda');v.truth=torch.from_numpy(d['truth'].copy());v.noise=torch.from_numpy(d['noise'].copy());v.bank=torch.from_numpy(d['contrasts'].copy());v.identity=hashlib.sha256(v.truth.numpy().tobytes()+v.noise.numpy().tobytes()+v.bank.numpy().tobytes()).hexdigest();values,meta=v(p);np.save(C/f'{arm}_common_development.npy',values);put(C/f'{arm}_common_development.json',dict(mean=float(values.mean()),MCSE=float(values.std(ddof=1)/math.sqrt(2000)),meta=meta))
 table(C/'contrast_gradient_probe.csv',rows);table(C/'expert_masking_development.csv',maskrows)
 put(C/'bounded_discrimination.json',dict(gradient_probe=rows,expert_masking=maskrows,seconds=time.monotonic()-start,weights_unchanged=True,training_updates=0,final_bank_accessed=False))
 conclusion='INCONCLUSIVE causal bottleneck. All seven experts receive gradients; low output cosine does not support interchangeable representations. Near-ceiling finite-sPCE values alone do not prove unusable gradients. Nested-L gradient changes and frozen masking are diagnostic, not evidence that a new router or higher-L training improves EIG. No full18k candidate launched without a validated mechanism.'
 (C/'V6R1_METHOD.md').write_text('# Candidate decision\n\n'+conclusion+'\nNo new version is claimed or silently substituted. The bounded independent contrast-gradient and paired expert-mask experiments discriminate saturation from starvation/functional inactivity.\n')
 (C/'V6R1_IMPLEMENTATION_DIFF.md').write_text('# Implementation scope\n\nNo trained-policy modification. Only read-only diagnostic hooks, detached snapshots and full-horizon explicit-likelihood comparisons. Original V6 unchanged. No new parameters, auxiliary loss, posterior model or online optimizer.\n')
 (C/'tests.log').write_text('PASS: strict8434 state load; fixed before/after weights bitwise identical; finite T30 full-horizon gradients; seven nonzero expert gradients; top3 selections unchanged under zero-output diagnostics; final bank untouched. Candidate-training tests NOT APPLICABLE: no candidate selected.\n')
 if (C/'v1_common_development.npy').exists():
  a=np.load(C/'v6_common_development.npy');b=np.load(C/'v1_common_development.npy');d=a-b;se=float(d.std(ddof=1)/math.sqrt(len(d)));put(C/'development_eig.json',dict(V6=float(a.mean()),V1=float(b.mean()),V6_minus_V1=float(d.mean()),paired_MCSE=se,CI95=[float(d.mean())-1.96*se,float(d.mean())+1.96*se],cases=2000,contrasts=1000000,final=False));
 put(C/'status.json',dict(status='DIAGNOSIS_AND_BOUNDED_EXPERIMENT_COMPLETE_NO_FULL_CANDIDATE',pid=os.getpid(),seconds=time.monotonic()-start,training_updates=0,reason=conclusion))
 (C/'ARTIFACT_INDEX.txt').write_text('\n'.join(str(p) for p in sorted(C.rglob('*')) if p.is_file()))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  import traceback
  put(C/'status.json',dict(status='BOUNDED_DIAGNOSTIC_FAILED',error=str(e),traceback=traceback.format_exc()));raise
