import os,sys,pathlib,json,time,hashlib,math,csv,fcntl,subprocess,traceback
R=pathlib.Path.home()/'Documents/objective_power_grid_temp';os.chdir(R);sys.path.insert(0,str(R))
OLD=R/'scratch_runs/11012026_174441_moe_sboed_v6_v1_reference_seed101'
C=pathlib.Path(sys.argv[1]);sys.path.insert(0,str(OLD/'source'))
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=pathlib.Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False));q.replace(p)
def table(p,rows):
 if not rows:return
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 import torch,numpy as np
 from v6_policy import V6Policy
 from src.research.engine import rollout
 from src.research.benchmarks import make_benchmark
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 held=[]
 for path in [R.parent/'objective_power_grid/.gpu0.lock',pathlib.Path.home()/'.cache/objective_power_grid/device_0.lock',R/'.gpu0.lock']:
  f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 put(C/'status.json',dict(status='RUNNING_DIAGNOSIS',pid=os.getpid(),started=time.time()))
 raw=json.loads((OLD/'physical.json').read_text());stats=[];grad=[];diag=[];initial=time.monotonic()
 provenance={};saved={}
 for arm,folder in [('v1','training_v1_matched_seed101'),('v6','training_v6_seed101')]:
  rr=json.loads((OLD/folder/'result.json').read_text());cp=pathlib.Path(rr['checkpoint']['path']);assert sha(cp)==rr['checkpoint']['sha256'];assert rr['update']==18000 and rr['primary']==36000000
  provenance[arm]=dict(checkpoint=str(cp),sha256=sha(cp),result=rr)
  p=V6Policy().to('cuda').eval();p.load_state_dict(torch.load(cp,map_location='cuda',weights_only=False)['state_dict'],strict=True);assert p.counts()['total_parameters']==8434
  before={k:v.detach().clone() for k,v in p.state_dict().items()};captures=[]
  def hook(m,inputs,output):
   z=m.last_features;dispatch=m.last_dispatch
   if dispatch:
    with torch.no_grad():
     out=torch.stack([e(z) for e in m.routed],dim=1);cos=torch.nn.functional.cosine_similarity(out[:,:,None,:],out[:,None,:,:],dim=-1)
     captures.append(dict(stage=m._last_stage,scores=dispatch['scores'].cpu(),ids=dispatch['indices'].cpu(),cos=cos.mean(0).cpu(),outnorm=out.norm(dim=-1).mean(0).cpu()))
  h=p.register_forward_hook(hook)
  data=np.load(OLD/'reference/development_bank.npz')
  with torch.device('cuda'),torch.no_grad():r=rollout(make_benchmark(raw),p,30,256,202611020601,0,truth=torch.from_numpy(data['truth'][:256].copy()).cuda(),noise=torch.from_numpy(data['noise'][:256].copy()).cuda())
  saved[arm]=dict(policy=p,features=[x.detach() for x in r['features']],actions=[x.detach() for x in r['actions']])
  h.remove()
  for cap in captures:
   scores=cap['scores'];ids=cap['ids'];one=torch.nn.functional.one_hot(ids,7).sum(1).double();freq=one.mean(0);co=(one.T@one)/len(one)
   for e in range(7):stats.append(dict(arm=arm,stage=cap['stage'],expert=e,selection_frequency=float(freq[e]),mean_softmax=float(scores[:,e].mean()),entropy=float((-(scores*scores.clamp_min(1e-30).log()).sum(-1)).mean()),effective_experts=float((-(scores*scores.clamp_min(1e-30).log()).sum(-1)).exp().mean()),mean_output_norm=float(cap['outnorm'][e]),mean_cosine_other=float((cap['cos'][e].sum()-1)/6),co_selection=json.dumps(co[e].tolist())))
  groups={'shared':list(p.shared.parameters()),'router':list(p.router.parameters()),'head':list(p.emitter.parameters()),**{f'expert{e}':list(p.routed[e].parameters()) for e in range(7)}}
  for k in range(4):
   p.zero_grad(set_to_none=True);p.train();p.exploration_scale=0.
   with torch.device('cuda'):r=rollout(make_benchmark(raw),p,30,128,202611020700+k,2000);loss=-r['info'][:,-1].mean();loss.backward()
   diag.append(dict(arm=arm,batch=k,L=2000,spce=float(-loss.detach()),ceiling=math.log(2001),gap=float(math.log(2001)+loss.detach())))
   for name,ps in groups.items():
    norm=math.sqrt(sum(float(x.grad.square().sum()) for x in ps if x.grad is not None));grad.append(dict(arm=arm,batch=k,group=name,norm=norm,zero=int(norm<1e-12)))
  p.eval();assert all(torch.equal(v,before[k]) for k,v in p.state_dict().items()),'Diagnostic modified weights'
  del r,loss;torch.cuda.empty_cache()
 # Fixed-history ablation effect on decisions, distinct from EIG.
 for arm,x in saved.items():
  p=x['policy'];rows=[]
  for t in [0,9,19,29]:
   f=x['features'][t]
   with torch.no_grad():
    base=p(f,t);p.ablate_routed=True;masked=p(f,t);p.ablate_routed=False
    rows.append(dict(stage=t,latent_rms=float((base-masked).square().mean().sqrt())))
  put(C/f'{arm}_decision_masking.json',rows)
 # Reuse existing 2000/1M paired intervention arrays; do not rerun expensive controls.
 oldpair=json.loads((OLD/'paired_eig_development.json').read_text());oldswaps=json.loads((OLD/'specialization_diagnostics.json').read_text())
 put(C/'archived_paired_development.json',oldpair);put(C/'archived_behavioral_interventions.json',oldswaps)
 table(C/'expert_stats.csv',stats);table(C/'gradient_stats.csv',grad);table(C/'diagnosis.csv',diag);put(C/'provenance.json',provenance)
 summary={}
 for arm in ['v1','v6']:
  a=[x for x in stats if x['arm']==arm];g=[x for x in grad if x['arm']==arm];d=[x for x in diag if x['arm']==arm]
  summary[arm]=dict(min_expert_mean_selection=min(np.mean([x['selection_frequency'] for x in a if x['expert']==e]) for e in range(7)),mean_offdiag_cosine=float(np.mean([x['mean_cosine_other'] for x in a])),mean_spce=float(np.mean([x['spce'] for x in d])),mean_gap=float(np.mean([x['gap'] for x in d])),zero_gradient_fraction=float(np.mean([x['zero'] for x in g])),gradient_groups={name:dict(mean=float(np.mean([x['norm'] for x in g if x['group']==name])),CV=float(np.std([x['norm'] for x in g if x['group']==name])/(np.mean([x['norm'] for x in g if x['group']==name])+1e-30))) for name in groups})
 put(C/'diagnosis_summary.json',summary)
 # No automatic full run: first use the observed evidence for the single intervention decision.
 text='# V1/V6 routing and gradient diagnosis\n\n'+json.dumps(summary,indent=2)+'\n\nSaved weights remained bitwise unchanged. Primary gradients exclude balance loss. Diagnostics use observed histories and independent simulation draws, no inferred posterior claims. Existing paired 2000/1M interventions are retained; small deltas do not prove specialization. Gradient CV uses four fixed batches and is imprecise. No new full candidate is automatically launched.\n'
 (C/'V1_V6_ROUTING_GRADIENT_DIAGNOSIS.md').write_text(text)
 (C/'comparison.md').write_text('# Existing matched development evidence\n\n'+json.dumps(oldpair,indent=2)+'\nNo statistically established improvement. Final bank untouched.\n')
 put(C/'status.json',dict(status='DIAGNOSIS_COMPLETE_DECISION_PENDING',pid=os.getpid(),seconds=time.monotonic()-initial,weights_unchanged=True,summary=summary))
 print(json.dumps(summary),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:put(C/'status.json',dict(status='DIAGNOSIS_FAILED',error=str(e),traceback=traceback.format_exc()));raise
