import pathlib,os,sys,json,csv,hashlib,math,shutil
R=pathlib.Path.home()/'Documents/objective_power_grid_temp';C=pathlib.Path(sys.argv[1]);OLD=R/'scratch_runs/11012026_174441_moe_sboed_v6_v1_reference_seed101';sys.path.insert(0,str(R));sys.path.insert(0,str(OLD/'source'));os.chdir(R)
import numpy as np,torch
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v6_policy import V6Policy
from src.research.engine import rollout
from src.research.benchmarks import make_benchmark
def acquire():
 import fcntl
 files=[]
 for path in [R.parent/'objective_power_grid/.gpu0.lock',pathlib.Path.home()/'.cache/objective_power_grid/device_0.lock',R/'.gpu0.lock']:
  f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);files.append(f)
 return files
def read(p):return json.loads(p.read_text())
held=acquire();torch.set_num_threads(4);raw=read(OLD/'physical.json');provenance=read(C/'provenance.json');models={}
for arm in ['v1','v6']:
 p=V6Policy().cuda().eval();p.load_state_dict(torch.load(provenance[arm]['checkpoint'],map_location='cuda',weights_only=False)['state_dict'],strict=True);models[arm]=p
bank=np.load(C/'reference/development.npz')
with torch.device('cuda'),torch.no_grad():
 r=rollout(make_benchmark(raw),models['v1'],30,256,202611021001,0,truth=torch.from_numpy(bank['truth'][:256].copy()).cuda(),noise=torch.from_numpy(bank['noise'][:256].copy()).cuda())
 rows=[];transitions={}
 for low,high in [(0,10),(10,20),(20,30)]:
  features=torch.cat(r['features'][low:high]);x=features.cpu().numpy();scale=x.std(0);scale[scale<1e-8]=1.;z=(x-x.mean(0))/scale
  # Fixed unsupervised observed-history clusters; no theta, EIG or routing labels.
  rng=np.random.default_rng(202611021002+low);centers=z[rng.choice(len(z),4,replace=False)].copy()
  for _ in range(20):
   labels=((z[:,None]-centers[None])**2).sum(-1).argmin(1)
   for k in range(4):
    if np.any(labels==k):centers[k]=z[labels==k].mean(0)
  for arm,p in models.items():
   allids=[];allscores=[]
   for stage in range(low,high):p(r['features'][stage],stage);allids.append(p.last_dispatch['indices']);allscores.append(p.last_dispatch['scores'])
   ids=torch.cat(allids).cpu();scores=torch.cat(allscores).cpu();one=torch.nn.functional.one_hot(ids,7).sum(1).double()
   for k in range(4):
    which=torch.from_numpy(labels==k)
    for e in range(7):rows.append(dict(arm=arm,stages=f'{low+1}-{high}',observed_history_cluster=k,cluster_cases=int(which.sum()),expert=e,selection=float(one[which,e].mean()),gate=float(scores[which,e].mean())))
   seq=torch.stack(allids);changes=[]
   for a,b in zip(seq[:-1],seq[1:]):changes.append(float((a.sort(-1).values!=b.sort(-1).values).any(-1).double().mean()))
   transitions[f'{arm}_{low+1}_{high}']=float(np.mean(changes))
with (C/'history_cluster_routing.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(C/'routing_transition.json').write_text(json.dumps(transitions,indent=2))
for f in held:f.close()
# Audit source/config/initialization, preserve references without rewriting OLD.
ref=C/'reference';ref.mkdir(exist_ok=True)
for name in ['protocol_frozen.json','source_manifest_sha256.json','unit_test_results.json','throughput_feasibility.json']:
 shutil.copy2(OLD/name,ref/name)
for arm,folder in [('v1','training_v1_matched_seed101'),('v6','training_v6_seed101')]:shutil.copy2(OLD/folder/'initial_draw_receipt.json',ref/f'{arm}_initial_draw_receipt.json')
a=read(ref/'v1_initial_draw_receipt.json');b=read(ref/'v6_initial_draw_receipt.json');assert a==b
(C/'MATCHING_AND_LIMITATIONS.md').write_text('''# Matched-work audit and limits

Both arms start from identical fresh8434-parameter state (not historical weights),18k successful updates/B2000/L2000/36M fresh primary trajectories, Adam5e-5 betas(.8,.998),clip5,balance.001. Both share the LR interval360 adaptation; historical native interval1000 is not silently claimed. Initial truth/contrast/weight receipts exactly match. Routing exploration is the between-arm change; original unperturbed softmax gates and top3 retained. Current saved full checkpoints strictly load. Diagnostics leave weights unchanged. Core full-horizon scorer/simulator hashes and original source manifests are retained.

Per-stage selection/co-selection/gate distributions, output cosine, fixed-history cluster routing, route transitions and primary-gradient norms/CVs are recorded. Clusters use only observed history standardized within ten-stage blocks; cluster labels have no posterior or expert-role semantics. Four of seven experts are structurally inactive per decision undertop3; this is not batch-wide starvation. All experts had nonzero gradients across the tested batches. Gradient CV from four batches is uncertain. No gradient-conflict claim across unlike tensors.

The128-case/16k-contrast zero-output interventions preserve top3 identities and gate weights, with no renormalization. They can shift trajectory distribution. Fourteen masks are exploratory multiple comparisons: expert6's nominal interval is not family-wise significant (Holm correction required), and does not prove specialist competence. Existing single-step swap diagnostics on2k/1M also show small, uncertain effects.

Nested contrast gradients differ substantially and one V6 batch has negative L2000-vs-L8000 cosine. Gradients remain nonzero and highly variable: proximity to the finite ceiling does not by itself identify a dominant optimization bottleneck. No controlled training evidence yet shows that changing L improves EIG. Therefore the causal diagnosis remains INCONCLUSIVE and no speculative18k candidate is launched, as required by the task's inconclusive-evidence branch. The completed bounded contrast/gradient and masking experiments are the discriminating experiment; no V6R1 performance gain is claimed.

New common-development results are evaluation evidence only, conditional on one training seed. Final test bank never copied or accessed. Source/config/checkpoint hashes are explicit in provenance.json and reference manifests. Both controls were trained on the5080; no cross-GPU matched-training claim.
''')
with (C/'expert_stats.csv').open() as f:stats=list(csv.DictReader(f))
fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,arm in zip(axes,['v1','v6']):
 matrix=np.array([[float(next(x for x in stats if x['arm']==arm and int(x['stage'])==t and int(x['expert'])==e)['selection_frequency']) for e in range(7)] for t in range(30)])
 im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1);ax.set_title(arm.upper());ax.set_xlabel('Expert');ax.set_ylabel('Stage');fig.colorbar(im,ax=ax,label='Selection frequency')
fig.savefig(C/'routing_selection.png',dpi=160);plt.close(fig)
with (C/'gradient_stats.csv').open() as f:gr=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True);names=['shared','router','head']+[f'expert{e}' for e in range(7)]
for arm in ['v1','v6']:ax.plot(names,[np.mean([float(x['norm']) for x in gr if x['arm']==arm and x['group']==g]) for g in names],marker='o',label=arm)
ax.set_yscale('log');ax.set_ylabel('Primary full-horizon gradient norm');ax.legend();fig.savefig(C/'primary_gradient_norms.png',dpi=160);plt.close(fig)
# Report actual new common-bank numbers, without replacing historical evidence.
results=read(C/'development_eig.json');(C/'comparison.md').write_text('# New shared development-bank comparison\n\n'+json.dumps(results,indent=2)+'\n\n2000 paired cases,1M common contrasts. No established V6 improvement; case MCSE is not training-seed uncertainty. Original matched-bank result+0.02781±0.04025 remains immutable in archived_paired_development.json.\n')
(C/'ARTIFACT_INDEX.txt').write_text('\n'.join(str(p) for p in sorted(C.rglob('*')) if p.is_file()))
print('CONTEXT_AND_REPORTS_COMPLETE')
