"""Paired held-out EIG, role prediction calibration, and causal interventions."""
import copy,json,math,time,hashlib
from pathlib import Path
import numpy as np,torch
from src.research.journal_validation import CommonValidation
from src.research.engine import rollout,replay_likelihood
from src.research.moe_v4_information import targets

def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def stat(x):
 x=np.asarray(x,dtype=float);mu=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)));return dict(mean=mu,MC_SE=se,CI95=[mu-1.96*se,mu+1.96*se],N=len(x))
def load_checked(row):
 path=Path(row['path'])
 if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Checkpoint SHA256 mismatch: '+str(path))
 return torch.load(path,map_location='cuda:0',weights_only=False)['policy'].eval()

def evaluate_models(C,cfg,models,seed,N,L,tag):
 v=CommonValidation(cfg['benchmark'],30,seed,N,L,batch=16,chunk=16384,device='cuda:0');out=C/'evaluations'/tag;out.mkdir(parents=True,exist_ok=True);casefile=out/'paired_cases.npz'
 if not casefile.exists():np.savez(casefile,truth=v.truth.numpy(),noise=v.noise.numpy(),bank=v.bank.numpy())
 scores={};summary={}
 for name,p in models.items():
  x,meta=v(p);scores[name]=x;summary[name]=dict(**stat(x),metadata=meta);np.save(out/(name+'.npy'),x)
 write(out/'summary.json',summary);return scores,summary,v

def evaluate_milestone(C,cfg,models,state,milestone):
 eligible={k:p for k,p in models.items() if state[k]['updates']==milestone and not state[k].get('discard_incomplete_update',False)}
 if not eligible:write(C/'evaluations'/f'milestone{milestone}_incomplete.json',dict(reason='No common complete milestone',updates={k:v['updates'] for k,v in state.items()}));return
 scores,summary,v=evaluate_models(C,cfg,eligible,cfg['seeds']['development'],cfg['evaluation']['development_systems'],cfg['evaluation']['development_contrasts'],f'milestone{milestone}')
 paired={}
 for seed in [101,102]:
  a=f'v4_s{seed}';b=f'v1_context_s{seed}'
  if a in scores and b in scores:paired[str(seed)]=stat(scores[a]-scores[b])
 write(C/'evaluations'/f'milestone{milestone}'/'paired_v4_v1.json',paired)
 write(C/'latest_evaluation.json',dict(phase='development_milestone',milestone=milestone,results=summary,paired_v4_v1=paired,updates={k:v['updates'] for k,v in state.items()},matched_arms_reported_separately=True))

class PermutationIntervention(torch.nn.Module):
 kind='v4_role_swap';outputs_design=False
 def __init__(self,p,a,b,stage):super().__init__();self.policy=p;self.a=a;self.b=b;self.stage=stage
 def forward(self,f,stage):
  mapping=list(range(4));mapping[self.a],mapping[self.b]=mapping[self.b],mapping[self.a];self.policy.expert_permutation=mapping if stage==self.stage else None;return self.policy(f,stage)

@torch.no_grad()
def role_calibration(C,cfg,p,partition,adapter,seed,tag):
 N=cfg['evaluation']['role_systems'];L=cfg['evaluation']['role_contrasts'];v=CommonValidation(cfg['benchmark'],30,seed,N,L,batch=8,chunk=16384,device='cuda:0');out=C/'evaluations'/tag;out.mkdir(parents=True,exist_ok=True);truth=v.truth.cuda();noise=v.noise.cuda();bank=v.bank.cuda();pred_all=[];target_all=[];contributions=[];selection=torch.zeros(4,dtype=torch.float64);p=copy.deepcopy(p).cuda().eval();started=time.monotonic()
 with torch.device('cuda:0'):
  bank_labels=partition.classify(adapter,bank);prior_counts=torch.bincount(bank_labels,minlength=4);truth_labels=partition.classify(adapter,truth);b=adapter.b
  for start in range(0,N,8):
   n=min(8,N-start);r=rollout(b,p,30,n,seed+start*97451,0,truth=truth[start:start+n],noise=noise[start:start+n]);_,_,values=replay_likelihood(b,r['truth'][:,None],r['actions'],r['observations']);true=torch.stack([x[:,0] for x in values],1);all_sum=torch.full_like(true,-torch.inf);group_sum=torch.full_like(true,-torch.inf)
   for offset in range(0,L,16384):
    x=bank[offset:offset+16384];_,_,vals=replay_likelihood(b,x[None].expand(n,-1,-1),r['actions'],r['observations']);ll=torch.stack(vals,2);same=bank_labels[offset:offset+len(x)][None]==truth_labels[start:start+n,None];all_sum=torch.logaddexp(all_sum,torch.logsumexp(ll,1));group_sum=torch.logaddexp(group_sum,torch.logsumexp(ll.masked_fill(~same[:,:,None],-torch.inf),1))
   logZ=torch.logaddexp(true,all_sum);total=true-logZ+math.log(L+1);prior=(prior_counts[truth_labels[start:start+n]]+1).double()/(L+1);coarse=torch.logaddexp(true,group_sum)-logZ-prior[:,None].log();zero=true.new_zeros(n,1);role=targets(torch.cat((zero,total),1),torch.cat((zero,coarse),1));pred=torch.stack([p.role_predictions(f,t) for t,f in enumerate(r['features'])],1);pred_all.append(pred.cpu());target_all.append(role.cpu());delta=[]
   for stage,f in enumerate(r['features']):
    p.mask_expert=None;base=b.feasible(p(f,stage),r['actions'][:stage],30);ids=p.last_dispatch['indices'].cpu();selection+=torch.nn.functional.one_hot(ids,4).sum((0,1)).double();effects=[]
    for e in range(4):p.mask_expert=e;alt=b.feasible(p(f,stage),r['actions'][:stage],30);effects.append((base-alt).square().sum(-1).sqrt().cpu())
    p.mask_expert=None;delta.append(torch.stack(effects,1))
   contributions.append(torch.stack(delta,1))
 pred=torch.cat(pred_all).numpy();target=torch.cat(target_all).numpy();action_delta=torch.cat(contributions).numpy();np.save(out/'role_predictions.npy',pred);np.save(out/'role_targets.npy',target);np.save(out/'next_action_mask_deltas.npy',action_delta)
 flatp=pred.reshape(-1,4);flatt=target.reshape(-1,4);metrics=[];half=N//2;stage_baseline=np.empty_like(target);stage_baseline[:half]=target[half:].mean(0);stage_baseline[half:]=target[:half].mean(0)
 for e in range(4):
  mse=float(np.mean((flatp[:,e]-flatt[:,e])**2));variance=float(np.var(flatt[:,e]));bins=[];edges=np.quantile(flatp[:,e],np.linspace(0,1,6))
  for k in range(5):
   m=(flatp[:,e]>=edges[k])&(flatp[:,e]<=edges[k+1] if k==4 else flatp[:,e]<edges[k+1]);bins.append(dict(N=int(m.sum()),predicted_mean=float(flatp[m,e].mean()) if m.any() else None,realized_mean=float(flatt[m,e].mean()) if m.any() else None))
  stage_mse=float(np.mean((target[:,:,e]-stage_baseline[:,:,e])**2));metrics.append(dict(role=e,MSE=mse,cross_fitted_stage_only_baseline_MSE=stage_mse,skill_vs_stage_baseline=1-mse/stage_mse if stage_mse>1e-15 else None,R2=1-mse/variance if variance>1e-15 else None,prediction_mean=float(flatp[:,e].mean()),realized_mean=float(flatt[:,e].mean()),negative_target_fraction=float((flatt[:,e]<0).mean()),negative_prediction_fraction=float((flatp[:,e]<0).mean()),calibration_bins=bins))
 corr=np.corrcoef(flatt.T);report=dict(role_metrics=metrics,target_correlations=corr.tolist(),mean_action_change_when_selected=[float(action_delta[:,:,e].sum()/float(selection[e])) if selection[e]>0 else None for e in range(4)],expert_selected_slot_share=(selection/selection.sum()).tolist(),mean_physical_next_action_change_when_masked=action_delta.mean((0,1)).tolist(),seconds=time.monotonic()-started,cases_sha256=v.identity,contrasts=L,interpretation='Prediction and action changes do not prove beneficial causal competence; correlations are descriptive, trajectory observations are dependent')
 write(out/'role_calibration.json',report);return pred,target,report

def utility_diagnostics(C,cfg,p,seed,tag,pred):
 N=cfg['evaluation']['role_systems'];L=cfg['evaluation']['role_contrasts'];v=CommonValidation(cfg['benchmark'],30,seed,N,L,batch=8,chunk=16384,device='cuda:0');out=C/'evaluations'/tag;out.mkdir(exist_ok=True,parents=True);base,_=v(p);rows={};tests=[]
 for e in range(4):
  q=copy.deepcopy(p);q.mask_expert=e;alt,meta=v(q);np.save(out/f'mask_expert{e}.npy',alt);rows[f'mask{e}']=dict(full_minus_intervention=stat(base-alt),metadata=meta)
 for routing_seed in [86404991,86404992]:
  q=copy.deepcopy(p);q.routing='uniform_top2';q.routing_seed=routing_seed;q._routing_rng=None;alt,meta=v(q);np.save(out/f'uniform_top2_seed{routing_seed}.npy',alt);rows[f'uniform{routing_seed}']=dict(full_minus_intervention=stat(base-alt),same_K=2,same_selected_softmax_semantics=True,metadata=meta)
 q=copy.deepcopy(p);q.mask_all=True;alt,meta=v(q);rows['mask_all']=dict(full_minus_intervention=stat(base-alt),metadata=meta)
 masktests=[]
 for e in range(4):
  r=rows[f'mask{e}']['full_minus_intervention'];r['pvalue']=math.erfc(abs(r['mean']/r['MC_SE'])/math.sqrt(2)) if r['MC_SE'] else 1.;masktests.append(r)
 running=0.
 for rank,r in enumerate(sorted(masktests,key=lambda r:r['pvalue'])):running=max(running,min(1.,(4-rank)*r['pvalue']));r['Holm_pvalue_four_mask_family']=running
 for stage in [5,15,25]:
  expected=pred[:,stage];den=expected.sum(1);valid=np.isfinite(expected).all(1)&(den>0);coarse=np.divide(expected[:,0]+expected[:,2],den,out=np.zeros_like(den),where=valid);immediate=np.divide(expected[:,0]+expected[:,1],den,out=np.zeros_like(den),where=valid)
  definitions=[('coarse_share_at_least_point1',coarse>=.1),('immediate_above_average_remaining_step',immediate>=1/(30-stage))]
  for a in range(4):
   for b in range(a):
    alt,meta=v(PermutationIntervention(copy.deepcopy(p),a,b,stage));d=base-alt;np.save(out/f'swap_{a}_{b}_stage{stage}.npy',alt)
    for context,mask in definitions:
     high=valid&mask;low=valid&~mask
     if min(high.sum(),low.sum())<16:tests.append(dict(stage=stage,a=a,b=b,context=context,tested=False,high_N=int(high.sum()),low_N=int(low.sum()),reason='Insufficient observable-context coverage; no forced median split'));continue
     hi=stat(d[high]);lo=stat(d[low]);se=math.hypot(hi['MC_SE'],lo['MC_SE']);interaction=hi['mean']-lo['mean'];pvalue=math.erfc(abs(interaction/se)/math.sqrt(2)) if se else 1.;tests.append(dict(stage=stage,a=a,b=b,context=context,tested=True,high=hi,low=lo,interaction=interaction,SE=se,pvalue=pvalue))
 validtests=[x for x in tests if x['tested']];ordered=sorted(validtests,key=lambda x:x['pvalue']);running=0.
 for rank,r in enumerate(ordered):running=max(running,min(1.,(len(ordered)-rank)*r['pvalue']));r['Holm_pvalue']=running
 result=dict(interventions=rows,context_interactions=tests,minimum_cohort=16,predeclared_family=36,interpretation='Single-decision swaps share the complete prefix and preserve two distinct expert functions; downstream policy continues normally. Effects include nonlinear fusion interactions, not additive expert EIG.')
 write(out/'utility_diagnostics.json',result);return result

def final_assessment(C,cfg,index,state,partition,adapter):
 results={};models_by_seed={};score_by_seed={};selected={};paired={};supplemental={}
 for seed in [101,102]:
  tags=[f'{a}_s{seed}' for a in cfg['primary_arms'] if f'{a}_s{seed}' in index]
  if seed==102:tags=[f'v4_s102',f'v1_context_s102']
  sets=[{x['update'] for x in index[tag] if x['eligible']} for tag in tags];common=set.intersection(*sets)
  if not common:write(C/'evaluations'/f'seed{seed}_NO_COMMON_CHECKPOINT.json',dict(tags=tags,updates={t:state[t]['updates'] for t in tags}));continue
  update=max(common);models={};receipts={}
  for tag in tags:
   row=max((x for x in index[tag] if x['eligible'] and x['update']==update),key=lambda x:x['charged_seconds']);models[tag]=load_checked(row);receipts[tag]=row
  seed_eval=cfg['seeds']['confirmation'] if seed==101 else cfg['seeds']['confirmation_replication'];scores,summary,v=evaluate_models(C,cfg,models,seed_eval,cfg['evaluation']['final_systems'],cfg['evaluation']['final_contrasts'],f'final_seed{seed}_update{update}');delta=stat(scores[f'v4_s{seed}']-scores[f'v1_context_s{seed}']);paired[str(seed)]=delta;results[str(seed)]=dict(common_update=update,arms=summary,paired_v4_minus_v1=delta,checkpoints=receipts,training_seed_uncertainty_not_estimated=True);models_by_seed[seed]=models;score_by_seed[seed]=scores;selected[seed]=receipts
 write(C/'evaluations/final_comparison.json',results)
 # Supplemental native recipe is explicitly not a matched-B/L comparison.
 if 'native_v1_s101' in index:
  rows=[r for r in index['native_v1_s101'] if r['eligible']]
  if rows:
   row=max(rows,key=lambda x:x['update']);p=load_checked(row);ns,summary,nv=evaluate_models(C,cfg,{'native_v1_B2000_L2000':p},cfg['seeds']['confirmation'],cfg['evaluation']['final_systems'],cfg['evaluation']['final_contrasts'],'SUPPLEMENTAL_native_recipe');supplemental['v4_minus_native_v1']=stat(score_by_seed[101]['v4_s101']-ns['native_v1_B2000_L2000']) if 101 in score_by_seed else None;write(C/'evaluations/SUPPLEMENTAL_native_recipe/comparison_limit.json',dict(native_recipe=True,B=2000,L=2000,not_training_matched_to_v4=True,checkpoint=row))
 # Matched COMPUTE CEILINGS: actual timestamps and updates remain explicit.
 compute={}
 for target in cfg['compute_checkpoints_seconds']:
  chosen={}
  for tag,rows in index.items():
   candidates=[r for r in rows if r['eligible'] and r['charged_seconds']<=target]
   if candidates:chosen[tag]=max(candidates,key=lambda x:x['charged_seconds'])
  if not chosen:continue
  models={tag:load_checked(r) for tag,r in chosen.items()};x,summary,v=evaluate_models(C,cfg,models,cfg['seeds']['development'],cfg['evaluation']['development_systems'],cfg['evaluation']['development_contrasts'],f'compute_ceiling{target}');compute[str(target)]=dict(checkpoints=chosen,results=summary,actual_times_are_not_identical=True)
 write(C/'evaluations/matched_compute.json',compute)
 for seed,models in models_by_seed.items():
  p=models[f'v4_s{seed}'];role_seed=cfg['seeds']['role_diagnostics']+(10 if seed==102 else 0);pred,truth,report=role_calibration(C,cfg,p,partition,adapter,role_seed,f'roles_seed{seed}');utility_diagnostics(C,cfg,p,role_seed,f'roles_seed{seed}',pred)
 # Training/evaluation contrast sensitivity on fresh independent development
 # cases, never used to select a checkpoint or alter the frozen training L.
 if 101 in models_by_seed:
  validator=CommonValidation(cfg['benchmark'],30,86404104,256,1000000,batch=16,chunk=16384,device='cuda:0');audit={}
  for tag in ['v4_s101','v1_context_s101']:
   arrays={L:validator(models_by_seed[101][tag],contrasts=L)[0] for L in [131072,262144,1000000]};audit[tag]={str(L):dict(EIG=stat(x),paired_change_vs_1M=stat(x-arrays[1000000])) for L,x in arrays.items()}
  write(C/'evaluations/contrast_sensitivity.json',audit)
 positive=len(paired)==2 and all(x['CI95'][0]>0 for x in paired.values());negative=bool(paired) and all(x['CI95'][1]<0 for x in paired.values());decision='REPLICATED_MATCHED_RECIPE_DEVELOPMENT_GAIN' if positive else 'NEGATIVE' if negative else 'INCONCLUSIVE'
 write(C/'decision.json',dict(decision=decision,paired_v4_v1=paired,supplemental_native_recipe=supplemental,primary_and_native_reference_gains_supported_in_this_pilot=bool(positive and supplemental.get('v4_minus_native_v1') and supplemental['v4_minus_native_v1']['CI95'][0]>0),no_STABLE_promotion=True,official_superiority_claim=False,specialization_not_inferred_from_EIG_gain=True,claim_limit='Conditional MC uncertainty for two training seeds and one frozen partition; native-recipe supplemental comparison separate'))
 from src.research.moe_v4_reports import final_reports
 final_reports(C,cfg,results,compute,decision)
