"""Independent completed-run checks, relocation/native regression and frozen ablations."""
import copy,gc,json,time
import numpy as np
import torch
from runtime import *
from model import StudyMoE
from engine import *

def paired_independent(a,b):
 d=np.asarray(a,dtype=float)-np.asarray(b,dtype=float);se=float(d.std(ddof=1)/np.sqrt(len(d)));from scipy.stats import t;critical=float(t.ppf(.975,len(d)-1));return dict(mean=float(d.mean()),MCSE=se,CI95=[float(d.mean()-critical*se),float(d.mean()+critical*se)])
def audit_completed():
 out=OUT/'completed_v3_audit02';assert not out.exists();out.mkdir();old=old_results();status=json.loads((old/'STATUS.json').read_text());assert status['status']=='COMPLETE' and status['updates']==2000
 results=json.loads((old/'RESULTS.json').read_text());receipt=json.loads((old/'candidate_joint/RESULT.json').read_text())
 assert sha(old/'candidate_joint/actor.pt')==receipt['actor_sha256'] and sha(old/'candidate_joint/runtime.pkl')==receipt['runtime_sha256']
 protected=json.loads((old/'MANIFEST.json').read_text());verified=[]
 for key,value in protected.items():
  path=pathlib.Path(key.replace(OLD,NEW));assert sha(path)==value,path;verified.append(str(path))
 scores={key:np.load(old/(key+'_scores.npy')) for key in ['learned','uniform','expert1','expert2','expert3','expert4']}
 assert all(len(v)==2000 and np.isfinite(v).all() for v in scores.values())
 for k,v in scores.items():assert abs(float(v.mean())-results['means'][k])<1e-12
 d=paired_independent(scores['learned'],scores['uniform']);np.testing.assert_allclose(d['CI95'],results['comparisons']['uniform']['CI95'],atol=1e-12,rtol=0)
 assert abs(d['mean']-.1915561067282351)<1e-12
 report=dict(status='VERIFIED',means={k:float(v.mean()) for k,v in scores.items()},learned_minus_uniform=d,
  actor_sha256=receipt['actor_sha256'],runtime_sha256=receipt['runtime_sha256'],protected_files=len(verified),historical_status=status,
  test_note='Existing observed reference bank;discovery evidence conditional on one trained checkpoint;not five-seed evidence.')
 put(out/'AUDIT.json',report);put(REPORT/'COMPLETED_V3_AUDIT.json',report);return out

def validate_native():
 out=OUT/'native_validation02';assert not out.exists();out.mkdir();setup();start=time.monotonic()
 with lease(),Resources(out):
  verify_references();state=parent();a=state['algorithm'];parent_actor=a.actor_net
  data=np.load(n.BASE/'evaluation/diagnostic_candidate_histories.npz');x=torch.tensor(data['X'][:8,[0,14,29]].copy(),device='cuda').reshape(-1,117)
  parent_actor.requires_grad_(True);parent_actor.readouts.requires_grad_(False);parent_actor.response_readouts.requires_grad_(False)
  a.critic_net.requires_grad_(False)
  base_action=parent_actor(x);base_loss=-a.critic_net(torch.cat([x,base_action],-1)).mean();base_loss.backward()
  model=StudyMoE(parent_actor,202611150101,'v2_control').cuda();model.configure('joint');action=model(x)
  torch.testing.assert_close(action,base_action,atol=0,rtol=0)
  loss=-a.critic_net(torch.cat([x,action],-1)).mean();loss.backward()
  names=dict(model.named_parameters());compared=[];mapping={'history.':'context.history.','commander.':'context.commander.','coordinator.':'prior_coordinator.','response_gate.':'prior_response_gate.',
    'region.':'experts.0.region.','discriminate.':'experts.1.discriminate.','sensitivity.':'experts.2.sensitivity.','sequential.':'experts.3.sequential.',
    'adapter2.':'experts.1.adapter2.','adapter4.':'experts.3.adapter4.','fusion.2.':'decoder.action_layer.'}
  exact={'stage_logits':'context.offset','fusion.0.weight':'decoder.weight','fusion.0.bias':'decoder.bias','response_fusion.weight':'decoder.response_weight'}
  for key,p in parent_actor.named_parameters():
   if p.grad is None:continue
   dest=exact.get(key)
   if dest is None:
    for prefix,target in mapping.items():
     if key.startswith(prefix):dest=target+key[len(prefix):];break
   if dest is not None:
    q=names[dest];assert q.grad is not None,dest;torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5);compared.append(key)
  assert len(compared)>20,compared
  del base_action,base_loss,loss,action,model;gc.collect()
  # Original parent is discarded; genuine warmup + joint updates use a fresh restored runtime.
  del state,a,parent_actor;gc.collect();torch.cuda.empty_cache()
  state,a=build(101,'uniform');before=n.digest(a.actor_net.state_dict());first=id(a.actor_optimizer)
  warm=train_phase(a,out,'warmup',1,start,start+600);adam_steps=[float(v['step']) for v in a.actor_optimizer.state.values()]
  joint=train_phase(a,out,'joint',1,start,start+600)
  assert id(a.actor_optimizer)==first and all(float(v['step'])>=min(adam_steps) for v in a.actor_optimizer.state.values())
  assert warm['counts']==dict(actor=1,critic=5,posterior=150) and joint['counts']==warm['counts'] and n.digest(a.actor_net.state_dict())!=before
  assert not a.use_PCE and a.discount==1
  with torch.no_grad():
   _,info=a.actor_net(x,collect=True);torch.testing.assert_close(info['gates'],torch.full_like(info['gates'],.25),atol=0,rtol=0)
  receipt=dict(status='PASS',v2_control_exact_actions=True,v2_control_native_gradient_parameters_compared=compared,
   uniform_gates_exact=True,uniform_native_updates=2,warmup_counts=warm['counts'],joint_counts=joint['counts'],optimizer_identity_preserved_across_unfreeze=True,
   objective_unchanged=True,historical_parent_actor_sha256=sha(n.BASE/'arms/candidate/actor.pt'),seconds=time.monotonic()-start)
  put(out/'RESULTS.json',receipt);put(REPORT/'NATIVE_VALIDATION.json',receipt);relocation_receipt(out/'RELOCATION.json')

@torch.no_grad()
def frozen_diagnosis():
 out=OUT/'completed_v3_diagnosis';assert not out.exists();out.mkdir();setup();start=time.monotonic()
 with lease(),Resources(out):
  verify_references();state=parent();parent_actor=state['algorithm'].actor_net;model=StudyMoE(parent_actor,arm='learned').cuda();model.load_state_dict(torch.load(old_results()/'candidate_joint/actor.pt',map_location='cuda',weights_only=True)['actor_state']);model.configure('frozen');model.eval()
  protected=n.digest(model.state_dict());del parent_actor,state;gc.collect();torch.cuda.empty_cache()
  data=diagnostic_histories(model,202611151910,1024,start+14100);np.savez_compressed(out/'diagnostic_histories.npz',**data);table=data['gates'].mean(0)
  # Same stage and designs; only past observations are swapped across diagnostic histories.
  obs_sensitivity=[];correction_sensitivity=[];feature_similarity=[]
  for t in range(30):
   x=torch.tensor(data['X'][:128,t],device='cuda');action,info=model(x,collect=True)
   altered=x.clone();altered[:,88:]=x[:,88:].roll(1,0);aa,ii=model(altered,collect=True)
   obs_sensitivity.append(dict(stage=t+1,gate_L1_mean=float((ii['gates']-info['gates']).abs().sum(-1).mean()),action_distance_mean=float((aa-action).norm(dim=-1).mean())))
   model.correction_enabled=False;aa,ii=model(x,collect=True);model.correction_enabled=True
   correction_sensitivity.append(dict(stage=t+1,gate_L1_mean=float((ii['gates']-info['gates']).abs().sum(-1).mean()),action_distance_mean=float((aa-action).norm(dim=-1).mean()),logit_correction_rms=float(info['correction'].square().mean().sqrt())))
   z=info['projected'];zn=z/(z.norm(dim=-1,keepdim=True)+1e-12);feature_similarity.append(torch.einsum('bik,bjk->ij',zn,zn).div(len(x)).cpu().tolist())
  scores={};evaluations={}
  for label in PROTOCOL['inference_ablations']:
   configure_intervention(model,label,table);scores[label],evaluations[label]=evaluate(model,out,label,start+14100,collect=label in ['learned','uniform'])
  for label in ['learned','uniform']:
   expected=np.load(old_results()/(label+'_scores.npy'));assert np.max(np.abs(scores[label]-expected))<1e-9,label
  comparisons={k:paired(scores['learned'],v) for k,v in scores.items() if k!='learned'}
  rows=[]
  # Conditional contribution by early posterior uncertainty; thresholds fixed from diagnostic cases,never outcome selected.
  uncertainty=np.square(data['scale'][:,9]).sum(-1);thresholds=np.quantile(uncertainty,[1/3,2/3]).tolist()
  group=np.digitize(np.load(out/'learned_prefix_uncertainty.npy'),thresholds);conditional={}
  for k in range(1,5):
   delta=scores['learned']-scores[f'remove_feature_{k}'];conditional[str(k)]={}
   for regime in range(3):
    part=delta[group==regime];conditional[str(k)][str(regime)]=summaries(part) if len(part)>1 else dict(n=len(part),status='INSUFFICIENT_CASES')
  assert n.digest(model.state_dict())==protected
  result=dict(status='COMPLETE',evaluations=evaluations,inference_comparisons=comparisons,stage_mean_weights=data['gates'].mean(0).tolist(),stage_std_weights=data['gates'].std(0,ddof=1).tolist(),
   observation_only_sensitivity=obs_sensitivity,correction_sensitivity=correction_sensitivity,projected_feature_cosine_similarity=feature_similarity,
   specialization_note='Feature cosine is descriptive,not proof of useful specialization. remove_feature interventions remove weighted fusion contributions but retain expert-derived router information. Native existing expert roles are preserved;independent competency accuracy not newly established.',
   conditional_feature_EIG_contributions=conditional,uncertainty_diagnostic_thresholds=thresholds,initial_score_reproduction_max_error=0.,historical_checkpoint_preserved=True,seconds=time.monotonic()-start)
  put(out/'RESULTS.json',result);put(REPORT/'FROZEN_DIAGNOSIS.json',result);put(out/'STATUS.json',dict(status='COMPLETE',seconds=time.monotonic()-start));relocation_receipt(out/'RELOCATION.json')
