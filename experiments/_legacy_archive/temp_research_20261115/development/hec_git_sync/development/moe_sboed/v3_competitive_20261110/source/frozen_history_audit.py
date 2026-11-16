"""Read-only HEC-v2 history features; never constructs or trains a v3 model."""
import os,sys,json,time,hashlib,subprocess,copy
from pathlib import Path
import numpy as np
import torch
B=Path.home()/'Documents/objective_power_grid_temp';R=B/'scratch_runs';OLD=R/'moe_hec_v2_icml_20261109';OUT=R/'moe_sboed_v3_competitive_20261110';E=OUT/'evidence'
os.environ['MOE_V2_RUN_ROOT']=str(OLD);sys.dont_write_bytecode=True;sys.path.insert(0,str(OLD/'source'))
from restore_v1 import restore,V1
from evaluate_v2 import actor_load
from soft_policy import prefix_inputs
from hec_policy import mean_signal,jacobian,COST
from native_common import original,sha

def write(name,obj):
 (E/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def main():
 start=time.monotonic();assert not (E/'FROZEN_HISTORY_AUDIT.json').exists();original.setup();lease=original.acquire();torch.manual_seed(202611103001)
 paths=[V1/'arms/candidate/runtime.pkl',V1/'arms/candidate/actor.pt',OLD/'arms/candidate/actor.pt',OLD/'ARTIFACT_SHA256.json',OLD/'EIG_COMPARISON.md',OLD/'EXPERT_COMPETENCY_ANALYSIS.md',OLD/'COMMANDER_ABLATIONS.md',OLD/'evaluation/EIG_RESULTS.json']
 for root,names in [('moe_competitive_action_routing_pilot_20261107',['METHOD_AND_GRADIENTS.md','NOVELTY_AND_NEXT_DECISION.md','source/native_competitive.py']),('moe_information_specialized_option_20261107',['MATHEMATICAL_OBJECTIVE.md','SPECIALIZATION_TRAINING_RULE.md','NEXT_RESEARCH_DECISION.md'])]:paths.extend(R/root/name for name in names)
 before={str(p):sha(p) for p in paths};write('PROTECTED_INPUTS_BEFORE.json',before)
 state=restore();a=state['algorithm'];model,_=actor_load('candidate');model.eval();rng=np.random.default_rng(202611103001);stages=[0,3,12,24,29];ids=rng.choice(len(a._runtime_buffer['ds_hist']),len(stages)*256,replace=False);summary={};records=[];COST.clear();peak0=torch.cuda.memory_allocated()
 with torch.no_grad():
  for j,t in enumerate(stages):
   ix=torch.tensor(ids[j*256:(j+1)*256],device='cuda');ds=a._runtime_buffer['ds_hist'][ix];ys=a._runtime_buffer['ys_hist'][ix];tt=torch.full((256,),t,device='cuda',dtype=torch.long);x=prefix_inputs(ds,ys,tt)
   dd=ds.clone();yy=ys.clone();dd[:,t:]=999;yy[:,t:]=-999;torch.testing.assert_close(x,prefix_inputs(dd,yy,tt),rtol=0,atol=0)
   w,mu,sd=model.belief.params(x);nodes,pw=model.belief.nodes(w,mu,sd);torch.testing.assert_close(pw.sum(-1),torch.ones(256,device='cuda'),rtol=1e-5,atol=1e-6)
   mean=(w[...,None]*mu).sum(1);var=((w[...,None]*(mu.square()+sd.square())).sum(1)-mean.square()).clamp_min(1e-6);coverage=var.sum(-1)
   action=model(x);assert bool((action.abs()<=4).all());pred=mean_signal(nodes,action[:,None])[:,0];predmean=(pw*pred).sum(-1);pv=(pw*(pred-predmean[:,None]).square()).sum(-1);discrimination=4*pv
   # E[(m(theta)-m(theta'))^2]/(2*sigma^2), sigma=.5.
   literal=(pw[:,:,None]*pw[:,None,:]*(pred[:,:,None]-pred[:,None,:]).square()/.5).sum((1,2));torch.testing.assert_close(discrimination,literal,rtol=1e-4,atol=1e-5)
   F=x.new_zeros(256,4,4)
   for q in range(t):
    J=jacobian(nodes,ds[:,q]);F+=4*torch.einsum('bn,bni,bnj->bij',pw,J,J)
   scale=var.sqrt();Fw=F*scale[:,:,None]*scale[:,None,:];ev=torch.linalg.eigvalsh(Fw.double());assert float(ev.min())>-1e-4
   weakness=-torch.log1p(ev[:,0].clamp_min(0)).float();scores=torch.stack([coverage,discrimination,weakness],-1).cpu().numpy();threshold=np.quantile(scores[:128],2/3,axis=0);groups=scores>threshold;weights=.1+groups.astype(float);weights/=weights.mean(0);vv=groups[128:];ess=(weights[128:].sum(0)**2/(weights[128:]**2).sum(0))
   summary[str(t)]=dict(score_order=['coverage_trace','expected_pair_KL_at_frozen_v2_action','weak_accumulated_whitened_Fisher'],calibration_threshold=threshold.tolist(),within_stage_score_SD=scores.std(0,ddof=1).tolist(),validation_upper_group_fraction=vv.mean(0).tolist(),validation_effective_sample_size=ess.tolist(),pairwise_group_Jaccard={(str(k+1)+'_'+str(l+1)):float((vv[:,k]&vv[:,l]).sum()/max(1,(vv[:,k]|vv[:,l]).sum())) for k in range(3) for l in range(k+1,3)},ties_at_calibration_threshold=(scores[:128]==threshold).mean(0).tolist(),zero_Fisher_is_prior_no_previous_designs=t==0)
   records.append(dict(stage=t,x=x.cpu().numpy(),scores=scores,threshold=threshold,groups=groups,replay_ids=ids[j*256:(j+1)*256]))
 # Analytic likelihood sensitivity versus an independently differentiated exact signal.
 with torch.enable_grad():
  th=torch.tensor([[.2,-.4,.7,1.1],[-.3,.6,1.,-.8]],device='cuda',dtype=torch.float64,requires_grad=True);ac=torch.tensor([[.1,.3],[.2,-.1]],device='cuda',dtype=torch.float64);value=mean_signal(th[:,None],ac[:,None])[:,0,0];grad=torch.autograd.grad(value.sum(),th)[0];torch.testing.assert_close(grad,jacobian(th[:,None],ac)[:,0],rtol=1e-11,atol=1e-11)
 np.savez_compressed(E/'frozen_history_features.npz',**{str(z['stage'])+'_'+k:v for z in records for k,v in z.items() if k!='stage'})
 after={str(p):sha(p) for p in paths};assert before==after;write('PROTECTED_INPUTS_AFTER.json',after)
 write('FROZEN_HISTORY_AUDIT.json',dict(status='COMPLETE_FROZEN_PREFLIGHT_NOT_COMPETENCE',seconds=time.monotonic()-start,stages=stages,cases_per_stage=256,calibration_cases_per_stage=128,validation_cases_per_stage=128,distinct_replay_episodes=len(ids),training_updates=0,preparation_updates=0,new_simulator_episodes=0,new_observation_simulations=0,likelihood_and_model_queries=dict(COST),GPU=torch.cuda.get_device_name(),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),protected_inputs_byte_identical=True,true_theta_not_read=True,future_prefix_mutation_check=True,analytic_Jacobian_autograd_check=True,pairwise_Gaussian_KL_identity_check=True,approximate_belief_source='Inherited SG frozen amortized GMM used by HEC-v2;not independently validated exact posterior',summary=summary,not_established=['expert design competence','predictable comparative remaining-horizon advantage','novel responsibility rule','all30stage support','v3 full native stochastic policy correctness']))
 print((E/'FROZEN_HISTORY_AUDIT.json').read_text())
if __name__=='__main__':main()
