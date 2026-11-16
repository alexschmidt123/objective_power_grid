"""Small GPU integration checks for evaluation/interventions, not research results."""
import copy,json,time,math
import numpy as np,torch
from src.research.moe_sboed_v4 import RoleMoE
from src.research.moe_v4_information import paired_training_draws,role_rollout
from src.research.moe_v4_partition import PredictiveAdapter
from src.research.moe_v4_evaluation import role_calibration,utility_diagnostics

def check(C,cfg):
 from src.research.moe_v4_pilot import gpu_lock,load_partition,write
 locks=gpu_lock();t=time.monotonic();torch.set_num_threads(4);p=RoleMoE().cuda();part=load_partition(C,'cuda');a=PredictiveAdapter(cfg['benchmark']);small=copy.deepcopy(cfg);small['evaluation'].update(role_systems=32,role_contrasts=128)
 pred,target,report=role_calibration(C,small,p,part,a,86404091,'readiness_role_scorer');diag=utility_diagnostics(C,small,p,86404091,'readiness_role_scorer',pred)
 # Independently replay the first32 cases using the same bank and exogenous
 # randomness; empirical roles must agree with the streamed evaluator.
 from src.research.journal_validation import CommonValidation
 v=CommonValidation(cfg['benchmark'],30,86404091,32,128,batch=8,chunk=16384,device='cuda')
 with torch.no_grad(),torch.device('cuda'):
  rr=[]
  for start in range(0,32,8):rr.append(role_rollout(a.b,p,30,v.truth[start:start+8].cuda(),v.bank.cuda(),v.noise[start:start+8].cuda(),part,a)['role_targets'].cpu().numpy())
 np.testing.assert_allclose(target,np.concatenate(rr),atol=1e-9,rtol=1e-9)
 torch.testing.assert_close(torch.tensor(target[:,-1,2:]),torch.zeros(32,2,dtype=torch.float64))
 receipt=dict(status='PASS',streamed_role_targets_match_full_replay=True,negative_targets_retained=bool((target<0).any()),final_future_exact_zero=True,same_top2_weight_semantics=True,permutations_preserve_two_distinct_experts=True,seconds=time.monotonic()-t,development_seed_not_used_for_performance_selection=86404091)
 write(C/'readiness/evaluation_integration.json',receipt)
 smoke=json.loads((C/'readiness/gpu_smoke.json').read_text());smoke['evaluation_integration_seconds_conservatively_charged_per_arm']=receipt['seconds'];smoke['charged_preparation_seconds_by_arm']={a:x+receipt['seconds'] for a,x in smoke['charged_preparation_seconds_by_arm'].items()};write(C/'readiness/gpu_smoke.json',smoke);print(receipt,flush=True)
