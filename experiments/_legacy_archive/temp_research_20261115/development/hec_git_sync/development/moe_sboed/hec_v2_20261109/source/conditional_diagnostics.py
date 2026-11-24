"""Joint-prefix conditional native-return contributions;not exact conditional EIG."""
import time,numpy as np,torch
from native_common import O,put
from restore_v1 import restore
from normalized_reward import normalize_native_post

def grouped_ci(values,mask):
 from scipy.stats import t
 n=len(values);v=values.reshape(n,-1);m=mask.reshape(n,-1);counts=m.sum(1);sums=(v*m).sum(1);total=counts.sum();cases=int((counts>0).sum())
 if cases<16:return dict(status='NOT_ESTIMABLE',distinct_cases=cases,p_two_sided=1.)
 mu=float(sums.sum()/total);loo=(sums.sum()-sums)/(total-counts);se=float(np.sqrt((n-1)/n*np.square(loo-loo.mean()).sum()));critical=float(t.ppf(.975,cases-1));return dict(status='ESTIMATED',distinct_cases=cases,mean=mu,MCSE=se,CI95=[mu-critical*se,mu+critical*se],p_two_sided=float(2*t.sf(abs(mu/max(se,1e-300)),cases-1)))

def suffix(actor,theta,pds,pys,start,noise,k=None,donor=None):
 from evaluate_v2 import lib
 acts=list(pds[:,:start].unbind(1));obs=[pys[:,t] for t in range(start)]
 for t in range(start,30):
  x=lib.native_x(acts,obs,len(theta));action=actor(x,replace_expert=k if t==start else None,donor_slots=donor if t==start else None);d=(theta.reshape(-1,2,2)-action[:,None]).square().sum(-1);y=(.1+(d+1e-4).reciprocal().sum(-1)).log()+.5*noise[:,t];acts.append(action);obs.append(y[:,None])
 return torch.stack(acts,1),torch.stack(obs,1)

def run_conditional(actor,hist,deadline):
 from evaluate_v2 import holm
 E=O/'evaluation';rng=np.random.default_rng(202611099210);n=128;R=4;stages=[3,12,24];state=restore('cuda');a=state['algorithm'];normalize_native_post(a);noise=torch.tensor(rng.standard_normal((n,3,R,30)),device='cuda',dtype=torch.float32);idx=np.arange(256,384);truth=torch.tensor(hist['truth'][idx],device='cuda',dtype=torch.float32);prefixds=torch.tensor(hist['ds'][idx],device='cuda',dtype=torch.float32);prefixys=torch.tensor(hist['ys'][idx],device='cuda',dtype=torch.float32);base=np.zeros((n,3,R));effects=np.zeros((n,3,4,R));groups=np.zeros((n,3),int);donor_indices=rng.integers(256,size=(n,3));medians=[];calls=0
 with torch.no_grad(),torch.device('cuda'):
  for j,t in enumerate(stages):
   median=float(np.median(hist['uncertainty_trace'][:256,t]));medians.append(median);groups[:,j]=(hist['uncertainty_trace'][idx,t]>median)
   for offset in range(0,n,16):
    if time.monotonic()>deadline:raise TimeoutError('Conditionaldiagnosticfixeddeadline')
    sl=slice(offset,offset+16);m=len(truth[sl]);th=truth[sl,None].expand(m,R,4).reshape(-1,4);ds=prefixds[sl,None].expand(m,R,30,2).reshape(-1,30,2);ys=prefixys[sl,None].expand(m,R,30,1).reshape(-1,30,1);eps=noise[sl,j].reshape(-1,30);first=suffix(actor,th,ds,ys,t,eps);calls+=len(th)*(30-t);score=a.post_approx.log_poi_post(29,*first,th);base[sl,j]=score.reshape(m,R).cpu().numpy();don=torch.tensor(hist['slots'][donor_indices[sl,j],t],device='cuda')[:,None].expand(m,R,4,128).reshape(-1,4,128)
    if j==0 and offset==0:
     repeated=suffix(actor,th,ds,ys,t,eps);calls+=len(th)*(30-t);assert torch.equal(first[0],repeated[0]) and torch.equal(first[1],repeated[1]),'Identicalcontinuationresimulationconsistency'
    for k in range(4):
     changed=suffix(actor,th,ds,ys,t,eps,k,don);calls+=len(th)*(30-t);assert torch.equal(changed[0][:,:t],ds[:,:t]) and torch.equal(changed[1][:,:t],ys[:,:t]);value=a.post_approx.log_poi_post(29,*changed,th);effects[sl,j,k]=(score-value).reshape(m,R).cpu().numpy()
 tests={}
 for j,t in enumerate(stages):
  for group in [0,1]:
   for k in range(4):tests[f'stage{t}_signal{group}_expert{k+1}']=grouped_ci(effects[:,j,k].mean(-1),groups[:,j]==group)
 holm(tests,'Holm24_p');np.savez_compressed(E/'conditional_expert_native_raw.npz',effects=effects,baseline_terminal_scores=base,regimes=groups,truth=truth.cpu().numpy(),prefix_actions=prefixds.cpu().numpy(),prefix_observations=prefixys.cpu().numpy(),future_noise=noise.cpu().numpy(),donor_indices=donor_indices,stages=stages)
 put(E/'CONDITIONAL_COMPETENCE.json',dict(status='COMPLETE',cases=128,replicates=4,stages=stages,simulator_observations=calls,estimator='pairedexpectedremainingnormalizednativevariationalreturn difference;prefixtermcancels;NOTexactconditionalMI',conditioning='theta retainedfromsamejointgeneratedprefix;noindependentpriorrestart',groups='stage-specificcalibrationmedianofapproximatebeliefvariancetrace',calibration_medians=medians,tests=tests,canonical_identical_continuation_bitwise_consistency=True,changed_policy_observations_resimulated=True))
