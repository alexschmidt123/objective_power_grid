"""CPU analysis only. Reuses unchanged v3 gates; no policy optimization."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from src.research.fh_moe_v3_context import *
from src.research.deepseek_sboed_experiment import write
STAGES=[1,10,20,29]

def ci(v):return [float(torch.quantile(v,.025)),float(torch.quantile(v,.975))]
def kappa(a,b):
 agree=(a==b).double().mean();p=sum((a==k).double().mean()*(b==k).double().mean() for k in [0,1]);return float((agree-p)/(1-p).clamp_min(1e-30))
def distribution(g):
 n=g.norm(dim=-1);unit=normalized(g);cov=torch.cov(g.T);m=g.mean(0)
 return dict(norm_quantiles=dict(zip(['min','p01','p05','p25','median','p75','p95','p99','max'],n.quantile(torch.tensor([0,.01,.05,.25,.5,.75,.95,.99,1.],dtype=torch.double)).tolist())),
  mean_raw_norm=float(n.mean()),raw_mean_norm=float(m.norm()),raw_cancellation_ratio=float(m.norm()/n.mean().clamp_min(1e-30)),
  unit_norm_range=[float(unit.norm(dim=-1).min()),float(unit.norm(dim=-1).max())],unit_mean_norm=float(unit.mean(0).norm()),
  covariance_trace=float(cov.trace()),covariance=cov.tolist(),covariance_eigenvalues=torch.linalg.eigvalsh(cov).tolist(),
  fraction_norm_le_1e12=float((n<=1e-12).double().mean()),
  opposing_unit_fraction=float(((unit@unit.mean(0))<0).double().mean()))

def bootstrap_context(g,seed,reps=200):
 rng=torch.Generator().manual_seed(seed);values=[]
 for _ in range(reps):
  a=g[torch.randint(len(g),(len(g),),generator=rng)];m=a.mean(0);v=a.var(0,unbiased=True).sum();values.append(torch.stack([m.norm(),m.norm()/(v/len(a)).sqrt().clamp_min(1e-30),v]))
 a=torch.stack(values)
 return dict(coherence_CI95=ci(a[:,0]),SNR_CI95=ci(a[:,1]),covariance_trace_CI95=ci(a[:,2]),resamples=reps,scope='fixed trained grouping and sampled context; does not include refitting uncertainty')

def saved_audit(v3,C):
 start=time.perf_counter();torch.set_num_threads(2);splits=[torch.load(v3/f'gradient_dataset_split{i}.pt',map_location='cpu',weights_only=False) for i in range(3)];out={}
 for t in STAGES:
  metric,bank=discovery_metrics(*[(s[t][0],s[t][1]) for s in splits]);x,g,raw=splits[2][t]
  label=responsibilities(bank['fit'],bank['centers'],x).argmax(-1);fit2=fit_conditional(splits[1][t][0],splits[1][t][1]);label2=responsibilities(fit2,bank['centers'],x).argmax(-1)
  n=raw.norm(dim=-1);cut=n.quantile(.05);trim=n>cut
  rawfit=fit_conditional(splits[0][t][0],splits[0][t][2]);rawdiff=(raw-predict(rawfit,x)).square().sum(-1)-(raw-splits[0][t][2].mean(0)).square().sum(-1)
  contexts={}
  for k in [0,1]:
   a=raw[label==k];u=g[label==k];m=u.mean(0)
   contexts[k]=dict(raw=distribution(a),unit=distribution(u),bootstrap=bootstrap_context(u,98266001+t*13+k),trimmed_unit_coherence=float(g[(label==k)&trim].mean(0).norm()),mean_pairwise_unit_cosine=float((len(u)*m.square().sum()-1)/(len(u)-1)))
  rng=torch.Generator().manual_seed(98266101+t);ks=[]
  for _ in range(200):
   ids=torch.randint(len(g),(len(g),),generator=rng);ks.append(kappa(label[ids],label2[ids]))
  out[t]=dict(historical_metrics=metric,all_raw=distribution(raw),contexts=contexts,
    assignment_kappa_CI95=ci(torch.tensor(ks)),raw_gradient_prediction_vs_constant=confidence(rawdiff),
    lower5percent_norm_cutoff=float(cut),nearzero_exclusion_description='descriptive sensitivity only; original gate unchanged',
    independent_split_unit_mean_cosines=[[float(normalized(splits[i][t][1].mean(0))@normalized(splits[j][t][1].mean(0))) for j in range(3)] for i in range(3)])
 write(C/'saved_signal_audit.json',dict(stages=out,CPU_seconds=time.perf_counter()-start,uncertainty='historical MSE mean +/-1.96SE; fixed-label bootstrap200 for context coherence,SNR,covariance trace; independent-fit kappa bootstrap200. All exploratory, not simultaneous CIs. Norm quantiles/covariance matrices are empirical population-of-saved-cases descriptions, not population parameters.'))
 print('SAVED_AUDIT_COMPLETE',time.perf_counter()-start)

def estimator_sets(d):
 c,p,n=d['C'],d['D'],d['negative_C']
 return dict(single_iid_L2000=c[:,0],mean4_iid_L2000=c.mean(1),mean4_iid_pairs_L2000=((c+p)/2).mean(1),mean4_antithetic_pairs_L2000=((c+n)/2).mean(1))

def bootstrap_ratio(num,den,seed,reps=1000):
 rng=torch.Generator().manual_seed(seed);vals=[]
 for _ in range(reps):
  ids=torch.randint(len(num),(len(num),),generator=rng);vals.append(num[ids].mean(0)/den[ids].mean(0).clamp_min(1e-30))
 return torch.stack(vals)

def new_audit(C):
 start=time.perf_counter();torch.set_num_threads(2);splits=[torch.load(C/f'new_split{i}.pt',map_location='cpu',weights_only=False) for i in range(3)];sets=[estimator_sets(d) for d in splits];gates={}
 for name in sets[0]:
  stages={}
  for j,t in enumerate(STAGES):
   r,_=discovery_metrics(*[(d['x'][:,j],normalized(s[name][:,j])) for d,s in zip(splits,sets)]);stages[t]=r
  gates[name]=stages
 c=torch.cat([d['C'] for d in splits]);p=torch.cat([d['D'] for d in splits]);n=torch.cat([d['negative_C'] for d in splits]);R=c.shape[1]
 basevar=c.var(1,unbiased=True).sum(-1);indpair=(c+p)/2;antipair=(c+n)/2
 iidvar=indpair.var(1,unbiased=True).sum(-1);antivar=antipair.var(1,unbiased=True).sum(-1)
 boot=bootstrap_ratio(antivar,iidvar,98266001);variance={}
 for j,t in enumerate(STAGES):
  within=basevar[:,j].mean();between=c.mean(1)[:,j].var(0,unbiased=True).sum()-within/R
  # Marginal symmetrized estimator means must agree in expectation. These are
  # empirical paired errors with CI, not a proof of absence of finite-sample bias.
  difference=(antipair-indpair).mean(1)[:,j]
  variance[t]=dict(single_bank_within_trajectory_variance=float(within),corrected_between_trajectory_variance=float(between),contrast_fraction_of_raw_variance=float(within/(within+between)),
   pair_variance_ratio=float(antivar[:,j].mean()/iidvar[:,j].mean()),pair_variance_ratio_CI95=[float(boot[:,j].quantile(.025)),float(boot[:,j].quantile(.975))],
   antithetic_independent_mean_difference_norm=float(difference.mean(0).norm()),componentwise_mean_difference_CI95=[confidence(difference[:,k]) for k in range(16)],
   minimum_raw_norm=float(c[:,:,j].norm(dim=-1).min()))
 sensitivity=torch.load(C/'contrast_sensitivity.pt',map_location='cpu',weights_only=False);small=sensitivity['small'];large=sensitivity['large'];contrast={}
 for j,t in enumerate(STAGES):
  diff=(large-small).mean(0)[:,j];contrast[t]=dict(L2000_variance=float(small[:,:,j].var(0,unbiased=True).sum(-1).mean()),L8000_variance=float(large[:,:,j].var(0,unbiased=True).sum(-1).mean()),finite_target_shift_norm=float(diff.mean(0).norm()),paired_difference_component_CI95=[confidence(diff[:,k]) for k in range(16)],mean_per_case_direction_cosine=float((normalized(small.mean(0)[:,j])*normalized(large.mean(0)[:,j])).sum(-1).mean()))
 obs=torch.load(C/'observation_contrast.pt',map_location='cpu',weights_only=False);observation={}
 for j,t in enumerate(STAGES):
  a=obs[:,:,:,j];within=a.var(2,unbiased=True).sum(-1).mean();ov=a.mean(2).var(1,unbiased=True).sum(-1).mean()-within/4
  tv=a.mean((1,2)).var(0,unbiased=True).sum()-ov/4-within/16
  observation[t]=dict(contrast_variance=float(within),noise_path_variance_corrected=float(ov),truth_variance_corrected=float(tv),interpretation='joint-law variance conditional on generating truth; histories change with noise; not a fixed-history posterior diagnostic')
 refs=torch.load(C/'reference_sensitivity.pt',map_location='cpu',weights_only=False);reference={}
 for name,d in refs.items():
  row={}
  for j,t in enumerate(STAGES):
   m=refs['shared600'];row[t]=dict(unit_mean_coherence=float(normalized(d['raw'][:,j]).mean(0).norm()),raw_direction_cosine_to_shared600=float((normalized(d['raw'][:,j])*normalized(m['raw'][:,j])).sum(-1).mean()),action_adjoint_cosine_to_shared600=float((normalized(d['action'][:,j])*normalized(m['action'][:,j])).sum(-1).mean()))
  reference[name]=row
 write(C/'variance_results.json',dict(gates=gates,variance=variance,contrast_sensitivity=contrast,observation_decomposition=observation,reference_sensitivity=reference,CPU_seconds=time.perf_counter()-start,raw_target='All L2000 averaged raw estimators preserve original finite-iid-contrast expected gradient; normalized label distributions differ. L8000 changes target.'))
 print('NEW_AUDIT_COMPLETE',time.perf_counter()-start)

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--v3-campaign',required=True);p.add_argument('--phase',choices=['saved','new'],required=True);a=p.parse_args();C=Path(a.campaign)
 if a.phase=='saved':saved_audit(Path(a.v3_campaign),C)
 else:new_audit(C)
if __name__=='__main__':main()
