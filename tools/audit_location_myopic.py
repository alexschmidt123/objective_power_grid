"""Bounded independent diagnostics; never changes source-run artifacts."""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp
import torch

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
args=parser.parse_args();run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str((run/'source').resolve()))
from src.research.benchmarks import LocationFinding
from src.research.location_posterior import LocationSMC,LocationMyopic
torch.set_num_threads(1)
cfg=json.loads((run/'run_config.json').read_text())
rows=json.loads((run/'eval/myopic_seed63777278.json').read_text())
report={};started=time.time()

def mean(theta,actions):
    sources=np.asarray(theta).reshape(-1,2,2)
    dist=((sources[:,None]-actions[None,:,None])**2).sum(-1)
    return np.log(.1+np.sum(1/(1e-4+dist),axis=-1))

def perm_mse(estimate,truth):
    return min(np.mean((estimate-truth)**2),np.mean((estimate.reshape(2,2)[::-1].ravel()-truth)**2))

# Independent NumPy observation/noise/MSE verification across all saved episodes.
noise_errors=[];mse_errors=[];distances=[];stage_distances=[]
for r in rows:
    theta=np.array(r['truth']);a=np.array(r['actions']);y=np.array(r['observations']).ravel()
    predicted=mean(theta,a)[0];eps=np.array(r['measurement_noise_standard']).ravel()
    noise_errors.append(float(np.max(np.abs(y-predicted-.5*eps))))
    mse_errors.append(abs(perm_mse(np.array(r['inference']['estimate']),theta)-r['inference']['mse']))
    d=np.sqrt(((a[:,None]-theta.reshape(1,2,2))**2).sum(-1)).min(-1)
    distances.append(d.min());stage_distances.append(d)
report['saved_observation_check']=dict(max_residual_error=max(noise_errors),max_mse_error=max(mse_errors),
    fraction_probe_within_001=float(np.mean(np.array(distances)<.01)),
    median_nearest_source_distance_by_stage=np.median(stage_distances,axis=0).tolist())
print('Saved observations and MSE independently checked',report['saved_observation_check'],flush=True)

# Recompute full-million-contrast score with separate NumPy implementation.
bank=np.load(run/'eval/contrast_bank_seed63777278.npy',mmap_mode='r');scores=[]
for case in [0,17,123,777]:
    r=rows[case];a=np.array(r['actions']);y=np.array(r['observations']).ravel()
    constant=math.log(.5*math.sqrt(2*math.pi))
    true_lp=float(np.sum(-.5*((y-mean(np.array(r['truth']),a)[0])/.5)**2-constant))
    total=-np.inf
    for start in range(0,len(bank),4096):
        mu=mean(bank[start:start+4096],a)
        lp=np.sum(-.5*((y[None]-mu)/.5)**2-constant,axis=1)
        total=np.logaddexp(total,logsumexp(lp))
    lower=true_lp-np.logaddexp(true_lp,total)+math.log(len(bank)+1)
    upper=true_lp-total+math.log(len(bank))
    scores.append(dict(case=case,lower=float(lower),upper=float(upper),
        lower_error=float(lower-r['spce']),upper_error=float(upper-r['nmc_upper_estimate'])))
report['independent_eig_reduction']=scores
print('Independent full-bank EIG',scores,flush=True)

# Reproduce cross-episode coupling: row zero history/RNG unchanged, row one altered.
torch.cuda.is_available=lambda:False
b=LocationFinding(cfg['benchmark']);smcs=[]
for second_y in [0.,20.]:
    s=LocationSMC(b,2,512,torch.Generator().manual_seed(192817),moves=4)
    s.update(torch.zeros(2,2,dtype=torch.float64),torch.tensor([[1.],[second_y]],dtype=torch.float64))
    smcs.append(s)
report['cross_episode_intervention']=dict(first_particle_max_change=float((smcs[0].theta[0]-smcs[1].theta[0]).abs().max()),
    first_means=[s.theta[0].mean(0).tolist() for s in smcs],
    tempering_steps=[s.diagnostics[0]['tempering_steps'] for s in smcs])
print('Cross-episode intervention',report['cross_episode_intervention'],flush=True)

# Truth-free multi-start numerical inference, using observations only.
fitted=[];rng=np.random.default_rng(8392751)
for case in range(16):
    r=rows[case];a=np.array(r['actions']);y=np.array(r['observations']).ravel()
    candidates=[]
    for init in rng.normal(size=(32,4)):
        fit=least_squares(lambda theta:(mean(theta,a)[0]-y)/.5,init,max_nfev=400,
                          ftol=1e-9,xtol=1e-9,gtol=1e-9)
        candidates.append(fit)
    best=min(candidates,key=lambda f:np.sum(f.fun**2))
    fitted.append(dict(case=case,estimate=best.x.tolist(),mse=perm_mse(best.x,np.array(r['truth'])),
        original_mse=r['inference']['mse'],mean_squared_standardized_residual=float(np.mean(best.fun**2))))
report['independent_observation_only_least_squares']=dict(cases=fitted,
    mean_mse=float(np.mean([r['mse'] for r in fitted])),
    original_subset_mean_mse=float(np.mean([r['original_mse'] for r in fitted])),
    note='Likelihood optimizer, not a posterior-mean estimator or convergence certificate; truth used only after fitting for scoring')
print('Independent observation-only fit',report['independent_observation_only_least_squares'],flush=True)
report['runtime_seconds']=time.time()-started
report['source_run']=str(run.resolve())
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print('AUDIT_OUTPUT',out,flush=True)
