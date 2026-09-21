"""Single-step SIR time scan: production entropy reduction and separate sPCE check."""
import argparse,csv,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.special import logsumexp
from src.config import load_config
from src.objectives.eig.vector import VectorEIGEngine

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--systems',type=int,default=512);p.add_argument('--seed',type=int,default=1001);a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(2)
 cfg=load_config('configs/sir_ode_eig.yaml');bank=Path(cfg.raw['data']['dataset_dir']);meta=json.loads((bank/'meta/meta.json').read_text());t=np.load(bank/'meta/t.npy');idx=np.load(bank/'meta/design_indices.npy');times=t[idx]
 train=np.load(bank/'train/I.npy',mmap_mode='r');test=np.load(bank/'test/I.npy',mmap_mode='r');nfit=len(train)-max(1,len(train)//4);P=min(cfg.raw['prior']['mc_samples'],nfit);pick=np.sort(np.random.default_rng(cfg.raw['prior']['mc_support_seed']).choice(nfit,P,replace=False));centres=np.asarray(train[pick][:,idx],dtype=float)
 sigma=cfg.raw['observation']['noise_sigma'];ctx=SimpleNamespace(centres_support=centres.T[:,:,None],log_p0=np.full(P,-np.log(P)),sigma_y=sigma);engine=VectorEIGEngine(ctx,torch.device('cpu'));rng=np.random.default_rng(a.seed);truth_ids=rng.choice(len(test),a.systems,replace=False);truth=np.asarray(test[truth_ids][:,idx],dtype=float);noise=rng.normal(size=a.systems);h0=float(engine.entropy(engine.log_p0));entropy=[];spce=[];rows=[]
 for j,tt in enumerate(times):
  y=truth[:,j]+sigma*noise;obs=torch.tensor(y[:,None,None],dtype=torch.float32)
  w=engine.update(engine.log_p0,j,obs);gain=(h0-engine.entropy(w)).numpy();entropy.append(gain)
  ll=-.5*((y[:,None]-centres[:,j][None,:])/sigma)**2;lltrue=-.5*noise**2;contrast=lltrue-logsumexp(np.column_stack([lltrue,ll]),axis=1)+np.log(P+1);spce.append(contrast)
  rows.append({'time':float(tt),'mean_entropy_reduction':float(gain.mean()),'entropy_standard_error':float(gain.std(ddof=1)/np.sqrt(a.systems)),'mean_spce':float(contrast.mean()),'spce_standard_error':float(contrast.std(ddof=1)/np.sqrt(a.systems))})
 with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(out/'paired_scores.npz',times=times,entropy=np.asarray(entropy),spce=np.asarray(spce),truth_ids=truth_ids,support_ids=pick)
 (out/'settings.json').write_text(json.dumps({**vars(a),'sigma':sigma,'support_particles':P,'population':meta['population'],'i0':meta['i0'],'prior':meta['prior'],'bank_filter_min_mean_infected':meta['min_mean_infected'],'times':times.tolist(),'note':'Uses stored grid indices, actual t[index]; entropy matches production SIR engine; sPCE is a separate empirical-bank diagnostic, not the legacy reported metric.'},indent=2))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,key,label in zip(axes,['entropy','spce'],['Production SIR entropy reduction','Separate single-step sPCE diagnostic']):
  vals=np.asarray(entropy if key=='entropy' else spce);mean=vals.mean(axis=1);se=vals.std(axis=1,ddof=1)/np.sqrt(a.systems);ax.plot(times,mean);ax.fill_between(times,mean-1.96*se,mean+1.96*se,alpha=.2);ax.set(xlabel='Measurement time (model units)',ylabel='Information (nats)',title=label);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(out/'sir_time_eig.png',dpi=180)
 result={'wall_seconds':time.monotonic()-start,'best_entropy_row':max(rows,key=lambda x:x['mean_entropy_reduction']),'best_spce_row':max(rows,key=lambda x:x['mean_spce']),'end':rows[-1],'begin':rows[0]};(out/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
