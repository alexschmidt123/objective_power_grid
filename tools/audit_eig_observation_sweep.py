"""Single-step paired EIG duration curves; reuse physical simulations across sigma."""
import argparse,csv,json,time,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from src.config import load_config
from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver
from src.objectives.eig.continuous_eig import OnlineEIG
from tools.audit_single_step_eig import MaxRocofObserver

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--sigmas',default='0.005,0.01,0.1');p.add_argument('--n-obs',default='0,1,5,10');p.add_argument('--systems',type=int,default=512);p.add_argument('--contrasts',type=int,default=1024);p.add_argument('--seed',type=int,default=1001);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();cfg=load_config('configs/ieee9_eig.yaml');sigmas=list(map(float,a.sigmas.split(',')));nobs=list(map(int,a.n_obs.split(',')));durations=np.round(np.linspace(.2,3.,29),10);levels=sorted(set([min(128,a.contrasts),min(512,a.contrasts),a.contrasts]));scores=np.empty((len(nobs),len(sigmas),len(levels),len(durations),a.systems));means_by_duration={};rows=[]
 sources=[Path(__file__),Path('tools/audit_single_step_eig.py'),Path('src/objectives/eig/continuous_eig.py'),Path('src/domains/swing/continuous_cuda.py'),Path('src/domains/swing/continuous.py'),Path('src/domains/swing/cuda.py'),Path('configs/ieee9_eig.yaml')]
 for f in sources:
  dst=out/'source'/f.name;dst.parent.mkdir(exist_ok=True);shutil.copy2(f,dst)
 settings={**vars(a),'sigmas':sigmas,'n_obs':nobs,'durations':durations.tolist(),'contrast_levels':levels,'amplitude':.05,'window':3.,'injection_bus':1,'physics':cfg.raw,'observation_specs':{},'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},'note':'T=1, equilibrium initial state. Same model draws across all cells. Noise shared across sigma/duration within each observation type; different dimensions are distinct observation experiments. N_obs=1 follows production first-step sampling.'}
 for i,N in enumerate(nobs):
  kw=dict(duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.)
  o=MaxRocofObserver(cfg,**kw) if N==0 else CudaContinuousSwingObserver(cfg,n_obs=N,**kw)
  settings['observation_specs'][str(N)]={'kind':'max_absolute_rocof' if N==0 else 'sampled_frequency','times':o.times.tolist(),'sigma_units':'Hz/s' if N==0 else 'Hz','rocof_sample_dt':getattr(o,'rocof_sample_dt',None)};(out/'settings.json').write_text(json.dumps(settings,indent=2))
  e=OnlineEIG(cfg,o,horizon=1,sigma=sigmas[0],contrasts=a.contrasts)
  for j,d in enumerate(durations):
   for offset in range(0,a.systems,a.batch_size):
    count=min(a.batch_size,a.systems-offset);rng=np.random.default_rng(np.random.SeedSequence([a.seed,offset]));theta,state=e.sample(rng,count);z=rng.normal(size=(count,o.n_obs));pred=o.propagate(theta.reshape(-1,6),state.reshape(-1,6),np.full(count*(a.contrasts+1),d));mu=pred.observations.reshape(count,a.contrasts+1,o.n_obs)
    for k,sigma in enumerate(sigmas):
     y=mu[:,0]+sigma*z;ll=-.5*np.sum(((y[:,None]-mu)/sigma)**2,axis=-1)
     for h,L in enumerate(levels):scores[i,k,h,j,offset:offset+count]=ll[:,0]-logsumexp(ll[:,:L+1],axis=1)+np.log(L+1)
   print(f'N_obs={N} duration={d:.1f} means='+','.join(f'{scores[i,k,-1,j].mean():.6f}' for k in range(len(sigmas)))+f' elapsed={time.monotonic()-start:.1f}s',flush=True)
  np.savez_compressed(out/f'Nobs{N}_paired_scores.npz',scores=scores[i],durations=durations,sigmas=sigmas,contrasts=levels)
  for k,sigma in enumerate(sigmas):
   for h,L in enumerate(levels):
    for j,d in enumerate(durations):
     x=scores[i,k,h,j];rows.append({'N_obs':N,'sigma':sigma,'contrasts':L,'duration_s':d,'mean_spce':x.mean(),'standard_error':x.std(ddof=1)/np.sqrt(a.systems)})
  with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(11,8));best=[]
 for i,(N,ax) in enumerate(zip(nobs,axes.flat)):
  for k,sigma in enumerate(sigmas):
   x=scores[i,k,-1];mean=x.mean(axis=1);se=x.std(axis=1,ddof=1)/np.sqrt(a.systems);ax.plot(durations,mean,label=f'sigma={sigma}');ax.fill_between(durations,mean-1.96*se,mean+1.96*se,alpha=.10)
   peak=int(mean.argmax());best.append({'N_obs':N,'sigma':sigma,'peak_duration':float(durations[peak]),'peak_spce':float(mean[peak]),'start_spce':float(mean[0]),'end_spce':float(mean[-1]),'mean_increasing':bool(np.all(np.diff(mean)>=0)),'mean_decreasing':bool(np.all(np.diff(mean)<=0))})
  label='Max |RoCoF|; sigma in Hz/s' if N==0 else f'N_obs={N} frequency; sigma in Hz'
  if N==1:label+='\nSample at 0.0015625 s'
  ax.set(title=label,xlabel='Hann duration (s)',ylabel='Single-step sPCE (nats)');ax.grid(alpha=.2);ax.legend()
 fig.suptitle(f'IEEE9, A=0.05 pu, W=3 s; {a.systems} systems, L={a.contrasts}');fig.tight_layout();fig.savefig(out/'observation_sweep.png',dpi=180);plt.close(fig)
 result={'wall_seconds':time.monotonic()-start,'completed_cells':len(nobs)*len(sigmas),'best_by_cell':best,'note':'Exploratory paired Monte Carlo curves; apparent peaks near zero are not established useful optima.'};(out/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
