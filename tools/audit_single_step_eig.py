"""Paired continuous-duration single-step EIG diagnostic using the production model."""
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from src.config import load_config
from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver
from src.objectives.eig.continuous_eig import OnlineEIG

from src.domains.swing.continuous_rocof import MaxRocofObserver

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--config',default='configs/ieee9_eig.yaml');p.add_argument('--systems',type=int,default=512);p.add_argument('--contrasts',type=int,default=1024);p.add_argument('--seed',type=int,default=1001);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--window',type=float,default=3.);p.add_argument('--amplitude',type=float,default=.05);p.add_argument('--sigma',type=float,default=.005);p.add_argument('--N-obs',type=int,default=0);p.add_argument('--duration-step',type=float,default=.1)
 a=p.parse_args()
 if not np.isfinite(a.duration_step) or a.duration_step<=0 or a.duration_step>2.8:p.error('duration-step must be in (0,2.8]')
 intervals=int(round(2.8/a.duration_step))
 if not np.isclose(intervals*a.duration_step,2.8,rtol=0,atol=1e-10):p.error('duration-step must divide the 0.2-to-3.0 interval exactly')
 out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 cfg=load_config(a.config)
 kwargs=dict(duration_bounds=(.2,3.),injection_bus=1,amplitude=a.amplitude,window=a.window)
 o=MaxRocofObserver(cfg,**kwargs) if a.N_obs==0 else CudaContinuousSwingObserver(cfg,n_obs=a.N_obs,**kwargs)
 if a.N_obs==0:
  # Independent compression of a sampled trajectory verifies peak extraction.
  ref=CudaContinuousSwingObserver(cfg,n_obs=int(round(a.window/o.rocof_sample_dt)),**kwargs);ref.times=np.arange(1,ref.n_obs+1)*o.rocof_sample_dt
  theta=np.tile(np.r_[(np.array(cfg.swing['M_lower_nodes'])+cfg.swing['M_upper_nodes'])/2,(np.array(cfg.swing['K_lower_nodes'])+cfg.swing['K_upper_nodes'])/2],(3,1));state=o.initial_state(3);ds=np.array([.2,1.5,3.])
  actual=o.propagate(theta,state,ds);trace=ref.propagate(theta,state,ds);f0=state[:,o.N+o.sim.observation_bus,None]/(2*np.pi);expected=np.max(np.abs(np.diff(np.c_[f0,trace.observations],axis=1)/o.rocof_sample_dt),axis=1)
  np.testing.assert_allclose(actual.observations[:,0],expected,atol=1e-10);print('ROCOF extraction matches sampled frequency trajectory: PASS',flush=True)
 e=OnlineEIG(cfg,o,horizon=1,sigma=a.sigma,contrasts=a.contrasts)
 durations=np.round(np.linspace(.2,3.,intervals+1),10);levels=sorted(set([min(128,a.contrasts),min(512,a.contrasts),a.contrasts]));scores=np.zeros((len(levels),len(durations),a.systems));rows=[]
 (out/'settings.json').write_text(json.dumps({**vars(a),'durations':durations.tolist(),'contrast_levels':levels,'observation_times':o.times.tolist(),'observation_kind':'max_absolute_rocof' if a.N_obs==0 else 'sampled_frequency','noise_units':'Hz/s' if a.N_obs==0 else 'Hz','rocof_sample_dt':getattr(o,'rocof_sample_dt',None),'physics':cfg.raw,'note':'T=1 from equilibrium, paired models/contrasts/noise across durations; no policy training','source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('src/objectives/eig/continuous_eig.py'),Path('src/domains/swing/continuous.py'),Path('src/domains/swing/continuous_cuda.py'),Path('src/domains/swing/continuous_rocof.py')]}},indent=2))
 for j,d in enumerate(durations):
  for offset in range(0,a.systems,a.batch_size):
   n=min(a.batch_size,a.systems-offset);rng=np.random.default_rng(np.random.SeedSequence([a.seed,offset]));initial=e.sample(rng,n);noise=rng.normal(size=(n,1,o.n_obs))
   r=e.rollout(None,rng,n,stochastic=False,initial=initial,noise=noise,selector=lambda stage,actions,obs:np.full(n,d));ll=r['log_likelihood']
   for k,L in enumerate(levels):scores[k,j,offset:offset+n]=ll[:,0]-logsumexp(ll[:,:L+1],axis=1)+np.log(L+1)
  for k,L in enumerate(levels):
   x=scores[k,j];delta=x-scores[k,j-1] if j else None
   rows.append({'duration_s':float(d),'contrasts':L,'mean_spce':float(x.mean()),'standard_error':float(x.std(ddof=1)/np.sqrt(a.systems)),'change_from_previous':None if delta is None else float(delta.mean()),'paired_change_se':None if delta is None else float(delta.std(ddof=1)/np.sqrt(a.systems))})
  print(f'duration={d:.2f} spce={scores[-1,j].mean():.6f} elapsed={time.monotonic()-start:.1f}s',flush=True)
  np.savez_compressed(out/'paired_scores.npz',durations=durations[:j+1],contrast_levels=levels,scores=scores[:,:j+1])
  with (out/'summary.partial.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (out/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4))
 for k,L in enumerate(levels):
  mean=scores[k].mean(axis=1);se=scores[k].std(axis=1,ddof=1)/np.sqrt(a.systems);ax.plot(durations,mean,label=f'L={L}');ax.fill_between(durations,mean-1.96*se,mean+1.96*se,alpha=.10)
 ax.set(xlabel='Hann duration (s)',ylabel='Single-step sPCE (nats)',title=f'IEEE9: {"max RoCoF" if a.N_obs==0 else "frequency"}, A={a.amplitude}, W={a.window}s, sigma={a.sigma}');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'duration_eig.png',dpi=180);plt.close(fig)
 result={'wall_seconds':time.monotonic()-start,'systems':a.systems,'best_duration':float(durations[np.argmax(scores[-1].mean(axis=1))]),'mean_monotone_on_grid':bool(np.all(np.diff(scores[-1].mean(axis=1))>=0)),'note':'Pointwise Monte Carlo intervals; not simultaneous confidence bands or proof of monotonicity.'};(out/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
