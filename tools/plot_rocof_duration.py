"""Single-probe observation-response diagnostic using the production observer."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from src.config import load_config
from src.hardware import hardware_info
from src.domains.swing.continuous_rocof import MaxRocofObserver
from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/ieee9_eig.yaml')
    p.add_argument('--seed',type=int,default=101)
    a=p.parse_args(); start=time.monotonic()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=load_config(a.config)
    kw=dict(duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.)
    obs=MaxRocofObserver(cfg,**kw)
    lower=np.r_[cfg.swing['M_lower_nodes'],cfg.swing['K_lower_nodes']]
    upper=np.r_[cfg.swing['M_upper_nodes'],cfg.swing['K_upper_nodes']]
    theta=np.vstack([(lower+upper)/2,np.random.default_rng(a.seed).uniform(lower,upper,(128,6))])
    durations=np.linspace(.2,3.,281)
    y=np.empty((len(durations),len(theta)))
    for j,d in enumerate(durations):
        y[j]=obs.propagate(theta,obs.initial_state(len(theta)),np.full(len(theta),d)).observations[:,0]
    assert np.isfinite(y).all()
    # Independent finite-difference compression of the sampled frequency trace.
    ref=CudaContinuousSwingObserver(cfg,n_obs=120,**kw)
    ref.times=np.arange(1,121)*obs.rocof_sample_dt
    ix=np.array([0,80,180,280]);params=np.repeat(theta[:1],len(ix),axis=0)
    state=ref.initial_state(len(ix));trace=ref.propagate(params,state,durations[ix]).observations
    f0=state[:,obs.N+obs.sim.observation_bus,None]/(2*np.pi)
    reconstructed=np.abs(np.diff(np.c_[f0,trace],axis=1)/obs.rocof_sample_dt).max(axis=1)
    np.testing.assert_allclose(reconstructed,y[ix,0],rtol=1e-10,atol=1e-10)
    q=np.quantile(y[:,1:],[.05,.5,.95],axis=1)
    with (out/'duration_rocof.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['duration_s','prior_midpoint_hz_s','prior_mean_hz_s','prior_p05_hz_s','prior_median_hz_s','prior_p95_hz_s'])
        w.writerows(zip(durations,y[:,0],y[:,1:].mean(axis=1),*q))
    with (out/'per_system_rocof.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['duration_s','system','max_rocof_hz_s'])
        w.writerows((d,k,v) for d,row in zip(durations,y) for k,v in enumerate(row))
    (out/'latent_parameters.json').write_text(json.dumps(theta.tolist()))
    sources=[Path(__file__),Path(a.config),Path('src/domains/swing/continuous_rocof.py'),Path('src/domains/swing/continuous_cuda.py')]
    config={'T':1,'context_horizon':3,'N_obs':0,'window_s':3.,'duration_bounds_s':[.2,3.],'plot_step_s':.01,'amplitude_pu':.05,'physical_bus':1,'noise_sigma_hz_s':.005,'noise_added_to_curve':False,'initial_state':'equilibrium for each independent T=1 candidate','systems':128,'seed':a.seed,'rocof_sample_dt_s':obs.rocof_sample_dt,'hardware':hardware_info(),'physical_config':cfg.raw,'source_hashes':{str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}}
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.8))
    for k in range(1,13):ax.plot(durations,y[:,k],color='#85a9c7',alpha=.25,lw=.7)
    ax.fill_between(durations,q[0],q[2],color='#3786b5',alpha=.15,label='Prior 5–95% range (128 systems)')
    ax.plot(durations,q[1],color='#3786b5',lw=2,label='Prior median')
    ax.plot(durations,y[:,0],color='#b44924',lw=2,label='Prior-midpoint parameters')
    ax.set(xlabel='Probe duration (s)',ylabel='Maximum absolute RoCoF (Hz/s)',title='IEEE9 · One probe from equilibrium · 3 s recording window',xlim=(.2,3.))
    ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.text(.5,.015,'Physical bus 1 · Hann pulse amplitude 0.05 pu · 40 Hz RoCoF sampling · Noise-free response',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(out/'duration_rocof.png',dpi=200);fig.savefig(out/'duration_rocof.svg');plt.close(fig)
    result={'wall_seconds':time.monotonic()-start,'frequency_reconstruction_check':'passed','midpoint_peak_duration_s':float(durations[np.argmax(y[:,0])]),'midpoint_peak_rocof_hz_s':float(y[:,0].max()),'selected_midpoint_values':{str(durations[i]):float(y[i,0]) for i in [0,10,30,80,180,280]},'note':'Prior spread is not a confidence interval. This response curve is not an EIG curve or a sequential T=3 rollout.'}
    (out/'completion.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
