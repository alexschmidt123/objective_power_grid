"""T=1 signed RoCoF at a specified elapsed time versus probe duration."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from src.config import load_config
from src.hardware import hardware_info
from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--time',type=float,default=3.5)
    a=p.parse_args();start=time.monotonic()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    cfg=load_config('configs/ieee9_eig.yaml')
    delta=.025
    obs=CudaContinuousSwingObserver(cfg,n_obs=2,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=a.time)
    obs.times=np.array([a.time-delta,a.time])
    assert np.allclose(obs.times/obs.sim.ode_dt,np.rint(obs.times/obs.sim.ode_dt))
    lo=np.r_[cfg.swing['M_lower_nodes'],cfg.swing['K_lower_nodes']]
    hi=np.r_[cfg.swing['M_upper_nodes'],cfg.swing['K_upper_nodes']]
    durations=np.round(np.linspace(.2,3.,29),2)
    theta=np.tile((lo+hi)/2,(len(durations),1))
    response=obs.propagate(theta,obs.initial_state(len(theta)),durations)
    rocof=np.diff(response.observations,axis=1)[:,0]/delta
    assert np.isfinite(rocof).all()
    # Verify against two separately terminated trajectories at the same times.
    values=[]
    for t in obs.times:
        ref=CudaContinuousSwingObserver(cfg,n_obs=1,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=float(t))
        ref.times=np.array([t]);values.append(ref.propagate(theta,ref.initial_state(len(theta)),durations).observations[:,0])
    np.testing.assert_allclose(rocof,(values[1]-values[0])/delta,atol=1e-12,rtol=1e-10)
    with (out/'observation_vs_duration.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['duration_s','signed_rocof_hz_s']);w.writerows(zip(durations,rocof))
    config={'T':1,'observation_kind':'signed_rocof_at_fixed_time','observation_dimension':1,'observation_time_s':a.time,'window_s':a.time,'difference_interval_s':delta,'difference_times_s':obs.times.tolist(),'noise_added':False,'noise_sigma_hz_s':.005,'latent_selection':'prior midpoint','theta_MK':theta[0].tolist(),'physical_bus':1,'amplitude_pu':.05,'initial_state':'equilibrium','hardware':hardware_info(),'physical_config':cfg.raw,'source_hashes':{str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in [Path(__file__),Path('src/domains/swing/continuous_cuda.py'),Path('configs/ieee9_eig.yaml')]}}
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot(durations,rocof,'o-',ms=3);ax.axhline(0,color='gray',lw=.8)
    ax.set(xlabel='Probe duration (s)',ylabel=f'Signed RoCoF at {a.time:g} s (Hz/s)',title='IEEE9 · T=1 · Prior-midpoint parameters');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'observation_vs_duration.png',dpi=180);plt.close(fig)
    (out/'completion.json').write_text(json.dumps({'wall_seconds':time.monotonic()-start,'independent_termination_check':'passed'},indent=2))
    print((out/'observation_vs_duration.csv').read_text())

if __name__=='__main__':main()
