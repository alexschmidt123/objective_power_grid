"""Analyze carried-state effects on signed endpoint RoCoF; no policy training."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from src.config import load_config
from src.hardware import hardware_info
from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver
from src.domains.swing.continuous import ContinuousSwingObserver

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    cfg=load_config('configs/ieee9_eig.yaml');kw=dict(n_obs=2,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.5)
    o=CudaContinuousSwingObserver(cfg,**kw);o.times=np.array([3.475,3.5])
    lo=np.r_[cfg.swing['M_lower_nodes'],cfg.swing['K_lower_nodes']];hi=np.r_[cfg.swing['M_upper_nodes'],cfg.swing['K_upper_nodes']]
    theta=np.vstack([(lo+hi)/2,np.random.default_rng(101).uniform(lo,hi,(128,6))]);durations=np.round(np.linspace(.2,3.,141),8)
    prefixes=[[],[.3],[.8],[1.5],[.3,.6],[.3,1.2],[.8,1.2]]
    all_curves=[];all_states=[];summaries=[];rows=[]
    def observe(params,state,d):
        r=o.propagate(params,state,d);return (r.observations[:,1]-r.observations[:,0])/.025,r.terminal_state
    for prefix in prefixes:
        state=o.initial_state(len(theta))
        for d in prefix:_,state=observe(theta,state,d)
        curves=np.array([observe(theta,state,d)[0] for d in durations]);all_curves.append(curves);all_states.append(state)
        valid=(durations>= (prefix[-1]+.01 if prefix else .2)) & (durations<= (2.99 if len(prefix)==1 else 2.98 if not prefix else 3.))
        v=curves[valid];ds=durations[valid];ix=np.argmax(v[:,1:],axis=0)
        summary={'prefix_s':prefix,'stage':len(prefix)+1,'midpoint_initial_state':state[0].tolist(),'midpoint_range_hz_s':[float(v[:,0].min()),float(v[:,0].max())],'midpoint_peak_duration_s':float(ds[np.argmax(v[:,0])]),'prior_peak_duration_p05_median_p95_s':np.quantile(ds[ix],[.05,.5,.95]).tolist()}
        if prefix:
            delta=v-all_curves[0][valid];summary.update(midpoint_max_change_vs_equilibrium_hz_s=float(np.max(np.abs(delta[:,0]))),prior_p95_abs_change_hz_s=float(np.quantile(np.abs(delta[:,1:]),.95)))
        summaries.append(summary)
        for j,d in enumerate(durations):
            if valid[j]:
                for k,value in enumerate(curves[j]):rows.append([json.dumps(prefix),len(prefix)+1,d,k,value])
    # Counterfactual state ablations at the midpoint for prefix [.3,.6].
    s=all_states[4][:1].copy();variants={'full carry':s,'reset to equilibrium':o.initial_state(1),'angles only':np.c_[s[:,:3],np.zeros((1,3))],'speeds only':np.c_[np.zeros((1,3)),s[:,3:]]}
    ablation={name:np.array([observe(theta[:1],x,d)[0][0] for d in durations]) for name,x in variants.items()}
    # Gauge invariance and independent CPU integration for a valid third duration.
    shifted=s.copy();shifted[:,:3]+=.1
    base=observe(theta[:1],s,1.)[0];np.testing.assert_allclose(base,observe(theta[:1],shifted,1.)[0],atol=1e-11)
    cpu=ContinuousSwingObserver(cfg,**kw);cpu.times=o.times.copy();r=cpu.propagate(theta[:1],s,1.);cpu_value=np.diff(r.observations,axis=1)[:,0]/.025
    np.testing.assert_allclose(base,cpu_value,atol=1e-6,rtol=1e-4)
    sensitivities=[]
    for j in range(6):
        h=1e-5 if j<3 else 2*np.pi*1e-5
        plus=s.copy();minus=s.copy();plus[0,j]+=h;minus[0,j]-=h
        derivative=(observe(theta[:1],plus,1.)[0]-observe(theta[:1],minus,1.)[0])/(2*h)
        sensitivities.append(float(derivative[0] if j<3 else derivative[0]*2*np.pi))
    with (out/'history_duration_observations.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['prefix_s','stage','duration_s','system','signed_rocof_hz_s']);w.writerows(rows)
    with (out/'state_ablation.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['duration_s',*ablation]);w.writerows(zip(durations,*ablation.values()))
    (out/'latent_parameters.json').write_text(json.dumps(theta.tolist()))
    (out/'carried_states.json').write_text(json.dumps([{'prefix':p,'states':s.tolist()} for p,s in zip(prefixes,all_states)]))
    result={'histories':summaries,'sensitivity_at_prefix_0p3_0p6_duration_1':{'angle_derivatives_hz_s_per_rad':sensitivities[:3],'frequency_derivatives_hz_s_per_hz':sensitivities[3:]},'cpu_gpu_observation_abs_difference':float(abs(base-cpu_value)[0]),'common_angle_shift_check':'passed','wall_seconds':time.monotonic()-start}
    # Small comparison table at shared feasible durations.
    result['selected_midpoint_observations']={str(d):[float(x[np.argmin(abs(durations-d)),0]) for x in all_curves] for d in [1.6,2.,2.6,3.]}
    result['ablation_at_duration_1']={name:float(values[np.argmin(abs(durations-1.))]) for name,values in ablation.items()}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'run_config.json').write_text(json.dumps({'T':3,'observation_kind':'signed_endpoint_rocof','observation_time_s':3.5,'difference_interval_s':.025,'noise_sigma_hz_s':.005,'noise_added':False,'hardware':hardware_info(),'prefixes':prefixes,'prior_seed':101,'prior_systems':128,'physical_config':cfg.raw,'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('src/domains/swing/continuous_cuda.py'),Path('src/domains/swing/continuous.py'),Path('src/domains/swing/cuda.py')]},'note':'Feasible domains use T=3 increasing-duration rule. Reset/angle/speed ablations are counterfactual diagnostics only.'},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    for prefix,curves in zip(prefixes,all_curves):
        valid=(durations>= (prefix[-1]+.01 if prefix else .2)) & (durations<= (2.99 if len(prefix)==1 else 2.98 if not prefix else 3.))
        axs[0].plot(durations[valid],curves[valid,0],label='Equilibrium' if not prefix else str(prefix))
    axs[0].set_title('Same latent parameters, different probe histories');axs[0].legend(title='Previous durations (s)',fontsize=8)
    for name,values in ablation.items():axs[1].plot(durations[durations>=.61],values[durations>=.61],label=name)
    axs[1].set_title('Stage 3 after [0.3, 0.6] s: state ablations');axs[1].legend(fontsize=8)
    for ax in axs:ax.set(xlabel='Current duration (s)',ylabel='Signed RoCoF at stage time 3.5 s (Hz/s)');ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.5)
    fig.tight_layout();fig.savefig(out/'carried_state_effect.png',dpi=180);plt.close(fig)
    (out/'completion.json').write_text(json.dumps({'checks':'passed','wall_seconds':time.monotonic()-start},indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
