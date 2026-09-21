"""Paired T=1 MSC/MOCU duration audits using production carried-state control."""
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from src.config import load_config
from src.domains.swing.continuous_rocof import MaxRocofObserver
from src.objectives.eig.continuous_eig import OnlineEIG
from src.control.continuous_control import CarriedStateControl,continuous_decision


def run_duration(engine,control,theta,initial,noise,duration,batch_size):
    """Truth is column zero; only columns 1: enter the controller posterior."""
    records=[];states_saved=[];weights_saved=[];requirements_saved=[]
    B,P,D=theta.shape
    for start in range(0,B,batch_size):
        stop=min(B,start+batch_size);th=theta[start:stop];n=len(th)
        prediction=engine.observer.propagate(th.reshape(-1,D),initial[start:stop].reshape(-1,D),np.full(n*P,duration))
        states=prediction.terminal_state.reshape(n,P,D)
        means=prediction.observations.reshape(n,P,1)
        y=means[:,0]+engine.sigma*noise[start:stop]
        ll=-.5*np.sum(((y[:,None]-means[:,1:])/engine.sigma)**2,axis=-1)
        weights=np.exp(ll-logsumexp(ll,axis=1,keepdims=True))
        # Production solver checks sampled monotonicity, retains infeasible mass.
        requirements=control.requirements(th.reshape(-1,D),states.reshape(-1,D),allow_infeasible=True).reshape(n,P)
        states_saved.append(states);weights_saved.append(weights);requirements_saved.append(requirements)
        for i in range(n):
            oracle=float(requirements[i,0])
            base={'system':start+i,'duration_s':float(duration),'observation_rocof_hz_s':float(y[i,0]),
                  'oracle_msc':oracle if np.isfinite(oracle) else None,'oracle_feasible':bool(np.isfinite(oracle)),
                  'true_MK':th[i,0].tolist(),'true_terminal_state':states[i,0].tolist()}
            for objective in ['msc','mocu']:
                row={**base,'objective':objective}
                try:
                    decision=continuous_decision(requirements[i,1:],weights[i],control.coverage,objective)
                    _,_,safe=control.metrics(th[i,1:],states[i,1:],np.full(P-1,decision['u_ctrl']))
                    mass=float(weights[i]@safe)
                    if mass<control.coverage-1e-12:raise ValueError('Direct posterior safety verification failed')
                    r,f,ok=control.metrics(th[i,:1],states[i,:1],[decision['u_ctrl']])
                    row.update(decision,status='ok',posterior_safe_mass=mass,physical_safe=bool(ok[0]),
                               terminal_rocof_hz_s=float(r[0]),terminal_nadir_hz=float(f[0]))
                except ValueError as exc:
                    row.update(status='failed',error=str(exc))
                records.append(row)
    return records,np.concatenate(states_saved),np.concatenate(weights_saved),np.concatenate(requirements_saved)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',default='configs/ieee9_mocu.yaml')
    p.add_argument('--systems',type=int,default=128)
    p.add_argument('--particles',type=int,default=128)
    p.add_argument('--seed',type=int,default=1001)
    p.add_argument('--coverage',type=float,default=.9)
    p.add_argument('--sigma',type=float,default=.005)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--estimate-only',action='store_true')
    a=p.parse_args()
    if min(a.systems,a.particles,a.batch_size)<2 or not 0<a.coverage<1 or a.sigma<=0:p.error('Invalid audit settings')
    a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    cfg=load_config(a.config);cfg.raw['control']['posterior_coverage']=a.coverage
    observer=MaxRocofObserver(cfg,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.)
    engine=OnlineEIG(cfg,observer,horizon=1,sigma=a.sigma,contrasts=a.particles)
    control=CarriedStateControl(cfg,observer.sim)
    count=min(8,a.systems) if a.estimate_only else a.systems
    rng=np.random.default_rng(a.seed);theta,initial=engine.sample(rng,count);noise=rng.normal(size=(count,1))
    durations=np.round(np.arange(.2,3.001,.1),2)
    settings={**vars(a),'output':str(a.output),'durations_s':durations.tolist(),'T':1,'N_obs':0,
              'window_s':3,'amplitude_pu':.05,'physical_probe_bus':1,'noise_units':'Hz/s',
              'controller_excludes_truth':True,'control_spec':cfg.raw['control'],
              'uncertainty':'Pointwise Monte Carlo SE across evaluation systems, not training-seed SD.',
              'oracle_note':'Minimum safe continuous control for true parameters and that duration\'s terminal state.',
              'source_hashes':{str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),Path('src/control/continuous_control.py'),Path('src/domains/swing/continuous_rocof.py')]}}
    (a.output/'settings.json').write_text(json.dumps(settings,indent=2))
    np.savez_compressed(a.output/'paired_inputs.npz',theta=theta,initial_states=initial,standard_normal_noise=noise)
    if a.estimate_only:
        t=time.monotonic();rows,*_=run_duration(engine,control,theta,initial,noise,1.5,a.batch_size)
        measured=time.monotonic()-t;estimate=measured*a.systems/count*len(durations)
        report={'timing_systems':count,'particles':a.particles,'one_duration_seconds':measured,
                'projected_total_seconds':estimate,'padded_seconds':estimate*1.5,
                'failed_rows':sum(r['status']!='ok' for r in rows),'note':'Shared computation for both objectives; linear extrapolation, not a performance result.'}
        (a.output/'timing.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True);return
    all_rows=[];summary=[]
    for d in durations:
        t=time.monotonic()
        try:
            rows,states,weights,requirements=run_duration(engine,control,theta,initial,noise,d,a.batch_size)
        except Exception as exc:
            (a.output/'failure.json').write_text(json.dumps({'duration_s':float(d),'error':str(exc)}));raise
        all_rows.extend(rows)
        np.savez_compressed(a.output/f'duration_{d:.2f}_posterior.npz',states=states,weights=weights,requirements=requirements)
        with (a.output/'records.partial.jsonl').open('a') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        for objective in ['msc','mocu']:
            selected=[r for r in rows if r['objective']==objective];key='msc' if objective=='msc' else 'posterior_mocu'
            failed=sum(r['status']!='ok' for r in selected)
            row={'objective':objective,'duration_s':float(d),'systems':count,'failed_systems':failed}
            for field in [key,'u_ctrl','physical_safe','oracle_msc','posterior_safe_mass','posterior_ess']:
                vals=[r.get(field) for r in selected]
                complete=all(v is not None for v in vals)
                row['mean_'+field]=float(np.mean(vals)) if complete else None
                row['se_'+field]=float(np.std(vals,ddof=1)/np.sqrt(count)) if complete else None
            summary.append(row)
        (a.output/'summary.partial.json').write_text(json.dumps(summary,indent=2))
        print(f'duration={d:.2f} complete; failed rows={sum(r["status"]!="ok" for r in rows)}; seconds={time.monotonic()-t:.1f}',flush=True)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for objective in ['msc','mocu']:
        folder=a.output/objective;folder.mkdir()
        selected=[r for r in summary if r['objective']==objective]
        with (folder/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
        key='msc' if objective=='msc' else 'posterior_mocu'
        y=np.array([np.nan if r['mean_'+key] is None else r['mean_'+key] for r in selected]);se=np.array([np.nan if r['se_'+key] is None else r['se_'+key] for r in selected])
        fig,ax=plt.subplots(figsize=(7,4.4));ax.plot(durations,y,lw=2);ax.fill_between(durations,y-1.96*se,y+1.96*se,alpha=.18)
        ax.set(xlabel='Hann injection duration (s)',ylabel=('Selected control (pu)' if objective=='msc' else 'Posterior MOCU (pu)'),title=f'IEEE9 single-step {objective.upper()}: RoCoF, q={a.coverage:g}, sigma={a.sigma:g} Hz/s')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/f'ieee9_{objective}_vs_duration.png',dpi=200);plt.close(fig)
    (a.output/'records.json').write_text(json.dumps(all_rows))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    failed=sum(r['status']!='ok' for r in all_rows)
    report={'objectives':['msc','mocu'],'durations':29,'systems':count,'particles':a.particles,'records':len(all_rows),'failed_records':failed,'status':'complete' if failed==0 else 'complete_with_failures','wall_seconds':time.monotonic()-started}
    (a.output/'completion.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
