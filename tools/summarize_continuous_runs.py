"""Validate and collect completed fresh T=3/4/5 EIG runs into one result folder."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

METHODS=('dad','rl_sboed','step_dad','myopic','fixed','random')

def collect_campaign(campaign, horizons=(3,4,5), training_seed=101, evaluation_seed=1001, n_obs=0, noise_sigma=.005):
    """Validate full EIG runs and collect one comparison without rerunning methods."""
    root=Path(campaign).resolve()
    summaries={};provenance={};sensor_signature=None
    for t in horizons:
        run=root/f'T{t}'/'run'
        cfg=json.loads((run/'run_config.json').read_text())
        done=json.loads((run/'completion.json').read_text())
        rows=json.loads((run/'rollouts.json').read_text())
        summary=json.loads((run/'summary.json').read_text());s=cfg['settings']
        signature=(cfg['observation_kind'],s['window'],cfg.get('rocof_sample_dt'))
        if sensor_signature is None:sensor_signature=signature
        assert signature==sensor_signature, 'Cannot combine different sensors or recording windows'
        assert done['objective']=='eig' and not done['smoke_only']
        assert set(done['methods'])==set(METHODS) and done['evaluation_seeds']==[evaluation_seed]
        assert (s['T'],s['seed'],s['N_obs'],s['noise_sigma'])==(t,training_seed,n_obs,noise_sigma)
        assert len(rows)==done['records']==6*s['eval_systems']
        identities=None
        for m in METHODS:
            selected=[r for r in rows if r['method']==m]
            keys={(r['evaluation_seed'],r['system']):r['true_MK'] for r in selected}
            assert len(keys)==len(selected)==s['eval_systems']
            if identities is None:identities=keys
            assert identities==keys
            vals=[r['terminal_spce_nats'] for r in selected]
            assert all(map(math.isfinite,vals))
            expected=next(r['mean_spce_nats'] for r in summary if r['method']==m)
            assert math.isclose(statistics.mean(vals),expected,abs_tol=1e-12)
            for r in selected:
                ds=r['duration_sequence_s']
                assert len(ds)==t and all(math.isfinite(x) and .2<=x<=3. for x in ds)
                assert all(b-a>=s['min_duration_separation']-1e-9 for a,b in zip(ds,ds[1:]))
        history=json.loads((run/'models/rl_sboed_training.json').read_text())
        assert all(r['redq']['target_entropy']==-1. for r in history if r.get('redq'))
        summaries[str(t)]=summary
        provenance[str(t)]={'run':str(run),'sha256':{f:hashlib.sha256((run/f).read_bytes()).hexdigest() for f in ('run_config.json','summary.json','rollouts.json','completion.json')}}
    out=root/'combined_results';out.mkdir(exist_ok=False)
    for t in horizons:
        dest=out/f'T{t}';dest.mkdir()
        for f in ('run_config.json','summary.json','rollouts.json','completion.json'):
            shutil.copy2(root/f'T{t}'/'run'/f,dest/f)
    lines=['# IEEE9 EIG — fresh full-budget runs','',
           f'Train {training_seed}; eval {evaluation_seed}; N_obs={n_obs}; sigma={noise_sigma}. All six methods freshly trained/evaluated as applicable.',
           f'Observation: {sensor_signature[0]}; recording window: {sensor_signature[1]} s; RoCoF difference interval: {sensor_signature[2]} s.',
           'One seed pair: across-seed standard deviations and publication-level superiority are not established.','',
           '| Method | '+' | '.join(f'T={t} sPCE (nats)' for t in horizons)+' |','|---|'+'---:|'*len(horizons)]
    for m in METHODS:
        vals=[next(r['mean_spce_nats'] for r in summaries[str(t)] if r['method']==m) for t in horizons]
        lines.append('| '+m+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    for f,data in [('summary_by_horizon.json',summaries),('provenance.json',provenance),('completion.json',{'status':'complete','horizons':list(horizons),'methods':METHODS,'training_seed':training_seed,'evaluation_seed':evaluation_seed})]:
        (out/f).write_text(json.dumps(data,indent=2)+'\n')
    print('RESULT_COLLECTION_COMPLETE',out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    p.add_argument('--horizons',default='3,4,5')
    p.add_argument('--training-seed',type=int,default=101)
    p.add_argument('--evaluation-seed',type=int,default=1001)
    p.add_argument('--n-obs',type=int,default=0)
    p.add_argument('--noise-sigma',type=float,default=.005)
    a=p.parse_args()
    collect_campaign(a.campaign,tuple(map(int,a.horizons.split(','))),a.training_seed,a.evaluation_seed,a.n_obs,a.noise_sigma)

if __name__=='__main__':main()
