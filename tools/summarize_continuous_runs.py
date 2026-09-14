"""Validate and collect completed fresh T=3/4/5 EIG runs into one result folder."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

METHODS=('dad','rl_sboed','step_dad','myopic','fixed','random')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    a=p.parse_args();root=Path(a.campaign).resolve()
    summaries={};provenance={}
    for t in (3,4,5):
        run=root/f'T{t}'/'run'
        cfg=json.loads((run/'run_config.json').read_text())
        done=json.loads((run/'completion.json').read_text())
        rows=json.loads((run/'rollouts.json').read_text())
        summary=json.loads((run/'summary.json').read_text());s=cfg['settings']
        assert done['objective']=='eig' and not done['smoke_only']
        assert set(done['methods'])==set(METHODS) and done['evaluation_seeds']==[1001]
        assert (s['T'],s['seed'],s['N_obs'],s['noise_sigma'])==(t,101,0,.005)
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
    for t in (3,4,5):
        dest=out/f'T{t}';dest.mkdir()
        for f in ('run_config.json','summary.json','rollouts.json','completion.json'):
            shutil.copy2(root/f'T{t}'/'run'/f,dest/f)
    lines=['# IEEE9 EIG — fresh full-budget runs','',
           'Train 101; eval 1001; max absolute RoCoF; sigma 0.005 Hz/s. All six methods freshly trained/evaluated as applicable.',
           'One seed pair: across-seed standard deviations and publication-level superiority are not established.','',
           '| Method | T=3 sPCE (nats) | T=4 sPCE (nats) | T=5 sPCE (nats) |','|---|---:|---:|---:|']
    for m in METHODS:
        vals=[next(r['mean_spce_nats'] for r in summaries[str(t)] if r['method']==m) for t in (3,4,5)]
        lines.append('| '+m+' | '+' | '.join(f'{x:.6f}' for x in vals)+' |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    for f,data in [('summary_by_horizon.json',summaries),('provenance.json',provenance),('completion.json',{'status':'complete','horizons':[3,4,5],'methods':METHODS,'training_seed':101,'evaluation_seed':1001})]:
        (out/f).write_text(json.dumps(data,indent=2)+'\n')
    print('RESULT_COLLECTION_COMPLETE',out,flush=True)

if __name__=='__main__':main()
