"""Check saved IEEE9 baselines against current physics and paired evaluator."""
from pathlib import Path
import hashlib,json
import numpy as np
from src.config import load_config
from src.objectives.eig.continuous_eig import OnlineEIG
from src.domains.swing.continuous_rocof import EndpointRocofObserver

ROOT=Path(__file__).resolve().parents[1]
BASELINES={3:'experiments/09142026/09142026_ieee9_eig_endpoint_rocof_T3_train101_eval1001/T3/run',
4:'experiments/09152026/09152026_endpoint_rocof_T4-5_train101_eval1001_rerun12h/T4/run',
5:'experiments/09152026/09152026_endpoint_rocof_T4-5_train101_eval1001_rerun12h/T5/run'}

def verify():
    report={}
    for horizon,relative in BASELINES.items():
        base=ROOT/relative
        metadata=json.loads((base/'run_config.json').read_text())
        assert (base/'completion.json').exists()
        assert (base/'exit_code').read_text().strip()=='0'
        settings=metadata['settings']
        for key,value in dict(T=horizon,seed=101,eval_seed=1001,eval_systems=128,contrasts=128,
                window=3.5,observation_kind='endpoint_rocof',noise_sigma=.005,
                updates=2000,batch_size=32,validation_systems=128,validate_every=100).items():
            assert settings[key]==value,(horizon,key,settings[key])
        checked=[]
        for name,digest in metadata['source_hashes'].items():
            if name.startswith('src/domains/swing/'):
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
                checked.append(name)
        cfg=load_config(ROOT/'configs/ieee9_eig.yaml')
        # Compare the entire physical model, including the full node priors.
        cfg.raw['swing_equation']['T_obs_sec']=3.5
        cfg.raw['swing_equation'].pop('probe_durations',None)
        assert json.loads(json.dumps(cfg.raw['swing_equation']))==metadata['physical_config']['swing_equation']
        observer=EndpointRocofObserver(cfg,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05,window=3.5)
        e=OnlineEIG(cfg,observer,horizon=horizon,sigma=.005,contrasts=128,min_separation=.01)
        rng=np.random.default_rng(1001)
        theta,states=e.sample(rng,128)
        noise=rng.normal(size=(128,horizon,1))
        rows=[r for r in json.loads((base/'rollouts.json').read_text()) if r['method']=='dad']
        assert len(rows)==128
        for row in rows:
            np.testing.assert_array_equal(theta[row['system'],0],row['true_MK'])
        differences=[]
        for row in rows[:2]:
            system=row['system']
            def selector(stage,actions,observations):return [row['duration_sequence_s'][stage]]
            result=e.rollout(None,np.random.default_rng(1001+system),1,stochastic=False,
                initial=(theta[system:system+1],states[system:system+1]),selector=selector,
                noise=noise[system:system+1])
            np.testing.assert_allclose(np.asarray(result['observations'])[:,0,:],row['observations_rocof_hz_s'],rtol=1e-5,atol=1e-7)
            delta=float(result['info'][0,-1])-row['terminal_spce_nats']
            assert abs(delta)<1e-4,delta
            differences.append(delta)
        report[str(horizon)]={'baseline':relative,'all_128_true_parameters_identical':True,
            'replayed_baseline_systems':2,'spce_replay_differences':differences,
            'physical_source_hashes_match':checked,'settings':settings}
    output=ROOT/'experiments/09162026/09162026_ieee9_moe_endpoint_validation/comparability.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('COMPARABILITY_OK',output)

if __name__=='__main__':verify()
