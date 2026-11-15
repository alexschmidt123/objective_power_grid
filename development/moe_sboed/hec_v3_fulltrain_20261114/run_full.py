"""User-authorized V3-only continuation; no baseline training; 12h global ceiling."""
import argparse,copy,gc,json,pathlib,random,subprocess,time
import numpy as np
import torch
from v3_native import n,ROOT,WORK
from feature_policy import FeatureMoE
from v3_training import configure,phase,Stop
from evaluate_feature import evaluate,references

def main():
    start=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args();out=args.output.expanduser()
    assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    validation=ROOT/'scratch_runs/hec_v3_fulltrain_20261114_validation01/RESULTS.json'
    smoke=ROOT/'scratch_runs/hec_v3_fulltrain_20261114_smoke01/RESULTS.json'
    assert json.loads(validation.read_text())['maximum_score_error']<1e-9
    assert json.loads(smoke.read_text())['counts']==dict(actor=1,critic=5,posterior=150)
    references()
    cfg=dict(phases=[['warmup',200],['joint',1800]],seed=202611145101,soft_all_phase_seconds=37800,hard_all_phase_seconds=43200,
        training_stop_seconds=34200,evaluation_deadline_seconds=37800,baseline_training=False,initialization='Original terminal V2 runtime16001,not the 200-update pilot',
        objective='Unchanged native full-horizon variational EIG,discount1',optimizer='One fresh Adam at inherited terminal LR;state/scheduler preserved across history unfreeze',
        target='Online actor copy at initialization;native Polyak thereafter',uniform_and_single='Inference-only ablations,not separately optimized policies',
        source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    n.put(out/'CONFIG.json',cfg)
    protected={str(p):n.sha(p) for p in [*sorted(pathlib.Path(__file__).parent.glob('*.py')),n.BASE/'arms/candidate/actor.pt',n.BASE/'arms/candidate/runtime.pkl']}
    n.put(out/'MANIFEST.json',protected);n.original.setup();lease=None
    while lease is None:
        if time.monotonic()-start>600:raise TimeoutError('GPU unavailable;no jobs interrupted')
        try:lease=n.original.acquire()
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    try:
        state=n.load_parent();a=state['algorithm'];a.actor_net=FeatureMoE(a.actor_net).cuda();a.actor_target_net=copy.deepcopy(a.actor_net);a.actor_target_net.configure('frozen')
        n.original.restore(state['rng']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']%(2**32));random.seed(cfg['seed'])
        for name,steps in cfg['phases']:
            configure(a,name);result=phase(a,'candidate_'+name,steps,out,start,cfg['training_stop_seconds'])
            if result['status']!='COMPLETE':raise Stop('Incomplete training;partial artifacts preserved;no final comparison claimed')
        del state,a;gc.collect();torch.cuda.empty_cache()
        result=evaluate(out,start+cfg['evaluation_deadline_seconds'])
        assert all(n.sha(p)==h for p,h in protected.items())
        n.put(out/'STATUS.json',dict(status='COMPLETE',accepted=result['accepted'],updates=2000,new_episodes=2000000,seconds=time.monotonic()-start,historical_artifacts_preserved=True))
    except Stop as exc:n.put(out/'STATUS.json',dict(status='INCOMPLETE',reason=str(exc),seconds=time.monotonic()-start))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=repr(exc),seconds=time.monotonic()-start));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
