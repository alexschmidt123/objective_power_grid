"""Bounded V2-based feature V3 pilot, with native-update-matched V2 control."""
import argparse,copy,gc,json,pathlib,random,subprocess,time
import numpy as np
import torch
from v3_native import n,ROOT,WORK
from feature_policy import FeatureMoE
from v3_training import configure,phase,Stop
from evaluate_feature import evaluate,references
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args();out=args.output.expanduser()
    assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    prerequisite=ROOT/'scratch_runs/hec_v3_feature_20261114_validation01'
    assert json.loads((prerequisite/'RESULTS.json').read_text())['maximum_score_error']<1e-9
    smoke=ROOT/'scratch_runs/hec_v3_feature_20261114_native_smoke02'
    assert json.loads((smoke/'RESULTS.json').read_text())['counts']==dict(actor=1,critic=5,posterior=150)
    assert json.loads((ROOT/'scratch_runs/hec_v3_feature_20261114_control_smoke01/RESULTS.json').read_text())['counts']==dict(actor=1,critic=5,posterior=150)
    references()  # Fail before training if a frozen reference identity changed.
    cfg=dict(candidate_phases=[['experts',50],['router',50],['joint',100]],control_updates=200,wall_seconds=7200,seed=202611145101,
        training_objective='Original full-horizon native variational EIG;unchanged native actor,critic,posterior implementations',
        actor_optimizer='New Adam per parameter-selection phase,starting at current terminal LR;control inherits native optimizer',
        target='New candidate target copies its online actor so frozen projection/offset buffers agree;control retains native target lag',
        evaluation='Exact original V2 bank/scorer,2000 cases/1M contrasts;frozen baseline vectors and checkpoint certificates checked',
        source_commit=subprocess.check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    n.put(out/'CONFIG.json',cfg)
    files=[*sorted(pathlib.Path(__file__).parent.glob('*.*')),n.BASE/'arms/candidate/actor.pt',n.BASE/'arms/candidate/runtime.pkl']
    protected={str(p):n.sha(p) for p in files if p.is_file()};n.put(out/'MANIFEST.json',protected)
    n.original.setup();lease=None
    while lease is None:
        try:lease=n.original.acquire()
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic()
    try:
        state=n.load_parent();a=state['algorithm'];model=FeatureMoE(a.actor_net).cuda();target=copy.deepcopy(model);target.configure('frozen')
        a.actor_net=model;a.actor_target_net=target;n.original.restore(state['rng'])
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']%(2**32));random.seed(cfg['seed'])
        for name,steps in cfg['candidate_phases']:
            configure(a,name);result=phase(a,'candidate_'+name,steps,out,start,cfg['wall_seconds'])
            if result['status']!='COMPLETE':raise Stop('Candidate incomplete')
        del state,a,model,target;gc.collect();torch.cuda.empty_cache()
        state=n.load_parent();a=state['algorithm'];n.original.restore(state['rng'])
        torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']%(2**32));random.seed(cfg['seed'])
        result=phase(a,'control',200,out,start,cfg['wall_seconds'])
        if result['status']!='COMPLETE':raise Stop('Control incomplete')
        del state,a;gc.collect();torch.cuda.empty_cache()
        result=evaluate(out,start+cfg['wall_seconds'])
        assert all(n.sha(p)==h for p,h in protected.items())
        n.put(out/'STATUS.json',dict(status='COMPLETE',accepted=result['accepted'],historical_artifacts_preserved=True,seconds=time.monotonic()-start))
    except Stop as exc:n.put(out/'STATUS.json',dict(status='INCOMPLETE',reason=str(exc)))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
