"""One genuine native update; this smoke checkpoint never initializes the pilot."""
import argparse,copy,pathlib,time
import torch
from v3_native import n,ROOT
from feature_policy import FeatureMoE
from v3_training import configure,phase
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args();out=args.output.expanduser()
    assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True);n.original.setup();lease=n.original.acquire();start=time.monotonic()
    protected={str(p):n.sha(p) for p in [n.BASE/'arms/candidate/actor.pt',n.BASE/'arms/candidate/runtime.pkl']}
    try:
        state=n.load_parent();a=state['algorithm'];model=FeatureMoE(a.actor_net).cuda();target=copy.deepcopy(model);target.configure('frozen')
        a.actor_net=model;a.actor_target_net=target
        configure(a,'joint');before=n.digest(model.state_dict());n.original.restore(state['rng'])
        assert n.digest(target.state_dict())==before
        result=phase(a,'smoke',1,out,start,600);assert result['status']=='COMPLETE' and result['counts']==dict(actor=1,critic=5,posterior=150)
        assert n.digest(model.state_dict())!=before and all(n.sha(p)==h for p,h in protected.items())
        n.put(out/'RESULTS.json',dict(status='COMPLETE',counts=result['counts'],historical_artifacts_preserved=True,
            frozen_actor_state_preserved=True,seconds=time.monotonic()-start,claim='Native integration check only;no benchmark superiority claim'))
        n.put(out/'STATUS.json',dict(status='COMPLETE',native_updates=1))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=repr(exc)));raise
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
