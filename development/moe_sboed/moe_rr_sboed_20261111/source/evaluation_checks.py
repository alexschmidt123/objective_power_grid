"""Disposable exact scorer and changed-policy trajectory checks."""
from common import *
from rr_policy import RRPolicy
from evaluation import load_rr,simulate
from frozen_evaluator import contrast_scores

def main():
    original.setup();lease=original.acquire();torch.manual_seed(CFG['validation_seed'])
    state=restore();a=state['algorithm'];model=RRPolicy(a.actor_net,CFG)
    with torch.no_grad():model.correction.weight.normal_(0,.01)
    for arm in ['full','control']:
        root=O/'arms'/arm;root.mkdir(parents=True,exist_ok=True)
        torch.save(dict(actor_state=model.state_dict(),fixture_only=True),root/'actor.pt')
    model=load_rr('full');rng=np.random.default_rng(CFG['validation_seed'])
    with torch.no_grad(),torch.device('cuda'):
        truth=torch.tensor(rng.standard_normal((4,4)),device='cuda');noise=torch.tensor(rng.standard_normal((4,30)),device='cuda');u=torch.tensor(rng.random((4,30)),device='cuda')
        data=simulate(model,truth,noise,u,collect=True);again=simulate(model,truth,noise,u)
        for x,y in zip(data['actions'],again['actions']):torch.testing.assert_close(x,y,rtol=0,atol=0)
        uni=simulate(model,truth,noise,u,uniform=True,collect=True)
        torch.testing.assert_close(uni['probabilities'],torch.full((4,30,6),1/6,device='cuda'))
        negatives=torch.tensor(rng.standard_normal((17,4)),device='cuda');bench=type('Bench',(),{'sigma':.5,'calls':0})()
        score,_=contrast_scores(bench,data,negatives,5,'cuda');literal=[]
        for i in range(4):
            theta=torch.cat([truth[i:i+1],negatives]);ll=theta.new_zeros(18)
            for t in range(30):
                action=data['actions'][t][i];obs=data['observations'][t][i,0]
                distance=(theta.reshape(-1,2,2)-action).square().sum(-1);mean=(.1+(distance+1e-4).reciprocal().sum(-1)).log()
                ll+=-2*(obs-mean).square()-math.log(.5*math.sqrt(2*math.pi))
            literal.append(ll[0]-torch.logsumexp(ll,0)+math.log(18))
        torch.testing.assert_close(score[:,-1],torch.stack(literal).cpu(),rtol=1e-10,atol=1e-10)
        stage=12;prefix=(data['actions'][:stage],data['observations'][:stage])
        resumed=simulate(model,truth,noise,u,prefix=prefix)
        for x,y in zip(data['actions'],resumed['actions']):torch.testing.assert_close(x,y,rtol=0,atol=0)
        for k in range(4):
            changed=simulate(model,truth,noise,u,disable=k,prefix=prefix,intervene_once=True)
            for t in range(stage):torch.testing.assert_close(changed['observations'][t],data['observations'][t],rtol=0,atol=0)
            assert all(torch.isfinite(x).all() for x in changed['actions'])
        initial=RRPolicy(a.actor_net,CFG);baseline=simulate(initial,truth,noise,u)
        from evaluate_v2 import simulate as legacy
        reference=legacy(a.actor_net,truth,noise,u,'candidate',gates=torch.full((30,4),.25,device='cuda'))
        # Scorer inputs promote physical actions losslessly to float64.
        for x,y in zip(baseline['actions'],reference['actions']):torch.testing.assert_close(x,y.double(),rtol=0,atol=0)
        for x,y in zip(baseline['observations'],reference['observations']):torch.testing.assert_close(x,y,rtol=0,atol=0)
        from evaluate_v2 import actor_load
        for name in ['sg','ordinary','vsoed']:
            reference,_=actor_load(name);reference=reference.eval().requires_grad_(False)
            trajectory=legacy(reference,truth,noise,u,name)
            value,_=contrast_scores(bench,trajectory,negatives,5,'cuda')
            assert value.shape==(4,30) and torch.isfinite(value).all()
    put(O/'evidence/EVALUATION_CHECKS.json',dict(status='PASS',fixture_only=True,literal_positive_once_score=True,
        resimulated_entire_changed_policy=True,conditional_prefix_resume_identity=True,initial_uniform_HEC_trajectory_bitwise_identity=True,
        main_bank_access=False,action_bounds=True,uniform_distinct_pair_probability=True,SG_conventional_native_vsoed_compatible=True))
    print('RR_EVALUATION_CHECKS_PASS',flush=True)

if __name__=='__main__':main()
