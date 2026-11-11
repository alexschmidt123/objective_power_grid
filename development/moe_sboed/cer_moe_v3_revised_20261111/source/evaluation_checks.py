"""Tiny independent bank and literal scorer; never scientific evaluation."""
from common import *
from integration import make_policy
from evaluation import load_cer,simulate
from frozen_evaluator import contrast_scores

def main():
    original.setup();leases=original.acquire();torch.manual_seed(CFG['validation_seed'])
    state=restore();a=state['algorithm'];policy=make_policy(a)
    for arm in ['full','control']:
        root=O/'arms'/arm;root.mkdir(parents=True,exist_ok=True)
        torch.save(dict(actor_state=policy.state_dict(),critic_state=a.critic_net.state_dict(),fixture_only=True),root/'actor.pt')
    model,critic=load_cer('full');generator=np.random.default_rng(CFG['validation_seed'])
    with torch.no_grad(),torch.device('cuda'):
        truth=torch.tensor(generator.standard_normal((4,4)),device='cuda');noise=torch.tensor(generator.standard_normal((4,30)),device='cuda');u=torch.tensor(generator.random((4,30)),device='cuda')
        data=simulate(model,critic,truth,noise,u,collect=True)
        repeated=simulate(model,critic,truth,noise,u)
        for x,y in zip(data['actions'],repeated['actions']):torch.testing.assert_close(x,y,rtol=0,atol=0)
        for k in range(4):
            forced=simulate(model,critic,truth,noise,u,expert=k,collect=True)
            torch.testing.assert_close(forced['selected'],torch.full((4,30),k,device='cuda',dtype=torch.long),rtol=0,atol=0)
        uniform=simulate(model,critic,truth,noise,u,uniform=True,collect=True)
        torch.testing.assert_close(uniform['probabilities'],torch.full((4,30,4),.25,device='cuda'))
        negative=torch.tensor(generator.standard_normal((17,4)),device='cuda')
        bench=type('Bench',(),{'sigma':.5,'calls':0})()
        scores,_=contrast_scores(bench,data,negative,5,'cuda')
        literal=[]
        for case in range(4):
            theta=torch.cat([truth[case:case+1],negative]);ll=torch.zeros(18,device='cuda',dtype=torch.float64)
            for t in range(30):
                ac=data['actions'][t][case];y=data['observations'][t][case,0]
                distance=(theta.reshape(-1,2,2)-ac).square().sum(-1)
                mean=(.1+(distance+1e-4).reciprocal().sum(-1)).log()
                ll+=-2*(y-mean).square()-math.log(.5*math.sqrt(2*math.pi))
            literal.append(ll[0]-torch.logsumexp(ll,0)+math.log(18))
        torch.testing.assert_close(scores[:,-1],torch.stack(literal).cpu(),rtol=1e-10,atol=1e-10)
        # Full conditional suffix plumbing on two genuine simulated prefixes.
        from value_signatures import conditional_particles,suffix_values
        from evaluation import Deployment
        x=data['X'][:2,12];acts,_,_=model.distribution(x,critic)
        th,pw,ess=conditional_particles(model.belief,x,128,torch.Generator(device='cuda').manual_seed(7))
        result=suffix_values(Deployment(model,critic),x,acts,th,pw,2,torch.Generator(device='cuda').manual_seed(9))
        assert result.shape==(2,4,2) and torch.isfinite(result).all()
    put(O/'evidence/EVALUATION_CHECKS.json',dict(status='PASS',fixture_only=True,complete30step_resimulation=True,
         each_expert_all30stages=True,uniform_probabilities=True,new_critic_checkpoint_topology=True,
         bitwise_repeat_with_same_routing_uniforms=True,literal_positive_once_prior_contrast_score=True,
         posterior_conditional_suffix_plumbing=True,main_evaluation_bank_access=False))
    print('CER_EVALUATION_CHECKS_PASS',flush=True)

if __name__=='__main__':main()
