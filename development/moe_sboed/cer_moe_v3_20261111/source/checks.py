"""Independent mathematical fixtures plus real native shape/likelihood checks."""
from common import *
from policy import CERPolicy, expected_loss
from value_signatures import gmm_log, prefix_log_likelihood, terminal_log_q, conditional_particles

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actions=torch.nn.Parameter(torch.tensor([[-1.,.2],[.4,.6],[.7,-.8],[1.2,1.1]],dtype=torch.float64))
        self.log_temperature=torch.nn.Parameter(torch.tensor(.1,dtype=torch.float64))
        self.epsilon=.1
    def distribution(self,x,critic,uniform=False):
        acts=self.actions[None].expand(len(x),-1,-1)
        q=critic(torch.cat([x[:,None].expand(-1,4,-1),acts],-1))[...,0]
        p=torch.full_like(q,.25) if uniform else .9*(q/self.log_temperature.exp()).softmax(-1)+.025
        return acts,q,p

def gradient_fixture():
    toy=Toy();x=torch.tensor([[.3],[-.4]],dtype=torch.float64)
    critic=lambda z:(-(z[...,-2]-.5*z[...,0]).square()-.7*z[...,-1].square())[...,None]
    loss=expected_loss(toy,critic,x)
    grads=torch.autograd.grad(loss,tuple(toy.parameters()))
    eps=1e-6
    for p,g in zip(toy.parameters(),grads):
        for i in range(p.numel()):
            original_value=float(p.flatten()[i])
            with torch.no_grad():p.flatten()[i]=original_value+eps
            plus=float(expected_loss(toy,critic,x))
            with torch.no_grad():p.flatten()[i]=original_value-eps
            minus=float(expected_loss(toy,critic,x))
            with torch.no_grad():p.flatten()[i]=original_value
            assert abs((plus-minus)/(2*eps)-float(g.flatten()[i]))<1e-7
    acts,q,p=toy.distribution(x,critic)
    # Independent score-function identity, including exploration normalization.
    dlog=[torch.autograd.grad(p[:,k].log().sum(),toy.log_temperature,retain_graph=True)[0] for k in range(4)]
    score=sum(float((p[:,k]*q[:,k]).mean())*dlog[k]/len(x) for k in range(4))
    # x affects Q; compute per-row identity directly rather than a pooled shortcut.
    score=0
    for i in range(len(x)):
        for k in range(4):
            score=score+(p[i,k]*q[i,k]).detach()*torch.autograd.grad(p[i,k].log(),toy.log_temperature,retain_graph=True)[0]/len(x)
    torch.testing.assert_close(score,-grads[1],rtol=1e-10,atol=1e-10)
    assert float(p.min())>=.025
    generator=torch.Generator().manual_seed(101)
    draws=torch.multinomial(p[0].detach().expand(200000,-1),1,generator=generator).flatten()
    frequency=torch.bincount(draws,minlength=4)/len(draws)
    torch.testing.assert_close(frequency.double(),p[0].detach(),rtol=0,atol=.004)

def main():
    original.setup();gradient_fixture()
    # CPU fixtures must not compete with a training GPU.
    state=restore('cpu');a=state['algorithm'];parent=a.actor_net
    from src.research.vendor.vsoed.utils import Net
    native=Net([117,256,256,256,2],torch.nn.ReLU,torch.tensor([[-4.,4.]]*2),'actor')
    native.load_state_dict(torch.load(OLD/'inputs/vsoed.pt',map_location='cpu',weights_only=False)['actor_state'])
    policy=CERPolicy(native,parent.belief)
    with torch.no_grad():
        for t in range(30):
            ds=torch.randn(3,30,2);ys=torch.randn(3,30,1)
            tt=torch.full((3,),t,dtype=torch.long);x=prefix_inputs(ds,ys,tt)
            altered_ds=ds.clone();altered_ds[:,t:]=float('nan')
            altered_ys=ys.clone();altered_ys[:,t:]=float('nan')
            # Multiplication masking cannot erase NaNs, so use finite adversarial data.
            altered_ds[:,t:]=999;altered_ys[:,t:]=-999
            torch.testing.assert_close(x,prefix_inputs(altered_ds,altered_ys,tt),rtol=0,atol=0)
            torch.testing.assert_close(x,a.form_actor_input(t,ds[:,:t],ys[:,:t]),rtol=0,atol=0)
            acts,q,p=policy.distribution(x,a.critic_net)
            assert acts.shape==(3,4,2) and torch.isfinite(acts).all() and bool((acts.abs()<=4).all())
            for k in range(4):torch.testing.assert_close(acts[:,k],native(x),rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(p.sum(-1),torch.ones(3))
            action,info=policy.select(x,a.critic_net,u=torch.tensor([.1,.5,.9]))
            torch.testing.assert_close(action,acts[torch.arange(3),info['selected']],rtol=0,atol=0)
        # Literal Gaussian likelihood and GMM log-density independently evaluated.
        th=torch.randn(3,11,4);w,mu,sd=parent.belief.params(x)
        direct=torch.distributions.MixtureSameFamily(torch.distributions.Categorical(w),torch.distributions.Independent(torch.distributions.Normal(mu,sd),1))
        torch.testing.assert_close(gmm_log(th,w,mu,sd),direct.log_prob(th.transpose(0,1)).T,rtol=1e-5,atol=1e-5)
        literal=torch.zeros(3,11)
        for t in range(29):
            m=mean_signal(th,ds[:,t:t+1])[:,0]
            literal+=-2*(ys[:,t,0,None]-m).square()
        torch.testing.assert_close(prefix_log_likelihood(x,th),literal,rtol=1e-5,atol=1e-5)
        terminal=terminal_log_q(parent.belief,ds,ys,th[:,0])
        assert terminal.shape==(3,) and torch.isfinite(terminal).all()
        # Prior stage: target/proposal are both exactly N(0,I); ESS must be N.
        x0=prefix_inputs(ds,ys,torch.zeros(3,dtype=torch.long))
        _,pw,ess=conditional_particles(parent.belief,x0,128,torch.Generator().manual_seed(123))
        torch.testing.assert_close(ess,torch.full((3,),128.),rtol=1e-4,atol=1e-4)
    a.critic_net.requires_grad_(False)
    # Distinct gradients once heads leave their identical initialization.
    with torch.no_grad():
        for k,head in enumerate(policy.experts):head[-1].bias.add_(.01*k)
    loss=expected_loss(policy,a.critic_net,x)
    loss.backward()
    assert all(norm([p.grad for p in head.parameters()])>0 for head in policy.experts)
    assert norm([p.grad for p in policy.encoder.parameters()])>0
    assert all(p.grad is None for p in policy.belief.parameters())
    put(O/'evidence/IMPLEMENTATION_CHECKS.json',dict(status='PASS',all30steps=True,complete_physical_selection=True,
        finite_difference_and_score_identity=True,categorical_sampling_frequency=True,epsilon_in_policy_gradient=True,
        future_mutation_invariance=True,explicit_likelihood=True,prior_posterior_importance_identity=True,
        terminal_native_density_shape=True,no_shared_action_head=True,all_expert_gradients=True,
        native_training_integration='PENDING_GROUP_SUPPORT; NOT_YET_VALIDATED',training_updates=0))
    print('CER_IMPLEMENTATION_CHECKS_PASS',flush=True)

if __name__=='__main__':main()
