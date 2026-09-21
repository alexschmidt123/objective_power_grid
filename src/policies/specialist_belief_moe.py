"""Independent belief experts with an antisymmetric paired-utility router."""
import torch
from src.policies.belief_moe import BeliefMoEPolicy

ARCHITECTURE='specialist_belief_moe_v5'


def encoder(width,hidden):
    return torch.nn.Sequential(torch.nn.Linear(width,hidden),torch.nn.Tanh(),
        torch.nn.Linear(hidden,hidden),torch.nn.Tanh())


class SpecialistBeliefMoE(torch.nn.Module):
    def __init__(self,particle_width,context_width,n_experts=2,hidden=64):
        super().__init__()
        if n_experts!=2:raise ValueError('Paired router currently supports exactly two experts')
        self.n_experts=n_experts
        self.particle_width,self.context_width=particle_width,context_width
        self.register_buffer('regime_centers',torch.zeros(2,particle_width//2))
        self.register_buffer('regimes_initialized',torch.tensor(False))
        width=2*hidden+context_width
        self.expert_encoders=torch.nn.ModuleList([encoder(particle_width,hidden) for _ in range(n_experts)])
        self.experts=torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.Linear(width,hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,hidden),
            torch.nn.Tanh(),torch.nn.Linear(hidden,1)) for _ in range(n_experts)])
        self.router_encoder=encoder(particle_width,hidden)
        self.router=torch.nn.Sequential(torch.nn.Linear(width+2,hidden),torch.nn.Tanh(),
            torch.nn.Linear(hidden,hidden),torch.nn.Tanh(),torch.nn.Linear(hidden,1))

    @torch.no_grad()
    def regime_owner(self,inputs,stage,update=False):
        # Only posterior M/K estimates define the neighborhoods. No truth,
        # expert identity, action target, or hand-picked physical threshold.
        tokens,weights,_=inputs
        points=(weights[...,None]*tokens[...,:self.particle_width//2]).sum(1)
        if stage==0:
            # The physical prior is common before any observation. Avoid
            # inventing regimes from Monte Carlo differences between priors.
            return torch.zeros(len(points),dtype=torch.long)
        if not bool(self.regimes_initialized):
            if not update:return torch.zeros(len(points),dtype=torch.long)
            axis=points.var(0,unbiased=False).argmax()
            centers=torch.stack([points[points[:,axis].argmin()],points[points[:,axis].argmax()]])
            for _ in range(8):
                owner=torch.cdist(points,centers).argmin(1)
                for k in range(2):
                    if (owner==k).any():centers[k]=points[owner==k].mean(0)
            self.regime_centers.copy_(centers)
            self.regimes_initialized.fill_(True)
        owner=torch.cdist(points,self.regime_centers).argmin(1)
        if update:
            # Slowly update the same labeled centers; no repeated relabeling.
            for k in range(2):
                if (owner==k).any():
                    self.regime_centers[k].lerp_(points[owner==k].mean(0),.05)
        return owner

    def expert_mean(self,inputs,k):
        raw=self.experts[k](BeliefMoEPolicy.aggregate(self.expert_encoders[k],inputs)).squeeze(-1)
        return 4*torch.tanh(raw/4)

    def distributions(self,inputs):
        means=torch.stack([self.expert_mean(inputs,k) for k in range(self.n_experts)],-1)
        return torch.distributions.Normal(means,torch.full_like(means,.35))

    def gap(self,inputs,latent):
        features=BeliefMoEPolicy.aggregate(self.router_encoder,inputs)
        actions=torch.sigmoid(latent)
        # Swapping proposals must negate the prediction, and identical proposals
        # must have zero value difference. Expert identity cannot create a bias.
        forward=self.router(torch.cat([features,actions],-1)).squeeze(-1)
        reverse=self.router(torch.cat([features,actions.flip(-1)],-1)).squeeze(-1)
        return .5*(forward-reverse)

    def q_values(self,inputs,latent):
        delta=self.gap(inputs,latent)
        return torch.stack([.5*delta,-.5*delta],-1)

    def choose(self,inputs,*,stochastic=False,exploration=.15):
        distribution=self.distributions(inputs)
        with torch.no_grad():
            values=self.q_values(inputs,distribution.mean)
            selected=values.argmax(-1)
            if stochastic:
                selected=torch.where(torch.rand(len(selected))<exploration,
                    torch.randint(2,selected.shape),selected)
        all_latent=distribution.sample() if stochastic else distribution.mean
        latent=all_latent.gather(1,selected[:,None]).squeeze(1)
        logp=distribution.log_prob(latent[:,None]).gather(1,selected[:,None]).squeeze(1)
        return latent,logp,selected,values,distribution.mean

    def expert_parameters(self,k):
        return list(self.expert_encoders[k].parameters())+list(self.experts[k].parameters())

    def router_parameters(self):
        return list(self.router_encoder.parameters())+list(self.router.parameters())
