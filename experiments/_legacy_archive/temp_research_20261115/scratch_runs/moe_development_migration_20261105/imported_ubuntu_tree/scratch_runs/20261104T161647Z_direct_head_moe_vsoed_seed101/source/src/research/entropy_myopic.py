"""History-only Myopic: expected one-step finite-posterior entropy reduction.

Matches the SIR VectorEIGEngine criterion, with continuous candidate search.
Prior support is fixed per episode; no MAP fit, truth injection, resampling,
posterior rejuvenation, learned policy or lookahead. Entropy is categorical
entropy of the declared finite posterior approximation, not differential entropy.
"""
import torch


def posterior_entropy(log_weights):
    logp=log_weights.log_softmax(-1)
    return -(logp.exp()*logp).sum(-1)


def expected_entropy_reduction(log_weights,likelihood):
    """likelihood: candidates x fantasies x hypotheses (one episode).

    Fantasy observations must be sampled from the CURRENT posterior predictive.
    No self-normalized likelihood-ratio objective or local curvature surrogate.
    """
    return posterior_entropy(log_weights)-posterior_entropy(log_weights[None,None]+likelihood).mean(-1)


class EntropyMyopic:
    implementation='one_step_posterior_entropy_reduction_v1'
    def __init__(self,benchmark,batch,horizon,seed,settings):
        self.b,self.batch,self.horizon,self.settings=benchmark,batch,horizon,settings
        count=int(settings['planner_particles'])
        if count<2:raise ValueError('Myopic needs at least two posterior hypotheses')
        self.rngs=[torch.Generator().manual_seed(int(seed)+1000003*i) for i in range(batch)]
        self.theta=[benchmark.sample((count,),rng) for rng in self.rngs]
        self.states=[benchmark.initial((count,)) for _ in range(batch)]
        self.log_weights=[torch.full((count,),-torch.log(torch.tensor(float(count),dtype=torch.float64)),dtype=torch.float64) for _ in range(batch)]
        self.assimilated=0;self.diagnostics=[]
    def assimilate(self,actions,observations):
        while self.assimilated<len(actions):
            j=self.assimilated
            for i in range(self.batch):
                theta=self.theta[i];a=actions[j][i:i+1].detach();y=observations[j][i:i+1].detach()
                means,state=self.b.forward(theta,self.states[i],a.expand(len(theta),-1))
                self.states[i]=state
                self.log_weights[i]=(self.log_weights[i]+self.b.log_likelihood(y,means)).log_softmax(-1)
            self.assimilated+=1
    def candidate_scores(self,episode,candidates,actions):
        theta=self.theta[episode];state=self.states[episode];lw=self.log_weights[episode];rng=self.rngs[episode]
        fantasies=int(self.settings['fantasies'])
        if fantasies<1:raise ValueError('Myopic needs positive fantasy count')
        physical=self.b.feasible(candidates,actions,self.horizon)
        # One shared posterior index/noise set pairs candidates for a lower-noise comparison.
        ids=torch.multinomial(lw.exp(),fantasies,replacement=True,generator=rng)
        noise=torch.randn(fantasies,self.b.obs_dim,generator=rng,dtype=torch.float64)*self.b.sigma
        scores=[]
        chunk=int(self.settings.get('myopic_candidate_chunk',8))
        for start in range(0,len(physical),chunk):
            designs=physical[start:start+chunk];c=len(designs);p=len(theta)
            means,_=self.b.forward(theta[None].expand(c,p,-1),state[None].expand(c,p,self.b.state_dim),designs[:,None].expand(c,p,-1))
            hypothetical=means[:,ids]+noise[None]
            ll=self.b.log_likelihood(hypothetical[:,:,None],means[:,None])
            scores.append(expected_entropy_reduction(lw,ll))
        return torch.cat(scores)
    @torch.no_grad()
    def select(self,actions,observations):
        if len(actions)!=len(observations) or len(actions)<self.assimilated:raise ValueError('Invalid Myopic history')
        self.assimilate(actions,observations)
        selected=[];details=[]
        count=int(self.settings['search_candidates']);rounds=int(self.settings['search_rounds'])
        if count<1 or rounds<1:raise ValueError('Myopic search budget must be positive')
        for i in range(self.batch):
            history=[a[i:i+1].detach() for a in actions]
            rng=self.rngs[i];best=None;best_score=-torch.inf
            for round_index in range(rounds):
                candidates=(self.b.random_latent(count,rng) if round_index==0 else
                            best+(.7/round_index)*torch.randn(count,self.b.action_dim,generator=rng,dtype=torch.float64))
                if round_index:candidates[0]=best
                scores=self.candidate_scores(i,candidates,history)
                if not torch.isfinite(scores).all():raise FloatingPointError('Nonfinite Myopic entropy gains')
                index=int(scores.argmax())
                # Rescore incumbent each round; never compare stale fantasy estimates.
                best=candidates[index].clone();best_score=float(scores[index])
            lw=self.log_weights[i];ess=float(lw.exp().square().sum().reciprocal())
            details.append(dict(expected_one_step_entropy_reduction=best_score,
                posterior_entropy=float(posterior_entropy(lw)),posterior_ess=ess,
                posterior_max_weight=float(lw.exp().max()),posterior_particles=len(lw),
                posterior_degenerate=ess<2.,fantasies=int(self.settings['fantasies']),
                search_candidates=count,search_rounds=rounds,
                entropy_definition='categorical entropy of fixed prior hypotheses',
                posterior_approximation='fixed prior support, exact history likelihood reweighting'))
            selected.append(best)
        self.diagnostics.append(dict(stage=len(actions),implementation=self.implementation,
            objective='expected_one_step_posterior_entropy_reduction',truth_excluded=True,episodes=details))
        return torch.stack(selected)
