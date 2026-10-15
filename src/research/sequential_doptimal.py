"""History-only sequential local Bayesian D-optimal design.

MAP fitting uses the exact transformed prior and Gaussian observation likelihood.
Covariance uses positive Gauss-Newton curvature: this is a local Laplace
approximation, not exact EIG. Multistart selects one mode, not a mode mixture.
"""
import numpy as np
import torch
from scipy.optimize import least_squares


class SequentialDOptimal:
    def __init__(self, benchmark, batch, horizon, seed, settings):
        self.b, self.batch, self.horizon = benchmark, batch, horizon
        self.settings = settings
        self.rngs = [torch.Generator().manual_seed(seed+1000003*i) for i in range(batch)]
        self.previous = [None]*batch
        self.diagnostics = []

    def means(self, z, actions):
        theta = self.b.inverse(z)
        state = self.b.initial((len(z),))
        values = []
        for action in actions:
            mean, state = self.b.forward(theta, state, action.expand(len(z),-1))
            values.append(mean)
        return torch.cat(values,-1) if values else torch.empty(len(z),0,dtype=torch.float64)

    def prior_residual(self, z):
        if self.b.name.startswith('ieee'):
            # Logistic density is the exact transformed uniform prior.
            energy = (torch.nn.functional.softplus(z)+torch.nn.functional.softplus(-z)-2*np.log(2)).clamp_min(0)
            return z.sign()*(2*energy).sqrt()
        return z

    def fit(self, actions, observations, episode):
        d = self.b.param_dim
        target = torch.cat(observations,-1)
        def residual(x):
            z = torch.as_tensor(x,dtype=torch.float64).reshape(1,d)
            data = (self.means(z,actions)-target)/self.b.sigma
            return torch.cat((data,self.prior_residual(z)),-1)[0].numpy()
        starts = [self.previous[episode]] if self.previous[episode] is not None else []
        while len(starts)<self.settings.get('myopic_map_restarts',4):
            starts.append(self.b.transform(self.b.sample((1,),self.rngs[episode]))[0].numpy())
        fits = [least_squares(residual,x,max_nfev=self.settings.get('myopic_map_max_nfev',40),
                              ftol=1e-6,xtol=1e-6,gtol=1e-6) for x in starts]
        fit = min(fits,key=lambda r:float(r.fun@r.fun))
        if not np.isfinite(fit.x).all() or not np.isfinite(fit.jac).all():
            raise FloatingPointError('Nonfinite Myopic posterior fit')
        self.previous[episode] = fit.x.copy()
        precision = torch.from_numpy(fit.jac.T@fit.jac)+1e-8*torch.eye(d,dtype=torch.float64)
        covariance = torch.linalg.inv(precision)
        return torch.from_numpy(fit.x)[None],covariance,dict(map_converged=bool(fit.success),
                 map_nfev=int(fit.nfev),map_cost=float(fit.cost),map_restarts=len(fits))

    def scores(self, z, covariance, candidates, actions):
        # Central differences include the full carried-state dependence on z.
        d = self.b.param_dim
        eps = self.settings.get('myopic_jacobian_step',1e-4)
        offsets = eps*torch.eye(d,dtype=torch.float64)
        parameters = torch.cat((z[:,None]+offsets,z[:,None]-offsets),1).reshape(-1,d)
        theta = self.b.inverse(parameters)
        state = self.b.initial((len(parameters),))
        for action in actions:
            _,state = self.b.forward(theta,state,action.expand(len(parameters),-1))
        physical = self.b.feasible(candidates,actions,self.horizon)
        n,k = len(parameters),len(candidates)
        mean,_ = self.b.forward(theta[:,None].expand(n,k,d),state[:,None].expand(n,k,self.b.state_dim),
                               physical[None].expand(n,k,self.b.action_dim))
        mean = mean.reshape(len(z),2*d,k,self.b.obs_dim)
        jacobian = ((mean[:,:d]-mean[:,d:])/(2*eps)).permute(0,2,3,1)
        noise_scaled = jacobian@covariance@jacobian.transpose(-1,-2)/self.b.sigma**2
        gain = .5*torch.linalg.slogdet(torch.eye(self.b.obs_dim,dtype=torch.float64)+noise_scaled).logabsdet
        return gain.mean(0)

    @torch.no_grad()
    def select(self, actions, observations):
        if len(actions)!=len(observations):raise ValueError('History lengths differ')
        selected=[];details=[]
        for i in range(self.batch):
            a=[x[i:i+1].detach() for x in actions];y=[x[i:i+1].detach() for x in observations]
            if a:
                z,cov,diag=self.fit(a,y,i)
            else:
                # Prior averaging avoids the zero-gradient symmetric location mean.
                z=self.b.transform(self.b.sample((self.settings.get('myopic_prior_samples',16),),self.rngs[i]))
                scale=3. if self.b.name.startswith('ieee') else 1.
                cov=scale*torch.eye(self.b.param_dim,dtype=torch.float64)
                diag=dict(map_converged=True,map_nfev=0,map_restarts=0)
            count=self.settings.get('myopic_design_candidates',64)
            candidates=self.b.random_latent(count,self.rngs[i])
            best_score=-float('inf');best=None
            for round_index in range(self.settings.get('myopic_design_rounds',3)):
                if round_index:
                    candidates=best+(.7/round_index)*torch.randn(count,self.b.action_dim,generator=self.rngs[i],dtype=torch.float64)
                    candidates[0]=best
                scores=self.scores(z,cov,candidates,a)
                index=int(scores.argmax())
                if float(scores[index])>best_score:best=candidates[index].clone();best_score=float(scores[index])
            if not np.isfinite(best_score):raise FloatingPointError('Nonfinite Myopic design utility')
            selected.append(best)
            details.append(diag|dict(approximate_one_step_information=best_score,posterior_mean_z=z.mean(0).tolist(),
                           posterior_covariance_z=cov.tolist()))
        self.diagnostics.append(dict(stage=len(actions),implementation='sequential_bayesian_d_optimal',
                  objective='local_linear_Gaussian_information_surrogate',truth_excluded=True,episodes=details))
        return torch.stack(selected)
