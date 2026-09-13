"""Ordered online control objectives. No equilibrium observation/control banks."""
import numpy as np
import torch
from scipy.special import logsumexp
from src.domains.swing.continuous import ContinuousParticleBelief
from src.objectives.eig.continuous_eig import OnlineEIG
from src.control.continuous_control import CarriedStateControl, continuous_decision


class OnlineControl(OnlineEIG):
    def __init__(self,cfg,observer,*,objective='msc',**kwargs):
        super().__init__(cfg,observer,**kwargs)
        self.objective=objective
        self.control=CarriedStateControl(cfg,observer.sim)

    def stage_scores(self,theta,states,likelihood):
        # Column zero generates observations only. Exclude evaluator truth from
        # the controller's posterior support during training AND evaluation.
        weights=np.exp(likelihood[:,1:]-logsumexp(likelihood[:,1:],axis=1,keepdims=True))
        batch,count=weights.shape
        posterior_theta=theta[:,1:].reshape(-1,theta.shape[-1])
        posterior_states=states[:,1:].reshape(-1,states.shape[-1])
        req=self.control.requirements(posterior_theta,posterior_states,allow_infeasible=self.objective=='msc').reshape(batch,count)
        decisions=[continuous_decision(r,w,self.control.coverage,self.objective) for r,w in zip(req,weights)]
        chosen=np.repeat([r['u_ctrl'] for r in decisions],count)
        _,_,safe=self.control.metrics(posterior_theta,posterior_states,chosen)
        masses=(weights*safe.reshape(batch,count)).sum(axis=1)
        if np.any(masses<self.control.coverage-1e-12):
            raise ValueError('Continuous posterior control failed direct safety verification')
        for row,mass in zip(decisions,masses):
            row['posterior_safe_mass']=float(mass)
            row['control_solver_tolerance']=self.control.tolerance
        key='msc' if self.objective=='msc' else 'posterior_mocu'
        return -np.asarray([r[key] for r in decisions]), decisions

    def improve(self,policy,optimizer,rng,batch,**kwargs):
        from src.objectives.msc.continuous_numerical import improve_numerical
        return improve_numerical(self,policy,optimizer,rng,batch,
            scale=self.numerical_gradient_scale,directions=self.numerical_gradient_directions,**kwargs)

    def belief(self,particles):
        return ControlParticleBelief(self.observer,particles,self.sigma,self.control,self.objective)


class ControlParticleBelief(ContinuousParticleBelief):
    def __init__(self,observer,particles,sigma,control,objective):
        super().__init__(observer,particles,sigma)
        self.control,self.objective=control,objective

    def expected_design_utility(self,duration,*,noise_samples,rng):
        prediction=self.predict(duration)
        means=prediction.observations
        indices=rng.choice(len(means),size=int(noise_samples),p=np.exp(self.log_weights))
        y=means[indices]+self.sigma*rng.normal(size=(len(indices),self.observer.n_obs))
        loglike=-.5*np.sum(((y[:,None]-means[None])/self.sigma)**2,axis=-1)
        logw=loglike+self.log_weights
        weights=np.exp(logw-logsumexp(logw,axis=1,keepdims=True))
        req=self.control.requirements(self.particles,prediction.terminal_state,allow_infeasible=self.objective=='msc')
        scores=[]
        for w in weights:
            row=continuous_decision(req,w,self.control.coverage,self.objective)
            scores.append(-row['msc' if self.objective=='msc' else 'posterior_mocu'])
        return float(np.mean(scores))
