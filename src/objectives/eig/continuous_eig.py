"""Continuous-duration, no-reset IEEE9 EIG experiment.

Independent online likelihoods replace all equilibrium observation banks.
Uses a sequential prior-contrastive lower-bound estimator (sPCE, nats), with
the true trajectory included in the denominator. Report the contrast count:
these results must not be pooled with legacy bank entropy-reduction scores.
Continuous DAD uses deterministic pathwise gradients; RL uses REDQ.
Step-DAD refines the DAD policy using posterior predictive future simulations.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from datetime import datetime
import numpy as np
import torch
from scipy.special import logsumexp

from src.config import load_config
from src.domains.swing.continuous import ContinuousParticleBelief
from src.objectives.eig.continuous_pathwise import improve_pathwise
from src.objectives.eig.continuous_redq import REDQPolicy, REDQTrainer


def feasible_duration(unit, history, bounds, min_separation, horizon=None):
    """Map [0,1] onto the remaining intervals by their lengths, without snapping.

    Uniform unit samples are uniform over feasible duration length. The mapping
    depends only on past actions, so latent-action score gradients remain valid.
    """
    lo, hi = bounds
    unit = np.asarray(unit, dtype=float).reshape(-1)
    if not np.all(np.isfinite(unit)) or np.any((unit < 0) | (unit > 1)):
        raise ValueError('Unit action must be finite and in [0,1]')
    past = [np.broadcast_to(np.asarray(a, dtype=float), unit.shape) for a in history]
    result = np.empty_like(unit)
    gap = float(min_separation) + 1e-10
    if horizon is not None:
        lower = past[-1] + gap if past else np.full_like(unit, lo)
        upper = hi - (horizon-len(past)-1)*gap
        if len(past) >= horizon or np.any(lower > upper+1e-12):
            raise ValueError('No increasing duration remains')
        return lower + unit*np.maximum(upper-lower, 0.)
    for row, q in enumerate(unit):
        intervals = [(lo, hi)]
        for a in past:
            left, right = a[row]-gap, a[row]+gap
            remaining = []
            for start, end in intervals:
                if right <= start or left >= end:
                    remaining.append((start, end))
                else:
                    if left > start:
                        remaining.append((start, left))
                    if right < end:
                        remaining.append((right, end))
            intervals = remaining
        lengths = np.asarray([b-a for a,b in intervals])
        if not len(lengths) or lengths.sum() <= 0:
            raise ValueError('No feasible duration remains')
        distance = q*lengths.sum()
        index = min(int(np.searchsorted(np.cumsum(lengths), distance, side='right')), len(lengths)-1)
        offset = distance-lengths[:index].sum()
        result[row] = min(intervals[index][1], intervals[index][0]+offset)
    return result


class DurationPolicy(torch.nn.Module):
    def __init__(self, horizon, n_obs, *, fixed=False):
        super().__init__()
        self.horizon, self.n_obs, self.fixed = horizon, n_obs, fixed
        self.log_std = torch.nn.Parameter(torch.tensor(-.3))
        if fixed:
            self.sequence = torch.nn.Parameter(torch.zeros(horizon))
        else:
            self.stage_bias = torch.nn.Parameter(torch.zeros(horizon))
            self.network = torch.nn.Sequential(
                torch.nn.Linear(1+horizon*(n_obs+2), 64), torch.nn.Tanh(),
                torch.nn.Linear(64,64), torch.nn.Tanh(), torch.nn.Linear(64,1))
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(1+horizon*(n_obs+2),64), torch.nn.Tanh(),torch.nn.Linear(64,1))

    def forward(self, features, stage):
        mean = (self.sequence[stage].expand(len(features)) if self.fixed else
                self.network(features).squeeze(-1)+self.stage_bias[stage])
        return torch.distributions.Normal(mean, self.log_std.clamp(-3.,1.).exp()), self.critic(features).squeeze(-1)


def improve_objective(engine, policy, optimizer, rng, batch, **kwargs):
    if hasattr(engine, 'improve'):
        return engine.improve(policy, optimizer, rng, batch, **kwargs)
    return improve_pathwise(engine, policy, optimizer, rng, batch, **kwargs)


class OnlineEIG:
    def __init__(self, cfg, observer, *, horizon, sigma, contrasts, min_separation=.01):
        self.observer, self.horizon, self.sigma, self.contrasts = observer, int(horizon), float(sigma), int(contrasts)
        sw = cfg.swing
        self.lower = np.r_[sw['M_lower_nodes'],sw['K_lower_nodes']]
        self.upper = np.r_[sw['M_upper_nodes'],sw['K_upper_nodes']]
        self.n_obs = observer.n_obs
        self.min_separation = float(min_separation)
        # Conservative guarantee: any selected history leaves positive measure.
        if (not np.isfinite(self.min_separation) or self.min_separation <= 0 or
            2*(self.min_separation+1e-10)*(self.horizon-1) >= observer.bounds[1]-observer.bounds[0]):
            raise ValueError('Duration separation must be positive and leave room for every stage')
        if self.horizon < 1 or not np.isfinite(self.sigma) or self.sigma <= 0 or self.contrasts < 1:
            raise ValueError('Positive horizon, sigma and contrast count required')

    def sample(self, rng, batch):
        theta = rng.uniform(self.lower,self.upper,size=(batch,self.contrasts+1,len(self.lower)))
        states = self.observer.initial_state(batch*(self.contrasts+1)).reshape(theta.shape)
        return theta,states

    def features(self, actions, observations, batch, stage):
        features = np.zeros((batch,1+self.horizon*(self.n_obs+2)),dtype=np.float32)
        features[:,0] = stage/self.horizon
        lo,hi = self.observer.bounds
        for k,(action,obs) in enumerate(zip(actions,observations)):
            start = 1+k*(self.n_obs+2)
            features[:,start] = (np.asarray(action)-lo)/(hi-lo)
            features[:,start+1:start+1+self.n_obs] = np.asarray(obs)/.05
            features[:,start+1+self.n_obs] = 1.
        return torch.from_numpy(features)

    def rollout(self, policy, rng, batch, *, stochastic, initial=None,
                history=None, stage_start=0, selector=None, noise=None):
        theta,states = self.sample(rng,batch) if initial is None else initial
        theta,states = theta.copy(),states.copy()
        particles = theta.shape[1]
        actions,observations = copy.deepcopy(history) if history is not None else ([],[])
        likelihood = np.zeros((batch,particles))
        infos,logprobs,values = [],[],[]
        means_record = []
        unit_actions = []
        decisions = []
        controller_seconds=[]
        for stage in range(stage_start,self.horizon):
            features = self.features(actions,observations,batch,stage)
            if selector is not None:
                duration = np.asarray(selector(stage,actions,observations),dtype=float).reshape(batch)
            else:
                distribution,value = policy(features,stage)
                latent = distribution.sample() if stochastic else distribution.mean
                unit = torch.sigmoid((latent*getattr(policy,'unit_scale',1.)).double()).detach().numpy()
                unit_actions.append(unit)
                duration = feasible_duration(unit,
                    actions, self.observer.bounds, self.min_separation, self.horizon)
                logprobs.append(distribution.log_prob(latent.detach()))
                values.append(value)
            if (not np.all(np.isfinite(duration)) or np.any(duration < self.observer.bounds[0]) or
                np.any(duration > self.observer.bounds[1]) or
                any(np.any(np.abs(duration-a) < self.min_separation) for a in actions)):
                raise ValueError('Selected duration violates bounds or non-repetition constraint')
            if (
                (actions and np.any(duration-actions[-1] < self.min_separation)) or
                np.any(duration > self.observer.bounds[1]-(self.horizon-stage-1)*self.min_separation+1e-12)):
                raise ValueError('increasing durations require room for later probes')
            prediction = self.observer.propagate(theta.reshape(-1,theta.shape[-1]),
                states.reshape(-1,states.shape[-1]),np.repeat(duration,particles))
            means = prediction.observations.reshape(batch,particles,self.n_obs)
            states = prediction.terminal_state.reshape(states.shape)
            z = rng.normal(size=(batch,self.n_obs)) if noise is None else noise[:,stage,:]
            y = means[:,0,:]+self.sigma*z
            likelihood += -.5*np.sum(((y[:,None,:]-means)/self.sigma)**2,axis=-1)
            if hasattr(self, 'stage_scores'):
                control_start=time.monotonic()
                score, decision = self.stage_scores(theta,states,likelihood)
                controller_seconds.append(time.monotonic()-control_start)
                infos.append(score)
                decisions.append(decision)
            else:
                infos.append(likelihood[:,0]-logsumexp(likelihood,axis=1)+np.log(particles))
            actions.append(duration.copy())
            observations.append(y.copy())
            means_record.append(means[:,0,:].copy())
        return {'info':np.stack(infos,axis=1),'logprobs':logprobs,'values':values,
                'actions':actions,'observations':observations,'states':states,
                'true_means':means_record,'unit_actions':unit_actions,
                'decisions':decisions,'controller_seconds':controller_seconds,'theta':theta,'log_likelihood':likelihood}


def improve_policy(engine, policy, optimizer, rng, batch, *, critic=False,
                   initial=None, history=None, stage_start=0):
    result = engine.rollout(policy,rng,batch,stochastic=True,initial=initial,
                            history=history,stage_start=stage_start)
    info = result['info']
    previous = np.c_[np.zeros(batch),info[:,:-1]]
    returns = torch.as_tensor(info[:,-1,None]-previous,dtype=torch.float32)
    logp = torch.stack(result['logprobs'],dim=1)
    if critic:
        value = torch.stack(result['values'],dim=1)
        loss = -(logp*(returns-value.detach())).mean()+.5*((value-returns)**2).mean()
    else:
        # Leave-one-out baseline does not depend on this trajectory's action.
        baseline = (returns.sum(0,keepdim=True)-returns)/max(batch-1,1) if batch>1 else 0.
        loss = -(logp*(returns-baseline)).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(),5.)
    optimizer.step()
    return float(info[:,-1].mean())


def fixed_initializations(engine):
    """Interior ordered sequences; avoid saturated endpoint-logit initializations."""
    lo,hi=engine.observer.bounds
    gap=engine.min_separation+1e-10
    free=hi-lo-(engine.horizon-1)*gap
    candidates=[]
    for label,start,end in [('short',.05,.35),('central',.3,.7),
                            ('long',.65,.95),('spread',.05,.95)]:
        durations=lo+np.arange(engine.horizon)*gap+free*np.linspace(start,end,engine.horizon)
        history=[];logits=[]
        for k,d in enumerate(durations):
            lower=history[-1]+gap if history else lo
            upper=hi-(engine.horizon-k-1)*gap
            unit=(d-lower)/(upper-lower)
            if not 0.<unit<1.:raise ValueError('Fixed initialization must be interior')
            logits.append(float(np.log(unit/(1-unit))))
            history.append(d)
        candidates.append((label,logits))
    return candidates


def train(engine, method, args, directory, *, initial_sequence=None):
    torch.manual_seed(args.seed)
    policy = REDQPolicy(args.T,engine.n_obs) if method=='rl_sboed' else DurationPolicy(args.T,engine.n_obs,fixed=method=='fixed')
    optimizer = torch.optim.Adam(policy.parameters(),lr=args.learning_rate)
    redq = REDQTrainer(policy,args) if method=='rl_sboed' else None
    rng = np.random.default_rng(args.seed)
    records=[]
    best=-math.inf
    best_state=None
    best_update=None
    start=time.monotonic()
    # Fixed must compete with simple feasible sequences before optimization.
    # Candidate selection uses the same validation prior as checkpoint selection,
    # never evaluator systems. Retain the strongest as the update-zero checkpoint.
    initialization=[]
    if method=='dad' and initial_sequence is not None:
        # A history-independent Fixed policy is a valid member of the DAD class.
        # Compare using validation only; never choose initialization on evaluation data.
        candidates=[('random',copy.deepcopy(policy.state_dict()))]
        with torch.no_grad():
            policy.network[-1].weight.zero_()
            policy.network[-1].bias.zero_()
            # Move endpoint logits into the interior so DAD can actually learn.
            policy.stage_bias.copy_(torch.as_tensor(initial_sequence).clamp(-4.,4.))
        candidates.append(('interior_from_new_fixed',copy.deepcopy(policy.state_dict())))
        winner=None
        for label,state in candidates:
            policy.load_state_dict(state)
            with torch.no_grad():
                result=engine.rollout(policy,np.random.default_rng(args.seed+900000),
                                     args.validation_systems,stochastic=False)
            score=float(result['info'][:,-1].mean())
            initialization.append({'candidate':label,'validation_utility':score})
            if winner is None or score>winner[0]:winner=(score,copy.deepcopy(state),label)
        policy.load_state_dict(winner[1])
        (directory/'dad_initialization.json').write_text(json.dumps(
            {'candidates':initialization,'selected':winner[2]},indent=2)+'\n')
    if method=='fixed':
        candidates=fixed_initializations(engine)
        winner=None
        for label,sequence in candidates:
            with torch.no_grad():
                policy.sequence.copy_(torch.tensor(sequence))
                result=engine.rollout(policy,np.random.default_rng(args.seed+900000),
                    args.validation_systems,stochastic=False)
            score=float(result['info'][:,-1].mean())
            initialization.append({'candidate':label,'validation_utility':score,
                'durations_s':[float(a[0]) for a in result['actions']]})
            if winner is None or score>winner[0]:
                winner=(score,copy.deepcopy(policy.state_dict()))
        policy.load_state_dict(winner[1])
        (directory/'fixed_initialization.json').write_text(json.dumps(initialization,indent=2)+'\n')
    for update in range(args.updates+1):
        training_score = None
        if update:
            training_score=(redq.step(engine,rng,args.batch_size) if redq is not None else
                            improve_objective(engine,policy,optimizer,rng,args.batch_size))
        if update % args.validate_every == 0 or update==args.updates:
            with torch.no_grad():
                validation=engine.rollout(policy,np.random.default_rng(args.seed+900000),
                                          args.validation_systems,stochastic=False)
            score=float(validation['info'][:,-1].mean())
            records.append({'update':update,'training_utility':training_score,'validation_utility':score})
            if redq is not None:
                records[-1]['redq']=dict(redq.last_diagnostics)
            print(f'[online-design] {method} update={update} validation_utility={score:.6f}',flush=True)
            if score>best:
                best,best_state=score,copy.deepcopy(policy.state_dict())
                best_update=update
                temporary=directory/(method+'.pth.tmp')
                torch.save({'state_dict':best_state,'method':method,'horizon':args.T,'n_obs':engine.n_obs,
                            'protocol':protocol_name(args),'settings':vars(args)},temporary)
                os.replace(temporary,directory/(method+'.pth'))
            (directory/(method+'_training.json')).write_text(json.dumps(records,indent=2)+'\n')
    policy.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'method':method,'horizon':args.T,'n_obs':engine.n_obs,
                'protocol':protocol_name(args),'settings':vars(args)},directory/(method+'.pth'))
    (directory/(method+'_training.json')).write_text(json.dumps(records,indent=2)+'\n')
    diagnostic={'best_update':best_update,'best_validation_utility':best,
        'completed_updates':args.updates,'validation_systems':args.validation_systems,
        'last_validation_utility':records[-1]['validation_utility'],
        'recent_validation_change':records[-1]['validation_utility']-records[max(0,len(records)-6)]['validation_utility'],
        'convergence_established':False,
        'note':'Validation history is evidence to inspect, not proof of convergence; equal updates are not equal optimization quality.'}
    diagnostic['validation_gain_from_initial']=best-records[0]['validation_utility']
    (directory/(method+'_training_diagnostics.json')).write_text(json.dumps(diagnostic,indent=2)+'\n')
    return policy,time.monotonic()-start


def myopic_duration(belief,rng,args,actions=(),min_separation=.01):
    lo,hi=0.,1.
    centre,spread=(lo+hi)/2,(hi-lo)/2
    chosen=centre
    best=-math.inf
    # Continuous random-search/CEM optimizer; proposals are never snapped to a catalog.
    seed=int(rng.integers(2**31))
    for iteration in range(args.search_rounds):
        proposals = rng.uniform(lo,hi,args.search_candidates) if iteration==0 else np.clip(
            rng.normal(centre,spread,args.search_candidates),lo,hi)
        durations=feasible_duration(proposals,actions,belief.observer.bounds,min_separation,args.T)
        scorer=getattr(belief,'expected_design_utility',belief.expected_information)
        scores=np.asarray([scorer(d,noise_samples=args.fantasies,
                           rng=np.random.default_rng(seed)) for d in durations])
        k=int(np.argmax(scores))
        if scores[k]>best:
            best,chosen=float(scores[k]),float(durations[k])
        elite=proposals[np.argsort(scores)[-max(2,len(proposals)//4):]]
        centre,spread=float(elite.mean()),max(float(elite.std()),(hi-lo)*.01)
    return chosen


def refine(engine,dad,belief,actions,observations,stage,args,rng):
    policy=copy.deepcopy(dad)
    optimizer=torch.optim.Adam(policy.parameters(),lr=args.learning_rate)
    # All fantasy truths and alternatives are sampled from the planner's own
    # posterior. The held-out evaluator truth is neither accepted nor used here.
    def conditional_batch(generator):
        indices=generator.choice(len(belief.particles),
            size=(args.batch_size,args.contrasts+1),p=np.exp(belief.log_weights))
        initial=(belief.particles[indices],belief.states[indices])
        history=([np.repeat(a[0],args.batch_size) for a in actions],
                 [np.repeat(y,args.batch_size,axis=0) for y in observations])
        return initial,history
    validation_rng=np.random.default_rng(int(rng.integers(2**31)))
    validation_initial,validation_history=conditional_batch(validation_rng)
    val_seed=int(validation_rng.integers(2**31))
    def score(candidate):
        with torch.no_grad():
            r=engine.rollout(candidate,np.random.default_rng(val_seed),args.batch_size,
                stochastic=False,initial=validation_initial,history=validation_history,stage_start=stage)
        return float(r['info'][:,-1].mean())
    best=score(policy)
    best_state=copy.deepcopy(policy.state_dict())
    for _ in range(args.refinement_updates):
        initial,history=conditional_batch(rng)
        improve_objective(engine,policy,optimizer,rng,args.batch_size,initial=initial,
                       history=history,stage_start=stage)
        candidate=score(policy)
        if candidate>best:
            best,best_state=candidate,copy.deepcopy(policy.state_dict())
    policy.load_state_dict(best_state)
    return policy


def evaluate(engine,policies,args,record_path=None):
    rows=[]
    methods=args.methods.split(',')
    for method in methods:
        start=time.monotonic()
        # Identical evaluator truths, contrast particles and measurement-noise
        # draws across methods. Planner RNG/support are separate from evaluation.
        rng=np.random.default_rng(args.eval_seed)
        theta,states=engine.sample(rng,args.eval_systems)
        noise=rng.normal(size=(args.eval_systems,args.T,engine.n_obs))
        for system in range(args.eval_systems):
            planner_rng=np.random.default_rng(args.eval_seed+700000+system)
            particles=planner_rng.uniform(engine.lower,engine.upper,
                                          size=(args.planner_particles,len(engine.lower)))
            belief=(engine.belief(particles) if hasattr(engine,'belief') else
                    ContinuousParticleBelief(engine.observer,particles,args.noise_sigma))
            assimilated=0
            planning_ess=[]
            decision_seconds=[]
            adapted_policy=policies.get('dad')
            def selector(stage,actions,observations):
                nonlocal assimilated,adapted_policy
                decision_start=time.monotonic()
                # Only completed stages enter the online planner.
                while method in {'myopic','step_dad'} and assimilated<len(actions):
                    belief.update(float(actions[assimilated][0]),observations[assimilated][0])
                    assimilated+=1
                if method in {'myopic','step_dad'}:
                    planning_ess.append(float(1/np.exp(2*belief.log_weights).sum()))
                if method=='random':
                    duration=feasible_duration([planner_rng.uniform()],actions,
                        engine.observer.bounds,engine.min_separation,engine.horizon)
                elif method=='myopic':
                    duration=[myopic_duration(belief,planner_rng,args,actions,engine.min_separation)]
                else:
                    policy=adapted_policy if method=='step_dad' else policies[method]
                    if method=='step_dad' and stage>0:
                        policy=refine(engine,policy,belief,actions,observations,stage,args,planner_rng)
                        adapted_policy=policy
                    with torch.no_grad():
                        distribution,_=policy(engine.features(actions,observations,1,stage),stage)
                        duration=feasible_duration(torch.sigmoid((distribution.mean*getattr(policy,'unit_scale',1.)).double()).detach().numpy(),
                            actions,engine.observer.bounds,engine.min_separation,engine.horizon)
                decision_seconds.append(time.monotonic()-decision_start)
                return duration
            with torch.no_grad() if method!='step_dad' else torch.enable_grad():
                result=engine.rollout(None,np.random.default_rng(args.eval_seed+system),1,
                    stochastic=False,initial=(theta[system:system+1],states[system:system+1]),
                    selector=selector,noise=noise[system:system+1])
            rows.append({'method':method,'system':system,'terminal_spce_nats':float(result['info'][0,-1]),
                'planning_ess_before_action':planning_ess,
                'decision_seconds_per_stage':decision_seconds,
                'duration_sequence_s':[float(a[0]) for a in result['actions']],
                'observations_hz':[y[0].tolist() for y in result['observations']],
                'true_terminal_state':result['states'][0,0].tolist(),
                'evaluation_seed':args.eval_seed, 'true_MK':theta[system,0].tolist()})
            if getattr(args,'N_obs',engine.n_obs)==0:
                rows[-1]['observations_rocof_hz_s']=rows[-1].pop('observations_hz')
            if hasattr(engine,'control'):
                row=rows[-1]
                row.pop('terminal_spce_nats')
                row.update(result['decisions'][-1][0])
                row['terminal_controller_seconds']=result['controller_seconds'][-1]
                row['counterfactual_control_scoring_seconds']=sum(result['controller_seconds'][:-1])
                decision_start=time.monotonic()
                r,f,safe=engine.control.metrics(theta[system,:1],result['states'][0,:1], [row['u_ctrl']])
                row.update(safe=bool(safe[0]),rocof_max_hz_s=float(r[0]),delta_f_nadir_hz=float(f[0]))
                row.update(engine.control.oracle(theta[system,0],result['states'][0,0]))
                row['evaluation_safety_oracle_seconds']=time.monotonic()-decision_start
                logw=result['log_likelihood'][0,1:]
                row['posterior_weights']=np.exp(logw-logsumexp(logw)).tolist()
                row['posterior_MK']=result['theta'][0,1:].tolist()
                row['posterior_terminal_states']=result['states'][0,1:].tolist()
            if record_path is not None:
                with Path(record_path).open('a') as stream:
                    stream.write(json.dumps(rows[-1])+'\n')
        print(f'[online-design] evaluated {method}: {time.monotonic()-start:.1f}s',flush=True)
    return rows


def protocol_name(args):
    objective=getattr(args,'objective','eig')
    if getattr(args,'observation_kind',None)=='endpoint_rocof':
        return f'continuous_no_reset_increasing_{objective}_endpoint_rocof_v1'
    return 'continuous_no_reset_increasing_spce_v5' if objective=='eig' else f'continuous_no_reset_increasing_{objective}_v3'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/ieee9_eig.yaml')
    p.add_argument('--objective',choices=['eig','msc','mocu'],default=None)
    p.add_argument('--experiment_type','--experiment-type',choices=['eig_based','msc_based','objective_based'],default=None)
    p.add_argument('--coverage',type=float,default=None)
    p.add_argument('--numerical-gradient-scale',type=float,default=.01)
    p.add_argument('--numerical-gradient-directions',type=int,default=4)
    p.add_argument('--output',default=None)
    p.add_argument('--T',type=int,default=3)
    p.add_argument('--N-obs','--N_obs',dest='N_obs',type=int,default=0)
    p.add_argument('--noise-sigma','--noise_sigma',dest='noise_sigma',type=float,default=.005)
    p.add_argument('--duration-min',type=float,default=.2)
    p.add_argument('--duration-max',type=float,default=3.)
    p.add_argument('--min-duration-separation',type=float,default=.01,
                   help='Minimum pairwise separation in seconds; applies to every method and stage')
    p.add_argument('--bus',type=int,default=1,help='One-based physical injection bus')
    p.add_argument('--amplitude',type=float,default=.05)
    p.add_argument('--window',type=float,default=3.)
    p.add_argument('--observation-kind',choices=['max_absolute_rocof','endpoint_rocof','sampled_frequency'],default=None,
                   help='Explicit sensor variant; endpoint_rocof reports signed RoCoF at window end')
    p.add_argument('--seed',type=int,default=101)
    p.add_argument('--eval-seed',type=int,default=1001)
    p.add_argument('--eval-seeds',default=None)
    p.add_argument('--methods','--method',default='dad,rl_sboed,step_dad,myopic,fixed,random')
    p.add_argument('--updates',type=int,default=2000)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--contrasts',type=int,default=128)
    p.add_argument('--validation-systems',type=int,default=64)
    p.add_argument('--validate-every',type=int,default=100)
    p.add_argument('--eval-systems',type=int,default=128)
    p.add_argument('--planner-particles',type=int,default=128)
    p.add_argument('--search-candidates',type=int,default=16)
    p.add_argument('--search-rounds',type=int,default=3)
    p.add_argument('--fantasies',type=int,default=32)
    p.add_argument('--refinement-updates',type=int,default=16)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--redq-critics',type=int,default=10)
    p.add_argument('--redq-updates-per-batch',type=int,default=5)
    p.add_argument('--preflight-max-hours',type=float,default=0.,
                   help='Measure this host before training; stop if padded projected total exceeds this budget')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--estimate-only',action='store_true',help='Bounded labpc timing diagnostic; no formal training/evaluation')
    args=p.parse_args()
    from src.config import resolve_config_path
    args.config=str(resolve_config_path(args.config))
    requested={'eig_based':'eig','msc_based':'msc','objective_based':'mocu'}.get(args.experiment_type)
    if args.objective is not None and requested is not None and args.objective!=requested:
        p.error('Objective and experiment_type disagree')
    args.objective=args.objective or requested or {'eig_based':'eig','msc_based':'msc','objective_based':'mocu'}[load_config(args.config).raw['experiment']['experiment_type']]
    if args.output is None:
        now=datetime.now()
        sigma_token=format(args.noise_sigma,'.12g').replace('.','p')
        window_token=format(args.window,'.12g').replace('.','p')
        label=(now.strftime('%m%d%Y_%H%M%S_%f')+
            f'_ieee9_nonreset_{args.objective}_T{args.T}_Nobs{args.N_obs}_sigma{sigma_token}_W{window_token}_train{args.seed}')
        args.output=str(Path(__file__).resolve().parents[3]/'experiments'/now.strftime('%m%d%Y')/label)
    eval_seeds=[int(x) for x in args.eval_seeds.split(',')] if args.eval_seeds else [args.eval_seed]
    if not eval_seeds or len(set(eval_seeds))!=len(eval_seeds):p.error('Use distinct evaluation seeds')
    if args.N_obs<0:p.error('N_obs must be nonnegative')
    args.observation_kind=args.observation_kind or ('max_absolute_rocof' if args.N_obs==0 else 'sampled_frequency')
    if (args.observation_kind=='sampled_frequency') != (args.N_obs>0):
        p.error('RoCoF variants require N_obs=0; sampled_frequency requires positive N_obs')
    for key in ['T','updates','batch_size','contrasts','validation_systems','validate_every',
                'eval_systems','planner_particles','search_candidates','search_rounds','fantasies','refinement_updates']:
        if getattr(args,key)<1:p.error(key+' must be positive')
    if args.redq_critics<2 or args.redq_updates_per_batch<1:p.error('Invalid REDQ budgets')
    if args.smoke:
        args.updates,args.batch_size,args.contrasts=2,4,8
        args.validation_systems,args.validate_every,args.eval_systems=4,1,2
        args.planner_particles,args.search_candidates,args.search_rounds=8,4,1
        args.fantasies,args.refinement_updates=4,1
        args.numerical_gradient_directions=1
    if len(set(args.methods.split(',')))!=len(args.methods.split(',')):p.error('Duplicate methods')
    if args.numerical_gradient_scale<=0 or args.numerical_gradient_directions<1:p.error('Invalid numerical gradient settings')
    if not set(args.methods.split(',')) <= {'dad','rl_sboed','step_dad','myopic','fixed','random'}:
        p.error('Unsupported method')
    root=Path(__file__).resolve().parents[3]
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    import sys,traceback
    original_hook=sys.excepthook
    def record_failure(kind,value,tb):
        (output/'failure.json').write_text(json.dumps({'exception':kind.__name__,'message':str(value),
            'traceback':''.join(traceback.format_exception(kind,value,tb))},indent=2)+'\n')
        (output/'exit_code').write_text('1\n')
        original_hook(kind,value,tb)
    sys.excepthook=record_failure
    run_started=time.monotonic()
    cfg=load_config(args.config)
    cfg.raw=copy.deepcopy(cfg.raw)
    for key in ['poster_eig_protocol','poster_mocu_protocol','bank_quality','control_safety_calibration']:
        cfg.raw.pop(key,None)
    for key in ['alpha','enforce_bayes_loss_alignment']:
        cfg.raw.get('control',{}).pop(key,None)
    for block in cfg.raw.get('training',{}).values():
        if isinstance(block,dict):
            block.pop('undercontrol_penalty',None);block.pop('violation_penalty',None)
    if cfg.swing.get('reset_after_probe',False):
        raise ValueError('Reset settings are retired; use the canonical non-reset configuration')
    if args.coverage is not None:
        cfg.raw.setdefault('control',{})['posterior_coverage']=args.coverage
    if args.objective!='eig':
        if 'control' not in cfg.raw or 'posterior_coverage' not in cfg.raw['control']:
            raise ValueError('Control objective requires a declared control scenario and coverage')
        args.coverage=float(cfg.raw['control']['posterior_coverage'])
        if not 0 < args.coverage < 1: raise ValueError('Coverage must be in (0,1)')
    cfg.raw['swing_equation'].update(reset_after_probe=False,T_obs_sec=args.window,
        probe_buses=[args.bus-1],probe_amplitudes=[args.amplitude],
        probe_duration_bounds=[args.duration_min,args.duration_max])
    cfg.raw['swing_equation'].pop('probe_durations',None)
    cfg.raw['data']={'observation_backend':'online_continuous_swing','uses_probe_bank':False,'uses_control_bank':False}
    cfg.raw['experiment']={'mode':'continuous_duration_no_reset','experiment_type':{'eig':'eig_based','msc':'msc_based','mocu':'objective_based'}[args.objective],
                           'step_number':args.T,'methods':args.methods.split(',')}
    from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver
    observer_kwargs=dict(duration_bounds=(args.duration_min,args.duration_max),
        injection_bus=args.bus,amplitude=args.amplitude,window=args.window)
    if args.N_obs==0:
        from src.domains.swing.continuous_rocof import MaxRocofObserver,EndpointRocofObserver
        observer_class=EndpointRocofObserver if args.observation_kind=='endpoint_rocof' else MaxRocofObserver
        observer=observer_class(cfg,**observer_kwargs)
    else:
        observer=CudaContinuousSwingObserver(cfg,n_obs=args.N_obs,**observer_kwargs)
    if args.objective=='eig':
        engine=OnlineEIG(cfg,observer,horizon=args.T,sigma=args.noise_sigma,contrasts=args.contrasts,
                         min_separation=args.min_duration_separation)
    else:
        from src.objectives.msc.continuous_msc import OnlineControl
        engine=OnlineControl(cfg,observer,objective=args.objective,horizon=args.T,sigma=args.noise_sigma,
                             contrasts=args.contrasts,min_separation=args.min_duration_separation)
    if hasattr(engine,'control'):
        engine.numerical_gradient_scale=args.numerical_gradient_scale
        engine.numerical_gradient_directions=args.numerical_gradient_directions
    cfg.raw['observation']={'N_obs':args.N_obs, 'noise_sigma':args.noise_sigma,
        'dimension':observer.n_obs,
        'sampling':{'max_absolute_rocof':'max_absolute_rocof_over_fixed_recording_window',
                    'endpoint_rocof':'signed_rocof_at_window_end',
                    'sampled_frequency':'uniform_within_fixed_recording_window'}[args.observation_kind],
        'noise_units':'Hz/s' if args.N_obs==0 else 'Hz'}
    from src.hardware import hardware_info
    metadata={'protocol':protocol_name(args),'settings':vars(args),'hardware':hardware_info(),
        'comparison_revision':'ordered_pathwise_dad_redq_stepdad_v1',
        'gradient_backend':'pathwise chain rule; numerical state/duration Jacobians (1e-5); max-RoCoF uses the primal active sample for its branch derivative',
        'history_representation':'ordered stage slots; no permutation pooling',
        'duration_order':'strictly_increasing',
        'method_implementations':{'dad':'deterministic continuous DAD with pathwise sPCE gradients',
            'rl_sboed':'REDQ-style off-policy actor, replay and critic ensemble; telescoping sPCE rewards; gamma=1',
            'step_dad':'prefix-conditioned pathwise refinement after each observation; warm-start previous adapted policy',
            'fixed':'validation-selected feasible initialization followed by pathwise sequence optimization',
            'myopic':'one-step posterior-particle information search',
            'random':'uniform over remaining feasible duration intervals'},
        'publication_validation_complete':False,
        'observation_kind':args.observation_kind,
        'observation_dimension':observer.n_obs,
        'noise_units':'Hz/s' if args.N_obs==0 else 'Hz',
        'rocof_sample_dt':getattr(observer,'rocof_sample_dt',None),
        'numerical_gradient_caveat':'Max-RoCoF is piecewise smooth. The active sample is fixed in numerical Jacobians; peak ties remain nondifferentiable.' if args.observation_kind=='max_absolute_rocof' else None,
        'recording_window_s':observer.window,'observation_times_within_stage_s':observer.times.tolist(),
        'measurement_bus_physical':cfg.swing.get('observation_bus',1),
        'prior_lower':engine.lower.tolist(),'prior_upper':engine.upper.tolist(),
        'physical_config':cfg.raw,'uses_probe_bank':False,'estimator':'sequential prior-contrastive lower bound',
        'bound_ceiling_nats':math.log(args.contrasts+1),'smoke_not_performance':args.smoke,
        'source_hashes':{}}
    if args.observation_kind=='endpoint_rocof':
        metadata.update(gradient_backend='pathwise chain rule; numerical state/duration Jacobians (1e-5); signed endpoint RoCoF',
            rocof_difference_times_within_stage_s=[observer.window-observer.rocof_sample_dt,observer.window],
            stage_start_times_s=[k*observer.window for k in range(args.T)],
            observation_absolute_times_s=[(k+1)*observer.window for k in range(args.T)],
            total_simulated_time_s=args.T*observer.window,
            interstage_gap_s=0.,initial_state='equilibrium once; full state carry thereafter')
    if args.objective!='eig':
        metadata.update(protocol=protocol_name(args),estimator='independent finite-particle posterior control',
            gradient_backend='antithetic parameter-perturbation numerical gradient for deterministic DAD/Fixed/Step-DAD; REDQ for RL-sBOED',
            posterior_includes_evaluator_truth=False,uses_control_bank=False,
            terminal_control_initial_state='each particle state at end of last probe recording',
            control_contingency_timing='immediate after last probe window; zero decision latency assumed',
            control_safety_window_s=engine.control.spec.T_obs_sec,
            oracle_kind='continuous bracketed minimum for true M/K and actual terminal state')
        metadata.pop('bound_ceiling_nats')
        metadata['method_implementations'].update(
            dad='deterministic adaptive policy optimizing terminal utility; antithetic zeroth-order numerical training adaptation',
            fixed='validation-selected deterministic ordered sequence with numerical optimization',
            step_dad='posterior-prefix-conditioned numerical refinement of matching deterministic DAD',
            myopic='continuous search minimizing expected next-stage control objective',
            rl_sboed='REDQ with telescoping negative control-objective rewards; gamma=1')
    metadata['control_action_space']='continuous bounded magnitude' if hasattr(engine,'control') else None
    metadata['optimization_caveat']='MSC/MOCU numerical training estimates a Gaussian-smoothed parameter objective; validation and evaluation use the unperturbed deterministic policy and exact numerically refined controller. Check scale/direction sensitivity before publication.' if hasattr(engine,'control') else None
    for path in (root/'src').rglob('*.py'):

        metadata['source_hashes'][str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'run_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.estimate_only or args.preflight_max_hours>0:
        torch.manual_seed(args.seed)
        policy=DurationPolicy(args.T,engine.n_obs)
        optimizer=torch.optim.Adam(policy.parameters(),lr=args.learning_rate)
        rng=np.random.default_rng(args.seed)
        start=time.monotonic()
        for _ in range(3):
            improve_objective(engine,policy,optimizer,rng,args.batch_size)
        seconds=(time.monotonic()-start)/3
        redq_policy=REDQPolicy(args.T,engine.n_obs)
        redq=REDQTrainer(redq_policy,args)
        start=time.monotonic()
        train_redq='rl_sboed' in args.methods.split(',')
        if train_redq:
            for _ in range(3):redq.step(engine,rng,args.batch_size)
        redq_seconds=(time.monotonic()-start)/3 if train_redq else 0.
        timings_start=time.monotonic()
        with torch.no_grad():
            engine.rollout(policy,np.random.default_rng(args.seed+800000),args.validation_systems,stochastic=False)
        validation_seconds=time.monotonic()-timings_start
        evaluation_args=copy.copy(args)
        evaluation_args.eval_systems=1
        timings_start=time.monotonic()
        print('[preflight] Timing one simulated evaluation system per method; not performance results',flush=True)
        evaluate(engine,{'dad':policy,'rl_sboed':redq_policy,
                         'fixed':DurationPolicy(args.T,engine.n_obs,fixed=True)},evaluation_args)
        evaluation_seconds=time.monotonic()-timings_start
        train_seconds=(int(bool({'dad','step_dad'} & set(args.methods.split(','))))+
                       int('fixed' in args.methods.split(',')))*seconds+redq_seconds
        projected_hours=(args.updates*train_seconds+
            3*(args.updates//args.validate_every+1)*validation_seconds+
            len(eval_seeds)*args.eval_systems*evaluation_seconds)/3600
        estimate={'pathwise_training_update_seconds':seconds,'redq_training_update_seconds':redq_seconds,
                  'requested_training_hours_excluding_validation_and_evaluation':args.updates*train_seconds/3600,
                  'validation_batch_seconds':validation_seconds,
                  'all_methods_one_evaluation_system_seconds':evaluation_seconds,
                  'projected_total_hours':projected_hours,
                  'padded_projected_total_hours':1.25*projected_hours,
                  'note':'Three updates and one evaluation system per method; runtime extrapolation only, not convergence or performance evidence'}
        (output/'timing_estimate.json').write_text(json.dumps(estimate,indent=2)+'\n')
        print(json.dumps(estimate),flush=True)
        if args.estimate_only:return
        if estimate['padded_projected_total_hours']>args.preflight_max_hours:
            (output/'preflight_blocked.json').write_text(json.dumps(estimate,indent=2)+'\n')
            raise RuntimeError('Measured cost exceeds approved run budget; full training was not started')
        print('[preflight] Budget check passed; beginning formal training from fresh seeds',flush=True)
    snapshot=output/'source_snapshot'
    (snapshot/'configs').mkdir(parents=True,exist_ok=True)
    shutil.copy2(args.config,snapshot/'configs'/Path(args.config).name)
    for relative in metadata['source_hashes']:
        target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,target)
    for relative in ['scripts/continuous_eig.sh','scripts/continuous.sh','run.sh','sweep_run.sh']:
        target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,target)
    models=output/'models'
    models.mkdir()
    policies,times={},{}
    methods=set(args.methods.split(','))
    if 'step_dad' in methods:
        methods.add('dad')
    for method in ['fixed','dad','rl_sboed']:
        if method in methods:
            initial=(policies['fixed'].sequence.detach().clone()
                     if method=='dad' and 'fixed' in policies and args.objective=='eig' else None)
            policies[method],times[method]=train(engine,method,args,models,initial_sequence=initial)
    rows=[]
    for evaluation_seed in eval_seeds:
        evaluation_args=copy.copy(args)
        evaluation_args.eval_seed=evaluation_seed
        rows.extend(evaluate(engine,policies,evaluation_args,record_path=output/'rollouts.partial.jsonl'))
    (output/'rollouts.json').write_text(json.dumps(rows,indent=2)+'\n')
    summary=[]
    for method in args.methods.split(','):
        metric={'eig':'terminal_spce_nats','msc':'msc','mocu':'posterior_mocu'}[args.objective]
        values=np.asarray([r[metric] for r in rows if r['method']==method])
        summary.append({'method':method,'mean_spce_nats':float(values.mean()),
            'sd_across_systems_nats':float(values.std(ddof=1)) if len(values)>1 else None,
            'n_systems':len(values),'training_seconds':times.get('dad' if method=='step_dad' else method,0.),
            'training_reused_from':'dad' if method=='step_dad' else None,
            'mean_decision_seconds':float(np.mean([sum(r['decision_seconds_per_stage'])+r.get('terminal_controller_seconds',0.) for r in rows if r['method']==method])),
            'minimum_planning_ess':min((e for r in rows if r['method']==method for e in r['planning_ess_before_action']),default=None)})
        if args.objective!='eig':
            item=summary[-1]
            item['mean_'+args.objective]=item.pop('mean_spce_nats')
            item['sd_across_systems_'+args.objective]=item.pop('sd_across_systems_nats')
            selected=[r for r in rows if r['method']==method]
            item['mean_u_ctrl']=float(np.mean([r['u_ctrl'] for r in selected]))
            item['safety_rate']=float(np.mean([r['safe'] for r in selected]))
            item['safe_count']=sum(r['safe'] for r in selected)
            item['oracle_infeasible_count']=sum(not r['oracle_feasible'] for r in selected)
            oracle=[r['oracle_msc'] for r in selected if r['oracle_feasible']]
            item['mean_oracle_msc']=float(np.mean(oracle)) if len(oracle)==len(selected) else None
            item['safety_rate_sd_across_seeds']=None
            item['note']='System dispersion is not a run-level SD; seed SD requires multiple evaluation seeds. Oracle is conditional on each method actual terminal state.'
        selected=[r for r in rows if r['method']==method]
        seed_scores=[float(np.mean([r[metric] for r in selected if r['evaluation_seed']==seed])) for seed in eval_seeds]
        summary[-1]['evaluation_seeds']=eval_seeds
        summary[-1]['sd_of_seed_means']=float(np.std(seed_scores,ddof=1)) if len(seed_scores)>1 else None
        if args.objective!='eig':
            seed_safety=[float(np.mean([r['safe'] for r in selected if r['evaluation_seed']==seed])) for seed in eval_seeds]
            summary[-1]['safety_rate_sd_across_seeds']=float(np.std(seed_safety,ddof=1)) if len(seed_safety)>1 else None
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'completion.json').write_text(json.dumps({'objective':args.objective,'methods':args.methods.split(','),'evaluation_seeds':eval_seeds,'records':len(rows),'wall_seconds':time.monotonic()-run_started,'smoke_only':args.smoke},indent=2)+'\n')
    (output/'exit_code').write_text('0\n')
    sys.excepthook=original_hook
    print('CONTINUOUS_'+args.objective.upper()+'_COMPLETE',output,flush=True)


if __name__=='__main__':
    main()
