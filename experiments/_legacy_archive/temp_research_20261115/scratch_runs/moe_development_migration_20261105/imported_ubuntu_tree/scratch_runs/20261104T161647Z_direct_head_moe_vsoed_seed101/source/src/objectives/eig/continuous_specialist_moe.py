"""Paired utility specialization with validation-protected training blocks.

Training uses online posterior-predictive simulations only. Antithetic Gaussian
exploration estimates gradients of the smoothed remaining-horizon sPCE objective.
Deployment uses deterministic proposals and selects checkpoints by their EIG.
Assignments depend on independently sampled central-proposal utility comparisons;
gradients use a separate set of paired fantasies, avoiding selection on the same
simulation noise. No diversity bonus or forced equal-use inference is employed.
"""
import copy
import json
import time
import numpy as np
import torch
from src.policies.specialist_belief_moe import SpecialistBeliefMoE,ARCHITECTURE
from src.objectives.eig.continuous_belief_moe import ParticleContext,rollout,fantasy_initial


class ContinuousSpecialistMoE(SpecialistBeliefMoE):
    def __init__(self,engine,planner_particles=128):
        super().__init__(2*len(engine.lower),1+engine.horizon*(engine.n_obs+2)+4)
        self.planner_particles=planner_particles
        self.horizon,self.n_obs=engine.horizon,engine.n_obs
        self.routing_value_kind='centered_paired_remaining_eig'

    def rollout(self,engine,rng,batch,*,stochastic,initial=None,history=None,
                stage_start=0,selector=None,noise=None):
        if history is not None or stage_start or selector is not None:
            raise ValueError('Use the explicit posterior continuation interface')
        return rollout(engine,self,rng,batch,stochastic=stochastic,initial=initial,noise=noise)


class ValidationGuard:
    """Restore weights AND optimizer states after a degrading training block.

    Rejected blocks lower the learning rate; subsequent blocks draw fresh data.
    Stop after a bounded number of rejected blocks, rather than retry indefinitely.
    """
    def __init__(self,policy,optimizers,score,max_rejections=3):
        self.policy=policy;self.optimizers=optimizers
        if not np.isfinite(score):raise FloatingPointError('Non-finite initial validation')
        self.score=float(score);self.max_rejections=max_rejections
        self.rejections=0;self.total_rejections=0
        self.capture()

    def capture(self):
        self.state=copy.deepcopy(self.policy.state_dict())
        self.optimizer_states=[copy.deepcopy(o.state_dict()) for o in self.optimizers]

    def assess(self,score):
        if np.isfinite(score) and score>self.score+1e-6:
            self.score=float(score);self.rejections=0;self.capture()
            return 'accepted'
        rates=[[g['lr'] for g in o.param_groups] for o in self.optimizers]
        self.policy.load_state_dict(self.state)
        for optimizer,state,lr in zip(self.optimizers,self.optimizer_states,rates):
            optimizer.load_state_dict(copy.deepcopy(state))
            for group,old_lr in zip(optimizer.param_groups,lr):group['lr']=old_lr*.5
        self.rejections+=1;self.total_rejections+=1
        return 'early_stop' if self.rejections>=self.max_rejections else 'restored_lower_lr'


def expand_belief(belief,history,count):
    repeated=ParticleContext(belief.engine,np.repeat(belief.theta,count,axis=0),
        np.repeat(belief.states,count,axis=0),np.repeat(belief.logw,count,axis=0))
    past=tuple([np.repeat(x,count,axis=0) for x in seq] for seq in history)
    return repeated,past


def compare_proposals(engine,policy,belief,history,stage,latents,rng,fantasies):
    """Paired fantasy truths, alternatives and noises for all proposed actions."""
    repeated,past=expand_belief(belief,history,fantasies)
    batch=len(belief.theta); total=batch*fantasies
    initial=fantasy_initial(repeated,rng,engine.contrasts)
    noise=rng.normal(size=(total,engine.horizon-stage,engine.n_obs))
    rewards=[]
    with torch.no_grad():
        for latent in latents:
            branch=rollout(engine,policy,rng,total,stochastic=False,initial=initial,
                belief=repeated,history=past,stage_start=stage,noise=noise,
                first_latent=latent.detach().repeat_interleave(fantasies),first_logp=torch.zeros(total))
            rewards.append(branch['info'][:,-1].reshape(batch,fantasies))
    return np.stack(rewards,axis=1)


def make_optimizers(policy,lr):
    return [torch.optim.Adam(policy.expert_parameters(k),lr=lr) for k in range(2)]+[
        torch.optim.Adam(policy.router_parameters(),lr=lr)]


def smoothed_gradient(values,directions,scale):
    if scale<=0:raise ValueError('Exploration scale must be positive')
    return np.stack([(values[:,2*k]-values[:,2*k+1]).mean(1)*directions[:,k]/(2*scale)
                     for k in range(directions.shape[1])],1)


def improve(engine,policy,optimizers,rng,batch,fantasies=4,delta=1.25):
    with torch.no_grad():
        behavior=rollout(engine,policy,rng,batch,stochastic=True)
    stage=int(rng.integers(engine.horizon))
    belief,history,inputs=behavior['snapshots'][stage]
    means=policy.distributions(inputs).mean
    # Central utility labels and finite-difference gradients use separate draws.
    centers=compare_proposals(engine,policy,belief,history,stage,[means[:,0],means[:,1]],rng,fantasies)
    gap=centers[:,0]-centers[:,1]
    target=gap.mean(1)
    error=gap.std(1,ddof=1)/np.sqrt(fantasies)
    winner=(target<0).astype(int)
    uncertain=np.abs(target)<=2*error
    # Ambiguous utility comparisons use consistent posterior neighborhoods.
    # Confident utility winners override neighborhood ownership. This replaces
    # random reassignments that exposed both experts to the same population.
    owner=policy.regime_owner(inputs,stage,update=True).numpy()
    winner[uncertain]=owner[uncertain]
    # Broad, antithetic exploration can cross low-information plateaus. The
    # factor epsilon below estimates the Gaussian-smoothed objective gradient,
    # not a local derivative at an uninformative midpoint.
    directions=rng.normal(size=(batch,2))
    perturb=[]
    for k in range(2):
        offset=torch.as_tensor(delta*directions[:,k],dtype=torch.float32)
        perturb.extend([means[:,k]+offset,means[:,k]-offset])
    values=compare_proposals(engine,policy,belief,history,stage,perturb,rng,fantasies)
    derivatives=smoothed_gradient(values,directions,delta)
    norms=[]
    # Separate optimizers and encoders: an unassigned expert is not moved by
    # another expert's gradients, shared representations, or Adam momentum.
    for k in range(2):
        mask=torch.as_tensor(winner==k)
        optimizer=optimizers[k]
        optimizer.zero_grad(set_to_none=True)
        if not mask.any():norms.append(0.);continue
        gradient=torch.as_tensor(derivatives[:,k],dtype=torch.float32)
        loss=-(policy.expert_mean(inputs,k)[mask]*gradient[mask]).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Invalid expert gradient')
        loss.backward()
        norms.append(float(torch.nn.utils.clip_grad_norm_(policy.expert_parameters(k),1.)))
        optimizer.step()
    # Train the difference directly: large common EIG offsets cancel before
    # regression, so routing does not depend on fitting two absolute values.
    prediction=policy.gap(inputs,means.detach())
    router_loss=torch.nn.functional.smooth_l1_loss(prediction,torch.as_tensor(target,dtype=torch.float32))
    if not torch.isfinite(router_loss):raise FloatingPointError('Invalid router loss')
    optimizers[2].zero_grad();router_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.router_parameters(),5.);optimizers[2].step()
    return {'stage':stage,'expert_gradient_norms':norms,'assignment_counts':np.bincount(winner,minlength=2).tolist(),
        'resolved_comparisons':int((~uncertain).sum()),
        'resolved_assignment_counts':np.bincount(winner[~uncertain],minlength=2).tolist(),
        'ambiguous_regime_counts':np.bincount(winner[uncertain],minlength=2).tolist(),
        'regime_centers_normalized_MK':policy.regime_centers.tolist(),'router_loss':float(router_loss.detach()),
        'mean_paired_gap':float(target.mean()),'mean_smoothed_eig_gradient':derivatives.mean(0).tolist(),
        'behavior_trajectories':batch,'counterfactual_trajectories':6*batch*fantasies,
        'remaining_horizon':engine.horizon-stage,'exploration_standard_deviation':delta}


def validation(engine,policy,seed,systems):
    with torch.no_grad():
        result=rollout(engine,policy,np.random.default_rng(seed),systems,stochastic=False)
    records=[]
    for stage,record in enumerate(result['routing']):
        means=np.asarray(record['expert_latent_means'])
        records.append({'stage':stage,'selection_counts':np.bincount(record['selected_experts'],minlength=2).tolist(),
            'mean_absolute_latent_difference':float(np.abs(means[:,0]-means[:,1]).mean())})
    return float(result['info'][:,-1].mean()),records


def train(engine,args,directory):
    torch.manual_seed(args.seed)
    policy=ContinuousSpecialistMoE(engine,args.planner_particles)
    optimizers=make_optimizers(policy,args.learning_rate)
    rng=np.random.default_rng(args.seed)
    records=[];updates=[];start=time.monotonic();guard=None;best_update=0
    warmup_updates=min(100,max(args.updates//2,1))
    for update in range(args.updates+1):
        if update:updates.append({'update':update,**improve(engine,policy,optimizers,rng,
            args.batch_size,args.moe_fantasies,delta=1.25-.9*update/max(args.updates,1))})
        if update%args.validate_every and update!=args.updates:continue
        candidate,routing=validation(engine,policy,args.seed+900000,args.validation_systems)
        if guard is None:
            guard=ValidationGuard(policy,optimizers,candidate)
            decision='initial'
        elif update<=warmup_updates:
            if candidate>guard.score+1e-6:
                guard.score=candidate;guard.capture();decision='accepted'
            else:decision='warmup_exploration'
        else:decision=guard.assess(candidate)
        if decision in {'initial','accepted'}:
            best_update=update
            torch.save({'architecture':ARCHITECTURE,'state_dict':policy.state_dict(),
                'settings':vars(args),'best_update':best_update,'method':'moe_sboed'},directory/'moe_sboed.pth')
        records.append({'update':update,'candidate_validation_eig':candidate,'retained_validation_eig':guard.score,
            'validation_utility':guard.score,'decision':decision,'candidate_routing':routing,
            'learning_rates':[o.param_groups[0]['lr'] for o in optimizers],
            'elapsed_seconds':time.monotonic()-start})
        (directory/'moe_sboed_training.json').write_text(json.dumps(records,indent=2)+'\n')
        (directory/'moe_sboed_optimizer_diagnostics.json').write_text(json.dumps(updates,indent=2)+'\n')
        print(f'[specialist-moe] update={update}/{args.updates} candidate={candidate:.6f} '
            f'retained={guard.score:.6f} decision={decision} elapsed={time.monotonic()-start:.1f}s',flush=True)
        if decision=='early_stop':break
    policy.load_state_dict(guard.state)
    (directory/'moe_sboed_training_diagnostics.json').write_text(json.dumps({
        'architecture':ARCHITECTURE,'best_update':best_update,'completed_updates':update,
        'maximum_updates':args.updates,'early_stopped':update<args.updates,
        'rollback_warmup_updates':warmup_updates,
        'best_validation_utility':guard.score,'rejected_validation_blocks':guard.total_rejections,
        'teacher':None,'n_experts':2,'independent_expert_encoders':True,
        'routing':'antisymmetric regression of paired remaining-EIG differences',
        'actor_training':'antithetic Gaussian-smoothed remaining-horizon EIG gradient; deterministic continuation and validation',
        'responsibilities':'confident paired utility winner; stable posterior-MK neighborhoods for unresolved training comparisons',
        'exploration_standard_deviation_schedule':[1.25,.35],
        'gradient_fantasies_independent_of_assignment_fantasies':True,
        'behavior_trajectories':sum(r['behavior_trajectories'] for r in updates),
        'counterfactual_trajectories':sum(r['counterfactual_trajectories'] for r in updates),
        'specialization_established':False,'convergence_established':False},indent=2)+'\n')
    return policy,time.monotonic()-start
