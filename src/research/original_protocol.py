"""Published E1 training settings; only the primary-episode cap is scaled."""
from fractions import Fraction
import math

def score_allocation(budget, benchmark, initialization=None, batch=1024):
    if initialization is None: initialization=100014 if benchmark=='sir_ode' else 10014
    if batch != 1024: raise ValueError('Published ScoreBED batch is1024')
    if budget <= initialization+1: raise ValueError('Budget cannot cover native initialization')
    fraction=Fraction(95,100) if benchmark=='location_finding' else Fraction(399,400) if benchmark=='sir_ode' else None
    if fraction is None: raise ValueError(benchmark)
    remaining=budget-initialization;score=remaining*fraction.numerator//fraction.denominator;policy=remaining-score
    return dict(budget=budget,initialization=initialization,score=score,policy=policy,score_fraction=str(fraction),batch=batch,score_updates=math.ceil(score/batch),policy_updates=math.ceil(policy/batch),score_final_batch=score%batch or batch,policy_final_batch=policy%batch or batch,rounding='floor score allocation; remainder to policy; partial final minibatches only')

def vsoed_profile(benchmark,budget=1000000):
    if (budget-20000)%1000 or budget<=20000: raise ValueError('Native fresh batches require (budget-20000) divisible by1000')
    sir=benchmark=='sir_ode'
    return dict(updates=(budget-20000)//1000,new_trajectories=1000,replay_batch=10000,initial_replay=20000,replay_capacity=1000000,posterior_updates=5,critic_updates=5,posterior_lr=.0005 if sir else .001,actor_lr=.0005 if sir else .001,critic_lr=.001,lr_decay=.9999,discount=1. if sir else .9,exploration=5. if sir else .5,exploration_decay=.9999,target_rate=.1,transition=10000 if sir else 0,log_every=100,n_incre=1 if sir else 30,posterior_coordinate_reference='log_parameters' if sir else 'source_coordinates')

def sir_gmm_limits(log_mean,log_std):
    # Original SIR posterior is on log(beta),log(gamma); adapter uses Z.
    return [[(-6-float(mu))/log_std,(4-float(mu))/log_std] for mu in log_mean],[.5/log_std]*len(log_mean)

def dad_settings():
    return dict(batch_size=2000,contrasts=2000,learning_rate=5e-5,spce_adam_beta1=.8,spce_adam_beta2=.998,spce_lr_decay=.98,spce_lr_schedule_every=1000,spce_gradient_clip=None,first_design_revision=1,primary_episode_budget=1,spce_curriculum=1)
