import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.special import logsumexp
from src.domains.swing.continuous import StageResponse
from src.objectives.eig.continuous_eig import OnlineEIG, evaluate
from src.objectives.eig.continuous_belief_moe import (
    ContinuousBeliefMoE, ParticleContext, new_belief, fantasy_initial, rollout, improve, train)


class Observer:
    n_obs=1
    bounds=(.2,3.)
    def initial_state(self,n):return np.zeros((n,6))
    def propagate(self,theta,state,duration):
        q=np.asarray(duration)[:,None]
        return StageResponse(theta[:,:1]*q+.03*state[:,:1],state+q*.01)


def engine(t=3):
    cfg=SimpleNamespace(swing={'M_lower_nodes':[.1]*3,'M_upper_nodes':[.3]*3,
        'K_lower_nodes':[.1]*3,'K_upper_nodes':[.3]*3})
    return OnlineEIG(cfg,Observer(),horizon=t,sigma=.02,contrasts=8)


class BeliefMoETests(unittest.TestCase):
    def test_weighted_set_invariance_and_posterior_update(self):
        e=engine(); b=new_belief(e,np.random.default_rng(1),3,16)
        theta=b.theta.copy(); state=b.states.copy()
        duration=np.array([.3,.4,.5]); y=np.array([[.05],[.06],[.07]])
        predicted=theta[:,:,:1]*duration[:,None,None]+.03*state[:,:,:1]
        expected=-.5*np.sum(((y[:,None]-predicted)/e.sigma)**2,-1)
        expected-=logsumexp(expected,axis=1,keepdims=True)
        b.update(duration,y)
        np.testing.assert_allclose(b.logw,expected)
        p=ContinuousBeliefMoE(e,16); x=b.inputs([duration],[y],1)
        perm=np.random.default_rng(2).permutation(16)
        xp=(x[0][:,perm],x[1][:,perm],x[2])
        torch.testing.assert_close(p.distributions(x).mean,p.distributions(xp).mean)
        torch.testing.assert_close(p.q_values(x,p.distributions(x).mean),
                                   p.q_values(xp,p.distributions(xp).mean))

    def test_hard_selection_not_average_and_batched_agreement(self):
        e=engine(); p=ContinuousBeliefMoE(e,16)
        b=new_belief(e,np.random.default_rng(3),5,16); x=b.inputs([],[],0)
        z,_,selected,values,means=p.choose(x)
        torch.testing.assert_close(z,means.gather(1,selected[:,None]).squeeze(1))
        self.assertTrue(torch.equal(selected,values.argmax(1)))
        for i in range(5):
            scalar=p.choose(tuple(a[i:i+1] for a in x))[0]
            torch.testing.assert_close(z[i:i+1],scalar)

    def test_truth_does_not_enter_first_action_and_support_is_separate(self):
        e=engine(); p=ContinuousBeliefMoE(e,16)
        initial=e.sample(np.random.default_rng(4),4)
        alternative=(initial[0].copy(),initial[1].copy())
        alternative[0][:,0]=e.upper
        a=rollout(e,p,np.random.default_rng(5),4,stochastic=False,initial=initial)
        b=rollout(e,p,np.random.default_rng(5),4,stochastic=False,initial=alternative)
        np.testing.assert_array_equal(a['actions'][0],b['actions'][0])
        planner=a['snapshots'][0][0].theta
        self.assertFalse(np.any(np.all(planner==initial[0][:,0,None],axis=-1)))

    def test_counterfactual_sampling_uses_posterior_and_carried_states(self):
        e=engine(); b=new_belief(e,np.random.default_rng(6),2,8)
        b.states[:]=np.arange(8)[None,:,None]
        b.logw[:]=-np.inf; b.logw[:,3]=0
        theta,state=fantasy_initial(b,np.random.default_rng(7),8)
        np.testing.assert_array_equal(theta,np.repeat(b.theta[:,3:4],9,axis=1))
        self.assertTrue(np.all(state==3))

    def test_full_horizons_gradients_and_order(self):
        for t in (3,4,5):
            torch.manual_seed(8)
            e=engine(t);p=ContinuousBeliefMoE(e,16)
            result=rollout(e,p,np.random.default_rng(9),8,stochastic=False)
            self.assertEqual(result['info'].shape,(8,t))
            self.assertTrue(np.all(np.diff(np.stack(result['actions']),axis=0)>=.01))
            optimizer=torch.optim.Adam(p.parameters(),lr=.0003)
            before=copy.deepcopy(p.state_dict())
            diagnostic=improve(e,p,optimizer,np.random.default_rng(10),16)
            self.assertTrue(all(np.isfinite(diagnostic['expert_gradient_norms'])))
            self.assertTrue(all(v>0 for v in diagnostic['expert_gradient_norms']))
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in p.state_dict().items()))

    def test_training_checkpoint_and_paired_evaluation(self):
        e=engine();args=SimpleNamespace(seed=101,planner_particles=16,learning_rate=.0003,
            updates=2,batch_size=4,validate_every=1,validation_systems=4,T=3,
            eval_seed=1001,eval_systems=3,methods='moe_sboed',noise_sigma=.02,N_obs=0)
        with tempfile.TemporaryDirectory() as d:
            p,_=train(e,args,Path(d))
            checkpoint=torch.load(Path(d)/'moe_sboed.pth',weights_only=False)
            restored=ContinuousBeliefMoE(e,16);restored.load_state_dict(checkpoint['state_dict'])
            rows=evaluate(e,{'moe_sboed':p},args)
            duplicate=evaluate(e,{'moe_sboed':restored},args)
            expected,_=e.sample(np.random.default_rng(args.eval_seed),3)
            for i,(a,b) in enumerate(zip(rows,duplicate)):
                self.assertEqual(a['true_MK'],expected[i,0].tolist())
                self.assertEqual(a['terminal_spce_nats'],b['terminal_spce_nats'])
                self.assertEqual(a['duration_sequence_s'],b['duration_sequence_s'])
                self.assertAlmostEqual(sum(a['planner_posterior_weights']),1.)
                self.assertEqual(len(a['routing_trace']),3)


if __name__=='__main__':unittest.main()
