import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.objectives.eig.continuous_specialist_moe import (
    ContinuousSpecialistMoE,ValidationGuard,make_optimizers,improve,train,smoothed_gradient)
from src.objectives.eig.continuous_belief_moe import new_belief
from src.objectives.eig.continuous_eig import evaluate
from tools.tests.test_belief_moe import engine


class SpecialistTests(unittest.TestCase):
    def test_regime_ownership_is_stable_and_uses_posterior_only(self):
        e=engine();p=ContinuousSpecialistMoE(e,16)
        x=list(new_belief(e,np.random.default_rng(14),8,16).inputs([],[],0))
        x[0][:4,:,:6] = -.5
        x[0][4:,:,:6] = .5
        owner=p.regime_owner(tuple(x),1,update=True)
        self.assertTrue(torch.equal(owner,torch.tensor([0,0,0,0,1,1,1,1])))
        shuffled=torch.tensor([7,3,6,2,5,1,4,0])
        yp=tuple(t[shuffled].clone() for t in x)
        yp[0][:,:,6:]=torch.randn_like(yp[0][:,:,6:])
        yp[2][:]=torch.randn_like(yp[2])
        self.assertTrue(torch.equal(p.regime_owner(yp,1),owner[shuffled]))
        self.assertTrue(torch.equal(p.regime_owner(tuple(x),0),torch.zeros(8,dtype=torch.long)))

    def test_warmup_allows_delayed_improvement_before_rollback(self):
        e=engine(3)
        args=SimpleNamespace(seed=101,planner_particles=8,learning_rate=.0003,updates=6,
            validate_every=1,validation_systems=4,batch_size=4,moe_fantasies=2)
        scores=[(v,[]) for v in [2.,1.,1.,1.,2.1,2.2,2.3]]
        with tempfile.TemporaryDirectory() as d:
            with patch('src.objectives.eig.continuous_specialist_moe.validation',side_effect=scores):
                train(e,args,Path(d))
            history=json.loads((Path(d)/'moe_sboed_training.json').read_text())
            self.assertEqual([r['decision'] for r in history[1:4]],['warmup_exploration']*3)
            self.assertEqual(history[-1]['retained_validation_eig'],2.3)
            self.assertEqual(history[-1]['update'],6)

    def test_antithetic_gradient_matches_known_smoothed_quadratic(self):
        rng=np.random.default_rng(18);directions=rng.normal(size=(20000,2))
        means=np.array([-.2,.3]);targets=np.array([.8,-.7]);scale=1.25
        values=[]
        for k in range(2):
            for sign in (1,-1):
                values.append(-(means[k]+sign*scale*directions[:,k]-targets[k])**2)
        rewards=np.stack(values,1)[...,None]
        gradient=smoothed_gradient(rewards,directions,scale).mean(0)
        np.testing.assert_allclose(gradient,-2*(means-targets),atol=.06)

    def test_router_symmetry_and_identical_action_tie(self):
        p=ContinuousSpecialistMoE(engine(),16)
        x=new_belief(engine(),np.random.default_rng(1),8,16).inputs([],[],0)
        actions=torch.randn(8,2)
        torch.testing.assert_close(p.gap(x,actions),-p.gap(x,actions.flip(-1)))
        torch.testing.assert_close(p.gap(x,actions[:,:1].expand(-1,2)),torch.zeros(8),rtol=0,atol=0)

    def test_expert_and_router_updates_do_not_move_other_expert(self):
        e=engine();p=ContinuousSpecialistMoE(e,16);opts=make_optimizers(p,.001)
        x=new_belief(e,np.random.default_rng(2),8,16).inputs([],[],0)
        before=p.expert_mean(x,1).detach().clone()
        loss=p.expert_mean(x,0).square().sum();loss.backward();opts[0].step()
        torch.testing.assert_close(p.expert_mean(x,1),before,rtol=0,atol=0)
        opts[2].zero_grad();p.gap(x,torch.randn(8,2)).sum().backward();opts[2].step()
        torch.testing.assert_close(p.expert_mean(x,1),before,rtol=0,atol=0)

    def test_unassigned_expert_is_frozen_even_with_adam_momentum(self):
        e=engine();p=ContinuousSpecialistMoE(e,16);opts=make_optimizers(p,.001)
        # Seed momentum, which must not move an unassigned expert later.
        sum(v.sum() for v in p.expert_parameters(1)).backward();opts[1].step()
        before=[v.clone() for v in p.expert_parameters(1)]
        central=np.ones((4,2,4));central[:,0]=2
        perturbed=np.ones((4,4,4));perturbed[:,0]=2;perturbed[:,2]=2
        with patch('src.objectives.eig.continuous_specialist_moe.compare_proposals',side_effect=[central,perturbed]):
            result=improve(e,p,opts,np.random.default_rng(3),4)
        self.assertEqual(result['assignment_counts'],[4,0])
        for a,b in zip(before,p.expert_parameters(1)):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_validation_guard_restores_optimizer_reduces_lr_and_stops(self):
        p=torch.nn.Linear(2,1);o=torch.optim.Adam(p.parameters(),lr=.01)
        p(torch.ones(3,2)).sum().backward();o.step()
        state=copy.deepcopy(p.state_dict());optstate=copy.deepcopy(o.state_dict())
        guard=ValidationGuard(p,[o],2.)
        for attempt in range(3):
            o.zero_grad();p(torch.ones(3,2)).sum().backward();o.step()
            result=guard.assess(1.)
            for k,v in p.state_dict().items():torch.testing.assert_close(v,state[k],rtol=0,atol=0)
            for k,v in o.state_dict()['state'].items():
                for name,value in v.items():torch.testing.assert_close(value,optstate['state'][k][name],rtol=0,atol=0)
            self.assertAlmostEqual(o.param_groups[0]['lr'],.01*.5**(attempt+1))
        self.assertEqual(result,'early_stop')

    def test_router_can_learn_context_dependent_preferences(self):
        torch.manual_seed(4)
        e=engine();p=ContinuousSpecialistMoE(e,8)
        x=list(new_belief(e,np.random.default_rng(4),32,8).inputs([],[],0))
        x[2][:16,-1]=-1;x[2][16:,-1]=1
        actions=torch.tensor([[-1.,1.]]).repeat(32,1)
        target=torch.cat([torch.ones(16),-torch.ones(16)])*.2
        opt=torch.optim.Adam(p.router_parameters(),lr=.01)
        for _ in range(250):
            loss=(p.gap(tuple(x),actions)-target).square().mean()
            opt.zero_grad();loss.backward();opt.step()
        prediction=p.gap(tuple(x),actions)
        self.assertTrue(torch.equal(prediction>0,target>0))

    def test_training_and_checkpoint_selected_evaluation(self):
        e=engine(4)
        args=SimpleNamespace(seed=101,planner_particles=16,learning_rate=.0003,updates=4,
            validate_every=1,validation_systems=4,batch_size=4,moe_fantasies=2,
            methods='moe_sboed',eval_seed=900123,eval_systems=2,T=4,noise_sigma=.02,N_obs=0)
        with tempfile.TemporaryDirectory() as d:
            p,_=train(e,args,Path(d));saved=torch.load(Path(d)/'moe_sboed.pth',weights_only=False)
            restored=ContinuousSpecialistMoE(e,16);restored.load_state_dict(saved['state_dict'])
            a=evaluate(e,{'moe_sboed':p},args);b=evaluate(e,{'moe_sboed':restored},args)
            self.assertEqual([r['terminal_spce_nats'] for r in a],[r['terminal_spce_nats'] for r in b])
            self.assertTrue(all(np.all(np.diff(r['duration_sequence_s'])>=.01) for r in a))


if __name__=='__main__':unittest.main()
