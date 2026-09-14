import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from src.config import load_config
from src.domains.swing.continuous import StageResponse
from src.objectives.eig.continuous_eig import OnlineEIG, DurationPolicy, train, evaluate, feasible_duration


class UninformativeObserver:
    n_obs=5
    bounds=(.2,3.)
    def initial_state(self,n):
        return np.zeros((n,6))
    def propagate(self,theta,state,duration):
        # Response can depend on applied duration/history, but not on theta.
        final=state+np.asarray(duration).reshape(-1,1)
        return StageResponse(np.repeat(final[:,:1],5,axis=1),final)


class ContinuousEIGTests(unittest.TestCase):
    def test_increasing_t3_map_reserves_room_and_preserves_gradients(self):
        from src.objectives.eig.continuous_pathwise import feasible_duration_torch
        for value in [0.,.5,1.]:
            history=[]
            for stage in range(3):
                a=feasible_duration([value],history,(.2,3.),.01,3)
                self.assertLessEqual(a[0],3.+1e-12)
                if history:self.assertGreaterEqual(a[0]-history[-1][0],.01)
                history.append(a)
        q=torch.tensor([.7],dtype=torch.float64,requires_grad=True)
        past=torch.tensor([1.1],dtype=torch.float64,requires_grad=True)
        fn=lambda x,y:feasible_duration_torch(x,[y],(.2,3.),.01,3)
        self.assertTrue(torch.autograd.gradcheck(fn,(q,past)))
        np.testing.assert_allclose(fn(q,past).detach().numpy(),
            feasible_duration([.7],[[1.1]],(.2,3.),.01,3))

    def test_history_order_changes_policy_inputs(self):
        engine=OnlineEIG(self.config(),UninformativeObserver(),horizon=3,sigma=.005,contrasts=4)
        actions=[np.array([.5]),np.array([2.5])]
        obs=[np.full((1,5),.01),np.full((1,5),.03)]
        a=engine.features(actions,obs,1,2)
        b=engine.features(actions[::-1],obs[::-1],1,2)
        self.assertFalse(torch.equal(a,b))

    def test_torch_feasible_map_matches_numpy_and_has_history_gradient(self):
        from src.objectives.eig.continuous_pathwise import feasible_duration_torch
        q=torch.tensor([.8],dtype=torch.float64,requires_grad=True)
        previous=torch.tensor([1.1],dtype=torch.float64,requires_grad=True)
        fn=lambda x,y:feasible_duration_torch(x,[y],(.2,3.),.01)
        self.assertTrue(torch.autograd.gradcheck(fn,(q,previous),eps=1e-6,atol=1e-6))
        np.testing.assert_allclose(fn(q,previous).detach().numpy(),
            feasible_duration(q.detach().numpy(),[previous.detach().numpy()],(.2,3.),.01))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_two_stage_pathwise_gradient_includes_carried_state(self):
        from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver
        from src.objectives.eig.continuous_pathwise import SwingStage
        observer=CudaContinuousSwingObserver(self.config(),duration_bounds=(.2,3.),injection_bus=1,amplitude=.05)
        engine=OnlineEIG(self.config(),observer,horizon=3,sigma=.005,contrasts=1)
        theta,initial=engine.sample(np.random.default_rng(42),1)
        theta=torch.tensor(theta[0],dtype=torch.float64)
        initial=torch.tensor(initial[0],dtype=torch.float64)
        duration=torch.tensor([.7,1.4],dtype=torch.float64,requires_grad=True)
        _,state=SwingStage.apply(observer,theta,initial,duration[:1].expand(2))
        obs,final=SwingStage.apply(observer,theta,state,duration[1:].expand(2))
        loss=obs.square().sum()+final.square().sum()
        grad=torch.autograd.grad(loss,duration)[0].numpy()
        def total(d):
            first=observer.propagate(theta.numpy(),initial.numpy(),np.repeat(d[0],2))
            second=observer.propagate(theta.numpy(),first.terminal_state,np.repeat(d[1],2))
            return (second.observations**2).sum()+(second.terminal_state**2).sum()
        reference=[]
        for j in range(2):
            plus,minus=duration.detach().numpy().copy(),duration.detach().numpy().copy()
            plus[j]+=1e-4;minus[j]-=1e-4
            reference.append((total(plus)-total(minus))/2e-4)
        np.testing.assert_allclose(grad,reference,rtol=2e-3,atol=1e-7)

    def test_feasible_mapping_excludes_overlapping_intervals_and_endpoints(self):
        q=np.linspace(0,1,10001)
        history=[.2,1.5,1.505,3.]
        d=feasible_duration(q,history,(.2,3.),.01)
        self.assertTrue(np.all((d>=.2)&(d<=3.)))
        for a in history:
            self.assertTrue(np.all(np.abs(d-a)>=.01))
        self.assertTrue(np.all(np.diff(d)>=0))

    def test_saturated_policy_and_external_selector_obey_nonrepetition(self):
        engine=OnlineEIG(self.config(),UninformativeObserver(),horizon=3,sigma=.005,contrasts=4)
        policy=DurationPolicy(3,5,fixed=True)
        with torch.no_grad(): policy.sequence.fill_(100.)
        result=engine.rollout(policy,np.random.default_rng(1),2,stochastic=False)
        for i in range(3):
            for j in range(i):
                self.assertTrue(np.all(np.abs(result['actions'][i]-result['actions'][j])>=.01))
        with self.assertRaisesRegex(ValueError,'non-repetition|increasing'):
            engine.rollout(None,np.random.default_rng(1),2,stochastic=False,
                           selector=lambda *args:[3.,3.])

    def config(self):
        return load_config(Path(__file__).resolve().parents[2]/'configs/ieee9_eig.yaml')

    def test_uninformative_system_has_zero_sequential_information(self):
        engine=OnlineEIG(self.config(),UninformativeObserver(),horizon=3,sigma=.005,contrasts=8)
        result=engine.rollout(DurationPolicy(3,5),np.random.default_rng(101),4,stochastic=True)
        np.testing.assert_allclose(result['info'],0.,atol=1e-10)
        self.assertTrue(np.all(np.asarray(result['actions'])>=.2))
        self.assertTrue(np.all(np.asarray(result['actions'])<=3.))

    def test_fixed_policy_does_not_see_observations(self):
        policy=DurationPolicy(3,5,fixed=True)
        a,_=policy(torch.zeros((2,22)),1)
        b,_=policy(torch.randn((2,22)),1)
        torch.testing.assert_close(a.mean,b.mean)

    def test_legacy_bank_path_rejects_no_reset_before_bank_loading(self):
        from src.objectives.mocu.context import build_context_from_config
        cfg=self.config()
        cfg.raw['swing_equation']['reset_after_probe']=False
        with self.assertRaisesRegex(ValueError,'equilibrium-bank experiment pipeline is retired'):
            build_context_from_config(cfg,ensure_bank=False,experiment_type='eig_based')

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_all_six_methods_work_with_online_state_and_continuous_actions(self):
        from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver
        cfg=self.config()
        observer=CudaContinuousSwingObserver(cfg,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05)
        engine=OnlineEIG(cfg,observer,horizon=3,sigma=.005,contrasts=4)
        args=SimpleNamespace(T=3,N_obs=5,seed=101,eval_seed=1001,learning_rate=.001,
            updates=1,batch_size=2,validate_every=1,validation_systems=2,
            methods='dad,rl_sboed,step_dad,myopic,fixed,random',eval_systems=1,
            planner_particles=4,noise_sigma=.005,contrasts=4,refinement_updates=1,
            search_rounds=1,search_candidates=2,fantasies=2)
        with tempfile.TemporaryDirectory() as directory:
            policies={m:train(engine,m,args,Path(directory))[0] for m in ['fixed','dad','rl_sboed']}
            rows=evaluate(engine,policies,args)
        self.assertEqual(len(rows),6)
        for row in rows:
            self.assertTrue(np.isfinite(row['terminal_spce_nats']))
            self.assertLessEqual(row['terminal_spce_nats'],np.log(5)+1e-10)
            self.assertEqual(len(row['duration_sequence_s']),3)
            durations=row['duration_sequence_s']
            for i in range(3):
                for j in range(i):
                    self.assertGreaterEqual(durations[i]-durations[j],.01)
            self.assertEqual(np.asarray(row['observations_hz']).shape,(3,5))
        # Same evaluator truth for all methods; not a separate test draw per method.
        for row in rows[1:]:
            np.testing.assert_array_equal(row['true_MK'],rows[0]['true_MK'])


if __name__=='__main__':
    unittest.main()
