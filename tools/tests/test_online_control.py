import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.integrate import solve_ivp
from src.config import load_config
from src.control.continuous_control import select_control, CarriedStateControl

ROOT=Path(__file__).resolve().parents[2]


class PosteriorControlTests(unittest.TestCase):
    def test_direct_safety_constraint_allows_nonmonotone_safe_sets(self):
        safe=np.array([[False,True,False,True],[False,False,True,True]])
        result=select_control([0,.1,.2,.3],safe,[.9,.1],.85)
        self.assertEqual(result['u_ctrl'],.1)
        self.assertAlmostEqual(result['posterior_safe_mass'],.9)
        self.assertEqual(select_control([0,.1,.2,.3],safe,[.9,.1],.95)['u_ctrl'],.3)

    def test_infeasible_posterior_is_not_silently_capped(self):
        with self.assertRaisesRegex(ValueError,'infeasible'):
            select_control([0,.1],[[False,True],[False,False]],[.5,.5],.9)

    def test_evaluator_truth_is_excluded_from_control_posterior(self):
        from src.objectives.msc.continuous_msc import OnlineControl
        engine=object.__new__(OnlineControl)
        engine.objective='msc'
        seen=[]
        def requirements(theta,states,**kwargs):
            seen.append((theta.copy(),states.copy()))
            return np.full(len(theta),.3)
        engine.control=SimpleNamespace(requirements=requirements,coverage=.9,tolerance=1e-5,
            metrics=lambda t,s,u:(None,None,np.ones(len(t),bool)))
        theta=np.arange(18).reshape(1,3,6)
        states=theta+100
        a,_=engine.stage_scores(theta,states,np.array([[0.,-1.,-2.]]))
        theta[0,0]=99999; states[0,0]=-99999
        b,_=engine.stage_scores(theta,states,np.array([[100000.,-1.,-2.]]))
        np.testing.assert_array_equal(a,b)
        for x,y in zip(seen[0],seen[1]):np.testing.assert_array_equal(x,y)
        self.assertEqual(seen[0][0].shape,(2,6))

    def test_canonical_configs_have_no_reset_catalog_or_bank(self):
        for p in (ROOT/'configs').glob('ieee*.yaml'):
            cfg=load_config(p)
            self.assertFalse(cfg.swing['reset_after_probe'])
            self.assertNotIn('probe_durations',cfg.swing)
            self.assertFalse(cfg.raw['data']['uses_probe_bank'])
            self.assertFalse(cfg.raw['data']['uses_control_bank'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_carried_control_matches_independent_cpu_integration(self):
        from src.domains.swing.continuous_cuda import CudaContinuousSwingObserver
        cfg=load_config(ROOT/'configs/ieee9_mocu.yaml')
        observer=CudaContinuousSwingObserver(cfg,duration_bounds=(.2,3.),injection_bus=1,amplitude=.05)
        control=CarriedStateControl(cfg,observer.sim)
        sw=cfg.swing
        theta=np.r_[(np.array(sw['M_lower_nodes'])+sw['M_upper_nodes'])/2,
                    (np.array(sw['K_lower_nodes'])+sw['K_upper_nodes'])/2]
        states=observer.initial_state(2)
        for duration in [1.75,2.4,2.97]:
            states=observer.propagate(np.tile(theta,(2,1)),states,duration).terminal_state
        states[1,3:]+=[.05,-.04,.025]
        controls=np.array([.35,.4])
        r,f,_=control.metrics(np.tile(theta,(2,1)),states,controls)
        sim,spec=observer.sim,control.spec
        dt=spec.ode_dt
        for i in range(2):
            def rhs(t,y):
                angle,omega=y[:3],y[3:]
                coupling=(sim.B*np.sin(angle[:,None]-angle[None,:])).sum(axis=1)
                power=sim.P_m.copy()
                power[spec.contingency.bus]+=spec.contingency.magnitude
                power[spec.profile.bus]+=spec.profile.amplitude_at(t,controls[i])
                return np.r_[omega,(power-coupling-(theta[3:]/(2*np.pi)+sim.D_nodes)*omega)/theta[:3]]
            times=np.linspace(0,spec.T_obs_sec,round(spec.T_obs_sec/dt)+1)
            sol=solve_ivp(rhs,(0,spec.T_obs_sec),states[i],t_eval=times,rtol=1e-10,atol=1e-12,max_step=.005)
            self.assertTrue(sol.success)
            reference_f=sol.y[3:]/(2*np.pi)
            self.assertAlmostEqual(f[i],reference_f.min(),delta=2e-6)
            self.assertAlmostEqual(r[i],np.abs(np.diff(reference_f,axis=1)/dt).max(),delta=2e-4)
        rr,ff,_=control.metrics(np.tile(theta,(2,1)),observer.initial_state(2),controls)
        self.assertGreater(max(np.abs(f-ff)),1e-5)



class ContinuousDecisionTests(unittest.TestCase):
    def control(self):
        control=object.__new__(CarriedStateControl)
        control.bounds=(0.,.5)
        control.grid=np.linspace(0,.5,17)
        control.tolerance=1e-6
        control.coverage=.9
        def metrics(theta,states,controls):
            safe=np.asarray(controls)>=np.asarray(states)[:,0]
            return np.zeros(len(safe)),np.zeros(len(safe)),safe
        control.metrics=metrics
        return control

    def test_continuous_requirement_is_not_snapped_to_scan(self):
        c=self.control()
        states=np.array([[.371234],[.289876]])
        req=c.requirements(np.zeros_like(states),states)
        np.testing.assert_allclose(req,states[:,0],atol=c.tolerance)
        self.assertGreater(np.min(np.abs(c.grid-req[0])),.001)
        row=c.decision(np.zeros_like(states),states,np.array([.95,.05]))
        self.assertAlmostEqual(row['u_ctrl'],.371234,delta=c.tolerance)
        self.assertGreaterEqual(row['posterior_safe_mass'],.9)

    def test_msc_keeps_infeasible_particle_mass_without_discarding_it(self):
        from src.control.continuous_control import continuous_decision
        row=continuous_decision([.3,np.inf],[.95,.05],.9,'msc')
        self.assertEqual(row['u_ctrl'],.3)
        self.assertAlmostEqual(row['posterior_safe_mass'],.95)
        with self.assertRaisesRegex(ValueError,'infeasible'):
            continuous_decision([.3,np.inf],[.95,.05],.99,'msc')
        with self.assertRaises(ValueError):
            continuous_decision([.3,np.inf],[.95,.05],.9,'mocu')

    def test_mocu_uses_continuous_requirements_and_zero_when_known(self):
        from src.control.continuous_control import continuous_decision
        row=continuous_decision([.371234],[1.],.9,'mocu')
        self.assertEqual(row['u_ctrl'],.371234)
        self.assertEqual(row['posterior_mocu'],0.)
        row=continuous_decision([.2,.4],[.95,.05],.9,'mocu')
        self.assertAlmostEqual(row['posterior_mocu'],.09)

    def test_numerical_optimizer_keeps_deterministic_policy_and_correct_sign(self):
        from src.objectives.msc.continuous_numerical import improve_numerical
        policy=torch.nn.Module()
        policy.sequence=torch.nn.Parameter(torch.tensor([.2]))
        def rollout(policy,rng,batch,*,stochastic):
            self.assertFalse(stochastic)
            return {'info':np.full((batch,1),-(float(policy.sequence)-1)**2)}
        optimizer=torch.optim.SGD(policy.parameters(),lr=.1)
        improve_numerical(SimpleNamespace(rollout=rollout),policy,optimizer,np.random.default_rng(42),2,scale=.02,directions=1024)
        self.assertAlmostEqual(float(policy.sequence),.36,delta=.02)

    def test_numerical_optimizer_restores_parameters_after_failed_rollout(self):
        from src.objectives.msc.continuous_numerical import improve_numerical
        policy=torch.nn.Module();policy.sequence=torch.nn.Parameter(torch.tensor([.2]))
        before=policy.sequence.detach().clone()
        def rollout(*args,**kwargs):raise ValueError('infeasible')
        with self.assertRaisesRegex(ValueError,'infeasible'):
            improve_numerical(SimpleNamespace(rollout=rollout),policy,torch.optim.SGD(policy.parameters(),lr=.1),np.random.default_rng(42),2)
        torch.testing.assert_close(before,policy.sequence)

class RLObjectiveTests(unittest.TestCase):
    def test_redq_rewards_telescope_for_all_three_objectives(self):
        from src.objectives.eig.continuous_eig import DurationPolicy,OnlineEIG
        from src.objectives.eig.continuous_redq import REDQPolicy,REDQTrainer
        args=SimpleNamespace(T=3,N_obs=5,seed=101,redq_critics=2,redq_updates_per_batch=1)
        engine=SimpleNamespace(horizon=3,features=lambda a,o,b,t:torch.zeros((b,22)))
        for values in [[.2,.4,.7],[-.4,-.37,-.38],[-.04,-.03,-.01]]:
            trainer=REDQTrainer(REDQPolicy(3,5),args)
            result={'info':np.array([values]),'actions':[np.array([1.])]*3,
                    'observations':[np.zeros((1,5))]*3,'unit_actions':[np.array([.5])]*3}
            trainer.observe(engine,result,1)
            self.assertAlmostEqual(trainer.replay[:3,23].sum(),values[-1],places=6)
            np.testing.assert_array_equal(trainer.replay[:3,24],[0,0,1])


if __name__=='__main__':unittest.main()
