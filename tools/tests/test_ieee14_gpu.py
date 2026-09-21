"""IEEE14 CUDA validation, run only after active scientific jobs finish."""
import copy,unittest
import numpy as np
import torch
from scipy.special import logsumexp
from tools.tests.test_ieee14_cpu import config
from src.domains.swing.continuous import ContinuousSwingObserver
from src.objectives.eig.continuous_eig import OnlineEIG,DurationPolicy
from src.objectives.eig.continuous_pathwise import pathwise_rollout
from src.policies.direct_moe import DirectMoEPolicy

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class IEEE14GPUTests(unittest.TestCase):
    def make(self,c=None,bus=1):
        from src.domains.swing.continuous_rocof import EndpointRocofObserver
        return EndpointRocofObserver(c or config(),duration_bounds=(.2,3),injection_bus=bus,amplitude=.05,window=3.5)

    def test_endpoint_state_carry_and_time_step_convergence(self):
        c=config()
        lo=np.r_[c.swing['M_lower_nodes'],c.swing['K_lower_nodes']]
        hi=np.r_[c.swing['M_upper_nodes'],c.swing['K_upper_nodes']]
        theta=np.vstack([lo,hi,(lo+hi)/2])
        for bus in (1,4):
            gpu=self.make(c,bus)
            fine_cfg=config();fine_cfg.raw['swing_equation']['ode_dt']=float(c.swing['ode_dt'])/2
            fine=self.make(fine_cfg,bus)
            # Reference error must be below RK4 discretization error to measure convergence.
            cpu=ContinuousSwingObserver(c,duration_bounds=(.2,3),injection_bus=bus,amplitude=.05,n_obs=2,window=3.5,rtol=1e-12,atol=1e-14)
            cpu.times=np.array([3.475,3.5])
            gs=gpu.initial_state(3);fs=gs.copy();cs=gs.copy()
            for duration in (.3,.7,1.2):
                a=gpu.propagate(theta,gs,duration);b=fine.propagate(theta,fs,duration);ref=cpu.propagate(theta,cs,duration)
                rocof=np.diff(ref.observations,axis=1)/.025
                np.testing.assert_allclose(a.observations,rocof,atol=2e-6,rtol=2e-4)
                np.testing.assert_allclose(a.terminal_state,ref.terminal_state,atol=2e-6,rtol=2e-4)
                coarse_err=np.max(abs(a.terminal_state-ref.terminal_state))
                fine_err=np.max(abs(b.terminal_state-ref.terminal_state))
                # RK4 should approach 1/16 error on halving dt; allow 1/8 plus a reference floor.
                self.assertLessEqual(fine_err,coarse_err/8+2e-10)
                np.testing.assert_allclose(a.observations,b.observations,atol=2e-6,rtol=2e-4)
                gs,fs,cs=a.terminal_state,b.terminal_state,ref.terminal_state

    def direction(self,T,method,prefix=0):
        torch.manual_seed(101)
        observer=self.make();engine=OnlineEIG(config(),observer,horizon=T,sigma=.005,contrasts=4)
        p=DirectMoEPolicy(T,1) if method=='moe_sboed' else DurationPolicy(T,1,fixed=method=='fixed')
        if method=='moe_sboed':params=list(p.parameters())
        else:params=[v for n,v in p.named_parameters() if n in ('stage_bias','sequence') or n.startswith('network.')]
        bases=[v.detach().clone() for v in params]
        rng=np.random.default_rng(912)
        zs=[torch.tensor(rng.normal(size=v.shape),dtype=v.dtype) for v in params]
        norm=sum(float(z.square().sum()) for z in zs)**.5;zs=[z/norm for z in zs]
        kw={}
        if prefix:
            theta,state=engine.sample(np.random.default_rng(81),2);actions=[];observations=[]
            for k in range(prefix):
                d=np.full(2,.3+.2*k)
                out=observer.propagate(theta.reshape(-1,10),state.reshape(-1,10),np.repeat(d,5))
                state=out.terminal_state.reshape(state.shape)
                actions.append(d);observations.append(out.observations.reshape(2,5,1)[:,0])
            kw=dict(initial=(theta,state),history=(actions,observations),stage_start=prefix)
        def score(eps):
            with torch.no_grad():
                for v,b,z in zip(params,bases,zs):v.copy_(b+eps*z)
            return pathwise_rollout(engine,p,np.random.default_rng(187),2,**kw)[0][:,-1].mean()
        value=score(0)
        gs=torch.autograd.grad(value,params)
        gradient=sum(float((g*z).sum()) for g,z in zip(gs,zs))
        ref=float((score(.001)-score(-.001)).detach())/.002
        self.assertAlmostEqual(gradient,ref,delta=max(4e-5,.01*abs(ref)))
        score(0)
        with torch.no_grad():fwd=engine.rollout(p,np.random.default_rng(187),2,stochastic=False,**kw)
        self.assertAlmostEqual(float(value.detach()),float(fwd['info'][:,-1].mean()),places=9)

    def test_full_and_conditional_gradients(self):
        for T in (3,4,5):
            for method in ('dad','fixed','moe_sboed'):
                with self.subTest(T=T,method=method):self.direction(T,method)
        for prefix in (1,2):
            with self.subTest(prefix=prefix):self.direction(3,'dad',prefix)

    def test_scalar_reference_spce(self):
        c=config();obs=self.make();engine=OnlineEIG(c,obs,horizon=3,sigma=.005,contrasts=4)
        torch.manual_seed(101);policy=DirectMoEPolicy(3,1)
        with torch.no_grad():out=engine.rollout(policy,np.random.default_rng(81),2,stochastic=False)
        theta,_=engine.sample(np.random.default_rng(81),2)
        cpu=ContinuousSwingObserver(c,duration_bounds=(.2,3),injection_bus=1,amplitude=.05,n_obs=2,window=3.5)
        cpu.times=np.array([3.475,3.5])
        for row in range(2):
            state=cpu.initial_state(5);loglikes=np.zeros(5)
            for stage in range(3):
                r=cpu.propagate(theta[row],state,out['actions'][stage][row])
                means=np.diff(r.observations,axis=1)[:,0]/.025
                y=out['observations'][stage][row,0]
                for particle in range(5):loglikes[particle]-=.5*((y-means[particle])/.005)**2
                expected=loglikes[0]-logsumexp(loglikes)+np.log(5)
                self.assertAlmostEqual(float(out['info'][row,stage]),expected,delta=2e-4)
                state=r.terminal_state

if __name__=='__main__':unittest.main()
