"""Independent network, continuous propagation and Bayesian checks; CPU only."""
import copy,re,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import logsumexp
from src.config import load_config
from src.domains.swing.continuous import ContinuousSwingObserver,ContinuousParticleBelief
from src.domains.swing.simulator import generate_ieee14_coupling_matrix,ieee14_physical_input_map

ROOT=Path(__file__).resolve().parents[2]
def config():
    return load_config(ROOT/'configs/ieee14_mocu.yaml')

def reference_network():
    text=(ROOT/'tools/reference_data/case14.m').read_text()
    block=re.search(r'mpc.branch\s*=\s*\[(.*?)\];',text,re.S).group(1)
    rows=np.array([[float(v) for v in line.split('%')[0].replace(';','').split()]
                   for line in block.splitlines() if line.split('%')[0].strip()])
    lap=np.zeros((14,14))
    for row in rows:
        if row[10]==0:continue
        a,b=int(row[0])-1,int(row[1])-1
        v=np.zeros(14);v[a]=1;v[b]=-1
        lap+=np.outer(v,v)/row[3]
    kept=[0,1,2,5,7];drop=[i for i in range(14) if i not in kept]
    transform=np.zeros((14,5));transform[kept]=np.eye(5)
    transform[drop]=-np.linalg.solve(lap[np.ix_(drop,drop)],lap[np.ix_(drop,kept)])
    reduced=transform.T@lap@transform
    coupling=-reduced;np.fill_diagonal(coupling,0)
    return coupling,transform

def reference_sequence(observer,theta,durations):
    """Absolute-time, segmented Radau solution, independent of production propagate."""
    B,input_map=reference_network()
    cfg=config();sw=cfg.swing
    M,K=theta[:5],theta[5:];D=np.array(sw['D_nodes'])
    state=np.r_[sw['theta0_nodes'],sw['omega0_nodes']].astype(float)
    output=[]
    for stage,duration in enumerate(durations):
        start=stage*observer.window
        def rhs(t,y):
            local=t-start
            pulse=.5*(1-np.cos(2*np.pi*local/duration)) if 0<=local<=duration else 0.
            electrical=np.array([sum(B[i,j]*np.sin(y[i]-y[j]) for j in range(5)) for i in range(5)])
            return np.r_[y[5:],(np.array(sw['P_m_nodes'])-electrical-(K/(2*np.pi)+D)*y[5:]
                          +observer.amplitude*input_map[observer.bus]*pulse)/M]
        # Exact pulse-end segmentation, then free response to the next probe.
        a=solve_ivp(rhs,(start,start+duration),state,method='Radau',rtol=2e-10,atol=1e-12,dense_output=True)
        b=solve_ivp(rhs,(start+duration,start+observer.window),a.y[:,-1],method='Radau',rtol=2e-10,atol=1e-12,dense_output=True)
        values=np.stack([(a if t<=duration else b).sol(start+t) for t in observer.times])
        state=b.y[:,-1]
        output.append((values[:,5+observer.sim.observation_bus]/(2*np.pi),state.copy()))
    return output

class IEEE14CPUTests(unittest.TestCase):
    def test_reference_case_reduction_and_bus_mapping(self):
        B,P=reference_network()
        np.testing.assert_allclose(generate_ieee14_coupling_matrix(),B,atol=2e-13)
        np.testing.assert_allclose(ieee14_physical_input_map(),P,atol=2e-14)
        np.testing.assert_allclose(P.sum(1),1,atol=2e-14)
        self.assertEqual(P.shape,(14,5))
        self.assertGreater(np.linalg.eigvalsh(np.diag(B.sum(1))-B)[1],0)

    def test_absolute_time_sequence_against_independent_solver(self):
        c=config();obs=ContinuousSwingObserver(c,duration_bounds=(.2,3),injection_bus=4,
                 amplitude=.05,n_obs=2,window=3.5)
        obs.times=np.array([3.475,3.5])
        theta=(np.r_[c.swing['M_lower_nodes'],c.swing['K_lower_nodes']]
               +np.r_[c.swing['M_upper_nodes'],c.swing['K_upper_nodes']])/2
        durations=[.31,.73,1.21]
        expected=reference_sequence(obs,theta,durations)
        state=obs.initial_state(1)
        for duration,(freq,terminal) in zip(durations,expected):
            actual=obs.propagate(theta[None],state,duration)
            np.testing.assert_allclose(actual.observations[0],freq,rtol=2e-5,atol=2e-8)
            np.testing.assert_allclose(actual.terminal_state[0],terminal,rtol=2e-5,atol=2e-7)
            state=actual.terminal_state
        reset=obs.propagate(theta[None],obs.initial_state(1),durations[-1])
        self.assertGreater(np.max(abs(reset.terminal_state-state)),1e-4)

    def test_batched_scalar_and_posterior_reference(self):
        c=config();obs=ContinuousSwingObserver(c,duration_bounds=(.2,3),injection_bus=1,amplitude=.05,n_obs=2,window=3.5)
        lo=np.r_[c.swing['M_lower_nodes'],c.swing['K_lower_nodes']]
        hi=np.r_[c.swing['M_upper_nodes'],c.swing['K_upper_nodes']]
        particles=np.random.default_rng(81).uniform(lo,hi,(4,10))
        belief=ContinuousParticleBelief(obs,particles,.005)
        reference_weights=np.full(4,-np.log(4));reference_states=obs.initial_state(4)
        for duration in (.3,.7):
            batch=obs.propagate(particles,reference_states,duration)
            scalar=[obs.propagate(particles[i:i+1],reference_states[i:i+1],duration) for i in range(4)]
            np.testing.assert_allclose(batch.observations,np.concatenate([r.observations for r in scalar]),atol=2e-8,rtol=2e-5)
            y=batch.observations[0]+np.array([.001,-.002])
            reference_weights-=.5*np.sum(((y-batch.observations)/.005)**2,axis=1)
            reference_weights-=logsumexp(reference_weights)
            belief.update(duration,y)
            np.testing.assert_allclose(belief.log_weights,reference_weights,atol=1e-10)
            np.testing.assert_allclose(belief.states,batch.terminal_state,atol=1e-10)
            self.assertAlmostEqual(float(np.exp(belief.log_weights).sum()),1.)
            reference_states=batch.terminal_state
        before=belief.states.copy()
        score=belief.expected_information(.9,noise_samples=8,rng=np.random.default_rng(1))
        self.assertTrue(np.isfinite(score));np.testing.assert_array_equal(before,belief.states)

    def test_prior_shape_equilibrium_and_unsupported_system(self):
        c=config();obs=ContinuousSwingObserver(c,duration_bounds=(.2,3),injection_bus=1,amplitude=0,n_obs=1,window=3.5)
        lo=np.r_[c.swing['M_lower_nodes'],c.swing['K_lower_nodes']]
        hi=np.r_[c.swing['M_upper_nodes'],c.swing['K_upper_nodes']]
        self.assertEqual(len(lo),10);self.assertTrue(np.all(hi>lo));self.assertTrue(np.all(lo>0))
        result=obs.propagate(((lo+hi)/2)[None],obs.initial_state(1),.5)
        np.testing.assert_array_equal(result.terminal_state,np.zeros((1,10)))
        c.raw['system']['name']='ieee30'
        with self.assertRaises(ValueError):ContinuousSwingObserver(c,duration_bounds=(.2,3),injection_bus=1,amplitude=.05)

if __name__=='__main__':unittest.main()
