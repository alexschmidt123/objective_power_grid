import unittest,torch,numpy as np,math
from pathlib import Path
from src.research.benchmarks import SIR
from src.research.engine import rollout,generator
from src.research.policies import make_policy
from src.research.sir_evaluation import contrast_scores
from src.research.research_redq import RLPolicy
class Tests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1);self.b=SIR({})
 def test_jax_torch_carried_ode_and_gradients(self):
  a=np.load(Path(__file__).parent/'fixtures/sir_scorebed_ode_golden.npz')
  theta=torch.from_numpy(a['theta']);actions=torch.from_numpy(a['actions']).requires_grad_();state=self.b.initial((4,));ys=[]
  for t in range(5):
   mean,state=self.b.forward(theta,state,actions[:,t]);ys.append(mean)
  ys=torch.stack(ys,1)
  np.testing.assert_allclose(ys.detach().numpy(),a['means'],rtol=1e-10,atol=1e-9)
  g=torch.autograd.grad(ys.sum(),actions)[0].numpy()
  j=a['time_jacobian'].sum((0,1,2))
  np.testing.assert_allclose(g,j,rtol=2e-6,atol=1e-7)
 def test_common_streamed_eig_equals_direct(self):
  p=make_policy(self.b,3,'dad',8);r=rollout(self.b,p,3,2,101,7)
  # Direct rollout uses independent banks per row, so compare each individually.
  for i in range(2):
   one=dict(truth=r['truth'][i:i+1],actions=[a[i:i+1] for a in r['actions']],observations=[y[i:i+1] for y in r['observations']])
   info,_=contrast_scores(self.b,one,r['particles'][i],chunk=3,device='cpu')
   torch.testing.assert_close(info,r['info'][i:i+1],rtol=1e-10,atol=1e-10)
 def test_rl_raw_latent_uses_full_feasible_interval(self):
  p=RLPolicy(self.b,3)
  self.assertFalse(p.outputs_design);self.assertEqual(p.bound,1.)
  f=torch.zeros(2,10,dtype=torch.float64)
  with torch.no_grad():
   p.head[-1].weight.zero_();p.head[-1].bias[0]=8.
  z,_=p.sample(f,deterministic=True)
  physical=self.b.feasible(z.double(),[],3)
  self.assertGreater(float(physical.min()),99.)
 def test_ode_reference_and_mass(self):
  from scipy.integrate import solve_ivp
  theta=torch.tensor([[.5,.1]],dtype=torch.float64);state=self.b.initial((1,));times=[5.,20.,80.]
  for t in times:
   y,state=self.b.forward(theta,state,torch.tensor([[t]],dtype=torch.float64))
   def rhs(_,v):
    q=.5*v[0]*v[1]/500;r=.1*v[1];return [-q,q-r,r]
   ref=solve_ivp(rhs,[0,t],[498,2,0],rtol=1e-11,atol=1e-12).y[:,-1]
   np.testing.assert_allclose(state[0,:3].numpy(),ref,rtol=1e-6,atol=1e-4)
   self.assertAlmostEqual(float(state[0,:3].sum()),500.,places=9)
if __name__=='__main__':unittest.main()
