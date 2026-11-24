"""Exact small fixtures; no model training or claim of an integrated v3 policy."""
import argparse,itertools,json,time
from pathlib import Path
import numpy as np
import torch

def run(out):
 start=time.monotonic();torch.set_default_dtype(torch.float64);torch.manual_seed(202611103001)
 # A two-step MDP: the first action changes the second information state.
 # Four complete categorical toy policies; every expert available at either step.
 gate=torch.randn(3,4,requires_grad=True);expert=torch.randn(3,4,requires_grad=True)
 probs=gate.softmax(-1);p=expert.sigmoid();rows=[];joint=[];rewards=[];scores=[]
 for k0,a0,k1,a1 in itertools.product(range(4),range(2),range(4),range(2)):
  h1=1+a0
  q0=p[0,k0] if a0 else 1-p[0,k0];q1=p[h1,k1] if a1 else 1-p[h1,k1]
  jp=probs[0,k0]*q0*probs[h1,k1]*q1
  r0=.1*a0;r1=1.7*(a1==a0)+.4*a0*a1
  score=(r0+r1)*(probs[0,k0].log()+q0.log())+r1*(probs[h1,k1].log()+q1.log())
  joint.append(jp);rewards.append(r0+r1);scores.append(score)
 joint=torch.stack(joint);rewards=torch.tensor(rewards);score=torch.stack(scores)
 objective=(joint*rewards).sum();exact=torch.autograd.grad(objective,[gate,expert],retain_graph=True)
 pg=torch.autograd.grad((joint.detach()*score).sum(),[gate,expert],retain_graph=True)
 for x,y in zip(exact,pg):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
 # Marginal mixture density gradient equals posterior responsibility weighted joint score.
 means=torch.tensor([[-1.,-.5],[.1,.5],[1.,-.3],[-.4,1.]],requires_grad=True)
 logits=torch.tensor([-.7,.2,.4,-.1],requires_grad=True);scales=torch.tensor([.3,.4,.5,.6])[:,None]
 z=torch.tensor([.23,-.31]);log_pi=torch.distributions.Normal(means,scales).log_prob(z).sum(-1)
 action=4*torch.tanh(z);logjac=(np.log(4.)+2*(np.log(2.)-z-torch.nn.functional.softplus(-2*z))).sum()
 log_pi=log_pi-logjac;log_g=logits.log_softmax(-1);lm=torch.logsumexp(log_g+log_pi,0);resp=(log_g+log_pi-lm).exp()
 gm=torch.autograd.grad(lm,[logits,means],retain_graph=True)
 gr=torch.autograd.grad((resp.detach()*(log_g+log_pi)).sum(),[logits,means],retain_graph=True)
 for x,y in zip(gm,gr):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)
 assert abs(float(resp.sum())-1)<1e-12 and bool((action.abs()<4).all())
 torch.testing.assert_close(torch.atanh(action/4),z,rtol=1e-12,atol=1e-12)
 # Density mass under the bounded transform: 2D Gauss-Legendre quadrature.
 nodes,weights=np.polynomial.legendre.leggauss(160);z1=torch.tensor(nodes*6);w1=torch.tensor(weights*6)
 zz=torch.cartesian_prod(z1,z1);ww=(w1[:,None]*w1[None,:]).flatten();aa=4*zz.tanh()
 lj=(np.log(4.)+2*(np.log(2.)-zz-torch.nn.functional.softplus(-2*zz))).sum(-1)
 lp=torch.distributions.Normal(means.detach()[None],scales[None]).log_prob(zz[:,None]).sum(-1)-lj[:,None]
 density=torch.logsumexp(log_g.detach()[None]+lp,-1).exp();mass=float((density*lj.exp()*ww).sum());assert abs(mass-1)<1e-10,mass
 # An external history/selected-expert multiplier is not the unchanged mixture gradient.
 q=torch.tensor([[1.,-1.,.4,.2],[-.7,1.,.3,-.2]])
 l=torch.tensor([[.2,-.2,.1,.5],[-.1,.6,-.3,.1]],requires_grad=True);g=l.softmax(-1)
 base=(g*q).sum(-1).mean();weights=torch.tensor([[1.,4.,.1,2.],[1.,.1,4.,2.]])
 altered=(weights*g*q).sum(-1).mean();gb=torch.autograd.grad(base,l,retain_graph=True)[0];ga=torch.autograd.grad(altered,l,retain_graph=True)[0]
 diff=float(torch.linalg.vector_norm(gb-ga));assert diff>.1
 # Importance-resampling the histories and correcting exactly restores the same objective.
 contribution=(g*q).sum(-1);s=torch.tensor([.2,1.8]);dprime=s/s.sum();corrected=(dprime*(1/2)/dprime*contribution).sum()
 gc=torch.autograd.grad(corrected,l,retain_graph=True)[0];torch.testing.assert_close(gb,gc,rtol=1e-12,atol=1e-12)
 # Exploration must be included in the routed behavior distribution.
 eps=.2;b=(1-eps)*g+eps/4;bg=torch.autograd.grad((b*q).sum(-1).mean(),l)[0]
 torch.testing.assert_close(bg,(1-eps)*gb,rtol=1e-12,atol=1e-12)
 # Two-step native reward algebra telescopes at gamma=1; no alternate reward.
 lp0=torch.tensor([-.7,.2]);lp1=torch.tensor([.3,.1]);prior=torch.tensor([-2.,-2.]);increments=(lp0-prior)+(lp1-lp0)
 torch.testing.assert_close(increments,lp1-prior,rtol=1e-12,atol=1e-12)
 receipt=dict(status='PASS_COMPONENT_FIXTURES_ONLY',seconds=time.monotonic()-start,optimizer_updates=0,simulator_calls=0,exact_two_step_trajectories=len(joint),score_gradient_max_error=max(float((x-y).abs().max()) for x,y in zip(exact,pg)),marginal_responsibility_gradient_max_error=max(float((x-y).abs().max()) for x,y in zip(gm,gr)),bounded_density_mass=mass,unjustified_specialist_weight_gradient_change=diff,importance_corrected_history_gradient_matches=True,exploration_distribution_derivative_correct=True,variational_telescope_at_gamma1=True,full_native_v3_integration_test='NOT_IMPLEMENTED_NOT_PASSED',no_v3_checkpoint_created=True)
 out.mkdir(parents=True,exist_ok=True);(out/'GRADIENT_FIXTURES.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);run(ap.parse_args().output)
