import math,types,torch
def normalized_gmm_forward(net,inputs,thetas):
 assert not net.idxs_truncnorm and net.n_theta==4
 theta=thetas[:,:4];f=net.feature_net(inputs);lw=net.weight_net(f);mu=net.mu_net(f).sigmoid().reshape(-1,net.n_mixture,4);sd=net.sigma_net(f).sigmoid().reshape(-1,net.n_mixture,4);mu=mu*(net.mu_bounds[:,1]-net.mu_bounds[:,0])+net.mu_bounds[:,0];sd=sd*net.max_sigmas+1e-5;lp=torch.distributions.Normal(mu,sd).log_prob(theta[:,None]).sum(-1);core=torch.logsumexp(lp+lw,-1);prior=-.5*theta.square().sum(-1)-2*math.log(2*math.pi);return torch.logaddexp(core,prior+math.log(1e-27))-math.log1p(1e-27)
def normalize_native_post(a):
 for nets in a.post_approx.poi_post_nets.values():
  for net in nets:net.forward=types.MethodType(normalized_gmm_forward,net)

