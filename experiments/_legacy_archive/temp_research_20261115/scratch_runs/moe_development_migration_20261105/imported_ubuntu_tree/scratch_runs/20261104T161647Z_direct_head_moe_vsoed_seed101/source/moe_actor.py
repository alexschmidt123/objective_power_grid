"""Four soft mixed design-logit heads; native Net performs bounds once."""
import copy,hashlib
import torch
from torch import nn
def digest(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
class DirectDesignHeads(nn.Module):
 def __init__(self):
  super().__init__();self.gate=nn.Linear(256,4);self.heads=nn.ModuleList([nn.Sequential(nn.Linear(256,63),nn.ReLU(),nn.Linear(63,2)) for _ in range(4)]);self.last_probs=None
 def forward(self,z):
  p=self.gate(z).softmax(-1);values=torch.stack([h(z) for h in self.heads],dim=-2);self.last_probs=p.detach()
  return (p.unsqueeze(-1)*values).sum(-2)
def inject(a,seed=None):
 actor=a.actor_net;assert len(actor.net)==7
 assert [(m.in_features,m.out_features) for m in actor.net if isinstance(m,nn.Linear)]==[(117,256),(256,256),(256,256),(256,2)]
 assert sum(p.numel() for p in actor.parameters())==162306
 before={k:digest(getattr(a,k).state_dict()) for k in ['critic_net','critic_target_net']};post_before={str(k):[digest(n.state_dict()) for n in nets] for k,nets in a.post_approx.poi_post_nets.items()}
 rng=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [];device=next(actor.parameters()).device
 oldhidden=actor.net[4];oldout=actor.net[6];trunk={k:v.clone() for k,v in actor.state_dict().items() if k.startswith(('net.0.','net.2.'))}
 with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []):
  tail=DirectDesignHeads().to(device)
  with torch.no_grad():
   tail.gate.weight.zero_();tail.gate.bias.zero_()
   for k,h in enumerate(tail.heads):
    rows=slice(k*63,(k+1)*63);h[0].weight.copy_(oldhidden.weight[rows]);h[0].bias.copy_(oldhidden.bias[rows]);h[2].weight.copy_(4*oldout.weight[:,rows]);h[2].bias.copy_(oldout.bias)
 # Test actual fresh initialization against fixed neuron omission, before discarding native layers.
 probe=torch.linspace(-1,1,3*256,device=device).reshape(3,256)
 with torch.no_grad():
  expected=torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(probe,oldhidden.weight[:252],oldhidden.bias[:252])),oldout.weight[:,:252],oldout.bias)
  torch.testing.assert_close(tail(probe),expected,rtol=1e-5,atol=2e-7)
 tail.last_probs=None;actor.net=nn.Sequential(*list(actor.net.children())[:4],tail);a.actor_target_net=copy.deepcopy(actor)
 assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda,torch.cuda.get_rng_state_all()))
 assert all(torch.equal(v,actor.state_dict()[k]) for k,v in trunk.items())
 assert before=={k:digest(getattr(a,k).state_dict()) for k in before}
 assert post_before=={str(k):[digest(n.state_dict()) for n in nets] for k,nets in a.post_approx.poi_post_nets.items()}
 assert sum(p.numel() for p in actor.parameters())==162304
 assert all(torch.equal(x,y) and x.data_ptr()!=y.data_ptr() for x,y in zip(actor.parameters(),a.actor_target_net.parameters()))
 return dict(native_actor_parameters=162306,candidate_actor_parameters=162304,all_four_heads_active=True,initialization='fresh native neurons0:252 split into four contiguous63-row groups; final weights times4; shared output bias; zero gate',uniform_init_252_neuron_reference_verified=True,helpers_unchanged=True,outer_rng_unchanged=True,outer_cpu_rng_sha256=hashlib.sha256(rng.cpu().numpy().tobytes()).hexdigest(),outer_cuda_rng_sha256=[hashlib.sha256(r.cpu().numpy().tobytes()).hexdigest() for r in cuda],critic_initial_hashes=before,posterior_initial_hashes=post_before,actor_initial_hash=digest(actor.state_dict()))
