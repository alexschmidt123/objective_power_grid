import os,sys,json,copy,time
from pathlib import Path
C=Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_moe_st_routing_dev_103_105');S=C/'source'
sys.path.insert(0,str(S));import torch,yaml,numpy as np
torch.set_num_threads(4)
from src.research.moe_family import MoESBOEDPolicy,load_policy,component_groups
from src.research.sparse_moe import DSelectGate
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
raw=yaml.safe_load((C/'original.yaml').read_text());b=make_benchmark(raw);passed=[]
def ok(name):passed.append(name);print('PASS',name,flush=True)
torch.manual_seed(103);router=DSelectGate(193,1).double();context=torch.randn(8,193,dtype=torch.float64)
for gamma in [1.,.1,.001]:
 router.gamma=gamma
 w,soft,selectors,bits=router.straight_through(context);hard=router(context,True)[0]
 assert torch.equal(w,hard) and torch.equal(w.sum(-1),torch.ones(8,dtype=w.dtype))
ok('hard_forward_exact_at_three_gammas')
with torch.no_grad():router.bits.weight.zero_();router.bits.bias.zero_()
w,*_=router.straight_through(context);assert torch.equal(w,router(context,True)[0]) and (w.argmax(-1)==3).all()
ok('native_zero_logit_tie_convention')
router.gamma=1.
with torch.no_grad():router.bits.bias.copy_(torch.tensor([-.1,.1],dtype=torch.float64))
w,soft,*_=router.straight_through(context);cost=torch.tensor([0.,1.,3.,7.],dtype=torch.float64)
st_grad=torch.autograd.grad((w*cost).sum(),tuple(router.parameters()),retain_graph=True,allow_unused=True)
soft_grad=torch.autograd.grad((soft*cost).sum(),tuple(router.parameters()),allow_unused=True)
assert all(torch.equal(x,y) for x,y in zip(st_grad,soft_grad)) and float(st_grad[0].norm())>0
ok('router_surrogate_gradient_exactly_matches_relaxed_gradient')
torch.manual_seed(103);st=MoESBOEDPolicy('moe',b,30,training_routing_mode='straight_through')
torch.manual_seed(103);reference=MoESBOEDPolicy('moe',b,30)
assert all(torch.equal(v,reference.state_dict()[k]) for k,v in st.state_dict().items())
ok('identical_initial_weights_no_rng_change')
reference.set_routing(relaxed=False);st.train();g=torch.Generator().manual_seed(771001);truth=b.sample((8,),g);noise=torch.randn(8,30,1,generator=g,dtype=torch.float64)
with torch.no_grad():
 a=rollout(b,st,30,8,771001,0,truth=truth,noise=noise);r=rollout(b,reference,30,8,771001,0,truth=truth,noise=noise)
 forward_errors={key:max(float((x-y).abs().max()) for x,y in zip(a[key],r[key])) for key in ['actions','observations','latents']}
 print('FORWARD_ERRORS',forward_errors,flush=True)
 assert all(torch.allclose(x,y,atol=1e-12,rtol=1e-12) for key in ['actions','observations','latents'] for x,y in zip(a[key],r[key]))
ok('complete_hard_forward_rollout_equivalence_1e_minus12')
feature=r['features'][10]
reference.train();reference.core.routing_override=None;reference(feature,10);expected=reference.regularizers();st(feature,10);observed=st.regularizers()
assert all(torch.equal(x,y) for x,y in zip(expected,observed))
ok('native_relaxed_routing_regularization_unchanged')
for j in range(4):
 st.zero_grad(set_to_none=True)
 with torch.no_grad():
  st.router.bits.weight.zero_();st.router.bits.bias.copy_(torch.tensor([.1 if j&2 else -.1,.1 if j&1 else -.1],dtype=torch.float64))
 output=st(feature,10);output.square().sum().backward()
 norms=[sum(float(p.grad.square().sum()) for p in e.parameters() if p.grad is not None)**.5 for e in st.experts]
 assert norms[j]>0 and all(norms[k]==0 for k in range(4) if k!=j)
 assert st.router.bits.weight.grad.norm()>0
ok('selected_expert_receives_gradient_and_unselected_proposals_reach_router')
for variant in ['switchmoe','resmoe','switchresmoe']:
 try:MoESBOEDPolicy(variant,b,30,training_routing_mode='straight_through')
 except ValueError as e:assert 'restricted to moe' in str(e)
 else:raise AssertionError('Unsupported variant accepted')
ok('other_variants_explicitly_excluded')
fixture=torch.load(C/'compatibility_fixture.pt',weights_only=False)
for seed in [103,105]:
 old=Path(fixture[seed]['checkpoint']);p=load_policy(old,b)
 with torch.no_grad():
  for relaxed in [False,True]:
   p.set_routing(relaxed=relaxed);rr=rollout(b,p,30,8,771001,0,truth=fixture['truth'],noise=fixture['noise'])
   assert all(torch.equal(x,y) for x,y in zip(rr['actions'],fixture[seed][str(relaxed)]))
ok('historical_103_105_hard_and_relaxed_predictions_unchanged')
st.set_routing();st(torch.zeros_like(feature[:1]),0);seen=[]
handles=[e.register_forward_hook(lambda m,i,o,j=j:seen.append(j)) for j,e in enumerate(st.experts)]
st(feature[:1],10)
for h in handles:h.remove()
assert len(seen)==1
ok('hard_inference_executes_one_expert_per_case')
checkpoint=C/'test_checkpoint.pt';torch.save(st.checkpoint(),checkpoint);restored=load_policy(checkpoint,b)
assert restored.config['training_routing_mode']=='straight_through'
with torch.no_grad():assert torch.equal(st(feature,10),restored(feature,10))
ok('straight_through_checkpoint_roundtrip')
# One real-size GPU sPCE forward/backward; no optimizer update, separately accounted.
device='cuda' if torch.cuda.is_available() else 'cpu'
with torch.device(device):
 torch.manual_seed(103);gpu=MoESBOEDPolicy('moe',make_benchmark(raw),30,training_routing_mode='straight_through').train()
 start=time.perf_counter();rr=rollout(make_benchmark(raw),gpu,30,64,103+104729,1024)
 loss=-rr['info'][:,-1].mean();loss.backward()
 assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in gpu.parameters() if p.grad is not None)
 assert gpu.router.bits.weight.grad.norm()>0 and next(gpu.core.sum_encoder.parameters()).grad.norm()>0
 assert all(bool((torch.stack(rr['actions']).abs()<=4).all()) for _ in [0])
 if device=='cuda':torch.cuda.synchronize()
 seconds=time.perf_counter()-start
ok('real_batch64_L1024_finite_full_horizon_sPCE_and_encoder_router_gradients')
# Native CPU trainer smoke verifies optional setting is actually wired through.
from src.research.family_training import train_family
settings=copy.deepcopy(raw['research']['settings']);settings.update(moe_training_routing_mode='straight_through',contrasts=16,batch_size=8,validate_every=100,router_schedule_episodes=1000000)
smoke=C/'training_smoke';smoke.mkdir();p,record=train_family(b,30,settings,103,smoke,'moe',trajectory_budget=16)
assert record['generated_training_trajectories']==16 and record['settings']['training_routing_mode']=='straight_through'
ok('native_training_interface_two_update_smoke')
result=dict(status='passed',tests=passed,real_size_probe_seconds=seconds,probe_device=device,forward_max_errors=forward_errors,numerical_tolerance='weights bitwise; complete rollout rtol=atol=1e-12 (dense-vs-sparse batched GEMM); fixed-context surrogate gradient bitwise',diagnostic_primary_episodes=16+16+32+64+16,optimizer_smoke_episodes=16,full_training_launched=False)
(C/'correctness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
