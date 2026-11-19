"""Offline-only training/development diagnostics. Never imported by deployed actor.forward."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,inspect,io,json,math,sys,time
from unittest.mock import patch
M=Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_v5_belief_20261119';R=M/'reports/moe_sboed_v5_belief_20261119'
spec=importlib.util.spec_from_file_location('belief_reference_history',M/'src/research/moe_sboed_v5_history_20261119/run.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
import torch,numpy as np
from scipy.stats import spearmanr

def put(path,data):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(path,data)
def read(path):return json.loads(Path(path).read_text())
def load_runtime(path,device):
 old=torch.storage._load_from_bytes;torch.storage._load_from_bytes=lambda b:torch.load(io.BytesIO(b),map_location=device,weights_only=False)
 try:
  with Path(path).open('rb') as f:return p.n.cloudpickle.load(f)
 finally:torch.storage._load_from_bytes=old

def forbid(*args,**kwargs):raise AssertionError('Online optimization,gradient use or random search attempted')

def deployment_audit(actor,x):
 actor.configure('frozen');actor.eval();assert not any(v.requires_grad for v in actor.parameters())
 before=p.n.digest(actor.state_dict());sources={};calls=[];forbidden=[]
 functions=[]
 for module in actor.modules():
  for name in ['forward','compute','decode','packets','params','nodes']:
   fn=getattr(type(module),name,None)
   if fn is None:continue
   try:source=inspect.getsource(fn);file=inspect.getsourcefile(fn)
   except (TypeError,OSError):continue
   if file and str(M) in file:
    sources[file]=p.sha(file);functions.append(str(type(module))+'.'+name)
    import textwrap
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
     if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ['backward','step','minimize','optimize','train_native','mcmc']:forbidden.append((file,node.func.attr))
 assert not forbidden,forbidden
 with torch.no_grad():reference=actor(x)
 # Prevent gradient optimization,optimizer construction and stochastic search in the action path.
 import contextlib
 with contextlib.ExitStack() as stack:
  for owner,name in [(torch.autograd,'grad'),(torch.autograd,'backward'),(torch.Tensor,'backward'),(torch.optim.Optimizer,'__init__'),(torch,'enable_grad')]:stack.enter_context(patch.object(owner,name,forbid))
  for name in ['rand','randn','rand_like','randn_like','normal','multinomial']:stack.enter_context(patch.object(torch,name,forbid))
  with torch.no_grad():actual=actor(x)
 torch.testing.assert_close(actual,reference,rtol=0,atol=0);assert not actual.requires_grad and p.n.digest(actor.state_dict())==before
 return dict(status='PASS',fixed_actor_state_unchanged=True,all_actor_parameters_frozen=True,gradient_optimizer_construction_and_random_search_traps=True,actions_bitwise_equal=True,
  static_forward_methods=sorted(set(functions)),deployed_source_sha256=sources,scope='Runtime tests plus source inspection for this frozen implementation;fixed NN/sigma-point/probe loops remain,not iterative design optimization',offline_reference_and_suffix_code_not_in_deployed_path=True)


def logprior(theta):return -.5*theta.square().sum(-1)-2*math.log(2*math.pi)
def logmix(theta,w,mu,sd):
 return torch.logsumexp(w.clamp_min(1e-300).log()[None,:]-.5*((theta[:,None]-mu[None])/sd[None]).square().sum(-1)-sd.log().sum(-1)[None,:]-2*math.log(2*math.pi),-1)
def samples(w,mu,sd,n):
 k=torch.multinomial(w,n,replacement=True);return mu[k]+sd[k]*torch.randn(n,4,device=mu.device,dtype=mu.dtype)

@torch.no_grad()
def reference(actor,native,ds,ys,stage,seed):
 p.engine.seed_all(seed % (2**32));x=p.n.load_library().packed(ds[None],ys[None],stage)
 w,mu,sd=[v[0].double() for v in actor.query.inference.belief.params(x)]
 wn,mn,sn=[v[0].double() for v in native.params(x)]
 N=32768;particles=torch.cat([samples(w,mu,sd,24576),torch.randn(8192,4,device=mu.device,dtype=torch.float64)])
 proposal=torch.logaddexp(logmix(particles,w,mu,sd)+math.log(.75),logprior(particles)+math.log(.25))
 pred=p.signal(particles[:,None].reshape(N,1,2,2),ds[None,:stage].double())
 target=logprior(particles)-.5*((pred-ys[None,:stage].double())/.5).square().sum(-1)-stage*math.log(.5*math.sqrt(2*math.pi))
 logweight=target-proposal;weight=(logweight-logweight.max()).exp();weight=weight/weight.sum();ess=float(1/weight.square().sum());maxw=float(weight.max())
 means=[]
 for idx in [slice(0,16384),slice(16384,None)]:
  v=logweight[idx].softmax(0);means.append((v[:,None]*particles[idx]).sum(0))
 mean=(weight[:,None]*particles).sum(0);var=(weight[:,None]*(particles-mean).square()).sum(0)
 delta=logmix(particles,wn,mn,sn)-logmix(particles,w,mu,sd)
 return particles,weight,dict(ESS=ess,max_weight=maxw,mean=mean.tolist(),sd=var.sqrt().tolist(),
  reference_expected_logq_native_minus_inherited=float((weight*delta).sum()),
  limitation='Finite importance reference using .75 inherited-q + .25 prior;ESS can be low,not an exact posterior or guaranteed mode coverage. Half comparison is not valid convergence proof because stratified proposals differ.')

@torch.no_grad()
def run():
 assert not (R/'PREFLIGHT_RESULTS.json').exists();O.mkdir(parents=True,exist_ok=True);R.mkdir(parents=True,exist_ok=True)
 protocol=dict(study='NEW_V5_OFFLINE_BELIEF_AND_CRITIC_PREFLIGHT',seed=202611196101,episodes=64,stages=[5,15,25],reference_cases_per_stage=4,importance_particles=32768,
  suffix_rollouts=32,offline_only=True,training_updates=0,test_bank_read=False,
  proposed_change='Refresh actor and target frozen belief snapshots from their own native posterior BETWEEN fixed200-update training blocks;never in actor.forward or evaluation. No new inference losses or optimizer updates.',
  pilot_gate='Fresh development paired per-episode mean logq native-vs-inherited lower95% CI>0;all deployment checks pass. Reference/ranking limitations disclosed.',
  reference='Development-only finite importance reference and native remaining-horizon variational-return diagnostic,not a replacement EIG training objective or final scorer')
 put(R/'PROTOCOL.json',protocol);start=time.monotonic();p.setup()
 resource_out=O/('attempt'+str(len(list(O.glob('attempt[0-9]*')))+1).zfill(2));resource_out.mkdir()
 with p.lease(),p.engine.Resources(resource_out):
  path=p.O/'learned/runtime.pkl';cp=p.O/'learned/joint_actor.pt';before=p.sha(cp);state=load_runtime(path,'cuda');a=state['algorithm'];actor=a.actor_net;actor.configure('frozen');actor.eval();a.critic_net.requires_grad_(False);a.critic_net.eval()
  native=type(actor.query.inference.belief)(a.post_approx).cuda().eval();native.requires_grad_(False)
  rng=np.random.default_rng(protocol['seed']);truth=torch.tensor(rng.standard_normal((64,4)),device='cuda');noise=torch.tensor(rng.standard_normal((64,30)),device='cuda');ds=torch.zeros(64,30,2,device='cuda');ys=torch.zeros(64,30,device='cuda',dtype=torch.float64);X={};lib=p.n.load_library()
  cached=O/'development_histories.npz'
  if cached.exists():
   cached_data=np.load(cached);ds=torch.tensor(cached_data['designs'],device='cuda');ys=torch.tensor(cached_data['observations'],device='cuda');truth=torch.tensor(cached_data['truth'],device='cuda');X={stage:lib.packed(ds,ys,stage) for stage in protocol['stages']}
  else:
   for stage in range(30):
    x=lib.packed(ds,ys,stage)
    if stage in protocol['stages']:X[stage]=x.clone()
    action=actor(x);ds[:,stage]=action;ys[:,stage]=p.signal(truth.reshape(-1,2,2),action.double())+.5*noise[:,stage]
   np.savez(O/'development_histories.npz',designs=ds.cpu().numpy(),observations=ys.cpu().numpy(),truth=truth.cpu().numpy(),seed=protocol['seed'])
  audit=deployment_audit(actor,torch.cat(list(X.values())));put(R/'NO_ONLINE_OPTIMIZATION_AUDIT.json',audit)
  gaps=torch.stack([native.log_prob(X[s],truth.float())-actor.query.inference.belief.log_prob(X[s],truth.float()) for s in protocol['stages']],-1)
  score=p.engine.summaries(gaps.mean(-1).cpu().numpy());put(R/'POSTERIOR_LOG_SCORE.json',dict(per_episode_mean_difference=score,per_stage={str(s):p.engine.summaries(gaps[:,i].cpu().numpy()) for i,s in enumerate(protocol['stages'])},unit='log density nats,not EIG',fresh_development=True,truth_used_only_offline_scoring=True))
  references=[];critic_rows=[]
  for stage in protocol['stages']:
   for case in range(protocol['reference_cases_per_stage']):
    particles,weights,ref=reference(actor,native,ds[case],ys[case],stage,protocol['seed']+100*stage+case);references.append(dict(stage=stage,case=case,**ref))
    # Paired remaining-horizon native variational-return diagnostics. Not exact EIG.
    Mroll=protocol['suffix_rollouts'];theta=particles[torch.multinomial(weights,Mroll,replacement=True)];eps=torch.tensor(rng.standard_normal((Mroll,30)),device='cuda');x=X[stage][case:case+1];base,info=actor(x,collect=True)
    actions=[base[0]]
    for shift in [[.25,0],[-.25,0],[0,.25],[0,-.25]]:actions.append((base[0]+base.new_tensor(shift)).clamp(-4,4))
    for k in range(4):
     gates=.9*info['gates'];gates[:,k]+=.1;actions.append(actor.decode(x,info['z'],gates)[0][0])
    designs=torch.stack(actions);Q=a.critic_net(torch.cat([x.expand(9,-1),designs],-1)).flatten();returns=[]
    for candidate in designs:
     dd=ds[case:case+1].expand(Mroll,-1,-1).clone();yy=ys[case:case+1].expand(Mroll,-1).clone();dd[:,stage:]=0;yy[:,stage:]=0
     for t in range(stage,30):
      aa=candidate[None].expand(Mroll,-1) if t==stage else actor(lib.packed(dd,yy,t));dd[:,t]=aa;yy[:,t]=p.signal(theta.reshape(-1,2,2),aa.double())+.5*eps[:,t]
     inp=torch.cat([dd.flatten(1),yy.float()],-1);terminal=a.post_approx.poi_post_nets[29][0](inp,theta.float())
     # Prefix term is constant across candidate actions for these paired theta samples.
     prefix=native.log_prob(x.expand(Mroll,-1),theta.float());returns.append((terminal-prefix).cpu().numpy())
    returns=np.stack(returns);np.savez(O/f'suffix_stage{stage}_case{case}.npz',returns=returns,predicted_Q=Q.cpu().numpy())
    mean=returns.mean(-1);bestq=int(Q.argmax());bestmc=int(mean.argmax());corr=float(spearmanr(Q.cpu().numpy(),mean).statistic)
    critic_rows.append(dict(stage=stage,case=case,reference_ESS=ref['ESS'],predicted_best=bestq,rollout_mean_best=bestmc,spearman=corr if math.isfinite(corr) else None,
     Q_differences=(Q-Q[0]).cpu().tolist(),paired_return_differences=[p.engine.summaries(v-returns[0]) for v in returns],
     native_return='Frozen native terminal variational log-posterior minus prefix log-posterior;not exact remaining EIG. Importance posterior and32 rollouts add uncertainty;do not infer best action from means alone.'))
  put(R/'POSTERIOR_IMPORTANCE_REFERENCE.json',references);put(R/'CRITIC_RANKING_DIAGNOSTICS.json',critic_rows)
  assert p.sha(cp)==before
  result=dict(status='COMPLETE',deployment_audit='PASS',posterior_log_score=score,pilot_gate=score['CI95'][0]>0,seconds=time.monotonic()-start,
   source_checkpoint=str(cp),source_checkpoint_sha256=before,source_runtime=str(path),reference_reliable_case_count=sum(r['ESS']>=256 for r in references),reference_cases=len(references),
   critic_ranking_agreement=sum(r['predicted_best']==r['rollout_mean_best'] for r in critic_rows),critic_cases=len(critic_rows),training_updates=0,
   conclusion='Only posterior refresh is proposed for new V5 if the proper-score gate passes;critic diagnostics do not authorize a new critic objective. No guarantee of EIG improvement.')
  put(R/'PREFLIGHT_RESULTS.json',result);put(O/'STATUS.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':run()
