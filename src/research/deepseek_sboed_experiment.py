"""One predeclared DeepSeekMoE-sBOED run, canonical DAD rollout and evaluator."""
import time
_ENTRY_START=time.monotonic()
import argparse,copy,hashlib,json,math,subprocess,traceback
from pathlib import Path
import numpy as np,torch,yaml
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.dad_location_policy import LocationDAD
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.compute_budget import ComputeClock
from src.research.journal_validation import CommonValidation
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class RandomControl(torch.nn.Module):
    kind='untrained_random';outputs_design=False
    def __init__(self,raw,seed):
        super().__init__();self.b=make_benchmark(raw);self.seed=seed;self.g=None
    def forward(self,f,stage):
        if self.g is None:self.g=torch.Generator(device=f.device).manual_seed(self.seed)
        return self.b.random_latent(len(f),self.g)



def norm(parameters):
 values=[p.grad.detach().norm() for p in parameters if p.grad is not None]
 return float(torch.stack(values).norm()) if values else 0.

def save(model,path,update,cfg,opt=None,scheduler=None):
 state=dict(method=getattr(model,'method','DeepSeekMoE-sBOED') if isinstance(model,DeepSeekPolicy) else 'corrected_dense_DAD_control',horizon=30,action_scale=4.,obs_scale=3.,variant=getattr(model,'variant','dad'),balance_factor=getattr(model,'balance_factor',0.),state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},training_seed=101,update=update,configuration=cfg,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 if opt is not None:state.update(optimizer=opt.state_dict(),scheduler=scheduler.state_dict())
 torch.save(state,path)

def dense_reference(cfg):
 if sha(cfg['corrected_DAD_checkpoint'])!=cfg['corrected_DAD_checkpoint_sha256']:raise ValueError('Corrected DAD reference changed')
 x=torch.load(cfg['corrected_DAD_checkpoint'],map_location='cpu',weights_only=False);p=LocationDAD(30,4.,3.,first_design_revision=1);p.load_state_dict(x['state_dict'],strict=True);return p.eval()

def diagnostics(model,result,gradient,loss,info,update,seconds):
 actions=torch.stack([a.detach() for a in result['actions']],1);jumps=(actions[:,1:]-actions[:,:-1]).norm(dim=-1)
 row=dict(update=update,train_spce=float(info.detach()),total_loss=float(loss.detach()),preclip_gradient_norm=gradient,charged_seconds=seconds,action_boundary_fraction=float((actions.abs()>3.95).double().mean()),nearly_repeated_fraction=float((jumps<.001).double().mean()),action_std=actions.flatten(0,1).std(0).cpu().tolist())
 if isinstance(model,DeepSeekPolicy):
  row.update(routing=model.routing_statistics(),shared_gradient=norm(model.shared.parameters()) if model.shared is not None else None,router_gradient=norm(model.router.parameters()) if model.router is not None else None,routed_gradient=[norm(e.parameters()) for e in model.routed],normalized_late_context_RMS=float(model.last_features.square().mean().sqrt()),raw_encoder_has_no_bounded_saturating_activation=True)
 return row

def train(C,cfg,variant,updates,budget,validator,credit=0.,start=None,policy_factory=None):
 out=C/'models'/variant;out.mkdir(parents=True,exist_ok=False);clock=ComputeClock(out/'compute_clock.json',budget,credit_seconds=credit,hardware=torch.cuda.get_device_name(0),start=start);torch.manual_seed(101)
 model=(LocationDAD(30,4.,3.,first_design_revision=1) if variant=='dad' else (policy_factory(variant=variant) if policy_factory is not None else DeepSeekPolicy(variant=variant))).cuda();model.train();
 if variant=='full' and cfg.get('charge_readiness_probe',True):
  probe=json.loads((C/'readiness/native_full_minibatch_probe.json').read_text());B0=probe['batch_size'];L0=probe['contrasts']
  for unit,n in [('fresh_complete_primary_trajectories',B0),('true_parameter_stages',B0*30),('contrast_parameter_stages',B0*L0*30),('scalar_conditional_likelihood_evaluations',B0*(L0+1)*30),('policy_optimizer_updates',1)]:clock.count(unit,n,component='readiness_probe_discarded_weights')
 b=make_benchmark(cfg['benchmark']);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']));scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,cfg['lr_decay']);records=[];evaluations=[];bound=1.;completed=0;stop='incomplete';unstable=collapsed=0
 try:
  for update in range(1,updates+1):
   if not clock.can_start(max(2.,bound*1.5)):stop='compute_budget_exhaustion';break
   unit=time.monotonic();charged_unit=clock.spent();opt.zero_grad()
   with torch.device('cuda'):
    if cfg.get('training_likelihood_backend') == 'streamed_checkpointed':
     from src.research.deepseek_sboed_v3 import streamed_spce
     r=streamed_spce(b,model,30,cfg['batch_size'],101+update*104729,cfg['contrasts'],cfg['contrast_chunk'])
    else:
     r=rollout(b,model,30,cfg['batch_size'],101+update*104729,cfg['contrasts'])
    info=r['info'][:,-1].mean();balance=model.balance_loss() if isinstance(model,DeepSeekPolicy) else info*0;loss=-info+balance
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-horizon information objective')
    loss.backward()
    if hasattr(model,'mask_unused_gradients'):model.mask_unused_gradients()
    gradient=norm(model.parameters())
    if not math.isfinite(gradient):raise FloatingPointError('Nonfinite native gradient')
    if variant!='dad':torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
    opt.step();torch.cuda.synchronize()
   completed=update
   if (update-1)%1000==0:scheduler.step()
   B,L=cfg['batch_size'],cfg['contrasts'];clock.count('fresh_complete_primary_trajectories',B);clock.count('true_parameter_stages',B*30);clock.count('contrast_parameter_stages',B*L*30);clock.count('scalar_conditional_likelihood_evaluations',B*(L+1)*30);clock.count('policy_optimizer_updates',1)
   if 'work_counts' in r:
    for key,value in r['work_counts'].items():clock.count(key,value)
    clock.count('contrast_recomputation_parameter_stages',r['work_counts']['executed_parameter_stage_propagations']-B*(L+1)*30)
   if isinstance(model,DeepSeekPolicy):clock.count('shared_FFN_contexts',B*30 if model.shared is not None else 0);clock.count('routed_FFN_contexts',B*30*model.top_k)
   if update==1 or update%100==0:
    row=diagnostics(model,r,gradient,loss,info,update,clock.spent());row['learning_rate']=opt.param_groups[0]['lr'];records.append(row);print(variant,row,flush=True);write(C/'costs'/f'{variant}_progress.json',dict(records=records,clock=clock.report(),forward_parameter_stages=b.calls,counts=model.counts() if isinstance(model,DeepSeekPolicy) else dict(total_parameters=sum(p.numel() for p in model.parameters()))))
    unstable=unstable+1 if (gradient>1000 or row.get('normalized_late_context_RMS',0)>10) and update>=1000 else 0
    routing=row.get('routing',{});collapsed=collapsed+1 if routing and routing['distinct_selected_experts']<=model.top_k and update>=1000 else 0
    if unstable>=5:stop='persistent_gradient_instability';break
    if collapsed>=5:stop='persistent_routing_collapse';break
   if update%cfg['checkpoint_every']==0 or update==updates:
    path=out/f'update{update}.pt';save(model,path,update,cfg,opt,scheduler);clock.checkpoint(str(path))
   if variant=='full' and (update%cfg['development_every']==0 or update==cfg['control_updates']):
    with clock.exclude('validation'):
     values,meta=validator(model);np.save(C/'evaluations'/f'development_update{update}.npy',values);evaluations.append(dict(update=update,mean=float(values.mean()),MC_SE=float(values.std(ddof=1)/math.sqrt(len(values))),metadata=meta));write(C/'reports/full_development_curve.json',evaluations)
    model.train()
   bound=max(bound,clock.spent()-charged_unit)
  else:stop='reference_schedule_completion'
  if completed==0:raise RuntimeError('No native optimizer update completed')
  # Final serialization is charged; eligibility is assessed after checkpoint IO.
  path=out/f'update{completed}_final.pt';save(model,path,completed,cfg,opt,scheduler);clock.checkpoint(str(path));selected=clock.selected();state=torch.load(selected['identifier'],map_location='cpu',weights_only=False);model.load_state_dict(state['state_dict']);save(model,out/'selected.pt',state['update'],cfg)
  write(C/'costs'/f'{variant}_final.json',dict(stop_reason=stop,executed_updates=completed,selected_update=state['update'],checkpoint_sha256=sha(out/'selected.pt'),selected_checkpoint=selected,clock=clock.stop(),forward_parameter_stages=b.calls,CPU_process_seconds=time.process_time(),GPU_active_seconds=None,counts=model.counts() if isinstance(model,DeepSeekPolicy) else dict(total_parameters=sum(p.numel() for p in model.parameters())),reference_minibatch_preserved=True,planned_updates_this_phase=updates,cheap_control_only=variant!='full',training_reference_updates=50000,training_reference_episodes=100000000,legacy_episode_cap=None))
  return model.eval(),stop
 except Exception as exc:
  clock.stop();write(C/'reports'/f'{variant}_failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),updates=completed,clock=clock.report()));raise

def feature_diagnostics(model,validator,C):
 b=make_benchmark(validator.raw)
 with torch.no_grad(),torch.device('cuda'):r=rollout(b,model,30,32,10310001,0)
 stages=[];similarity=[]
 with torch.no_grad():
  for t in [1,5,15,29]:
   z=model.encode_context(r['features'][t],t);features=torch.stack([e(z) for e in model.routed],1);unit=torch.nn.functional.normalize(features,dim=-1);cos=torch.einsum('bed,bfd->bef',unit,unit).mean(0)
   stages.append(dict(stage=t,mean_pairwise_cosine=cos.cpu().tolist(),feature_variance_by_expert=features.var(0).mean(-1).cpu().tolist()));similarity.append(cos.cpu().numpy())
 np.save(C/'evaluations/routed_feature_similarity.npy',np.stack(similarity));write(C/'reports/feature_function_diagnostics.json',dict(stages=stages,warning='Different feature functions or routing indices alone do not establish useful specialization'))


def run(C,cfg):
 from tools.journal_campaign import exclusive_device_lock
 lock=exclusive_device_lock();torch.set_num_threads(2);started=time.monotonic();status='FAILED';setup_ready=time.monotonic();dev_begin=time.monotonic();dev=CommonValidation(cfg['benchmark'],30,cfg['development_seed'],128,262144,batch=16,chunk=16384,device='cuda:0');np.savez(C/'evaluations/development_cases.npz',truth=dev.truth.numpy(),noise=dev.noise.numpy());np.save(C/'evaluations/development_contrasts.npy',dev.bank.numpy());write(C/'costs/setup.json',dict(module_startup_and_common_setup_seconds=setup_ready-_ENTRY_START,development_bank_generation_seconds=time.monotonic()-dev_begin,OS_process_launch_seconds=None,not_GPU_active_time=True))
 try:
  # Cheap controls use identical nominal minibatches, updatecounts and simulator RNGs.
  for k,variant in enumerate(['dad','shared_only','routed_only','dense']):
   model,stop=train(C,cfg,variant,cfg['control_updates'],cfg['control_seconds'],dev)
   values,meta=dev(model);np.save(C/'evaluations'/f'cheap_{variant}.npy',values);write(C/'reports'/f'cheap_{variant}.json',dict(mean=float(values.mean()),MC_SE=float(values.std(ddof=1)/math.sqrt(len(values))),updates=cfg['control_updates'],metadata=meta,not_full_schedule_result=True))
   del model;torch.cuda.empty_cache()
  credit=json.loads((C/'readiness/native_full_minibatch_probe.json').read_text())['seconds']+(setup_ready-_ENTRY_START);model,stop=train(C,cfg,'full',cfg['reference_optimizer_updates'],cfg['main_training_seconds'],dev,credit=credit)
  values,meta=dev(model);dense=dense_reference(cfg);reference,reference_meta=dev(dense)
  random_policy=RandomControl(cfg['benchmark'],cfg['development_seed']+700001);random,random_meta=dev(random_policy)
  d=values-reference;se=float(d.std(ddof=1)/math.sqrt(len(d)));dr=values-random;ser=float(dr.std(ddof=1)/math.sqrt(len(dr)));passed=stop in ['reference_schedule_completion','compute_budget_exhaustion'] and float(d.mean())>=0 and float(d.mean())-1.96*se>=-.1 and float(dr.mean())-1.96*ser>0
  gate=dict(PASS=passed,stop_reason=stop,full=dict(mean=float(values.mean()),MC_SE=float(values.std(ddof=1)/math.sqrt(len(values)))),corrected_DAD=dict(mean=float(reference.mean()),MC_SE=float(reference.std(ddof=1)/math.sqrt(len(reference)))),random=dict(mean=float(random.mean()),MC_SE=float(random.std(ddof=1)/math.sqrt(len(random)))),paired_difference_vs_DAD=float(d.mean()),paired_MC_SE_vs_DAD=se,paired_difference_vs_random=float(dr.mean()),paired_MC_SE_vs_random=ser,selection='Latest eligible checkpoint; no official-data-based selection',development_seed=cfg['development_seed'],official_seed1001_used=False,development_metadata=meta);write(C/'readiness/final_development_gate.json',gate);np.save(C/'evaluations/final_development_full.npy',values);np.save(C/'evaluations/final_development_DAD.npy',reference);np.save(C/'evaluations/final_development_random.npy',random)
  feature_diagnostics(model,dev,C)
  if not passed:status='NEGATIVE_DEVELOPMENT_GATE_FAILED_NO_OFFICIAL_EVALUATION';return
  path=C/'models/full/selected.pt';write(C/'reports/final_freeze.json',dict(checkpoint_sha256=sha(path),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),gate=gate))
  official=CommonValidation(cfg['benchmark'],30,1001,2000,1000000,batch=16,chunk=16384,device='cuda:0');canonical=Path(cfg['canonical_evaluation_directory'])
  for filename,key in [('location_cases.npz','cases_sha256'),('location_contrasts.npy','contrast_sha256')]:
   if sha(canonical/filename)!=cfg[key]:raise ValueError('Official paired bank mismatch')
  cases=np.load(canonical/'location_cases.npz');official.truth=torch.as_tensor(cases['truth']);official.noise=torch.as_tensor(cases['noise']);official.bank=torch.as_tensor(np.load(canonical/'location_contrasts.npy'));official.identity=hashlib.sha256(official.truth.numpy().tobytes()+official.noise.numpy().tobytes()+official.bank.numpy().tobytes()).hexdigest()
  scores,meta=official(model);np.save(C/'evaluations/official_seed1001_scores.npy',scores);write(C/'reports/official_seed101_result.json',dict(EIG=float(scores.mean()),MC_SE=float(scores.std(ddof=1)/math.sqrt(2000)),training_seed=101,evaluation_seed=1001,trajectories=2000,contrasts=1000000,horizon=30,checkpoint_sha256=sha(path),metadata=meta,cases_sha256=cfg['cases_sha256'],contrast_sha256=cfg['contrast_sha256']));status='OFFICIAL_EVALUATION_COMPLETE_ONE_TRAINING_SEED'
 except Exception as exc:
  write(C/'reports/campaign_failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));raise
 finally:write(C/'registry.json',dict(status=status,total_campaign_wall_seconds=time.monotonic()-started,hardware=torch.cuda.get_device_name(0),historical_runs_untouched=True,official_bank_consumed_only_if_gate_passed=True));print('DEEPSEEK_STATUS',status,flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--deepseek-sboed',action='store_true');p.add_argument('--campaign',required=True);p.add_argument('--config',required=True);a=p.parse_args();C=Path(a.campaign).resolve();cfg=yaml.safe_load(Path(a.config).read_text());run(C,cfg)
if __name__=='__main__':main()
