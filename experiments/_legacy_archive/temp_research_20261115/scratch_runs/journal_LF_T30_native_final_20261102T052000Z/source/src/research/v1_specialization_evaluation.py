
"""Independent v1 evaluator and joint-distribution, common-prefix expert diagnostics."""
import copy,hashlib,itertools,json,math,time
from pathlib import Path
from statistics import NormalDist
import numpy as np,torch
from torch import nn
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.deepseek_sboed_experiment import write,sha
from src.research.journal_validation import CommonValidation
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
from src.research.v1_specialization_experiment import paired

def install_arrays(val,directory,cases_name='cases.npz',bank_name='contrasts.npy'):
 manifest=directory/'manifest.json'
 if manifest.exists():
  data_manifest=json.loads(manifest.read_text())['file_SHA256']
  assert sha(directory/cases_name)==data_manifest[cases_name]
  assert sha(directory/bank_name)==data_manifest[bank_name]
 data=np.load(directory/cases_name);bank=np.load(directory/bank_name)
 val.truth=torch.as_tensor(data['truth'],device='cpu');val.noise=torch.as_tensor(data['noise'],device='cpu');val.bank=torch.as_tensor(bank,device='cpu')
 assert len(val.truth)==val.systems and tuple(val.noise.shape[:2])==(val.systems,val.horizon) and len(val.bank)==val.contrasts
 val.identity=hashlib.sha256(val.truth.numpy().tobytes()+val.noise.numpy().tobytes()+val.bank.numpy().tobytes()).hexdigest()
 return val

class FrozenIntervention(nn.Module):
 kind='deepseek_moe_sboed';outputs_design=False
 def __init__(self,model,mode='full',stage=None,expert_set=None,seed=97668001):
  super().__init__();self.policy=copy.deepcopy(model).eval();self.mode=mode;self.stage=stage;self.expert_set=expert_set;self.seed=seed;self.generator=None
 def forward(self,f,t):
  m=self.policy
  if self.mode=='full' or (self.mode=='forced_set' and t!=self.stage):return m(f,t)
  z=m.encode_context(f,t);p=m.router(z).softmax(-1);weights,ids=p.topk(3,-1)
  if self.mode=='forced_set':
   ids=torch.tensor([self.expert_set],device=z.device).expand(len(z),-1);weights=p.gather(1,ids)
  elif self.mode=='uniform_subset':
   if self.generator is None:self.generator=torch.Generator(device=z.device).manual_seed(self.seed)
   ids=torch.rand(len(z),7,device=z.device,generator=self.generator).topk(3,-1).indices
   weights=weights.sum(-1,keepdim=True).expand(-1,3)/3
  elif self.mode=='uniform_weights':weights=weights.sum(-1,keepdim=True).expand(-1,3)/3
  shared=m.shared(z) if self.mode!='mask_shared' else z*0;route=z*0
  if self.mode!='mask_routed':
   for e in ids.unique().tolist():
    rows,slots=(ids==e).nonzero(as_tuple=True)
    route.index_add_(0,rows,m.routed[e](z[rows])*weights[rows,slots,None])
  return m.emitter(z+shared+route)

def canonical_validator(cfg):
 val=CommonValidation(cfg['benchmark'],30,1001,2000,1000000,batch=16,chunk=16384,device='cuda:0')
 root=Path(cfg['canonical_evaluation_directory'])
 assert sha(root/'location_cases.npz')==cfg['cases_sha256'];assert sha(root/'location_contrasts.npy')==cfg['contrast_sha256']
 return install_arrays(val,root,'location_cases.npz','location_contrasts.npy')

def evaluate_once(C,cfg,model,validator,label,checkpoint_path=None,evaluated_training_seed=None):
 p=C/'evaluations'/f'{label}_scores.npy';j=C/'reports'/f'{label}_result.json'
 if p.exists() or j.exists():raise RuntimeError('Preserve existing evaluation; inspect rather than duplicate '+label)
 scores,meta=validator(model);np.save(p,scores)
 result=dict(label=label,method=cfg['method'] if evaluated_training_seed is None else 'frozen_reference_control',EIG=float(scores.mean()),MC_SE=float(scores.std(ddof=1)/len(scores)**.5),training_seed=cfg['training_seed'] if evaluated_training_seed is None else evaluated_training_seed,evaluation_seed=validator.seed,trajectories=len(scores),contrasts=len(validator.bank),horizon=30,metadata=meta,checkpoint_sha256=sha(Path(checkpoint_path) if checkpoint_path else C/'models/full/selected.pt'))
 write(j,result);print('EVALUATION',json.dumps(result),flush=True);return scores

@torch.no_grad()
def common_prefix_scores(model,validator,stage,forced=None,return_features=False):
 policy=model if forced is None else FrozenIntervention(model,'forced_set',stage,forced).to(validator.device)
 b=make_benchmark(validator.raw);scores=[];features=[];latents=[];before=time.perf_counter()
 with torch.device(validator.device):
  for start in range(0,validator.systems,validator.batch):
   n=min(validator.batch,validator.systems-start)
   r=rollout(b,policy,30,n,validator.seed+start*97451,0,truth=validator.truth[start:start+n].to(validator.device),noise=validator.noise[start:start+n].to(validator.device))
   values,_=contrast_scores(b,r,validator.bank,validator.chunk);scores.extend(values[:,-1].cpu().tolist())
   features.append(r['features'][stage].cpu().numpy())
   if return_features:latents.append(model.encode_context(r['features'][stage],stage).cpu().numpy())
 f=np.concatenate(features);identity=hashlib.sha256(f.tobytes()).hexdigest()
 return np.asarray(scores),dict(prefix_SHA256=identity,seconds=time.perf_counter()-before,parameter_stage_propagations=b.calls,latent=np.concatenate(latents) if latents else None)

@torch.no_grad()
def competence(C,cfg,model,validator):
 """Group-conditional joint averages, NOT a true-theta posterior for a fixed history."""
 begin=time.perf_counter();sets=list(itertools.combinations(range(7),3));records=[];training_context={}
 path=C/'costs/context_gradient_steps.jsonl'
 if path.exists():
  for line in path.read_text().splitlines():
   d=json.loads(line);training_context[d['stage']]=dict(coordinate=d['coordinate'],threshold=d['threshold'])
 for stage in [1,10,20]:
  if time.perf_counter()-begin>600:break
  baseline,meta=common_prefix_scores(model,validator,stage,return_features=True)
  specification=training_context.get(stage)
  if specification is None:
   records.append(dict(stage=stage,status='NO_TRAINING_CONTEXT_CLASSIFIER'));continue
  group=(meta['latent'][:,specification['coordinate']]>specification['threshold']).astype(int)
  values=[];tested=[];work=meta['parameter_stage_propagations']
  for j,S in enumerate(sets):
   if time.perf_counter()-begin>600:break
   v,m=common_prefix_scores(model,validator,stage,S)
   assert m['prefix_SHA256']==meta['prefix_SHA256'],'Interventions did not share exactly the observed prefix'
   values.append(v);tested.append(j);work+=m['parameter_stage_propagations']
  matrix=np.stack(values) if values else np.empty((0,len(baseline)))
  np.savez(C/'evaluations'/f'context_competence_stage{stage}.npz',scores=matrix,baseline=baseline,group=group,tested=np.asarray(tested))
  # Conservative family for every pair of35 sets,2 contexts,3 stages.
  z=NormalDist().inv_cdf(1-.05/(2*3*2*595));summary=[]
  for cohort in [0,1]:
   mask=group==cohort;n=int(mask.sum())
   if n<32:summary.append(dict(context=cohort,N=n,status='UNDERPOWERED'));continue
   difference=matrix[:,mask]-baseline[None,mask];mean=difference.mean(-1);se=difference.std(-1,ddof=1)/n**.5
   summary.append(dict(context=cohort,N=n,status='FINITE_JOINT_GROUP_AVERAGE',advantage_over_learned=mean.tolist(),paired_MC_SE=se.tolist(),familywise_z=z))
  # Select diagnostic contextual fixed sets on even cases; validate on odd cases.
  # This selects no trained checkpoint and changes no deployed policy.
  exploration=np.arange(len(baseline))%2==0;confirmation=~exploration
  composite=None
  if len(values)==35 and all(int(((group==c)&exploration).sum())>=16 and int(((group==c)&confirmation).sum())>=16 for c in [0,1]):
   selected=[int(matrix[:,exploration&(group==c)].mean(-1).argmax()) for c in [0,1]]
   blind=int(matrix[:,exploration].mean(-1).argmax())
   context_values=np.where(group==0,matrix[selected[0]],matrix[selected[1]])
   composite=dict(selected_sets=[list(sets[j]) for j in selected],blind_set=list(sets[blind]),N_confirmation=int(confirmation.sum()),contextual_minus_blind=paired(context_values[confirmation],matrix[blind,confirmation]),learned_minus_contextual=paired(baseline[confirmation],context_values[confirmation]),selection_data='Even development cases only; reported comparisons on odd cases only',diagnostic_not_deployment=True)
  row=dict(stage=stage,training_only_context_classifier=specification,tested_sets=[list(sets[j]) for j in tested],contexts=summary,composite_crossfit=composite,parameter_stage_propagations=work,prefix_SHA256=meta['prefix_SHA256'])
  records.append(row);write(C/'reports/context_competence.json',dict(records=records,wall_seconds=time.perf_counter()-begin,estimand='Differences of full-horizon information utility averaged over independently generated histories in a predeclared observed-history group. Prefixes,truth,noise and contrast bank are paired. This is not a precise fixed-history posterior-Q estimate.',generating_theta_not_a_fixed_history_posterior=True))
 return records

def finish(C,cfg,model):
 checkpoint=C/'models/full/selected.pt';before=sha(checkpoint)
 banks=Path.home()/'Documents/deepseek_v1_specialization_banks'
 development=install_arrays(CommonValidation(cfg['benchmark'],30,cfg['final_development_seed'],512,262144,batch=16,chunk=16384,device='cuda:0'),banks/'development')
 comparisons={};base=None;dev_values={}
 for mode in ['full','uniform_weights','uniform_subset','mask_routed','mask_shared']:
  arrays=[];metas=[]
  for rep in range(3 if mode=='uniform_subset' else 1):
   v,m=development(model if mode=='full' else FrozenIntervention(model,mode,seed=97668001+rep).cuda());arrays.append(v);metas.append(m)
  values=np.mean(arrays,0);dev_values[mode]=values
  if base is None:base=values
  np.save(C/'evaluations'/f'development_{mode}.npy',values)
  comparisons[mode]=dict(EIG=float(values.mean()),MC_SE=float(values.std(ddof=1)/len(values)**.5),paired_intervention_minus_full=paired(values,base),replicates=len(arrays),metadata=metas,frozen_intervention=mode!='full')
 original=load_deepseek('reference_artifacts/v1/selected.pt','cuda')
 values,meta=development(original);np.save(C/'evaluations/development_frozen_v1.npy',values)
 comparisons['frozen_v1']=dict(EIG=float(values.mean()),MC_SE=float(values.std(ddof=1)/len(values)**.5),new_minus_v1=paired(base,values),metadata=meta)
 controls={}
 for variant in ['dense','shared_only']:
  p=Path('reference_artifacts/controls')/(variant+'.pt')
  if not p.exists():comparisons[variant]=dict(status='UNAVAILABLE');continue
  state=torch.load(p,map_location='cpu',weights_only=False)
  assert state['variant']==variant and state['configuration']['contrasts']==2000
  control=DeepSeekPolicy(variant=variant);control.load_state_dict(state['state_dict'],strict=True);control=control.cuda().eval();controls[variant]=control
  values,meta=development(control);np.save(C/'evaluations'/f'development_control_{variant}.npy',values)
  comparisons[variant]=dict(EIG=float(values.mean()),MC_SE=float(values.std(ddof=1)/len(values)**.5),new_minus_control=paired(base,values),counts=control.counts(),checkpoint_sha256=sha(p),metadata=meta,scope='Previously trained reference, original v1 architecture variant; historical throughput/training update differences remain limitations')
 write(C/'reports/development_comparisons.json',comparisons)
 # Freeze was written before all candidate scoring. Official result never selects a model.
 official=canonical_validator(cfg)
 scores=evaluate_once(C,cfg,model,official,'official_seed1001')
 reference=np.load('reference_artifacts/v1/official_seed1001_scores.npy')
 assert official.identity==json.loads(Path('reference_artifacts/v1/official_seed101_result.json').read_text())['metadata']['paired_draws_sha256']
 paired_results={'v1':paired(scores,reference)}
 references=Path('reference_artifacts/evaluation_reference_scores.json')
 if references.exists():
  d=json.loads(references.read_text());assert d['paired_draws_sha256']==official.identity
  for label,values in d['scores'].items():paired_results[label]=paired(scores,values)
 write(C/'reports/official_paired_comparisons.json',paired_results)
 confirmation=install_arrays(CommonValidation(cfg['benchmark'],30,cfg['confirmation_seed'],2000,1000000,batch=16,chunk=16384,device='cuda:0'),banks/'confirmation')
 evaluate_once(C,cfg,model,confirmation,'confirmation_seed'+str(cfg['confirmation_seed']))
 # Reference controls are evaluated independently; do not conflate interventions with retraining.
 for variant,control in controls.items():
  v=evaluate_once(C,cfg,control,official,'control_'+variant+'_seed1001',checkpoint_path=Path('reference_artifacts/controls')/(variant+'.pt'),evaluated_training_seed=101)
  paired_results[variant]=paired(scores,v)
 write(C/'reports/official_paired_comparisons.json',paired_results)
 context=competence(C,cfg,model,development)
 # All neural decisions use the unchanged v1 forward; histories are observed simulator outputs.
 b=make_benchmark(cfg['benchmark'])
 with torch.no_grad(),torch.device('cuda'):r=rollout(b,model,30,32,97669001,0)
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(cfg['training_seed']);initial=DeepSeekPolicy().cuda().eval()
 latencies=[];features=[]
 with torch.no_grad():
  for t,f in enumerate(r['features']):
   z=model.encode_context(f,t);p=model.router(z).softmax(-1);weights,ids=p.topk(3,-1);all_features=torch.stack([e(z) for e in model.routed],1)
   route=(all_features.gather(1,ids[:,:,None].expand(-1,-1,16))*weights[:,:,None]).sum(1);shared=model.shared(z)
   full=b.feasible(model.emitter(z+shared+route),r['actions'][:t],30);masked=b.feasible(model.emitter(z+shared),r['actions'][:t],30)
   initial_features=torch.stack([e(z) for e in initial.routed],1)
   unit=torch.nn.functional.normalize(all_features,dim=-1)
   features.append(dict(stage=t,expert_function_change_L2=(all_features-initial_features).norm(dim=-1).mean(0).cpu().tolist(),shared_norm=float(shared.norm(dim=-1).mean()),routed_norm=float(route.norm(dim=-1).mean()),routed_action_removal_RMS=float((full-masked).square().mean().sqrt()),cosine=torch.einsum('bed,bfd->bef',unit,unit).mean(0).cpu().tolist(),utilization=torch.nn.functional.one_hot(ids,7).sum(1).double().mean(0).cpu().tolist()))
  for rep in range(12):
   for t,f in enumerate(r['features']):
    torch.cuda.synchronize();start=time.perf_counter();model(f[:1],t);torch.cuda.synchronize()
    if rep>=2:latencies.append((time.perf_counter()-start)*1000)
 write(C/'reports/expert_features.json',dict(stages=features,parameter_stage_propagations=b.calls))
 write(C/'reports/inference_latency.json',dict(mean_ms=float(np.mean(latencies)),median_ms=float(np.median(latencies)),p95_ms=float(np.quantile(latencies,.95)),decisions=len(latencies),frozen_forward_only=True))
 assert sha(checkpoint)==before
 from src.research.v1_specialization_reports import write_results
 write_results(C,cfg)
