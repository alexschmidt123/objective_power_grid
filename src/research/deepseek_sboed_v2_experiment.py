"""One throughput-only v2 primary run and predeclared full-budget controls."""
import time
_ENTRY_START=time.monotonic()
import argparse,json,hashlib,math,subprocess
from pathlib import Path
import numpy as np,torch,yaml
from src.research.deepseek_sboed import load_deepseek
from src.research.deepseek_sboed_v2 import DeepSeekV2Policy,load_v2
from src.research.deepseek_sboed_experiment import train,write,sha,feature_diagnostics,dense_reference
from src.research.journal_validation import CommonValidation
from tools.deepseek_v1_diagnostics import FrozenIntervention


def paired(a,b):
 d=np.asarray(a)-np.asarray(b);return dict(difference=float(d.mean()),paired_MC_SE=float(d.std(ddof=1)/math.sqrt(len(d))),interval95=[float(d.mean()-1.96*d.std(ddof=1)/math.sqrt(len(d))),float(d.mean()+1.96*d.std(ddof=1)/math.sqrt(len(d)))])

def evaluate(C,cfg,model,seed,label):
 val=CommonValidation(cfg['benchmark'],30,seed,2000,1000000,batch=16,chunk=16384,device='cuda:0')
 if seed==1001:
  root=Path(cfg['canonical_evaluation_directory']);cases=root/'location_cases.npz';contrasts=root/'location_contrasts.npy';expected={cases.name:cfg['cases_sha256'],contrasts.name:cfg['contrast_sha256']}
 else:
  root=C/'evaluations/confirmation_reserved';cases=root/'cases.npz';contrasts=root/'contrasts.npy';manifest=json.loads((root/'manifest.json').read_text());expected=manifest['file_SHA256']
 for path in [cases,contrasts]:
  if sha(path)!=expected[path.name]:raise ValueError('Paired evaluation bank changed')
 data=np.load(cases);val.truth=torch.as_tensor(data['truth']);val.noise=torch.as_tensor(data['noise']);val.bank=torch.as_tensor(np.load(contrasts));val.identity=hashlib.sha256(val.truth.numpy().tobytes()+val.noise.numpy().tobytes()+val.bank.numpy().tobytes()).hexdigest();scores,meta=val(model)
 np.save(C/'evaluations'/f'{label}_seed{seed}_scores.npy',scores);result=dict(model=label,EIG=float(scores.mean()),MC_SE=float(scores.std(ddof=1)/math.sqrt(len(scores))),training_seed=101,evaluation_seed=seed,trajectories=2000,contrasts=1000000,horizon=30,metadata=meta,comparison_type='historical_consumed_benchmark' if seed==1001 else 'new_reserved_confirmation_bank')
 write(C/'reports'/f'{label}_seed{seed}_result.json',result);print('OFFICIAL_RESULT',result,flush=True);return scores


def interventions(C,cfg,model,label,validator):
 rows=[];base=None
 for mode in ['full','shared_only','routed_only','uniform_weights_fixed_selection','uniform_subset_scale_preserved']:
  arrays=[];meta=[]
  for rep in range(3 if mode=='uniform_subset_scale_preserved' else 1):
   p=FrozenIntervention(model,mode,10310007+rep).cuda();v,m=validator(p);arrays.append(v);meta.append(m)
  values=np.mean(arrays,0)
  if mode=='full':base=values
  row=dict(mode=mode,mean=float(values.mean()),MC_SE=float(values.std(ddof=1)/math.sqrt(len(values))),paired_vs_full=paired(values,base),metadata=meta,frozen_intervention=True,not_retrained=True);rows.append(row);np.save(C/'evaluations'/f'{label}_{mode}_development.npy',values)
 write(C/'reports'/f'{label}_frozen_interventions.json',rows)


def run(C,cfg,smoke=False):
 from tools.journal_campaign import exclusive_device_lock
 lock=exclusive_device_lock();torch.set_num_threads(2);start=time.monotonic();status='FAILED';reference=Path(cfg['reference_campaign'])
 acceptance=json.loads((C/'readiness/native_backend_equivalence_and_speed.json').read_text())
 if not acceptance['PASS']:raise RuntimeError('Equivalence/native speedup gate failed; no training')
 if smoke:
  out=C/'readiness/canonical_smoke';out.mkdir(exist_ok=False)
  for name in ['models','reports','costs','evaluations']:(out/name).mkdir()
  smoke_cfg=dict(cfg,control_updates=500,development_every=5000,checkpoint_every=1)
  model,stop=train(out,smoke_cfg,'full',2,60,None,policy_factory=DeepSeekV2Policy)
  write(C/'readiness/canonical_smoke_result.json',dict(PASS=True,stop=stop,updates=2,batch=2000,contrasts=2000,checkpoint_sha256=sha(out/'models/full/selected.pt'),weights_reused_in_main=False));return
 try:
  val_old=CommonValidation(cfg['benchmark'],30,1030,128,262144,batch=16,chunk=16384,device='cuda:0');val_new=CommonValidation(cfg['benchmark'],30,1031,256,262144,batch=16,chunk=16384,device='cuda:0')
  # Fresh initialization. The v1frozen weights were ONLY used for diagnostics.
  model,stop=train(C,cfg,'full',50000,14400,val_old,start=_ENTRY_START,policy_factory=DeepSeekV2Policy)
  a,meta=val_old(model);b,meta_new=val_new(model);old=np.load(reference/'evaluations/final_development_full.npy');fresh_old=np.load(C/'diagnostics/full_0.npy');old_delta=paired(a,old);new_delta=paired(b,fresh_old)
  passed=stop in ['reference_schedule_completion','compute_budget_exhaustion'] and old_delta['difference']>=0 and new_delta['difference']>=0 and old_delta['interval95'][0]>=-.1 and new_delta['interval95'][0]>=-.1
  write(C/'readiness/v2_development_gate.json',dict(PASS=passed,stop_reason=stop,seed1030=dict(v2_mean=float(a.mean()),v1_mean=float(old.mean()),paired=old_delta),seed1031=dict(v2_mean=float(b.mean()),v1_mean=float(fresh_old.mean()),paired=new_delta),rule='Notworsepointestimateonbothmatchedbanks; pairedlower95>=-.1;stabletraining',official_data_used=False));np.save(C/'evaluations/v2_final_development1030.npy',a);np.save(C/'evaluations/v2_final_development1031.npy',b)
  path=C/'models/full/selected.pt';write(C/'reports/v2_checkpoint_freeze.json',dict(checkpoint_sha256=sha(path),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),selection='Latest valid4hourcheckpoint',development_gate=passed,all_choices_frozen_before_official=True))
  feature_diagnostics(model,val_new,C);interventions(C,cfg,model,'v2',val_new)
  if passed:
   historical=evaluate(C,cfg,model,1001,'v2');original_official=np.load(reference/'evaluations/official_seed1001_scores.npy');write(C/'reports/historical_paired_v2_minus_v1.json',paired(historical,original_official))
   # Open reservedbank for scoring ONLY after the v2checkpoint is frozen.
   original=load_deepseek(reference/'models/full/selected.pt','cuda');v1confirm=evaluate(C,cfg,original,1032,'v1');v2confirm=evaluate(C,cfg,model,1032,'v2');write(C/'reports/confirmation_paired_v2_minus_v1.json',paired(v2confirm,v1confirm));status='PRIMARY_EVALUATIONS_COMPLETE_CONTROLS_PENDING'
  else:status='PRIMARY_NEGATIVE_DEVELOPMENT_NO_OFFICIAL_EVALUATION_CONTROLS_PENDING'
  del model;torch.cuda.empty_cache()
  # Architecture controls are retrained, not presented as frozen removals or pilots.
  for variant in ['shared_only','routed_only','dense']:
   control,stop=train(C,cfg,variant,50000,14400,val_old,policy_factory=DeepSeekV2Policy);values,metadata=val_new(control);np.save(C/'evaluations'/f'retrained_{variant}_development.npy',values);write(C/'reports'/f'retrained_{variant}_development.json',dict(mean=float(values.mean()),MC_SE=float(values.std(ddof=1)/math.sqrt(len(values))),metadata=metadata,stop_reason=stop,comparable_allowance_seconds=14400,not_official_evaluation=True));del control;torch.cuda.empty_cache()
  status='COMPLETE_ONE_PRIMARY_SEED_AND_FULL_BUDGET_CONTROLS' if passed else 'COMPLETE_NEGATIVE_PRIMARY_AND_FULL_BUDGET_CONTROLS_NO_OFFICIAL'
 except Exception as exc:
  import traceback
  write(C/'reports/v2_campaign_failure.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()));raise
 finally:write(C/'v2_registry.json',dict(status=status,campaign_wall_seconds=time.monotonic()-start,hardware=torch.cuda.get_device_name(0),original_results_preserved=True,main_compute_cap_seconds=14400,architecture_controls_each_cap_seconds=14400));print('V2_STATUS',status,flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('--deepseek-sboed-v2',action='store_true');p.add_argument('--campaign',required=True);p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();C=Path(a.campaign).resolve();cfg=yaml.safe_load(Path(a.config).read_text());run(C,cfg,a.smoke)
if __name__=='__main__':main()
