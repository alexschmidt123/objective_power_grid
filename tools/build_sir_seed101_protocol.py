"""Freeze method-native schedules using measured timing, never EIG ranking."""
import json,math,copy,sys
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
READY=ROOT/'scratch_runs/10272026/10272026_sir_fourhour_readiness'

def build(timing):
 from src.research.compute_budget import native_update_plan,native_score_phase_plan
 base=yaml.safe_load((ROOT/'configs/sir_journal_corrected.yaml').read_text())
 jp=base['research']['journal_protocol']
 for key in ('historical_method_settings','historical_vsoed_profile','method_settings','method_stopping','historical_campaign','replication_seeds'):
  jp.pop(key,None)
 jp.update(version='sir_seed101_native_fourhour_20261027_v1',status='VALIDATED_FROZEN_SOURCE',
   checkpoint_selection_mode='final_compute_budget',checkpoint_selection='latest_fully_serialized_compute_eligible',
   development_training_seeds=[101],training_seeds=[101],evaluation_seeds=[1001],replication_launch_authorized=False,
   stopping_rule=dict(minimum_episodes=1,minimum_checkpoints=1,validation_every_episodes=1000000000,patience=1000000,
    minimum_improvement=.01,noise_multiplier=1.96,maximum_episodes=100000000,maximum_training_seconds=14400.),
   method_settings={},method_stopping={},
   native_schedule_policy='Preserve verified finite native schedules; safe measured update guards and whole-method charged time are both explicit',
   final_evaluation=dict(systems=2048,contrasts=1000000,seed=1001,resolution_status='COMMON_1M_FROZEN_LOWER_BOUND_TAIL_UNCERTAINTY_DISCLOSED'))
 jp['compute_budget'].update(seconds=14400.,checkpoint_seconds=300.,finalization_reserve_seconds=60.,require_prelaunch_start=True,
   hardware='NVIDIA GeForce RTX 4090',safety_cap_is_measured_compute_limit=True)
 jp['validation'].update(seed=97402611,systems=128,contrasts=65536,batch_size=16,chunk=4096,device='cuda_if_available')
 base['research']['settings'].update(updates=50000,primary_episode_budget=1,training_bound=1,threads=2,
   eval_systems=2048,eval_contrasts=1000000,evaluation_population_systems=2048,evaluation_batch_size=16,evaluation_contrast_chunk=4096)
 base['experiment']['methods']=['random','fixed','dad','rl_sboed','vsoed','scorebed']
 base['study_analysis']=dict(training_seed=101,primary_training_seconds=14400,development_evaluation_seed=1001,
   final_systems=2048,final_contrasts=1000000,reserved_confirmation_seed=98502691,reserved_historical_seed=8831001,
   native_timing_file=str(READY/'native_timing.json'),matched_update_count_across_methods=False)
 base.pop('project_scope',None)
 base.pop('current_scope_replacement',None)
 plans={};folder=READY/'frozen_configs';folder.mkdir(exist_ok=True)
 # Timing helper exports per-method complete-native-unit and initialization costs.
 for method,reference,batch,lr in [('dad',50000,2000,5e-5),('fixed',50000,2000,1e-4),('rl_sboed',20001,100,None),('vsoed',10001,1000,None)]:
  row=timing[method]
  plan=native_update_plan(14400.,float(row['unit_seconds']),reference,
      initialization_seconds=float(row.get('initialization_seconds',0.)),finalization_seconds=60.,safety_factor=1.25)
  maximum=int(plan['safe_max_updates'])
  settings=dict(reference_optimizer_updates=reference,reference_schedule_episodes=reference*batch,
       reference_training_episodes=reference*batch,batch_size=batch,updates=reference,
       spce_trajectory_budget_unit='primary_episodes',primary_episode_budget=1,safety_cap_is_measured_compute_limit=True,
       native_timing_plan=plan,native_profile_status='VERIFIED_PINNED_SOURCE_AND_MEASURED_RTX4090_UNITS')
  cap=maximum*batch
  if method in ('dad','fixed'):
   settings.update(learning_rate=lr,contrasts=2000,first_design_revision=1,spce_gradient_clip=None,
       spce_adam_beta1=.8,spce_adam_beta2=.998,spce_lr_decay=.98,spce_lr_schedule_every=1000,spce_curriculum=1)
  elif method=='rl_sboed':
   settings.update(rl_original_reference=True,rl_native_replay_capacity=10000000,rl_native_contrasts=100000)
  elif method=='vsoed':
   settings['reference_schedule_episodes']=20000+10001*1000
   settings['reference_training_episodes']=settings['reference_schedule_episodes'];cap=20000+maximum*1000
  jp['method_settings'][method]=settings;jp['method_stopping'][method]=dict(maximum_episodes=cap)
  plans[method]=plan|dict(maximum_new_training_trajectories=cap)
 jp['native_vsoed_profile']=dict(updates=10001,new_trajectories=1000,replay_batch=10000,initial_replay=20000,
   replay_capacity=1000000,posterior_updates=5,critic_updates=5,posterior_lr=.001,actor_lr=.0003,critic_lr=.001,
   lr_decay=.9999,discount=1.,exploration=5.,exploration_decay=.9999,target_rate=.1,transition=10000,
   log_every=10,n_incre=1,posterior_coordinate_reference='log_parameters')
 score=timing['scorebed']
 phase=native_score_phase_plan(14400.,float(score['initialization_seconds']),float(score['score_update_seconds']),
   float(score['policy_update_seconds']),score_reference_steps=50000,policy_reference_steps=500,
   finalization_seconds=60.,safety_factor=1.25)
 phase.update(hardware='NVIDIA GeForce RTX 4090',score_schedule_reference_steps=50000,policy_schedule_reference_steps=500)
 scoreupdates=int(phase['score_updates']);policyupdates=int(phase['policy_updates']);cap=100014+(scoreupdates+policyupdates)*1024
 jp['method_settings']['scorebed']=dict(scorebed_batch_size=1024,scorebed_reference_sampler=True,scorebed_restarts=1,
   scorebed_compute_phase_plan=phase,scorebed_pretraining_episodes=scoreupdates*1024,score_schedule_reference_steps=50000,
   policy_schedule_reference_steps=500,reference_schedule_episodes=100014+(50000+500)*1024,
   safety_cap_is_measured_compute_limit=True,native_profile_status='PINNED_DOUBLE_PEND_50K_SCORE_500_POLICY_SIR_ADAPTER')
 jp['method_stopping']['scorebed']=dict(maximum_episodes=cap);plans['scorebed']=phase|dict(maximum_new_training_trajectories=cap)
 jp['method_settings']['random']={}
 for method in base['experiment']['methods']:
  raw=copy.deepcopy(base);raw['experiment']['methods']=[method]
  (folder/(method+'.yaml')).write_text(yaml.safe_dump(raw,sort_keys=False))
 (ROOT/'configs/sir_seed101_fourhour.yaml').write_text(yaml.safe_dump(base,sort_keys=False))
 (READY/'FROZEN_NATIVE_PLANS.json').write_text(json.dumps(plans,indent=2))
 print(json.dumps(plans,indent=2))

if __name__=='__main__':build(json.loads(Path(sys.argv[1]).read_text()))
