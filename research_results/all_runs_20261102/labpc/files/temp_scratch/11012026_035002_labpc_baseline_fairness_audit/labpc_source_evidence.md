# Source evidence

InspectionUTC 2026-11-01T03:50:02Z

/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:25:         return (-.5 * ((observation - mean) / self.sigma).square()
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:26:                 - math.log(self.sigma * math.sqrt(2 * math.pi))).sum(-1)
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:45: class LocationFinding(Benchmark):
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:54:         self.sigma = float(raw.get('observation', {}).get('noise_sigma', .5))
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:57:         bounds = raw.get('location_finding', {}).get('design_bounds')
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:58:         self.design_bounds = [float(x) for x in bounds] if bounds is not None else None
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:59:         self.random_design_distribution=raw.get('location_finding',{}).get('random_design_distribution','uniform_box' if self.design_bounds is not None else 'standard_normal')
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:62:         if self.design_bounds is not None:
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:63:             if len(self.design_bounds) != 2 or self.design_bounds[0] >= self.design_bounds[1]:
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:65:             self.action_scale = max(abs(x) for x in self.design_bounds)
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:71:         if self.design_bounds is not None:
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:72:             lo,hi = self.design_bounds
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:77:         if self.design_bounds is not None:
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:80:                 lo,hi=self.design_bounds
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:113:         self.sigma = float(raw.get('observation', {}).get('noise_sigma', 1.))
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/benchmarks.py:229:         self.sigma = float(raw['observation']['noise_sigma'])
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/journal_run.py:52:     metadata=dict(protocol='journal_validation_convergence_v1',phase=phase,smoke=args.smoke,workability_only=args.smoke,train_seed=seed,eval_seeds=evaluation_seeds,horizon=h,methods=methods,settings=settings,benchmark=raw,hardware=hardware_info(),checkpoint_selection='latest_budget_eligible_checkpoint' if protocol.get('compute_budget') else 'final_predeclared_fixed_budget' if protocol.get('checkpo
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/journal_run.py:102:                 if 'reference_schedule_episodes' not in s:raise ValueError('Matched-compute mode requires an explicit reference schedule, separate from episode safety guard')
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/journal_run.py:103:                 reference_episodes=int(s['reference_schedule_episodes'])
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/journal_run.py:112:             if method=='random':p=None;t=dict(training='none',training_seconds=0.,generated_training_trajectories=0,checkpoint_selection='not_applicable',convergence_status='not_applicable')
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/journal_run.py:198:                 t['reference_schedule_episodes']=reference_episodes
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:11: METHODS = ('dad', 'fixed', 'rl_sboed', 'vsoed', 'scorebed')
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:12: SEEDS = (101, 102, 103, 104, 105)
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:41:         checkpoint_selection='latest_budget_eligible_checkpoint',
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:46:         reference_optimizer_updates=50000, reference_training_episodes=100000000,
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:47:         reference_schedule_episodes=100000000)
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:50:             reference_training_episodes=3200000,reference_schedule_episodes=3200000),
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:52:             rl_native_replay_capacity=1000000, reference_schedule_episodes=20001*100),
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:53:         vsoed=dict(reference_schedule_episodes=20000+10001*1000),
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:60:             reference_schedule_episodes=10014+(177365+9335)*1024,
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:62:                 score_updates=177365,policy_updates=9335,phase_ratio=19,
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:71:         evaluation_seeds=[FINAL_SEED], checkpoint_selection_mode='final_compute_budget',
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:76:             validation_every_episodes=1000000000, patience=6, minimum_improvement=.05,
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:77:             noise_multiplier=1.96, maximum_episodes=1000000000,
/home/grads/g/g.lin/Documents/objective_power_grid_temp/src/research/location_baseline_campaign.py:338:     validator=CommonValidation(raw,30,FINAL_SEED,2000,1000000,16,16384,device='cuda:0')
/home/grads/g/g.lin/Documents/objective_power_grid_temp/README.md:5: The journal study compares exactly ten methods: Random, Optimized Fixed, DAD, simplified RL-sBOED, vsOED, ScoreBED and the four variants, on location finding and SIR ODE. Each trained method receives **one million total primary training episodes**, including pretraining. Residual models use a seed-specific500k base plus500k residual episodes; the exact frozen base is shared within each seed. The i

## Per-run record pointers
{"method": "dad", "seed": 101, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/dad.yaml", "config_sha256": "e61a7d2e014558d8b4ed04b1851a72ec3984932d61ab8f1a439ff36e2ea39c20", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/dad/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/dad/models/dad_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/dad_seed101.log", "status": "TRAINED"}
{"method": "fixed", "seed": 101, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/fixed.yaml", "config_sha256": "59521ab5577446e5dc9cbf5a864c277cde8e39aa9d75ffa519b800d718c069c9", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/fixed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/fixed/models/fixed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/fixed_seed101.log", "status": "TRAINED"}
{"method": "rl_sboed", "seed": 101, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/rl_sboed.yaml", "config_sha256": "6455a3ebd4c7eac109d1d6424da3fe4ed55aeddb8459d3015fa2b8794c3f27e0", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/rl_sboed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/rl_sboed/models/rl_sboed_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/rl_sboed/models/rl_sboed_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/rl_sboed_seed101.log", "status": "TRAINED"}
{"method": "vsoed", "seed": 101, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/vsoed.yaml", "config_sha256": "ff92340fa2e11b95b179b887f1c0e7b656772a269fd768a9a857b3fc879b797a", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/vsoed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/vsoed/models/vsoed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/vsoed_seed101.log", "status": "TRAINED"}
{"method": "scorebed", "seed": 101, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/scorebed.yaml", "config_sha256": "2a3734abe8d6bdcfaee19be55dff55c535d04879a60dec9d71a3f9e67ec53378", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/scorebed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/scorebed/models/scorebed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/scorebed_seed101.log", "status": "TRAINED"}
{"method": "dad", "seed": 102, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/dad.yaml", "config_sha256": "e61a7d2e014558d8b4ed04b1851a72ec3984932d61ab8f1a439ff36e2ea39c20", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/dad/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/dad/models/dad_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/dad/models/dad_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/dad_seed102.log", "status": "TRAINED"}
{"method": "fixed", "seed": 102, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/fixed.yaml", "config_sha256": "59521ab5577446e5dc9cbf5a864c277cde8e39aa9d75ffa519b800d718c069c9", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/fixed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/fixed/models/fixed_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/fixed/models/fixed_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/fixed_seed102.log", "status": "TRAINED"}
{"method": "rl_sboed", "seed": 102, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/rl_sboed.yaml", "config_sha256": "6455a3ebd4c7eac109d1d6424da3fe4ed55aeddb8459d3015fa2b8794c3f27e0", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/rl_sboed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/rl_sboed/models/rl_sboed_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/rl_sboed/models/rl_sboed_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/rl_sboed_seed102.log", "status": "TRAINED"}
{"method": "vsoed", "seed": 102, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/vsoed.yaml", "config_sha256": "ff92340fa2e11b95b179b887f1c0e7b656772a269fd768a9a857b3fc879b797a", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/vsoed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/vsoed/models/vsoed_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/vsoed/models/vsOED_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/vsoed_seed102.log", "status": "TRAINED"}
{"method": "scorebed", "seed": 102, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/scorebed.yaml", "config_sha256": "2a3734abe8d6bdcfaee19be55dff55c535d04879a60dec9d71a3f9e67ec53378", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/scorebed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/scorebed/models/scorebed_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/scorebed/models/ScoreBED_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/scorebed_seed102.log", "status": "FAILED_OR_INTERRUPTED"}
{"method": "dad", "seed": 103, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/dad.yaml", "config_sha256": "e61a7d2e014558d8b4ed04b1851a72ec3984932d61ab8f1a439ff36e2ea39c20", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/dad/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/dad/models/dad_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/dad/models/dad_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/dad_seed103.log", "status": "TRAINED"}
{"method": "fixed", "seed": 103, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/fixed.yaml", "config_sha256": "59521ab5577446e5dc9cbf5a864c277cde8e39aa9d75ffa519b800d718c069c9", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/fixed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/fixed/models/fixed_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/fixed/models/fixed_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/fixed_seed103.log", "status": "TRAINED"}
{"method": "rl_sboed", "seed": 103, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/rl_sboed.yaml", "config_sha256": "6455a3ebd4c7eac109d1d6424da3fe4ed55aeddb8459d3015fa2b8794c3f27e0", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/rl_sboed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/rl_sboed/models/rl_sboed_training.json", "resource_progress": ["/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/rl_sboed/models/rl_sboed_resource_progress.json"], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/rl_sboed_seed103.log", "status": "PENDING"}
{"method": "vsoed", "seed": 103, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/vsoed.yaml", "config_sha256": "ff92340fa2e11b95b179b887f1c0e7b656772a269fd768a9a857b3fc879b797a", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/vsoed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/vsoed/models/vsoed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/vsoed_seed103.log", "status": "PENDING"}
{"method": "scorebed", "seed": 103, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/scorebed.yaml", "config_sha256": "2a3734abe8d6bdcfaee19be55dff55c535d04879a60dec9d71a3f9e67ec53378", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/scorebed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/scorebed/models/scorebed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/scorebed_seed103.log", "status": "PENDING"}
{"method": "dad", "seed": 104, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/dad.yaml", "config_sha256": "e61a7d2e014558d8b4ed04b1851a72ec3984932d61ab8f1a439ff36e2ea39c20", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/dad/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/dad/models/dad_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/dad_seed104.log", "status": "PENDING"}
{"method": "fixed", "seed": 104, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/fixed.yaml", "config_sha256": "59521ab5577446e5dc9cbf5a864c277cde8e39aa9d75ffa519b800d718c069c9", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/fixed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/fixed/models/fixed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/fixed_seed104.log", "status": "PENDING"}
{"method": "rl_sboed", "seed": 104, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/rl_sboed.yaml", "config_sha256": "6455a3ebd4c7eac109d1d6424da3fe4ed55aeddb8459d3015fa2b8794c3f27e0", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/rl_sboed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/rl_sboed/models/rl_sboed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/rl_sboed_seed104.log", "status": "PENDING"}
{"method": "vsoed", "seed": 104, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/vsoed.yaml", "config_sha256": "ff92340fa2e11b95b179b887f1c0e7b656772a269fd768a9a857b3fc879b797a", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/vsoed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/vsoed/models/vsoed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/vsoed_seed104.log", "status": "PENDING"}
{"method": "scorebed", "seed": 104, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/scorebed.yaml", "config_sha256": "2a3734abe8d6bdcfaee19be55dff55c535d04879a60dec9d71a3f9e67ec53378", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/scorebed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/scorebed/models/scorebed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/scorebed_seed104.log", "status": "PENDING"}
{"method": "dad", "seed": 105, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/dad.yaml", "config_sha256": "e61a7d2e014558d8b4ed04b1851a72ec3984932d61ab8f1a439ff36e2ea39c20", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/dad/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/dad/models/dad_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/dad_seed105.log", "status": "PENDING"}
{"method": "fixed", "seed": 105, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/fixed.yaml", "config_sha256": "59521ab5577446e5dc9cbf5a864c277cde8e39aa9d75ffa519b800d718c069c9", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/fixed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/fixed/models/fixed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/fixed_seed105.log", "status": "PENDING"}
{"method": "rl_sboed", "seed": 105, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/rl_sboed.yaml", "config_sha256": "6455a3ebd4c7eac109d1d6424da3fe4ed55aeddb8459d3015fa2b8794c3f27e0", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/rl_sboed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/rl_sboed/models/rl_sboed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/rl_sboed_seed105.log", "status": "PENDING"}
{"method": "vsoed", "seed": 105, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/vsoed.yaml", "config_sha256": "ff92340fa2e11b95b179b887f1c0e7b656772a269fd768a9a857b3fc879b797a", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/vsoed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/vsoed/models/vsoed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/vsoed_seed105.log", "status": "PENDING"}
{"method": "scorebed", "seed": 105, "config": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/configs/scorebed.yaml", "config_sha256": "2a3734abe8d6bdcfaee19be55dff55c535d04879a60dec9d71a3f9e67ec53378", "runconfig": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/scorebed/run_config.json", "training": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/scorebed/models/scorebed_training.json", "resource_progress": [], "log": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/logs/scorebed_seed105.log", "status": "PENDING"}

## Commands
[
  {
    "UTC": "2026-11-01T03:50:02Z",
    "argv": [
      "nvidia-smi",
      "-q",
      "-x"
    ]
  },
  {
    "UTC": "2026-11-01T03:50:02Z",
    "argv": [
      "lscpu",
      "-J"
    ]
  },
  {
    "UTC": "2026-11-01T03:50:02Z",
    "argv": [
      "git",
      "rev-parse",
      "HEAD"
    ]
  },
  {
    "UTC": "2026-11-01T03:50:02Z",
    "argv": [
      "git",
      "status",
      "--short"
    ]
  },
  {
    "UTC": "2026-11-01T03:50:03Z",
    "argv": [
      "ps",
      "-eo",
      "pid,ppid,etime,args",
      "--no-headers"
    ]
  }
]

Read-only snapshotsmayadvancenaturallywhilecollected. Activeprogress uses nearestexistingreceipt/workerledger, not an extrapolatedwallclock. No broadcheckpoint hashing, no payloadloads, no GPUforward/evaluation, no processesaltered. Currenthostsoftware only.

## Refined complete-method ledger evidence
[
  {
    "method": "dad",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/dad/models/dad_training.json#resource_ledger.complete_method_counts",
    "primary": 100000000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "fixed",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/fixed/models/fixed_training.json#resource_ledger.complete_method_counts",
    "primary": 3200000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "rl_sboed",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/rl_sboed/models/rl_sboed_resource_progress.json",
    "primary": 1030000,
    "components": "{\"policy\": 657088, \"critic\": 1314176, \"posterior\": null, \"score\": null, \"temperature\": 657088}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "vsoed",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": 10021000,
    "components": "{\"policy\": 10001, \"critic\": 50005, \"posterior\": 1500150, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "scorebed",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/scorebed/models/scorebed_training.json#resource_ledger.complete_method_counts",
    "primary": 191190814,
    "components": "{\"policy\": 9335, \"critic\": null, \"posterior\": null, \"score\": 177365, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "dad",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/dad/models/dad_resource_progress.json",
    "primary": 100000000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "fixed",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/fixed/models/fixed_resource_progress.json",
    "primary": 3200000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "rl_sboed",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/rl_sboed/models/rl_sboed_resource_progress.json",
    "primary": 1030000,
    "components": "{\"policy\": 657088, \"critic\": 1314176, \"posterior\": null, \"score\": null, \"temperature\": 657088}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "vsoed",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/vsoed/models/vsOED_resource_progress.json",
    "primary": 10020000,
    "components": "{\"policy\": 10000, \"critic\": 50000, \"posterior\": 1500000, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "scorebed",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/scorebed/models/ScoreBED_resource_progress.json",
    "primary": 171325214,
    "components": "{\"policy\": 0, \"critic\": null, \"posterior\": null, \"score\": 167300, \"temperature\": null}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "dad",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/dad/models/dad_resource_progress.json",
    "primary": 100000000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "fixed",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/fixed/models/fixed_resource_progress.json",
    "primary": 3200000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "rl_sboed",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/rl_sboed/models/rl_sboed_resource_progress.json",
    "primary": 380000,
    "components": "{\"policy\": 241088, \"critic\": 482176, \"posterior\": null, \"score\": null, \"temperature\": 241088}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "vsoed",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "scorebed",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/scorebed/models/scorebed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "dad",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/dad/models/dad_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "fixed",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/fixed/models/fixed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "rl_sboed",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/rl_sboed/models/rl_sboed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "vsoed",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "scorebed",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/scorebed/models/scorebed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "dad",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/dad/models/dad_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "fixed",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/fixed/models/fixed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "rl_sboed",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/rl_sboed/models/rl_sboed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "vsoed",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  },
  {
    "method": "scorebed",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/scorebed/models/scorebed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:53:31Z"
  }
]
Historicalreusedruns oftenlackstandaloneprogressfile; finaltraining.resource_ledger includescomplete_method_counts, alloptimizercomponents, CPU/validationcost. These are usedinstead of assumingmissingcost. Liveworkerelapsed and olderchargedcheckpointreceipt aredifferent snapshots; no cross-snapshotthroughput division.

## Refined complete-method ledger evidence
[
  {
    "method": "dad",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/dad/models/dad_training.json#resource_ledger.complete_method_counts",
    "primary": 100000000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "fixed",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/fixed/models/fixed_training.json#resource_ledger.complete_method_counts",
    "primary": 3200000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "rl_sboed",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/rl_sboed/models/rl_sboed_training.json#resource_ledger.complete_method_counts",
    "primary": 1037700,
    "components": "{\"policy\": 662016, \"critic\": 1324032, \"posterior\": null, \"score\": null, \"temperature\": 662016}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "vsoed",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": 10021000,
    "components": "{\"policy\": 10001, \"critic\": 50005, \"posterior\": 1500150, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "scorebed",
    "seed": "101",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed101/scorebed/models/scorebed_training.json#resource_ledger.complete_method_counts",
    "primary": 191190814,
    "components": "{\"policy\": 9335, \"critic\": null, \"posterior\": null, \"score\": 177365, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "dad",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/dad/models/dad_training.json#resource_ledger.complete_method_counts",
    "primary": 100000000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "fixed",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/fixed/models/fixed_training.json#resource_ledger.complete_method_counts",
    "primary": 3200000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "rl_sboed",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/rl_sboed/models/rl_sboed_training.json#resource_ledger.complete_method_counts",
    "primary": 1038000,
    "components": "{\"policy\": 662208, \"critic\": 1324416, \"posterior\": null, \"score\": null, \"temperature\": 662208}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "vsoed",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": 10021000,
    "components": "{\"policy\": 10001, \"critic\": 50005, \"posterior\": 1500150, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "scorebed",
    "seed": "102",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed102/scorebed/models/ScoreBED_resource_progress.json",
    "primary": 171325214,
    "components": "{\"policy\": 0, \"critic\": null, \"posterior\": null, \"score\": 167300, \"temperature\": null}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "dad",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/dad/models/dad_training.json#resource_ledger.complete_method_counts",
    "primary": 100000000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "fixed",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/fixed/models/fixed_training.json#resource_ledger.complete_method_counts",
    "primary": 3200000,
    "components": "{\"policy\": 50000, \"critic\": null, \"posterior\": null, \"score\": null, \"temperature\": null}",
    "native_checkpoint_selection": "latest_budget_eligible_checkpoint",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "rl_sboed",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/rl_sboed/models/rl_sboed_resource_progress.json",
    "primary": 380000,
    "components": "{\"policy\": 241088, \"critic\": 482176, \"posterior\": null, \"score\": null, \"temperature\": 241088}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "vsoed",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "scorebed",
    "seed": "103",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed103/scorebed/models/scorebed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "dad",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/dad/models/dad_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "fixed",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/fixed/models/fixed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "rl_sboed",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/rl_sboed/models/rl_sboed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "vsoed",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "scorebed",
    "seed": "104",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed104/scorebed/models/scorebed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "dad",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/dad/models/dad_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "fixed",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/fixed/models/fixed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "rl_sboed",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/rl_sboed/models/rl_sboed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "vsoed",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/vsoed/models/vsoed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  },
  {
    "method": "scorebed",
    "seed": "105",
    "source": "/home/grads/g/g.lin/Documents/objective_power_grid_temp/scratch_runs/10312026_location_baselines_seed101_105_fourhour/runs/seed105/scorebed/models/scorebed_training.json#resource_ledger.complete_method_counts",
    "primary": "UNVERIFIED",
    "components": "{}",
    "native_checkpoint_selection": "UNVERIFIED",
    "timestamp_UTC": "2026-11-01T03:54:40Z"
  }
]
Historicalreusedruns oftenlackstandaloneprogressfile; finaltraining.resource_ledger includescomplete_method_counts, alloptimizercomponents, CPU/validationcost. These are usedinstead of assumingmissingcost. Liveworkerelapsed and olderchargedcheckpointreceipt aredifferent snapshots; no cross-snapshotthroughput division.
