import pathlib,sys,os,json,time,hashlib,subprocess
R=pathlib.Path.home()/'Documents/objective_power_grid_temp';C=pathlib.Path(__file__).parent.parent;os.chdir(R);sys.path.insert(0,str(R))
if '--jax' not in sys.argv:
 from src.research.compute_budget import ComputeClock
 from src.research.location_baseline_campaign import acquire
 held=acquire();out=C/'readiness/score_native_profile';out.mkdir(exist_ok=False)
 clock=ComputeClock(out/'compute_clock.json',1200,hardware='NVIDIA GeForce RTX 4090')
 plan=dict(scorebed_native_reference_schedule=True,score_schedule_reference_steps=50000,policy_schedule_reference_steps=20000,score_pretraining_episodes=50000*1024,raw=json.loads((C/'physical.json').read_text()),horizon=30,threads=4,selection_mode='final_compute_budget',root=str(R),torch_python=str(pathlib.Path.home()/'miniconda3/envs/mocu_optimized/bin/python'),rule=dict(minimum_episodes=1,minimum_checkpoints=8,validation_every_episodes=1000000000,patience=6,minimum_improvement=.05,noise_multiplier=1.96,maximum_episodes=1000000000,maximum_training_seconds=1200),validation=dict(seed=2026110205150103,systems=8,contrasts=1024,batch_size=8,chunk=1024,device='cuda'),compute_budget=dict(path=str(clock.path),seconds=1200,credit_seconds=0,hardware='NVIDIA GeForce RTX 4090',checkpoint_seconds=900))
 (out/'plan.json').write_text(json.dumps(plan));start=time.monotonic()
 rc=subprocess.call([str(R/'.venv_scorebed/bin/python'),str(__file__),'--jax','--output',str(out),'--journal-plan',str(out/'plan.json'),'--budget',str(10054+20*1024),'--horizon','30','--seed','101','--batch-size','1024','--restarts','5','--design-bounds','-4','4'],env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',WANDB_MODE='disabled'))
 if rc:raise RuntimeError('Author-native production profile failed')
 data=json.loads((out/'scorebed_official_training.json').read_text());phase=data['phase_timings'] if 'phase_timings' in data else data.get('phase_timing',{})
 (C/'readiness/score_profile_raw.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,metadata=data),indent=2));print('PROFILE_DONE',time.monotonic()-start,flush=True)
else:
 # Only probe executed unit COUNTS. Production batch,T,networks,loss,LR horizons,
 # sampler, five restart initializations and real diagnostic selection are retained.
 sys.argv.remove('--jax');path=R/'src/research/scorebed_worker.py';source=path.read_text()
 a="score_count=int(journal_plan['score_schedule_reference_steps'])*batch";b="policy_count=int(journal_plan['policy_schedule_reference_steps'])*batch*restarts"
 assert source.count(a)==1 and source.count(b)==1
 source=source.replace(a,'score_count=10*batch').replace(b,'policy_count=2*batch*restarts')
 exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
