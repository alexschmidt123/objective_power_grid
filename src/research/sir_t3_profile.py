"""Isolated GPU-locked T3 numerical/full-method cost probe; no main result claims."""
import os,time,json,sys,math,fcntl,copy,traceback,subprocess,signal
from pathlib import Path
import numpy as np,torch,yaml
R=Path(__file__).resolve().parents[2]
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--sir-t3-profile',action='store_true');ap.add_argument('--campaign',required=True);ap.add_argument('--method',required=True);args=ap.parse_args();C=Path(args.campaign).resolve()
 if R.name!='objective_power_grid_temp' or R/'scratch_runs/sir_ode_t3_baselines' not in C.parents:raise ValueError('TEMP T3 only')
 locks=[]
 for p in [Path.home()/'.cache/objective_power_grid/device_0.lock',R/'.gpu0.lock',R.parent/'objective_power_grid/.gpu0.lock']:
  p.parent.mkdir(parents=True,exist_ok=True);f=p.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if busy:raise RuntimeError('Foreign GPU processes present: '+busy)
 args.method=args.method; cfg=yaml.safe_load((C/'configs/sir_ode_t3_continuous_v1.yaml').read_text());raw=cfg['benchmark'];out=C/'profiling'/args.method;out.mkdir(exist_ok=False)
 args.method=args.method.removesuffix('_fast').removesuffix('_fma').removesuffix('_retry');torch.set_num_threads(4);start=time.monotonic();torch.cuda.init();records=[]
 from src.research.sir_t3_common import T3SIR as SIR
 from src.research.engine import rollout,replay_likelihood
 from src.research.sir_dad import SIRSumDAD
 from src.research.policies import Policy
 def record(label,t,**kw):
  torch.cuda.synchronize();row=dict(phase=label,seconds=time.monotonic()-t,**kw);records.append(row);write(out/'progress.json',dict(PID=os.getpid(),method=args.method,elapsed=time.monotonic()-start,records=records));print(row,flush=True)
 try:
  with torch.device('cuda:0'):
   if args.method=='numerical':
    th=torch.tensor([[.5,.1],[1.7,.4],[.08,.03],[.9,.015]],dtype=torch.float64);a=torch.tensor([[.10001],[99.8],[.1],[100.]],dtype=torch.float64);results=[]
    for backend in ['eager','fused_rk4']:
     b=SIR(dict(raw,sir_ode=dict(raw['sir_ode'],execution_backend=backend)));theta=th.clone().requires_grad_();action=a.clone().requires_grad_();t=time.monotonic();mean,state=b.forward(theta,b.initial((4,)),action);lp=b.log_likelihood(mean.detach()+.13,mean);grad=torch.autograd.grad((mean+lp).sum(),(action,theta));record(backend,t)
     results.append([mean.detach(),state.detach(),lp.detach(),*[g.detach() for g in grad]])
    errors=[]
    for x,y in zip(*results):torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-8);errors.append(float((x-y).abs().max()))
    b=SIR(raw);p=SIRSumDAD(3);r=rollout(b,p,3,4,2026102835,8);bank=torch.cat((r['truth'][:,None],r['particles']),1);ll,_,_=replay_likelihood(b,bank,r['actions'],r['observations']);torch.testing.assert_close(r['info'][:,-1],ll[:,0]-torch.logsumexp(ll,-1)+math.log(9));(-r['info'][:,-1].mean()).backward()
    acts=torch.stack(r['actions'],1)[:,:,0];assert (acts[:,1:]-acts[:,:-1]>=.1-1e-10).all() and acts.min()>=.1 and acts.max()<=100
    record('T3_full_horizon_validation',start,max_abs_errors=errors,steps=b.steps,physical_state='no resets; S,I,R carried forward')
   elif args.method in ['fixed','dad','fixed_shared']:
    b=SIR(raw);torch.manual_seed(101);p=SIRSumDAD(3) if args.method=='dad' else Policy(10,1,3,'fixed');opt=torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998));t=time.monotonic();f=torch.randn(2000,10,dtype=torch.float64);f[:,0]=2/3
    for _ in range(10):p(f,2).sum().backward();p.zero_grad()
    record('policy_forward_backward_10',t)
    for update in range(1,3):
     t=time.monotonic();opt.zero_grad();r=rollout(b,p,3,2000,101+update*104729,2000) if args.method!='fixed_shared' else {'info':__import__('src.research.sir_t3_common',fromlist=['fixed_shared_spce']).fixed_shared_spce(b,p,2000,2000,101+update*104729)[:,None]};torch.cuda.synchronize();forward=time.monotonic()-t
     tback=time.monotonic();loss=-r['info'][:,-1].mean();loss.backward();torch.cuda.synchronize();backward=time.monotonic()-tback;opt.step();record('native_update',t,update=update,forward_seconds=forward,backward_seconds=backward,B=2000,L=2000,loss=float(loss),fresh_trajectories=2000,likelihood_evaluations=2000*2001*3,peak_memory_bytes=torch.cuda.max_memory_allocated())
    t=time.monotonic();torch.save(dict(policy=p.state_dict(),optimizer=opt.state_dict()),out/'profile_checkpoint.pt');record('serialization',t)
   elif args.method=='rl_sboed':
    from src.research.sir_native_rl import NativeSIRRLPolicy,algorithm,Replay
    b=SIR(raw);p=NativeSIRRLPolicy(3,b);a=algorithm(p,3);replay=Replay(1000000,10);t=time.monotonic()
    with torch.no_grad():r=rollout(b,p,3,100,101+104729,100000)
    info=r['info'];reward=info-torch.cat((torch.zeros_like(info[:,:1]),info[:,:-1]),1);inputs=torch.stack(r['features'],1).reshape(300,-1).float();nxt=torch.stack(r['features'][1:]+[r['final_features']],1).reshape_as(inputs).float();actions=torch.stack(r['actions'],1)[...,0];previous=torch.cat((actions.new_zeros(100,1),actions[:,:-1]),1);lo=previous+b.gap;hi=b.end-(3-torch.arange(3,device=actions.device)-1)*b.gap;unit=(2*(actions-lo)/(hi-lo)-1).reshape(-1,1).float();done=torch.zeros(100,3);done[:,-1]=1;replay.add(torch.cat((inputs,unit,reward.reshape(-1,1).float(),done.reshape(-1,1),nxt),-1));record('native_collection',t,B=100,L=100000,transitions=300,warmup_required_transitions=100000,loss=None)
    # Diagnostic kernel timing ONLY: repeated valid transitions permit native
    # replay-batch shape; this is not a trained/native-warmed pilot checkpoint.
    t=time.monotonic()
    for _ in range(64):a.optimize_policy(replay.sample(4096,3));a._update_targets()
    record('critic_actor_64_updates_cost_only',t,replay_batch=4096,not_native_training=True,reason='Duplicated replay draws for kernel timing; no warm-up or learning claim')
   elif args.method=='vsoed':
    from src.research.sir_t3_native_vsoed import train_upstream_vsoed
    class Observer:
     def set_runtime(self,rt):self.runtime=rt
     def check(self,p,episodes,update,*a,**kw):record('native_actor_critic_posterior_cycle',start,update=update,episodes=episodes);return update>=3
     def finish(self,p):return dict(selected_checkpoint=dict(update=3))
    profile=cfg['methods']['vsoed'];profile=dict(profile,updates=3);b=SIR(raw);t=time.monotonic();observer=Observer();p,data=train_upstream_vsoed(b,3,{},101,out,profile,observer);assert b.calls==data['generated_training_trajectories']*3;assert len(observer.runtime['native_replay_buffer']['params'])==23000;record('complete_native_initialization_and_3cycles',t,data=data,verified_simulation_calls=b.calls,recovery_buffer_rows=len(observer.runtime['native_replay_buffer']['params']))
   elif args.method=='gpu_tests':
    import pytest
    t=time.monotonic()
    with torch.device('cpu'):
     rc=pytest.main(['tools/tests/test_sir_t3_campaign.py','-q','--disable-warnings','--junitxml='+str(C/'manifests/t3_gpu_tests.xml')]);assert rc==0;record('CPU_and_GPU_regressions',t)
   else:raise ValueError(args.method)
  write(out/'result.json',dict(status='PASS',method=args.method,total_seconds=time.monotonic()-start,records=records,hardware=torch.cuda.get_device_name(0),GPU_utilization_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True)))
 except Exception as e:write(out/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),elapsed=time.monotonic()-start,records=records));raise
if __name__=='__main__':main()
