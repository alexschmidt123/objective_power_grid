"""Official benchmark adapter;scientific training implementation stays frozen."""
import argparse,copy,json,pathlib,sys,time,hashlib
M=pathlib.Path.home()/'Documents/objective_power_grid';S=M/'src/research/hec_v3_study';O=M/'experiments/location_finding/hec_v3_study_20261115';CONFIG=M/'configs/moe_sboed_resume_20261115.json'
sys.path.insert(0,str(S))
import engine
from runtime import *
from model import StudyMoE
NAMES={'learned':'MoE-sBOED','uniform':'MoE-sBOED-uniform',**{f'expert{k}':f'MoE-sBOED-E{k}' for k in range(1,5)}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',choices=['train','experts'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['learned','uniform'],default='learned');p.add_argument('--seconds',type=float,default=14400);args=p.parse_args()
 cfg=json.loads(CONFIG.read_text());assert args.seed in cfg['seeds'];official=NAMES[args.arm];root=O/f'seed{args.seed}_{args.arm}'
 if args.task=='train':
  # Override only the scheduled evaluation list;never run the removed ablations.
  engine.PROTOCOL=copy.deepcopy(engine.PROTOCOL);engine.PROTOCOL.update(methods=cfg['methods'],arms=cfg['methods'],queue=cfg['queue'],inference_ablations=['learned'],benchmark_protocol=cfg)
  original_put=engine.put
  def named_put(path,data):
   value=copy.deepcopy(data)
   if isinstance(value,dict):
    value.update(method=official,seed=args.seed,benchmark_protocol_sha256=sha(CONFIG))
    if 'arm' in value:value['implementation_arm']=value.pop('arm')
    if 'evaluations' in value:value['evaluations']={NAMES.get(k,k):v for k,v in value['evaluations'].items()}
   original_put(path,value)
  engine.put=named_put
  engine.run_training(args.seed,args.arm,root)
 else:
  import numpy as np,torch
  from frozen_feature_policy import FeatureExpert
  out=root/'standalone_experts';assert not out.exists();out.mkdir();start=time.monotonic();deadline=start+max(1,args.seconds-15);setup()
  with lease(),engine.Resources(out):
   try:
    cp=root/'joint_actor.pt';initial=sha(cp);state=parent();model=StudyMoE(state['algorithm'].actor_net,202611150000+args.seed,'learned').cuda();model.load_state_dict(torch.load(cp,map_location='cuda',weights_only=True)['actor_state']);model.configure('frozen');model.eval();del state
    data=np.load(n.BASE/'evaluation/diagnostic_candidate_histories.npz');x=torch.tensor(data['X'][:2,[0,14,29]].copy(),device='cuda').reshape(-1,117);rows={};valid=[]
    for k in range(4):
     engine.guard(deadline);single=model.standalone(k).cuda().eval();single.requires_grad_(False)
     assert sum(isinstance(v,FeatureExpert) for v in single.modules())==1
     assert not any(name.startswith(('experts.','router.','prior_coordinator.','prior_response_gate.')) for name,_ in single.named_modules())
     calls=[];hook=single.expert.register_forward_hook(lambda *unused:calls.append(k))
     with torch.no_grad():
      expected=model(x,mode='expert',expert=k);actual=single(x);torch.testing.assert_close(actual,expected,atol=0,rtol=0)
     hook.remove();assert len(calls)==1
     name=f'MoE-sBOED-E{k+1}';torch.save(dict(state_dict=single.state_dict(),source_learned_checkpoint_sha256=initial,expert=k+1,seed=args.seed,method=name),out/(name+'.pt'))
     scores,result=engine.evaluate(single,out,name,deadline,collect=False);rows[name]=result;valid.append(dict(method=name,one_current_expert=True,router_absent=True,exact_trained_expert_path=True,shared_context=True))
     del single
    assert sha(cp)==initial;put(out/'RESULTS.json',dict(status='COMPLETE',seed=args.seed,source_checkpoint_sha256=initial,experts=rows,path_validation=valid,training_updates=0,scope='Extracted standalone policies inherited from corresponding learned checkpoint,not independently trained.',seconds=time.monotonic()-start));put(out/'STATUS.json',dict(status='COMPLETE',seed=args.seed,training_updates=0))
   except Exception as e:put(out/'STATUS.json',dict(status='INCOMPLETE',error=repr(e)));raise
if __name__=='__main__':main()
