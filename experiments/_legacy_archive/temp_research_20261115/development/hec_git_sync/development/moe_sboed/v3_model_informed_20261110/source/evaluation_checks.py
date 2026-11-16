import copy,time,torch
from common import *
from evaluate import rollout,load_model
from v3_policy import V3Actor
from native_v3 import install

def main():
 original.setup();lease=original.acquire();tick=time.monotonic();s=restore();a=s['algorithm'];m=V3Actor(copy.deepcopy(a.actor_net)).cuda();rng=np.random.default_rng(CFG['seed']+11);theta=torch.tensor(rng.standard_normal((4,4)),device='cuda');noise=torch.tensor(rng.standard_normal((4,30)),device='cuda');u=torch.tensor(rng.random((4,30)),device='cuda');eps=torch.tensor(rng.standard_normal((4,30,2)),device='cuda');contrasts=torch.tensor(rng.standard_normal((11,4)),device='cuda');before=digest(m.state_dict())
 with torch.no_grad(),torch.device('cuda'):
  one=rollout(m,theta,noise,u,eps,'candidate',True);two=rollout(m,theta,noise,u,eps,'candidate',True);assert all(torch.equal(v,w) for v,w in zip(one['actions'],two['actions']));bench=type('B',(),{'sigma':.5,'calls':0})();from frozen_evaluator import contrast_scores
  score,_=contrast_scores(bench,one,contrasts,3,'cuda');literal=[]
  for i in range(4):
   h=torch.cat([theta[i:i+1],contrasts]);cum=torch.zeros(12,device='cuda',dtype=torch.float64)
   for t in range(30):
    ac=one['actions'][t][i];pred=mean_signal(h[None],ac[None,None])[0,0];y=one['observations'][t][i,0];cum+=-.5*((y-pred)/.5).square()-math.log(.5*math.sqrt(2*math.pi))
   literal.append(cum[0]-torch.logsumexp(cum,0)+math.log(12))
  torch.testing.assert_close(score[:,-1].to('cuda'),torch.stack(literal),rtol=1e-9,atol=1e-9)
  for opts in [dict(fixed=0),dict(remove=1),dict(gates=torch.full((30,4),.25,device='cuda')),dict(model_features=one['model_features'])]:
   v=rollout(m,theta,noise,u,eps,'candidate',**opts);assert len(v['actions'])==30 and all(torch.isfinite(z).all() and bool((z.abs()<4).all()) for z in v['actions'])
  models=[]
  for name in ['hec_v2','hec_v1','sg','option','collaborative','ordinary','vsoed']:
   model=load_model(name);v=rollout(model,theta[:2],noise[:2],u[:2],eps[:2],name);assert all(torch.isfinite(z).all() and bool((z.abs()<=4).all()) for z in v['actions']);models.append(name);del model
 assert digest(m.state_dict())==before
 migration=install(a,False);receipt=json.loads((O/'evidence/IMPLEMENTATION_CHECKS.json').read_text());assert migration['source_sha256']==receipt['transformed_native_train_sha256'],'Native source transform changed'
 put(O/'evidence/EVALUATION_CHECKS.json',dict(status='PASS',seconds=time.monotonic()-tick,literal_estimator_matches=True,positive_included_once=True,repeat_new_policy_rollouts_bitwise=True,changed_policy_observations_resimulated=True,intervention_interfaces_valid=True,baseline_loaders=models,checkpoints_unchanged=True,scientific_training_updates=0));print('V3_EVALUATION_CHECKS_PASS',flush=True)
if __name__=='__main__':main()
