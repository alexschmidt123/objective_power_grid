from common import *
from prepare import conditions
from v2_tasks import native_logpost

@torch.no_grad()
def run_conditional(model,bank,hist,deadline):
 E=O/'evaluation';s=restore();a=s['algorithm'];rng=np.random.default_rng(CFG['conditional_seed']);n=CFG['conditional_cases'];r=CFG['conditional_replicates'];truth=torch.tensor(bank['truth'][256:256+n],device='cuda').repeat_interleave(r,0);allvalues=[];group_tests={};interaction={};total=0
 for t in CFG['conditional_stages']:
  if time.monotonic()>deadline:raise TimeoutError('Conditional panel deadline')
  cal=torch.tensor(hist['X'][:256,t],device='cuda');x=torch.tensor(hist['X'][256:256+n,t],device='cuda');scal=conditions(model,cal);sv=conditions(model,x);threshold=torch.quantile(scal,CFG['quantile'],dim=0);groups=(sv>threshold).cpu().numpy();ds0=torch.tensor(hist['ds'][256:256+n],device='cuda').repeat_interleave(r,0);ys0=torch.tensor(hist['ys'][256:256+n],device='cuda').repeat_interleave(r,0);noise=torch.tensor(rng.standard_normal((n*r,30)),device='cuda');u=torch.tensor(rng.random((n*r,30)),device='cuda');eps=torch.tensor(rng.standard_normal((n*r,30,2)),device='cuda');values=[]
  for fixed in [None,0,1,2,3]:
   ds=ds0.clone();ys=ys0.clone();ds[:,t:]=0;ys[:,t:]=0
   for j in range(t,30):
    xx=prefix_inputs(ds,ys,torch.full((n*r,),j,device='cuda',dtype=torch.long));ac,_=model.draw(xx.float(),u[:,j],eps[:,j],fixed=fixed if j==t else None);y=mean_signal(truth[:,None],ac[:,None])[:,0,0]+.5*noise[:,j];ds[:,j]=ac;ys[:,j,0]=y;total+=n*r
   terminal=native_logpost(a,ds.float(),ys.float(),truth.float(),torch.full((n*r,),30,device='cuda',dtype=torch.long));values.append(terminal.reshape(n,r).mean(-1).cpu().numpy())
  advantages=np.stack(values[1:],-1)-values[0][:,None];allvalues.append(advantages)
  for k in [1,2,3]:
   high=groups[:,k-1];v=advantages[:,k]
   for label,mask in [('high',high),('low',~high)]:
    group_tests[f't{t}_expert{k}_{label}']=stats(v[mask]) if mask.sum()>=16 else dict(status='INSUFFICIENT_CASES',n=int(mask.sum()))
   # Independent history groups; Welch-style normal CI for difference, not a paired test.
   if high.sum()>=16 and (~high).sum()>=16:
    from scipy.stats import t as student
    hi=v[high];lo=v[~high];vh=hi.var(ddof=1)/len(hi);vl=lo.var(ddof=1)/len(lo);se=math.sqrt(vh+vl);delta=float(hi.mean()-lo.mean())
    if se:
     df=(vh+vl)**2/(vh**2/(len(hi)-1)+vl**2/(len(lo)-1));crit=student.ppf(.975,df);p=float(2*student.sf(abs(delta/se),df))
    else:crit=0.;p=1. if delta==0 else 0.
    interaction[f't{t}_expert{k}']=dict(mean=delta,MCSE=se,CI95=[delta-crit*se,delta+crit*se],p_two_sided=p,n_high=len(hi),n_low=len(lo))
 holm({k:v for k,v in group_tests.items() if 'p_two_sided' in v});holm(interaction);np.savez_compressed(E/'conditional_native_advantages.npz',advantages=np.stack(allvalues));put(E/'CONDITIONAL_COMPETENCE.json',dict(status='COMPLETE',cases=n,replicates=r,stages=CFG['conditional_stages'],group_advantages=group_tests,high_minus_low_interactions=interaction,simulator_observations=total,estimand='one-step expert intervention then common mixture suffix;paired terminal parent-native variational score difference;not exact conditional MI',prefixes='distinct heldout jointly generated histories with their latent draw;one true theta per history,noise replicas clustered',groups='upper-third stage-specific thresholds from independent calibration histories',no_future_replay=True,no_true_theta_policy_input=True))
