"""Frozen v1 diagnostics, no optimization or new model training."""
import argparse,json,time,copy,math
from pathlib import Path
import numpy as np,torch,yaml
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.journal_validation import CommonValidation
from src.research.engine import rollout
from src.research.benchmarks import make_benchmark

class FrozenIntervention(DeepSeekPolicy):
    def __init__(self,original,mode,seed=10310007):
        super().__init__();self.load_state_dict(original.state_dict());self.mode=mode;self.seed=seed;self.generator=None;self.eval()
    def forward(self,f,stage):
        if self.mode=='shared_only':self.ablate_routed=True;return super().forward(f,stage)
        if self.mode=='routed_only':self.ablate_shared=True;return super().forward(f,stage)
        if self.mode=='full':return super().forward(f,stage)
        z=self.encode_context(f,stage);p=self.router(z).softmax(-1);weight,index=p.topk(3,-1);mass=weight.sum(-1,keepdim=True)
        if self.mode=='uniform_subset_scale_preserved':
            if self.generator is None:self.generator=torch.Generator(device=f.device).manual_seed(self.seed)
            index=torch.rand(len(z),7,device=f.device,generator=self.generator).topk(3,-1).indices
        # Both uniform interventions preserve the original selected total gate mass.
        weight=mass.expand(-1,3)/3;out=z+self.shared(z);r=z.new_zeros(z.shape)
        for e in index.unique().tolist():
            rows,slots=(index==e).nonzero(as_tuple=True);r.index_add_(0,rows,self.routed[e](z[rows])*weight[rows,slots,None])
        return self.emitter(out+r)

def canonical(theta):
    left,right=theta[:,:2],theta[:,2:];swap=(left[:,0]>right[:,0])|((left[:,0]==right[:,0])&(left[:,1]>right[:,1]));return torch.cat([torch.where(swap[:,None],right,left),torch.where(swap[:,None],left,right)],-1)

def run(C,reference):
    from tools.journal_campaign import exclusive_device_lock
    lock=exclusive_device_lock();torch.set_num_threads(2);begin=time.perf_counter();cfg=yaml.safe_load((reference/'configs/seed101.yaml').read_text());checkpoint=reference/'models/full/selected.pt';model=load_deepseek(checkpoint,'cuda');v=CommonValidation(cfg['benchmark'],30,1031,256,262144,batch=16,chunk=16384,device='cuda:0');results={};scores={}
    for mode in ['full','shared_only','routed_only','uniform_weights_fixed_selection','uniform_subset_scale_preserved']:
        reps=3 if mode=='uniform_subset_scale_preserved' else 1;arrays=[]
        for rep in range(reps):
            policy=FrozenIntervention(model,mode,10310007+rep).cuda();values,meta=v(policy);arrays.append(values);np.save(C/'diagnostics'/f'{mode}_{rep}.npy',values)
        average=np.mean(arrays,0);scores[mode]=average;results[mode]=dict(mean=float(average.mean()),MC_SE=float(average.std(ddof=1)/math.sqrt(len(average))),replicates=reps,metadata=meta,intervention='Frozen unchangedweights; no retraining; uniform preserves original routed feature mass')
        if mode!='full':
            d=average-scores['full'];results[mode].update(paired_difference_from_full=float(d.mean()),paired_MC_SE=float(d.std(ddof=1)/math.sqrt(len(d))))
        print('FROZEN_INTERVENTION',mode,results[mode],flush=True)
    original=CommonValidation(cfg['benchmark'],30,1030,128,262144,batch=16,chunk=16384,device='cuda:0');reproduced,_=original(model);saved=np.load(reference/'evaluations/final_development_full.npy');reproduction=dict(max_score_error=float(np.max(abs(reproduced-saved))),mean=float(reproduced.mean()),original_mean=float(saved.mean()))
    b=make_benchmark(cfg['benchmark']);model.eval()
    with torch.no_grad(),torch.device('cuda'):r=rollout(b,model,30,64,10310005,0)
    stage_rows=[]
    with torch.no_grad():
        for t,f in enumerate(r['features']):
            z=model.encode_context(f,t);p=model.router(z).softmax(-1);weight,index=p.topk(3,-1);features=torch.stack([model.shared(z)]+[e(z) for e in model.routed],1);unit=torch.nn.functional.normalize(features,dim=-1);cos=torch.einsum('bed,bfd->bef',unit,unit).mean(0);selected=features[:,1:].gather(1,index[:,:,None].expand(-1,-1,16));routed=(selected*weight[:,:,None]).sum(1)
            frequency=torch.nn.functional.one_hot(index,7).sum(1).double().mean(0);row=dict(stage=t,selection_fraction=frequency.cpu().tolist(),mean_softmax=p.mean(0).cpu().tolist(),mean_softmax_entropy=float(-(p*p.log()).sum(-1).mean()),effective_gate_count=float((-(p*p.log()).sum(-1)).exp().mean()),dominant_subset_fraction=float(torch.unique(index.sort(-1).values,dim=0,return_counts=True)[1].max()/len(index)),expert_feature_cosine_including_shared=cos.cpu().tolist(),mean_feature_norm=features.norm(dim=-1).mean(0).cpu().tolist(),shared_to_routed_norm_ratio=float((features[:,0].norm(dim=-1)/(routed.norm(dim=-1)+1e-12)).mean()),residual_z_norm=float(z.norm(dim=-1).mean()))
            stage_rows.append(row)
    # Belief-context diagnostics only, never policy inputs or new training labels.
    from src.research.pimoe import AdaptiveBelief,rng
    cpu_b=make_benchmark(cfg['benchmark']);prior=canonical(cpu_b.sample((8192,),rng(10310015)));prior_trace=float(torch.var(prior,dim=0).sum());context_rows=[]
    for episode in range(8):
        belief=AdaptiveBelief(cpu_b,2048,10310016+episode)
        for t in range(30):
            if t in [1,5,15,29]:
                theta=canonical(belief.theta);w=belief.weights;mu=(theta*w[:,None]).sum(0);relative=float(((theta-mu).square()*w[:,None]).sum()/prior_trace)
                with torch.no_grad():z=model.encode_context(r['features'][t][episode:episode+1],t);p=model.router(z).softmax(-1);index=p.topk(3,-1).indices
                context_rows.append(dict(episode=episode,stage=t,relative_canonical_posterior_variance=relative,selected=index[0].cpu().tolist(),softmax=p[0].cpu().tolist(),approximate_belief=True))
            belief.update(r['actions'][t][episode].cpu(),r['observations'][t][episode].cpu())
    # Native full-minibatch frozen-weight profile. Gradients only; NO optimizer step.
    model.train();timings=[]
    for k in range(4):
        model.zero_grad(set_to_none=True);a=time.perf_counter()
        with torch.device('cuda'):
            rr=rollout(b,model,30,2000,10319000+k,2000);loss=-rr['info'][:,-1].mean()+model.balance_loss();loss.backward();torch.cuda.synchronize()
        timings.append(time.perf_counter()-a)
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
        with torch.device('cuda'):
            rr=rollout(b,model,30,2000,10319010,2000);loss=-rr['info'][:,-1].mean()+model.balance_loss();loss.backward();torch.cuda.synchronize()
    events=[]
    for e in profile.key_averages():
        if any(token in e.key.lower() for token in ['nonzero','unique','scalar','synchronize','index_add','scatter','index','linear','addmm','mm']):events.append(dict(operator=e.key,count=e.count,self_CPU_us=e.self_cpu_time_total,self_CUDA_us=getattr(e,'self_cuda_time_total',0.)))
    profile.export_chrome_trace(str(C/'diagnostics/v1_native_profile.json'))
    progress=json.loads((reference/'costs/full_progress.json').read_text())['records'];curve=json.loads((reference/'reports/full_development_curve.json').read_text());late={}
    for update in [15000,20000,25000]:
        old=np.load(reference/'evaluations'/f'development_update{update}.npy');d=saved-old;late[str(update)]=dict(final_minus_checkpoint=float(d.mean()),paired_MC_SE=float(d.std(ddof=1)/math.sqrt(len(d))))
    summary=dict(interventions=results,reproduction=reproduction,native_frozen_weight_seconds=timings,native_profile_operators=events,stage_diagnostics=stage_rows,uncertainty_contexts=context_rows,recorded_gradient_max=max(x['preclip_gradient_norm'] for x in progress),recorded_gradient_median=float(np.median([x['preclip_gradient_norm'] for x in progress])),recorded_clip5_fraction=float(np.mean([x['preclip_gradient_norm']>5 for x in progress])),development_curve=curve,late_paired_improvement=late,wall_seconds=time.perf_counter()-begin,optimizer_updates=0,original_checkpoint_unchanged=True,counts=model.counts())
    (C/'reports/v1_diagnostics.json').write_text(json.dumps(summary,indent=2));print('DIAGNOSTICS_COMPLETE',json.dumps({k:summary[k] for k in ['reproduction','native_frozen_weight_seconds','recorded_gradient_max','recorded_gradient_median','recorded_clip5_fraction','late_paired_improvement','wall_seconds']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--reference',required=True);a=p.parse_args();C=Path(a.campaign);run(C,Path(a.reference))
if __name__=='__main__':main()
