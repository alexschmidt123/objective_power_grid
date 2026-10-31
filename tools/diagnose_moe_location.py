"""Saved-checkpoint diagnostics; never train or select by evaluation MSE."""
import argparse,json,math,time
from pathlib import Path
import numpy as np
import torch,yaml
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.policies import make_policy,history_features
from src.research.sparse_moe import SparseMoESBOED
from src.research.moe_architecture_controls import ArchitectureControl,ExistingRoutingView
from src.research.location_posterior import posterior_metrics

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def distribution(x,ceiling):
    x=np.asarray(x);return dict(n=len(x),mean=float(x.mean()),quantiles=dict(zip(['min','p10','median','p90','p99','max'],np.quantile(x,[0,.1,.5,.9,.99,1]).tolist())),within_01_of_ceiling=float(np.mean(x>ceiling-.1)),within_05_of_ceiling=float(np.mean(x>ceiling-.5)),ceiling=ceiling)
def load(root,rel,kind):
    saved=torch.load(root/rel/'models'/f'{kind}.pt',map_location='cpu',weights_only=False)
    if kind=='dad':p=make_policy(b,30,'dad',64,dict(first_design_revision=2))
    else:
        p=SparseMoESBOED(2,1,30,saved['hidden'],first_action=[0.,0.]);p.router.gamma=saved['gamma']
    p.load_state_dict(saved['state_dict']);return p,saved
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--probes',type=int,default=512);parser.add_argument('--inference',action='store_true');args=parser.parse_args()
    root=Path.cwd();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    raw=yaml.safe_load((root/'configs/location_finding_eig.yaml').read_text());b=make_benchmark(raw)
    paths=dict(original='experiments/10132026/10132026_bounded_location_restart/pilot',component='experiments/10142026/10142026_moe_component_clipping/pilot',slower='experiments/10142026/10142026_moe_delayed_sharpening/pilot',dad='experiments/10132026/10132026_bounded_location_restart/pilot')
    result=dict(probe_definition='Fresh on-the-fly L1024 objective samples from final checkpoints; not reconstructed historical training minibatches',training_L=1024,probe_seed=10000120,checkpoint_probes={},historical_training={},permutation={},cost={})
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    for name,rel in paths.items():
        kind='dad' if name=='dad' else 'moe_sboed';p,saved=load(root,rel,kind);st=time.perf_counter();modes=['default'] if kind=='dad' else ['relaxed','hard']
        result['checkpoint_probes'][name]={}
        for mode in modes:
            values=[];p.to(device);p.train(mode=='relaxed')
            with torch.no_grad(),torch.device(device):
                for start in range(0,args.probes,16):
                    r=rollout(b,p,30,min(16,args.probes-start),10000120+start*97451,1024)
                    values.extend(r['info'][:,-1].cpu().tolist())
            result['checkpoint_probes'][name][mode]=distribution(values,math.log(1025))
            write(out/f'{name}_{mode}_objective_scores.json',values)
        p.cpu().eval();rows=json.loads((root/rel/'eval'/f'{kind}_seed1001.json').read_text());take=rows[:8]
        for stage in [5,15,29]:
            actions=[torch.tensor([r['actions'][t] for r in take],dtype=torch.float64) for t in range(stage)]
            ys=[torch.tensor([r['observations'][t] for r in take],dtype=torch.float64) for t in range(stage)]
            f=history_features(b,actions,ys,8,30);g=f.clone();slots=g[:,1:].reshape(8,30,4);idx=torch.randperm(stage,generator=torch.Generator().manual_seed(31001+stage));slots[:,:stage]=f[:,1:].reshape(8,30,4)[:,idx]
            with torch.no_grad():
                a=b.feasible(p(f,stage),[],30);c=b.feasible(p(g,stage),[],30)
            result['permutation'][name+f'_stage{stage}']=dict(mean_action_change=float((a-c).norm(dim=-1).mean()),max_coordinate_change=float((a-c).abs().max()),pairs_kept_intact=True,stage_fixed=True)
        trace=root/rel/'models/moe_sboed_diagnostics.jsonl'
        if kind!='dad' and trace.exists():
            history=[json.loads(x) for x in trace.read_text().splitlines()];phase=[]
            for lo,hi in [(0,.2),(.2,.5),(.5,.8),(.8,1.00001)]:
                subset=[r for r in history if lo<=r['primary_episodes']/1e6<hi]
                if not subset:continue
                phase.append(dict(fraction=[lo,hi],mean_training_bound=float(np.mean([r['training_eig'] for r in subset])),gradient_quantiles=np.quantile([r['gradient_norm'] for r in subset],[.5,.95,.99]).tolist(),router_gradient_quantiles=np.quantile([r['gradient_before']['router'] for r in subset],[.5,.95,.99]).tolist(),expert_gradient_p95={k:float(np.quantile([r['gradient_before'].get(k,0.) for r in subset],.95)) for k in ['expert_0','expert_1','expert_2','expert_3']},hard_choice_frequency=np.mean([r['hard_choice_frequency'] for r in subset],axis=0).tolist(),execution_utilization=np.mean([r['execution_utilization'] for r in subset],axis=0).tolist()))
            result['historical_training'][name]=dict(phases=phase,actual_historical_per_trajectory_scores_available=False)
        result['cost'][name]=dict(checkpoint_probe_seconds=time.perf_counter()-st);write(out/'diagnosis.json',result);print('CHECKPOINT_DIAG',name,result['checkpoint_probes'][name],flush=True)
    moe=json.loads((root/paths['component']/'eval/moe_sboed_seed1001.json').read_text());dad=json.loads((root/paths['dad']/'eval/dad_seed1001.json').read_text())
    assert len(moe)==len(dad)==2000
    assert all(m['trajectory_id']==d['trajectory_id'] and m['truth']==d['truth'] and m['measurement_noise_standard']==d['measurement_noise_standard'] for m,d in zip(moe,dad))
    failures=[i for i,r in enumerate(moe) if r['inference']['mse']>1]
    paired=[dict(trajectory_id=i,moe_mse=moe[i]['inference']['mse'],dad_mse=dad[i]['inference']['mse'],moe_realized_information_score=moe[i]['spce'],dad_realized_information_score=dad[i]['spce'],information_difference=moe[i]['spce']-dad[i]['spce']) for i in failures]
    write(out/'paired_failures.json',paired)
    ranked=sorted(failures,key=lambda i:moe[i]['spce']);selected=[ranked[0],ranked[len(ranked)//2],ranked[-1]]
    result['mse_audit']=dict(failure_count=len(failures),mean_information_difference_on_failures=float(np.mean([r['information_difference'] for r in paired])),fraction_moe_lower_information=float(np.mean([r['information_difference']<0 for r in paired])),representatives=selected,selection='purposeful low/median/high realized information among MSE>1; not a population performance estimate',inference=[])
    representatives=[]
    for i in selected:
        for name,rows in [('moe',moe),('dad',dad)]:
            r=rows[i];a=np.asarray(r['actions']);truth=np.asarray(r['truth']).reshape(2,2)
            representatives.append(dict(method=name,trajectory_id=i,truth=r['truth'],actions=r['actions'],observations=r['observations'],min_measurement_distance_to_each_true_source=np.sqrt(((a[:,None]-truth[None])**2).sum(-1)).min(0).tolist(),original_inference=r['inference']))
            if args.inference:
                rr=dict(truth=torch.tensor([r['truth']],dtype=torch.float64),actions=[torch.tensor([x],dtype=torch.float64) for x in r['actions']],observations=[torch.tensor([x],dtype=torch.float64) for x in r['observations']])
                for seed in [21001,21002,21003]:
                    st=time.perf_counter();inferred,art=posterior_metrics(b,rr,dict(posterior_particles=32768,smc_moves=32),seed,True)
                    samples=art['posterior_samples'][0];v=inferred[0];estimate=np.asarray(v['estimate']).reshape(2,2)
                    direct=float(((estimate-truth)**2).mean());swapped=float(((estimate[::-1]-truth)**2).mean())
                    result['mse_audit']['inference'].append(dict(method=name,trajectory_id=i,seed=seed,particles=32768,moves=32,seconds=time.perf_counter()-st,metrics=v,label_invariant_mse=min(direct,swapped),source_separation_quantiles=np.quantile(np.linalg.norm(samples.reshape(-1,2,2)[:,0]-samples.reshape(-1,2,2)[:,1],axis=1),[.1,.5,.9]).tolist()))
                    write(out/'diagnosis.json',result);print('INFERENCE_DIAG',name,i,seed,v['mse'],flush=True)
    write(out/'representative_designs.json',representatives)
    result['parameter_counts']={}
    for method in ['sum_single','attention_single','attention_dense_moe','attention_sparse_moe','attention_matched_dense']:
        p=ArchitectureControl(method,30);total=sum(v.numel() for v in p.parameters());base=sum(v.numel() for n,v in p.named_parameters() if not n.startswith('experts.') and n!='first_design')
        active=base+sum(v.numel() for v in p.experts[0].parameters()) if method=='attention_sparse_moe' else total-2
        effective=total-(p.head[0].weight[:,16:].numel() if method=='sum_single' else 0)
        result['parameter_counts'][method]=dict(total=total,active_nonfirst=active,effective_nonpadding=effective,active_definition='static parameter count; sparse attention still computes all4 pools; inactive zero-padded head weights reported separately')
    write(out/'diagnosis.json',result);write(out/'exit_code.json',dict(exit_code=0));print('DIAG_COMPLETE',flush=True)
