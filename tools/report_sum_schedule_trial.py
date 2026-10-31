"""Post-training paired evaluation only; does not change training or checkpoints."""
import csv,json,math,time
from pathlib import Path
import numpy as np
import torch
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
from src.research.moe_architecture_controls import ExistingRoutingView
from tools.diagnose_frozen_sum_routing import exposure,saved_usage,digest

ROOT=Path('/home/grads/g/g.lin/Documents/objective_power_grid')
C=ROOT/'experiments/10142026/10142026_223452_sum_moe_slower_sharpening'
B=ROOT/'experiments/10142026/10142026_moe_controlled_diagnosis/encoder_correction/corrected_seed101'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def paired(a,b):
    d=np.asarray(a)-np.asarray(b);se=float(d.std(ddof=1)/math.sqrt(len(d)))
    return dict(mean=float(d.mean()),standard_error=se,conditional_95_interval=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)],n=len(d))

def diagnostics(run,out):
    rows=read(run/'eval/moe_sboed_seed1001.json');counts=saved_usage(rows)
    tr=exposure(run/'models/moe_sboed_diagnostics.jsonl');write(out/'gradient_exposure.json',tr)
    records=[json.loads(x) for x in (run/'models/moe_sboed_diagnostics.jsonl').open()]
    phases={}
    for name,lo,hi in [('whole',0,1000000),('warmup_0_30',0,300000),('anneal_30_95',300000,950000),('final_95_100',950000,1000000)]:
        rs=[r for r in records if lo<r['primary_episodes']<=hi]
        a=np.array([r['gradient_before']['router'] for r in rs]);post=np.array([r['gradient_after']['router'] for r in rs])
        phases[name]=dict(updates=len(rs),router_pre_quantiles_p0_p50_p95_p99_max=np.quantile(a,[0,.5,.95,.99,1]).tolist(),router_post_max=float(post.max()),router_fraction_above_5=float(np.mean(a>5)),router_fraction_above_100=float(np.mean(a>100)),router_fraction_above_1000=float(np.mean(a>1000)),mean_gate_mass=np.mean([r['router_probabilities'] for r in rs],axis=0).tolist(),expert_post_grad_fraction_above_1e_3=[float(np.mean([r['gradient_after'][f'expert_{j}']>1e-3 for r in rs])) for j in range(4)])
        for key in ['router_probabilities_by_stage','hard_choice_frequency_by_stage','execution_utilization_by_stage']:
            phases[name][key]=np.mean([r[key] for r in rs],axis=0).tolist()
    usage=dict(counts=counts.sum(0).tolist(),fractions=(counts.sum(0)/58000).tolist(),first_measurement_excluded=True,stage_fractions=(counts[1:]/2000).tolist())
    with (out/'training_stage_usage.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['training_phase','measurement_stage','statistic','expert0','expert1','expert2','expert3'])
        for phase,p in phases.items():
            for key in ['router_probabilities_by_stage','hard_choice_frequency_by_stage','execution_utilization_by_stage']:
                for stage,values in enumerate(p[key],2):w.writerow([phase,stage,key]+values)
    with (out/'stage_usage.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['measurement_stage','expert0','expert1','expert2','expert3'])
        for s in range(1,30):w.writerow([s+1]+(counts[s]/2000).tolist())
    write(out/'training_phases.json',phases);write(out/'usage.json',usage)
    return rows,dict(usage=usage,phases=phases,exposure=tr)

@torch.no_grad()
def relaxed(run,out,rows):
    cfg=read(run/'run_config.json');b=make_benchmark(cfg['benchmark'])
    saved=torch.load(run/'models/moe_sboed.pt',map_location='cpu',weights_only=False)
    assert saved['history_encoder_revision']==2 and saved['routing_k']==1 and saved['action_scale']==4
    hard=ExistingRoutingView(saved,'hard');soft=ExistingRoutingView(saved,'relaxed')
    truth=torch.tensor([r['truth'] for r in rows],dtype=torch.float64)
    noise=torch.tensor([r['measurement_noise_standard'] for r in rows],dtype=torch.float64)
    bank=torch.from_numpy(np.load(run/'eval/contrast_bank_seed1001.npy'));assert bank.shape==(1000000,4)
    reference=np.array([r['spce'] for r in rows]);scores=[];start=time.perf_counter()
    for i in range(0,2000,16):
        kw=dict(truth=truth[i:i+16],noise=noise[i:i+16],method='moe_sboed')
        if i==0:
            r=rollout(b,hard,30,16,1001,0,**kw);x,_=contrast_scores(b,r,bank,16384)
            np.testing.assert_allclose(torch.stack(r['actions'],1).numpy(),np.array([r['actions'] for r in rows[:16]]),rtol=0,atol=1e-10)
            np.testing.assert_allclose(x[:,-1].numpy(),reference[:16],rtol=0,atol=1e-9)
        r=rollout(b,soft,30,len(truth[i:i+16]),1001+i*97451,0,**kw)
        x,_=contrast_scores(b,r,bank,16384);scores.extend(x[:,-1].tolist())
        if (i//16)%25==0:print('relaxed',run.name,i+len(r['truth']) if 'truth' in r else i+16,flush=True)
    value=dict(hard_eig=float(reference.mean()),relaxed_eig=float(np.mean(scores)),hard_minus_relaxed=paired(reference,scores),evaluation_seconds=time.perf_counter()-start)
    write(out/'hard_relaxed.json',value);write(out/'relaxed_scores.json',scores)
    return value

def main():
    torch.set_num_threads(2);out=C/'diagnostics';out.mkdir(exist_ok=True)
    assert (C/'pilot/exit_code').read_text().strip()=='0'
    runs={'baseline':B,'slower':C/'pilot'};allrows={};result={};hashes={}
    for label,run in runs.items():
        d=out/label;d.mkdir(exist_ok=True)
        hashes[label]=digest(run/'models/moe_sboed.pt')
        rows,diag=diagnostics(run,d);allrows[label]=rows
        result[label]=dict(**diag,**relaxed(run,d,rows),training_seconds=read(run/'models/moe_sboed_training.json')['training_seconds'])
    assert digest(B/'eval/contrast_bank_seed1001.npy')==digest(C/'pilot/eval/contrast_bank_seed1001.npy')
    for key in ['truth','measurement_noise_standard']:
        np.testing.assert_array_equal([r[key] for r in allrows['baseline']],[r[key] for r in allrows['slower']])
    result['paired_slower_minus_baseline']=paired([r['spce'] for r in allrows['slower']],[r['spce'] for r in allrows['baseline']])
    result['dad_reference_eig']=9.76665434381799
    dad_run=ROOT/'experiments/10132026/10132026_bounded_location_restart/pilot'
    dad_rows=read(dad_run/'eval/dad_seed1001.json')
    for key in ['truth','measurement_noise_standard']:
        np.testing.assert_array_equal([r[key] for r in dad_rows],[r[key] for r in allrows['slower']])
    assert digest(dad_run/'eval/contrast_bank_seed1001.npy')==digest(C/'pilot/eval/contrast_bank_seed1001.npy')
    result['paired_slower_minus_dad']=paired([r['spce'] for r in allrows['slower']],[r['spce'] for r in dad_rows])
    result['unchanged_budget']=dict(primary_training_episodes=1000000,contrast_training_likelihood_episodes=1024000000,train_L=1024,heldout_trajectories=2000,heldout_L=1000000)
    for label,run in runs.items():result[label]['common_evaluation_summary']=read(run/'summary.json')[0]
    result['checkpoint_hashes']=hashes
    result['limitations']=['One training seed: no statistical superiority claim.','Paired Monte Carlo interval conditions on trained checkpoints and the shared contrastive bank.','Gradient exposure is aggregate minibatch evidence, not proof of useful specialization.']
    for label,run in runs.items():assert hashes[label]==digest(run/'models/moe_sboed.pt')
    write(out/'report.json',result)
    lines=['# Sum-MoE slower sharpening: seed 101','', 'Only gamma schedule changed: 1 through 30%, cosine to 0.001 at 95%. All original checkpoints preserved.','', '| Policy | Hard EIG | Relaxed EIG | Training seconds |','|---|---:|---:|---:|']
    for k in runs:lines.append(f"| {k} | {result[k]['hard_eig']:.6f} | {result[k]['relaxed_eig']:.6f} | {result[k]['training_seconds']:.1f} |")
    lines+=['| DAD reference | 9.766654 | — | — |','',f"Paired slower minus baseline: {result['paired_slower_minus_baseline']}",'', 'One seed does not establish superiority across training seeds. See report.json and per-policy CSV/JSON for stage usage, training exposure and router spikes.']
    for k in runs:
        lines+=['',f"{k} held-out expert frequencies (first action excluded): {result[k]['usage']['fractions']}",f"{k} final 5% training gradient exposure >0.001: {result[k]['phases']['final_95_100']['expert_post_grad_fraction_above_1e_3']}"]
    lines+=['','Post-initial hard selections: 58,000 decisions per policy.','', '| Expert (zero based) | Baseline usage % | Slower usage % | Baseline final 5% gradient exposure % | Slower final 5% gradient exposure % |','|---|---:|---:|---:|---:|']
    for j in range(4):
        vals=[100*result[k]['usage']['fractions'][j] for k in runs]+[100*result[k]['phases']['final_95_100']['expert_post_grad_fraction_above_1e_3'][j] for k in runs]
        lines.append('| '+str(j)+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    lines+=['','Gradient exposure means aggregate expert gradient norm after clipping >0.001 per minibatch; it does not establish useful specialization. The JSON also reports the 1e-8 sensitivity threshold and whole-training/final-20% exposure.','', '| Policy | Router raw p99 norm | Router raw maximum | Updates with raw norm >1000 (%) | Maximum clipped norm |','|---|---:|---:|---:|---:|']
    for k in runs:
        p=result[k]['phases']['whole'];q=p['router_pre_quantiles_p0_p50_p95_p99_max']
        lines.append(f"| {k} | {q[3]:.3f} | {q[4]:.3f} | {100*p['router_fraction_above_1000']:.4f} | {p['router_post_max']:.6f} |")
    lines+=['','Hard minus relaxed EIG with paired conditional Monte Carlo uncertainty:']
    for k in runs:lines.append(f"- {k}: {result[k]['hard_minus_relaxed']}")
    lines+=['',f"Slower minus DAD: {result['paired_slower_minus_dad']}",'','Each full training used 1,000,000 primary episodes and L=1024. Each held-out evaluation used 2,000 paired trajectories and L=1,000,000. The separate smoke/timing checks used 8/6,400 episodes, respectively. No architecture, objective, optimizer, clipping, regularization, model-size, or evaluation change was introduced.','', 'Implementation: slower.yaml changes only moe_router_warmup_fraction=0.3 and moe_router_finish_fraction=0.95 relative to resolved baseline settings. The existing routing-schedule implementation was reused; scientific source hashes and helper AST identity were checked before launch.','', 'Stage-wise held-out usage is in baseline/stage_usage.csv and slower/stage_usage.csv. Training-phase by experimental-stage exposure is in each training_stage_usage.csv. Complete gradient summaries and costs are in report.json. All original checkpoints and earlier results remain intact.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');write(C/'status.json',dict(status='complete',diagnostics=str(out),checkpoint_hashes=hashes));print(json.dumps({k:{x:result[k][x] for x in ['hard_eig','relaxed_eig']} for k in runs}),flush=True)
if __name__=='__main__':main()
