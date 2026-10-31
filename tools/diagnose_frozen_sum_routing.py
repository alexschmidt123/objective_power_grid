"""Frozen learned versus dominant-expert routing: no training-code changes."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.location_evaluation import contrast_scores
from src.research.sparse_moe import SparseMoESBOED


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def state_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenRoutingView(nn.Module):
    """Evaluation-only branch selection; underlying architecture/weights intact."""
    def __init__(self,model,expert=None):
        super().__init__();self.model=model;self.expert=expert
        self.kind=model.kind;self.last_route=None;self.eval()

    def forward(self,features,stage):
        if stage==0:
            self.last_route=None
            return self.model(features,stage)
        if self.expert is None:
            value=self.model(features,stage)
            self.last_route=self.model.last_route
            return value
        context=self.model.context(features)
        weights=features.new_zeros(len(features),4);weights[:,self.expert]=1.
        self.last_route={'weights':weights}
        # Direct call bypasses the router without changing the encoder, expert,
        # first design, or external bounded feasibility map and simulator.
        return self.model.experts[self.expert](context[:,self.expert])


def restore(saved):
    model=SparseMoESBOED(2,1,saved['horizon'],saved['hidden'],k=saved['routing_k'],
        first_action=[0.,0.],history_encoder_revision=saved['history_encoder_revision'],
        action_scale=saved['action_scale'],obs_scale=saved['obs_scale'])
    model.load_state_dict(saved['state_dict']);model.router.gamma=saved['gamma']
    model.eval()
    return model


def saved_usage(rows):
    counts=np.zeros((30,4),dtype=np.int64)
    for row in rows:
        assert [r['stage'] for r in row['routing']]==list(range(1,30))
        for route in row['routing']:
            w=np.asarray(route['weights']);j=route['selected_expert']
            assert (w>0).sum()==1 and w[j]==1
            counts[route['stage'],j]+=1
    assert (counts[1:].sum(1)==2000).all() and counts[0].sum()==0
    return counts


def exposure(trace):
    phases={key:dict(updates=0,primary_episodes=0,post_initial_decisions=0,
                    gate_mass=np.zeros(4),executions=np.zeros(4),hard_choices=np.zeros(4),
                    pre_grad=[[] for _ in range(4)],post_grad=[[] for _ in range(4)])
            for key in ['whole_training','final20percent']}
    previous=0;steps=0
    with trace.open() as f:
        for line in f:
            r=json.loads(line);steps+=1;assert r['update']==steps
            batch=r['primary_episodes']-previous;previous=r['primary_episodes']
            nstage=len(r['execution_utilization_by_stage']);assert nstage==29
            for key,p in phases.items():
                if key=='final20percent' and r['primary_episodes']<=800000:continue
                p['updates']+=1;p['primary_episodes']+=batch;p['post_initial_decisions']+=batch*nstage
                p['gate_mass']+=np.asarray(r['router_probabilities'])*batch*nstage
                p['executions']+=np.asarray(r['execution_utilization'])*batch*nstage
                p['hard_choices']+=np.asarray(r['hard_choice_frequency'])*batch*nstage
                for j in range(4):
                    p['pre_grad'][j].append(r['gradient_before'][f'expert_{j}'])
                    p['post_grad'][j].append(r['gradient_after'][f'expert_{j}'])
    assert previous==1000000 and steps==15625
    result={}
    for key,p in phases.items():
        result[key]=dict(updates=p['updates'],primary_episodes=p['primary_episodes'],
            post_initial_decisions=p['post_initial_decisions'],experts=[])
        for j in range(4):
            before=np.asarray(p['pre_grad'][j]);after=np.asarray(p['post_grad'][j])
            result[key]['experts'].append(dict(expert=j,
                gate_mass_equivalent_decisions=float(p['gate_mass'][j]),
                mean_relaxed_gate_mass=float(p['gate_mass'][j]/p['post_initial_decisions']),
                executed_nonzero_weight_decisions=int(round(p['executions'][j])),
                executed_fraction=float(p['executions'][j]/p['post_initial_decisions']),
                relaxed_argmax_decision_fraction=float(p['hard_choices'][j]/p['post_initial_decisions']),
                pre_gradient_updates_above_1e_8=int((before>1e-8).sum()),
                post_gradient_updates_above_1e_8=int((after>1e-8).sum()),
                post_gradient_non_negligible_fraction=float(np.mean(after>1e-8)),
                post_gradient_updates_above_1e_3=int((after>1e-3).sum()),
                post_gradient_fraction_above_1e_3=float(np.mean(after>1e-3)),
                pre_gradient_quantiles=np.quantile(before,[0,.5,.95,1]).tolist(),
                post_gradient_quantiles=np.quantile(after,[0,.5,.95,1]).tolist()))
    result['limitations']=[
        'Nonzero-weight execution during relaxed training is not exclusive hard selection; exposures can sum above100percent.',
        'Gate-mass-equivalent decisions are weighted exposure proxies, not separately simulated episodes.',
        'Gradient norms are aggregate per-minibatch expert norms, not individual parameter updates.',
        'Nonzero-gradient frequency does not measure Adam momentum-driven weight changes or useful learning.'
    ]
    return result


@torch.no_grad()
def evaluate_batch(benchmark,view,truth,noise,start,bank):
    routes=[]
    def capture(module,inputs,output):
        if inputs[1]>0:
            w=module.last_route['weights']
            assert ((w>0).sum(-1)==1).all()
            routes.append(w.argmax(-1).cpu().numpy())
    handle=view.register_forward_hook(capture)
    try:
        result=rollout(benchmark,view,30,len(truth),1001+start*97451,0,
                       truth=truth,noise=noise,method='moe_sboed')
    finally:handle.remove()
    assert len(routes)==29
    assert all(bool(((a>=-4)&(a<=4)).all()) for a in result['actions'])
    info,_=contrast_scores(benchmark,result,bank,16384)
    return result,info,np.stack(routes,axis=1)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--preflight-only',action='store_true')
    args=ap.parse_args();root=Path.cwd();source=Path(args.source).resolve();out=Path(args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    checkpoint=source/'models/moe_sboed.pt';rows_path=source/'eval/moe_sboed_seed1001.json'
    bank_path=source/'eval/contrast_bank_seed1001.npy';trace=source/'models/moe_sboed_diagnostics.jsonl'
    config=read(source/'run_config.json');rows=read(rows_path)
    assert len(rows)==2000 and config['train_seed']==101 and config['horizon']==30
    assert config['settings']['eval_contrasts']==1000000
    assert config['benchmark']['location_finding']['design_bounds']==[-4.,4.]
    assert config['benchmark']['observation']=={'noise_sigma':.5,'sampling':'normal_log_signal'}
    # Check that current imported scientific code exactly matches the source
    # frozen with the checkpoint. Never silently evaluate changed code.
    hashes=read(source/'source_manifest.json')['sha256']
    mismatched=[p for p,h in hashes.items() if p.startswith('src/') and digest(root/p)!=h]
    if mismatched:raise RuntimeError('Imported scientific source differs from saved run: '+str(mismatched))
    original_hashes={str(p.relative_to(root)):digest(p) for p in [checkpoint,rows_path,bank_path,trace,source/'run_config.json']}
    counts=saved_usage(rows);total=counts.sum(0);dominant=int(total.argmax())
    usage=dict(expert_indexing='zero-based0..3',first_measurement_excluded=True,
               trajectories=2000,decisions_per_trajectory=29,post_initial_decisions=58000,
               dominant_expert=dominant,counts=total.tolist(),frequencies=(total/58000).tolist(),
               by_measurement_stage=[dict(measurement_stage=s+1,counts=counts[s].tolist(),frequencies=(counts[s]/2000).tolist()) for s in range(1,30)],
               selection_rule='Largest usage count under original learned routing; no EIG-based expert selection.')
    write(out/'expert_usage.json',usage)
    training=exposure(trace);write(out/'training_exposure.json',training)
    with (out/'stage_usage.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['measurement_stage']+[f'expert{j}_count' for j in range(4)]+[f'expert{j}_fraction' for j in range(4)])
        for r in usage['by_measurement_stage']:writer.writerow([r['measurement_stage']]+r['counts']+r['frequencies'])
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['history_encoder_revision']==2 and saved['routing_k']==1
    benchmark=make_benchmark(config['benchmark'])
    learned=FrozenRoutingView(restore(saved));fixed=FrozenRoutingView(restore(saved),dominant)
    state_before=[state_digest(v.model) for v in [learned,fixed]];assert state_before[0]==state_before[1]
    truth=torch.tensor([r['truth'] for r in rows],dtype=torch.float64)
    noise=torch.tensor([r['measurement_noise_standard'] for r in rows],dtype=torch.float64)
    bank=torch.from_numpy(np.load(bank_path));assert bank.shape==(1000000,4)
    reference=np.array([r['spce'] for r in rows])
    saved_actions=np.array([r['actions'] for r in rows]);saved_observations=np.array([r['observations'] for r in rows])
    # Enforce the bypass: any fixed-policy router invocation fails immediately.
    def reject_router(module,inputs):raise AssertionError('Forced policy must bypass router')
    fixed_router_guard=fixed.model.router.register_forward_pre_hook(reject_router)
    calls=np.zeros(4,dtype=np.int64)
    handles=[expert.register_forward_pre_hook(lambda m,inputs,j=j:calls.__setitem__(j,calls[j]+len(inputs[0]))) for j,expert in enumerate(fixed.model.experts)]
    st=time.perf_counter()
    a,ai,ar=evaluate_batch(benchmark,learned,truth[:16],noise[:16],0,bank)
    b,bi,br=evaluate_batch(benchmark,fixed,truth[:16],noise[:16],0,bank)
    elapsed=time.perf_counter()-st
    assert np.allclose(torch.stack(a['actions'],1).numpy(),saved_actions[:16],rtol=0,atol=1e-10)
    assert np.allclose(ai[:,-1].numpy(),reference[:16],rtol=0,atol=1e-9)
    torch.testing.assert_close(a['actions'][0],b['actions'][0],rtol=0,atol=0)
    torch.testing.assert_close(a['observations'][0],b['observations'][0],rtol=0,atol=0)
    assert calls[dominant]==16*29 and calls.sum()==16*29
    assert (br==dominant).all()
    for h in handles:h.remove()
    assert state_before==[state_digest(v.model) for v in [learned,fixed]]
    preflight=dict(status='passed',host='labpc',paired_first_batch_seconds=elapsed,
        estimated_full_eig_seconds=elapsed*125,systems=2000,contrasts=1000000,training=False,
        posterior_inference=False,dominant_expert=dominant,
        checks=['scientific source matches frozen checkpoint source','same model state in both views',
                'learned actions and EIG reproduce saved first batch','identical first physical measurement and observation',
                'forced route bypasses router','only dominant expert executes after initial measurement','bounds and frozen weights'])
    write(out/'preflight.json',preflight)
    print('PREFLIGHT',json.dumps(preflight),flush=True)
    print('USAGE',json.dumps(usage['frequencies']),flush=True)
    if args.preflight_only:return
    started=time.perf_counter();results=[];reproduced_counts=np.zeros((30,4),dtype=np.int64)
    max_action_error=0.;max_observation_error=0.;max_score_error=0.
    for start in range(0,2000,16):
        end=start+16
        a,ai,ar=evaluate_batch(benchmark,learned,truth[start:end],noise[start:end],start,bank)
        b,bi,br=evaluate_batch(benchmark,fixed,truth[start:end],noise[start:end],start,bank)
        actions=torch.stack(a['actions'],1).numpy();observations=torch.stack(a['observations'],1).numpy()
        max_action_error=max(max_action_error,float(np.abs(actions-saved_actions[start:end]).max()))
        max_observation_error=max(max_observation_error,float(np.abs(observations-saved_observations[start:end]).max()))
        max_score_error=max(max_score_error,float(np.abs(ai[:,-1].numpy()-reference[start:end]).max()))
        assert max_action_error<1e-10 and max_observation_error<1e-10 and max_score_error<1e-9
        torch.testing.assert_close(a['actions'][0],b['actions'][0],rtol=0,atol=0)
        torch.testing.assert_close(a['observations'][0],b['observations'][0],rtol=0,atol=0)
        assert (br==dominant).all()
        for s in range(29):reproduced_counts[s+1]+=np.bincount(ar[:,s],minlength=4)
        for i in range(16):
            results.append(dict(trajectory_id=start+i,truth=rows[start+i]['truth'],measurement_noise_standard=rows[start+i]['measurement_noise_standard'],
                learned_spce=float(ai[i,-1]),fixed_spce=float(bi[i,-1]),delta_route=float(ai[i,-1]-bi[i,-1]),
                learned_selected_experts=ar[i].tolist(),fixed_expert=dominant,
                fixed_actions=[x[i].tolist() for x in b['actions']],fixed_observations=[x[i].tolist() for x in b['observations']]))
        if end%128==0 or end==2000:print('PAIRED_EIG',end,'/2000',flush=True)
    fixed_router_guard.remove()
    assert np.array_equal(reproduced_counts,counts)
    assert state_before==[state_digest(v.model) for v in [learned,fixed]]
    for p,h in original_hashes.items():assert digest(root/p)==h
    # A normal paired interval is conditional on the fixed trained checkpoint,
    # selected dominant expert and shared contrast bank; it is not seed variance.
    learned_scores=np.array([r['learned_spce'] for r in results]);fixed_scores=np.array([r['fixed_spce'] for r in results]);delta=learned_scores-fixed_scores
    se=float(delta.std(ddof=1)/math.sqrt(2000));mean=float(delta.mean());interval=[mean-1.96*se,mean+1.96*se]
    clear=interval[0]>0
    recommendation=('Test a matched193-input sum-single control, retaining the same encoder, stage feature, first-action treatment, head widths, optimizer and budget.' if clear else
                    'Test the existing slower-sharpening schedule with the corrected sum encoder, changing only that schedule.')
    report=dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
        source_run=str(source),train_seed=101,eval_seed=1001,systems=2000,L=1000000,
        learned_eig=float(learned_scores.mean()),fixed_dominant_eig=float(fixed_scores.mean()),dominant_expert=dominant,
        delta_route=mean,paired_mc_standard_error=se,paired_normal95_interval=interval,
        learned_eig_conditional_mc_se=float(learned_scores.std(ddof=1)/math.sqrt(2000)),
        fixed_eig_conditional_mc_se=float(fixed_scores.std(ddof=1)/math.sqrt(2000)),
        fraction_trajectories_learned_higher=float(np.mean(delta>0)),
        expert_usage=usage,training_exposure=training,
        interpretation=('Learned routing clearly exceeds the frozen dominant-expert intervention on these paired cases.' if clear else 'No clear positive learned-routing gain over the frozen dominant-expert intervention.'),
        limitations=['Interval conditions on one trained checkpoint, the selected dominant expert and one shared contrast bank; it is not training-seed or contrast-bank uncertainty.',
                     'Dominant expert selected by usage on these same held-out/development cases, not by maximizing evaluated EIG.',
                     'A poor fixed-expert result does not prove beneficial specialization: forced routing induces different, potentially unfamiliar histories.',
                     'Evaluation1001 has already informed development; this is not untouched test evidence.',
                     'MSE is not recomputed; this is the requested EIG-only intervention.'],
        recommended_exactly_one_next_experiment=recommendation,followup_implemented=False,
        eig_evaluation_seconds=time.perf_counter()-started,preflight=preflight,
        learned_reproduction_max_errors=dict(action=max_action_error,observation=max_observation_error,eig=max_score_error),
        source_artifacts_sha256=original_hashes,model_state_sha256=state_before[0],
        scientific_source_manifest=hashes)
    write(out/'paired_trajectories.json',results);write(out/'report.json',report)
    with (out/'paired_scores.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['trajectory_id','learned_spce','fixed_spce','delta_route'])
        for r in results:writer.writerow([r[k] for k in ['trajectory_id','learned_spce','fixed_spce','delta_route']])
    lines=['# Frozen routing diagnostic','',f"Dominant expert: {dominant} (zero-based; first measurement excluded).",'',
           '| Policy | EIG |','|---|---:|',f"| Learned DSelect-1 | {report['learned_eig']:.8f} |",f"| Fixed expert {dominant} | {report['fixed_dominant_eig']:.8f} |",'',
           f"Delta_route = {mean:.8f} nats; paired SE {se:.8f}; approximate95% interval [{interval[0]:.8f}, {interval[1]:.8f}].",'',
           '| Expert | Held-out selection | Training nonzero-gradient updates | Final20% nonzero-gradient updates |','|---|---:|---:|---:|']
    for j in range(4):
        whole=training['whole_training']['experts'][j];late=training['final20percent']['experts'][j]
        lines.append(f"| {j} | {100*usage['frequencies'][j]:.4f}% | {whole['post_gradient_updates_above_1e_8']}/{training['whole_training']['updates']} | {late['post_gradient_updates_above_1e_8']}/{training['final20percent']['updates']} |")
    lines+=['','Non-negligible gradient threshold: post-clipping expert norm >1e-8. These counts are gradient proxies, not actual Adam weight-update counts.','',
            '| Measurement stage | Expert0 | Expert1 | Expert2 | Expert3 |','|---|---:|---:|---:|---:|']
    for row in usage['by_measurement_stage']:lines.append('| '+str(row['measurement_stage'])+' | '+' | '.join(f'{100*v:.2f}%' for v in row['frequencies'])+' |')
    lines+=['',report['interpretation'],'',*report['limitations'],'', 'Exactly one recommended next experiment: '+recommendation,'','No follow-up implemented. All source artifact hashes and model weights verified unchanged.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'exit_code').write_text('0\n')
    print('RESULT',json.dumps({k:report[k] for k in ['learned_eig','fixed_dominant_eig','dominant_expert','delta_route','paired_mc_standard_error','paired_normal95_interval','recommended_exactly_one_next_experiment','eig_evaluation_seconds']}),flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
