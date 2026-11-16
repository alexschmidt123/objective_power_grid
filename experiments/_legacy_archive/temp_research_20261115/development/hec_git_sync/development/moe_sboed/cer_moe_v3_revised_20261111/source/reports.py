"""Summarize completed or incomplete evidence without inventing results."""
import json,pathlib,math

def write(root,destination=None):
    root=pathlib.Path(root);destination=pathlib.Path(destination or root);destination.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text()) if p.exists() else None
    arms={name:read(root/f'arms/{name}/RESULT.json') for name in ['full','control']}
    status=read(root/'evidence/EXPERIMENT_COMPLETION.json') or dict(status='RUNNING_OR_NOT_STARTED')
    result=dict(status=status,arms=arms,preparation=read(root/'preparation/RESULT.json'),
                validation={name:read(root/f'evidence/{name}.json') for name in ['IMPLEMENTATION_CHECKS','NATIVE_FULL_BATCH_CHECKS','PREPARATION_CHECKS','EVALUATION_CHECKS']},
                evaluation=read(root/'evaluation/EIG_RESULTS.json'),routing=read(root/'evaluation/COMPETENCY_AND_ROUTING.json'),
                latency=read(root/'evaluation/LATENCY.json'),evaluation_completion=read(root/'evaluation/COMPLETION.json'),
                preservation=read(root/'evidence/PRESERVATION_VERIFIED.json'))
    samples=[]
    if (root/'TRAINING_LOGS/resources.jsonl').exists():
        samples=[json.loads(line) for line in (root/'TRAINING_LOGS/resources.jsonl').read_text().splitlines()]
    result['resource_summary']=dict(samples=len(samples),
        **{key:dict(mean=sum(float(row[key]) for row in samples)/len(samples),maximum=max(float(row[key]) for row in samples))
           for key in ['CPU_percent_one_core','RSS_bytes','GPU_percent','GPU_memory_MiB','GPU_power_W']} if samples else {})
    (destination/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Revised CER-MoE V3 results','',f'Experiment status: {status["status"]}.','',
           'One revised candidate and one matched trained uniform-router control. No architecture sweep, extra reward or cost penalty. All previous artifacts are preserved.','',
           '## Training and exposure','', '```json',json.dumps(dict(preparation=result['preparation'],arms=arms),indent=2),'```','']
    evaluation=result['evaluation']
    if evaluation:
        lines+=['## Independent common-bank full-horizon EIG','',
                '2000 trajectories, T30, 1M prior contrasts plus positive once. Finite-contrast lower bound; MCSE conditional on this one training lineage. Changed policies resimulated whole trajectories.','',
                '|Policy|EIG nats|MCSE|','|---|---:|---:|']
        for name,row in evaluation['methods'].items():lines.append(f'|{name}|{row["mean"]:.6f}|{row["MCSE"]:.6f}|')
        lines+=['','Primary paired differences: candidate minus each comparison. Ordinary95% intervals are unadjusted; Holm7 p-values account for the primary family.','',
                '```json',json.dumps(evaluation['primary'],indent=2),'```','', '## Assessment','']
        for key,label in [('control','trained uniform-router control'),('uniform','uniform deployment of candidate'),('hec_v2','frozen HEC-v2')]:
            row=evaluation['primary'][key];supported=row['mean']>0 and row['Holm7_p']<.05
            lines.append(f'- Higher EIG than {label}: '+('supported on this paired bank' if supported else 'not established')+'.')
        individuals=[evaluation['primary'][f'expert{k}'] for k in range(1,5)]
        supported=all(v['mean']>0 and v['Holm7_p']<.05 for v in individuals)
        lines+=['- Higher EIG than every frozen candidate expert: '+('supported on this paired bank' if supported else 'not established')+'.',
                '- Useful specialization requires conditional strength and routing evidence below; slot differences or nonzero exposure alone are insufficient.',
                '- ICML-level novelty and training-seed robustness are not established by one pilot.','',
                '## Conditional competence and routing','', '```json',json.dumps(result['routing'],indent=2),'```','',
                'Conditional values use exact likelihoods with finite SNIS particles and a frozen normalized variational posterior, not exact conditional MI. Subgroup probes are descriptive and noisy, not additional rewards.','']
    else:lines+=['No completed independent EIG comparison is available yet. No advantage or useful competency is claimed.','']
    lines+=['## Costs and limitations','',
            'Training receipts above contain actual simulator calls, optimizer steps, CPU time, peak RAM/VRAM and wall time. TRAINING_LOGS/resources.jsonl records actual GPU/CPU utilization, device memory and power every 10 seconds. Two disposable full native integration executions add 2000 trajectories/60000 observations; their optimizer steps do not initialize either scientific arm. A disposable preparation check adds 128 critic rows without new simulator episodes. Scientific preparation uses replay/value queries without new simulator episodes.','',
            'Evaluation receipts record resimulation and positive/negative likelihood terms. Conditional diagnostics add128 initial T30 rollouts and128x4x16x(27+18+6) future observations. Inference latency includes Bayesian information, four proposals, native critic and categorical selection. Frozen external models have different historical training/provider budgets.','',
            '```json',json.dumps(dict(resource_summary=result['resource_summary'],evaluation_completion=result['evaluation_completion'],latency=result['latency'],preservation=result['preservation']),indent=2),'```','',
            'No final-test bank, post-hoc training extension, independent specialization/diversity reward or computational-cost objective was used. A selection floor avoids zero exposure but cannot guarantee retained or useful competencies. Negative and incomplete results remain labeled.','']
    (destination/'RESULTS.md').write_text('\n'.join(lines))
    return result

if __name__=='__main__':
    import sys
    write(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
