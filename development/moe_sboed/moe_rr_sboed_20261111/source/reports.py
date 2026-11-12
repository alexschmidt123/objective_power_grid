"""Evidence-derived reports; negative or incomplete runs remain explicit."""
import pathlib,json,math
DYNAMIC=['TRAINING_RESULTS.md','PAIRED_EIG_COMPARISON.md','ROUTING_AND_SPECIALIZATION.md','GPU_CPU_COMPUTE_LEDGER.md','ICML_NOVELTY_ASSESSMENT.md','NEXT_RESEARCH_DECISION.md']

def write(root,destination=None):
    root=pathlib.Path(root);dest=pathlib.Path(destination or root);dest.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text()) if p.exists() else None
    completion=read(root/'evidence/EXPERIMENT_COMPLETION.json') or dict(status='NOT_STARTED_OR_RUNNING')
    train={name:read(root/f'{"preparation" if name=="preparation" else "arms/"+name}/RESULT.json') for name in ['preparation','full','control']}
    evaluation=read(root/'evaluation/EIG_RESULTS.json');routing=read(root/'evaluation/ROUTING_RESULTS.json');latency=read(root/'evaluation/LATENCY.json')
    resource=[]
    if (root/'TRAINING_LOGS/resources.jsonl').exists():resource=[json.loads(v) for v in (root/'TRAINING_LOGS/resources.jsonl').read_text().splitlines()]
    summary={key:dict(mean=sum(float(r[key]) for r in resource)/len(resource),maximum=max(float(r[key]) for r in resource)) for key in ['CPU_percent_one_core','RSS_bytes','GPU_percent','GPU_memory_MiB','GPU_power_W']} if resource else {}
    goals={name:'NOT_ASSESSED' for name in ['higher_than_matched_uniform_top2','higher_than_original_HEC_v2','higher_than_uniform_HEC_base','useful_routed_corrections','useful_history_dependent_routing','conditional_specialization']}
    if evaluation:
        for source,name in [('control','higher_than_matched_uniform_top2'),('hec_v2','higher_than_original_HEC_v2'),('hec_uniform','higher_than_uniform_HEC_base')]:
            r=evaluation['primary'][source];goals[name]=bool(r['mean']>0 and r['Holm3_p']<.05)
        r=evaluation['mechanisms']['uniform_deployment'];goals['useful_history_dependent_routing']=bool(r['mean']>0 and r['Holm5_p']<.05)
        removal=any(evaluation['mechanisms'][f'remove{k}']['mean']>0 and evaluation['mechanisms'][f'remove{k}']['Holm5_p']<.05 for k in range(1,5))
        goals['useful_routed_corrections']=bool(goals['higher_than_uniform_HEC_base'] and removal)
        if routing:goals['conditional_specialization']=bool(routing.get('conditional_crossovers'))
    result=dict(completion=completion,training=train,evaluation=evaluation,routing=routing,latency=latency,resource_summary=summary,
                goals=goals,preservation=read(root/'evidence/PRESERVATION_VERIFIED.json'),
                evaluation_completion=read(root/'evaluation/COMPLETION.json'),
                validation={n:read(root/f'evidence/{n}.json') for n in ['IMPLEMENTATION_CHECKS','PHYSICS_CHECKS','PREPARATION_CHECKS','EVALUATION_CHECKS']})
    (dest/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    def save(name,title,body):
        (dest/name).write_text('# '+title+'\n\nExperiment status: '+completion['status']+'.\n\n'+body.rstrip()+'\n')
    save('TRAINING_RESULTS.md','RR-MoE native training',
        'One common 100-update uniform-Top2 preparation followed by 2000 learned and 2000 uniform joint updates. Each arm inherits the identical preparation actor/target,critic/posterior,replay,Adam/scheduler and simulator RNG. Frozen HEC is excluded from Adam and Polyak; no extra reward or training extension.\n\n'
        'Actual preparation work is counted once (100000 fresh episodes); each arm inherits it and adds 2M episodes. Shared preparation is not a free pretrained model. The disposable native integration validation adds 1000 episodes/30000 observations and never initializes scientific training.\n\n```json\n'+json.dumps(train,indent=2)+'\n```')
    text='No independent EIG results yet. No performance claim.\n'
    if evaluation:
        lines=['2000 trajectories,T30,1M prior contrasts plus the positive once. New common bank; complete resimulation for every changed policy. Finite-contrast lower bound; MCSE conditional on one trained lineage and the common contrast bank.','',
               '|Policy|EIG nats|MCSE|','|---|---:|---:|']
        for name,row in evaluation['methods'].items():lines.append(f'|{name}|{row["mean"]:.6f}|{row["MCSE"]:.6f}|')
        lines+=['','Primary differences are full RR minus comparison. Ordinary95% t intervals are unadjusted; Holm3 p-values cover the primary family.','',
                '```json',json.dumps(evaluation['primary'],indent=2),'```','',
                'Mechanistic contrasts: frozen learned actor versus uniform Top2 deployment and four individual adaptation removals. Full trajectories are resimulated; Holm5 applies.','',
                '```json',json.dumps(evaluation['mechanisms'],indent=2),'```','',
                'Frozen SG/conventional-MoE/vsOED/DAD/RL/ScoreBED contextual comparisons use Holm6 separately and are not training-budget matched. Historical quoted numbers from other banks were never paired with this bank.','',
                '```json',json.dumps(evaluation['external'],indent=2),'```']
        text='\n'.join(lines)
    save('PAIRED_EIG_COMPARISON.md','Independent RR-MoE EIG',text)
    save('ROUTING_AND_SPECIALIZATION.md','Routed specialist contributions',
        'Top2 selection frequencies sum to two selected adaptations per decision. Every HEC specialist pathway remains present in the frozen collaborative base. Only the two sampled new adaptation branches execute at deployment; all six categorical possibilities are enumerated for training gradients.\n\n'
        'Probability variation,balanced use and different actions do not by themselves establish intelligent routing. Frozen uniform deployment and adaptation-removal EIG tests are the main mechanistic evidence. A common-prefix panel retains the joint latent draw, resamples all future noise/routing, removes one adaptation at the next decision only, and then follows the unchanged full policy. Replicates are clustered by history; conditional tests have a separate Holm family. One lineage/noisy histories limit interpretation.\n\n```json\n'+json.dumps(routing,indent=2)+'\n```')
    save('GPU_CPU_COMPUTE_LEDGER.md','Actual compute ledger',
        'Cost is recorded only as a secondary measure,never an optimization term. Training receipts give actual optimizer/simulator calls,wall time,CPU time,peak RSS and CUDA allocations. Frozen-cache construction and internal HEC likelihood/Jacobian/provider queries are counted by model counters. Cache reuse preserves the same frozen inputs and values.\n\n'
        'Device utilization/power/memory and process CPU/RSS are sampled every10seconds. The full frozen HEC model still executes; RR does not claim two-expert cost for the entire policy. Inference latency includes frozen HEC providers and two selected new adaptations. Shared preparation costs are separate,not omitted from either lineage. Evaluation receipts count prior-likelihood terms and observations.\n\n```json\n'+json.dumps(dict(stages=train,resources=summary,latency=latency,evaluation_completion=result['evaluation_completion']),indent=2)+'\n```')
    success=all(goals.get(n) is True for n in ['higher_than_matched_uniform_top2','higher_than_original_HEC_v2','useful_routed_corrections','conditional_specialization'])
    save('ICML_NOVELTY_ASSESSMENT.md','Algorithmic assessment',
        'The specific pilot combines a frozen competent collaborative uniform HEC base with routed feature residuals and exact unordered Top2 actor-critic expectations. It preserves real HEC specialty channels instead of replacing them with four complete policies. Initial identity and explicit uniform preparation address the zero-head gradient block.\n\n'
        'This pilot does not establish literature novelty or seed-robust generalization. Residual adaptation,categorical selection and expected-value gradients are mechanisms whose combination must earn information gains and conditional contribution evidence; renaming them is insufficient. No novelty uncertainty was used to gate training. Formal prior-art comparison and multi-seed confirmation remain outside this one-pilot authorization.\n\n'
        '```json\n'+json.dumps(dict(primary_and_mechanistic_support=goals,all_final_success_criteria_supported=success,ICML_level_novelty_established=False),indent=2)+'\n```')
    save('NEXT_RESEARCH_DECISION.md','RR-MoE research decision',
        ('PREDECLARED SUCCESS CRITERIA SUPPORTED ON THIS DEVELOPMENT BANK' if success else 'SUCCESS NOT ESTABLISHED; PRESERVE NEGATIVE OR INCOMPLETE EVIDENCE')+
        '\n\n```json\n'+json.dumps(dict(completion=completion,criteria=goals,preservation=result['preservation']),indent=2)+'\n```\n\n'
        'No additional architecture,training seed,budget extension or automatic follow-up experiment is authorized. Checkpoint/source/report bytes remain preserved. Conditional probes are supporting diagnostics,not new objectives and not proof of exact conditional MI.')
    return result

if __name__=='__main__':
    import sys
    write(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
