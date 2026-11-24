"""Publish frozen architecture/execution ablations without altering models."""
import json,pathlib,sys
import numpy as np
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
WORK=ROOT/'development/hec_git_sync'
sys.path.insert(0,str(WORK/'development/moe_sboed/hec_v2_resolution_20261112'))
from resolution import paired
RUN=ROOT/'scratch_runs/hec_v3_causal_troubleshoot_20261113'
PREVIOUS=ROOT/'scratch_runs/hec_v3_same_V2_bank_20261113_02'
DEST=WORK/'reports/hec_v3_causal_troubleshoot_20261113'
def main():
    status=json.loads((RUN/'STATUS.json').read_text());assert status['status']=='COMPLETE' and status['historical_artifacts_preserved']
    result=json.loads((RUN/'RESULTS.json').read_text());base=json.loads((PREVIOUS/'RESULTS.json').read_text())
    bridge=np.load(RUN/'V2_head_bridge_scores.npy')
    derived=dict(bridge_minus_initial_router=paired(bridge,np.load(RUN/'initial_router_scores.npy')),
                 bridge_minus_original_V2=paired(bridge,np.load(PREVIOUS/'V2_scores.npy')))
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'SUMMARY.json').exists()
    (DEST/'SUMMARY.json').write_text(json.dumps(dict(ablation=result,additional_exploratory_comparisons=derived,reference=base,artifacts=str(RUN)),indent=2)+'\n')
    text=['# Frozen V3 troubleshooting: expert conversion versus routing','',
       '**Strongest observed result: reconnecting the trained V3 query cores to the original V2 joint fusion/head and inherited routing recovers V2-level historical-bank performance.**','',
       'All results use the exact same2000-case historical V2 bank,noise,T30 likelihood and1M contrasts. No training or checkpoint changes. This is diagnostic reuse,not fresh method validation.','',
       '| Policy/execution | Estimated EIG,nats |','|---|---:|',f"|Original V2|{base['means']['V2']:.8f}|",f"|Current V3|{base['means']['V3']:.8f}|",f"|V3 fixed E3|{base['means']['expert3']:.8f}|"]
    for name,item in result['statistics'].items():text.append(f"|{name}|{item['mean']:.8f}|")
    text.extend(['','## Expert design: loss in the decision-interface conversion','',
       'Before preparation,the separate heads exactly reproduce V2 one-hot pathways,not the original full four-pathway policy. V2 mixes weighted packet contributions BEFORE ReLU/final decoding;V3 samples a separately decoded design. These maps are not mathematically equivalent.','',
       'All four heads imitate the full V2 action with low heldout MSE,but their complete-episode EIG immediately after preparation is below V2.250 native updates per expert yield no statistically supported EIG improvement in this historical-bank test. Pointwise action imitation was an insufficient prerequisite for preserving sequential policy quality. This is an observed design/preparation weakness,not an incorrect-gradient bug or proof that standalone MoE experts cannot work.','',
       'The bridge and initial-router ablations use the same inherited routing function and same fine-tuned query mechanisms;they differ in whether information is jointly fused into one V2 design or converted to independent heads followed by categorical execution. Their paired exploratory comparison is:','',
       '```json',json.dumps(derived,indent=2),'```','',
       'Bridge minus current V3:+0.134650,paired sPCE95%CI[0.086676,0.182625],Holm10 p4.18e-7. Bridge minus original V2:+0.008398,CI[-0.015406,0.032203];not a demonstrated improvement. Bridge versus initial-router comparison is additional/exploratory. The bridge changes head/fusion/execution jointly and does not isolate a single head-capacity parameter.','',
       '## Routing quality','',
       'Initial routing,argmax,critic-greedy,three-step commitments and full-episode commitments do not significantly improve current V3 after the existing10-test Holm correction. Thus neither switching frequency nor sampling alone is established as the main cause. Critic-greedy is an offline diagnostic,not a deployment change. Earlier loss/gradient/optimizer checks pass;no coding bug has been established in those tested paths.','',
       '## Specific recommendation','',
       'The next change should address expert initialization/decision representation:preserve the full V2 decision basis in each standalone expert and specialize through its own query/correction,then validate full-episode EIG before router training. Do not treat low one-step imitation error as preservation of V2. This recommendation is not implemented here;no new training or architecture replacement was launched.','',
       result['limitations'],'',f'Artifacts:`{RUN}`',''])
    (DEST/'RESULTS.md').write_text('\n'.join(text));print(str(DEST))
if __name__=='__main__':main()
