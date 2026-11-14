"""Compact reports for exact-bank comparison and frozen routing diagnostics."""
import json,pathlib
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
COMPARE=ROOT/'scratch_runs/hec_v3_same_V2_bank_20261113_02'
DEBUG=ROOT/'scratch_runs/hec_v3_troubleshoot_native_20261113'
DEST=ROOT/'development/hec_git_sync/reports/hec_v3_troubleshoot_20261113'
def main():
    cs=json.loads((COMPARE/'STATUS.json').read_text());ds=json.loads((DEBUG/'STATUS.json').read_text())
    assert cs['status']==ds['status']=='COMPLETE' and cs['historical_artifacts_preserved'] and ds['historical_artifacts_preserved']
    compare=json.loads((COMPARE/'RESULTS.json').read_text());debug=json.loads((DEBUG/'RESULTS.json').read_text())
    assert all(v['matched'] for v in compare['V2_reproduction'].values())
    DEST.mkdir(parents=True,exist_ok=True);assert not (DEST/'SUMMARY.json').exists()
    (DEST/'SUMMARY.json').write_text(json.dumps(dict(historical_comparison=compare,routing_debug=debug,comparison_path=str(COMPARE),debug_path=str(DEBUG)),indent=2)+'\n')
    text=['# Exact original V2 experiment: V2 versus new four-pathway V3','',
       '**Same2000 cases,same generating truths/observation noise,same saved1M prior contrasts,sameT30 location-finding likelihood and bounds. No training.**','',
       '| Policy | Estimated EIG,nats |','|---|---:|']
    for key,value in compare['means'].items():text.append(f'|{key}|{value:.8f}|')
    text.extend(['','## Primary paired difference:V3 minus V2','','```json',json.dumps(compare['primary_V3_minus_V2'],indent=2),'```','',
       'V2 learned and uniform saved per-case scores reproduce to maximum errors2.49e-14 and2.66e-14,respectively. This validates the original experimental bank and scorer.','',
       compare['fairness'],'',compare['limitations'],'',
       'Uniform semantics:V2 weights feature packets uniformly;V3 chooses one standalone expert uniformly. These are their respective architecture-consistent ablations,not identical policies. Single-expert intervals in the raw summary are exploratory/unadjusted.','',f'Artifacts:`{COMPARE}`',''])
    (DEST/'FAIR_COMPARISON.md').write_text('\n'.join(text))
    text=['# Routing quality versus expert design: frozen diagnostics','',
       '## Routing quality','',
       'No implementation bug found in the tested actor loss/gradient/optimizer/freeze paths. The actual actor loss equals the explicit categorical expected native-Q loss;logit gradients match the analytic softmax plus tanh-cap derivative. Every router tensor belongs to the optimizer with250 Adam steps and finite gradients. Saved expert weights and selector-initial state reproduce exactly. These tests do not prove all code is bug-free.','',
       'On24 fresh diagnostic histories,average E4 probability rises from0.48244 initially to0.91954 after training. The critic argmax chooses E4 in7/24 histories,and E2 in8/24. The cap derivative averages0.94983;there is no evidence of complete tanh-gradient shutdown. Probabilities/rankings alone are not utility evidence.','',
       '## Independent native-return continuation test','','```json',json.dumps(debug['native_return_audit'],indent=2),'```','',
       'The independent-bank selected action does NOT significantly beat learned routing on native variational return. Its comparison with critic argmax is only nominally positive (p0.0435),exploratory/unadjusted;this is insufficient to confirm critic misranking as the root cause. This criterion is native terminal-minus-prefix logposterior,not16M prior-sPCE.','',
       '## Expert design','',
       'The four query mechanisms are distinct and their saved standalone weights remain intact. Distinct computations do not establish useful conditional complementarity. Conditional selection versus the selection-chosen fixed expert has CI[-0.03778,0.04527];this small diagnostic does not establish a conditional advantage. It also does not establish absence of one.','',
       '## Evidence-grounded next test','',
       'Do not launch another expert rewrite or longer router training from these results. The next targeted routing diagnostic should compare one-step expert selection with commitment to the same expert for the remaining horizon,using paired posterior-conditioned futures. The current one-step test returns to the learned continuation,whereas full-episode standalone policies change all future decisions. This distinction matters in sBOED and is unresolved here.','',
       f'Artifacts:`{DEBUG}`','']
    (DEST/'ROUTING_DIAGNOSIS.md').write_text('\n'.join(text));print(str(DEST))
if __name__=='__main__':main()
