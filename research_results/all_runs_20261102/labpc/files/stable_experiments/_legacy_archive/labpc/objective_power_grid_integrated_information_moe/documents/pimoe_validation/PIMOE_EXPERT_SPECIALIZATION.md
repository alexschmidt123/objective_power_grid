# Expert specialization: frozen seed101

28held-out belief states,4independent diagnostic episodes,336candidates,seed1006. Model unchanged: 3a536e77331796ca4a88ae08f1c13106c07a7c31fe6689061482081fca10f4f9.

Mean expert mass: [0.029799391556613243, 0.11640070598355559, 0.18574612463280937, 0.6680537778270218]. Effective expert count 2.231; max-alpha>0.95 in 27.68% of candidates. Soft assignment is not expert sampling or action selection.

Function outputs differ: pairwise meanJS range 0.03789–0.21415nats. Identical-region-input head check: 0.07151196908670063. Independent initial weights alone are not used as evidence.

## Routing and contribution

Full NLL 0.8668970534280442 vs uniform-membership 0.9099941549962932; ranking 0.9345238095238095 vs 0.8523809523809524. Full EIG RMSE 0.1742735068064596 vs uniform 0.17134857068590406. Learned routing improves these NLL/ranking measurements but not every information-error metric.

RemovingE1 changes EIG RMSE to 0.04686974915719257, ranking to 0.95, NLL to 0.8648350472287714. This is counterfactual distribution shift, not definitive causal necessity evidence. No retraining/removal was adopted.

Regional calibration is in regional_calibration.csv. Raw regional NLL cannot be compared as if all regions had equal intrinsic entropy. Responsibility correlations are descriptive, clustered within only4episodes; no exploration/exploitation labels or significance claims.

## Numerical information check

{
  "maximum_abs_quadrature_128_minus24": 0.09340655663055353,
  "mean_abs_quadrature_difference": 0.0016370219735204168,
  "fraction_abs_difference_over005": 0.01488095238095238,
  "same_input_head_JS_mean": 0.07151196908670063,
  "algorithm_changed": false,
  "scope": "diagnostic only; frozen planner still uses24nodes; no final-EIG-based tuning"
}

The full-model information error on new on-policy states exceeds the original0.15nat diagnostic tolerance. Preserve this validation limitation; do not claim universal credible EIG estimation or release unnecessary replication before resolving the gate without tuning the frozen model.

## Questions answered

1.Functionally different:yes on this diagnostic set. 2.Assignments correlate with physical-response uncertainty/candidate/stage; exact descriptive correlations saved. 3.Consistent across seeds:unverified. 4.Learned routing helps NLL/ranking relative to uniform, but not every metric. 5.Parameter-matched single superiority requires the controlled results below and official paired evaluation. 6.Sequential-EIG translation pending. 7.Extra work includes587Mdiagnostic forward-stage evaluations; timing and original phase-accounting limitations disclosed. 8.Four experts are not yet empirically justified. 9.Keep the frozen model as a research reference, not as proof that4experts are necessary. 10.Cross-seed reproducibility, causal specialization and generality remain unsupported.