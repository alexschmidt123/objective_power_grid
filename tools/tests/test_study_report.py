import copy
import json
import numpy as np
from tools.report_four_variant_study import stats,normalize_routes,paired_runs

def test_report_intervals_are_paired_not_independent_errors():
 r=stats([1.,2.,3.]);assert r['mean']==2.
 assert abs(r['mc_standard_error']-1/3**.5)<1e-12
 assert stats([])['mean'] is None

def test_per_step_statistics_reconstruction_preserves_requests_and_changes():
 rows=[dict(routing=[dict(weights=w,selected_expert=None) for w in [[1,0,0,0],[1,0,0,0],[0,1,0,0]]])]
 out=normalize_routes(rows)[0]['routing']
 assert [r['selected_expert'] for r in out]==[0,0,1]
 assert [r['strategy']['reroute_requests'] for r in out]==[1,2,3]
 assert [r['strategy']['strategy_changes'] for r in out]==[0,0,1]
 assert [r['strategy']['consecutive_duration'] for r in out]==[1,2,1]

def test_mismatched_case_or_bank_identity_never_gets_paired_interval():
 a=dict(status='complete',cases_sha256='a',bank_sha256='x',rows=[dict(spce=2),dict(spce=4)])
 b=dict(status='complete',cases_sha256='b',bank_sha256='x',rows=[dict(spce=1),dict(spce=3)])
 assert 'conditional_95_interval' not in paired_runs(a,b)
 b['cases_sha256']='a';assert paired_runs(a,b)['mean']==1
 b['bank_sha256']='y';assert 'conditional_95_interval' not in paired_runs(a,b)


def test_additional_and_pretrained_likelihood_costs_are_separate(tmp_path):
 from tools.report_four_variant_study import load_run
 base=tmp_path/'base';(base/'models').mkdir(parents=True)
 (base/'run_config.json').write_text(json.dumps(dict(settings=dict(contrasts=1024))))
 run=tmp_path/'continued';(run/'models').mkdir(parents=True);(run/'eval').mkdir()
 (run/'exit_code').write_text('0\n')
 (run/'summary.json').write_text(json.dumps([dict(method='sum_single',eval_seed=1001,evaluation_seconds=3.,mean_decision_seconds=.01)]))
 (run/'run_config.json').write_text(json.dumps(dict(horizon=3,settings=dict(contrasts=4096))))
 (run/'models/sum_single_training.json').write_text(json.dumps(dict(generated_training_trajectories=32,total_training_episodes=64,training_seconds=2.,base_training_seconds=1.,base_pretraining=dict(path=str(base/'models/sum_single.pt'),training_episodes=32))))
 (run/'eval/sum_single_seed1001.json').write_text(json.dumps([dict(truth=[1.],measurement_noise_standard=[0.],spce=x) for x in [2.,3.]]))
 np.save(run/'eval/contrast_bank_seed1001.npy',np.array([1.,2.]))
 item=load_run(tmp_path,run,'sum_single','control','contrast_control')
 assert item['additional_episodes']==32 and item['total_episodes']==64
 assert item['likelihood_cost']['additional_truth_inclusive_trajectories']==32*4097
 assert item['likelihood_cost']['base_truth_inclusive_trajectories']==32*1025
 assert item['likelihood_cost']['total_truth_inclusive_trajectories']==32*(4097+1025)
