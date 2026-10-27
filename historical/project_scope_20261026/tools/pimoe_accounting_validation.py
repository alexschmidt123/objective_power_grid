from pathlib import Path
import json,csv,math
import numpy as np
from scipy.stats import t
import argparse
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);args=p.parse_args();C=Path(args.campaign).resolve();read=lambda p:json.loads(p.read_text());ref=C/'reference';old=read(ref/'reports/resource_ledger.json');freeze=read(ref/'reports/final_model_freeze.json');ev=read(ref/'reports/final_evaluation.json');single=read(C/'location/seed101/single/reports/training_metadata.json')
audit=dict(original_raw_ledger_seconds=old['charged_development_training_seconds'],correct_training_development_charge_seconds=freeze['charged_seconds'],independent_official_evaluation_seconds=ev['evaluation_seconds'],residual_overhead_seconds=old['charged_development_training_seconds']-freeze['charged_seconds']-ev['evaluation_seconds'],original_raw_counter_scope='End-to-end including official evaluation; NOT training-only',original_native_MoE_optimizer_wall_seconds=sum(read(ref/'reports'/f'{n}_training.json')['wall_seconds'] for n in ['pimoe_pilot','pimoe_final']),single_native_optimizer_wall_seconds=single['original_pilot_wall_seconds']+single['new_training_wall_seconds'],single_conservative_complete_attributed_seconds=single['conservative_complete_credit_seconds'],training_only_likelihood_counter=None,GPU_active_seconds=None,missing_cost_note='Unknown training-only likelihood split retained null; end-to-end forward counter and shared dataset counters available. Native wall not GPU-active time.')
(C/'reports/original_budget_accounting_audit.json').write_text(json.dumps(audit,indent=2))
rows=list(csv.DictReader((C/'reports/matched_control_independent_states.csv').open()));paired=[]
for metric in ['NLL','EIG_RMSE','EIG_bias','selected_reference_EIG','selected_regret','coverage90']:
 d=[]
 for episode in range(4):
  a=[float(x[metric]) for x in rows if x['model']=='moe' and int(x['episode'])==episode];b=[float(x[metric]) for x in rows if x['model']=='single' and int(x['episode'])==episode];d.append(float(np.mean(a)-np.mean(b)))
 mean=float(np.mean(d));se=float(np.std(d,ddof=1)/2);delta=float(t.ppf(.975,3)*se);paired.append(dict(metric=metric,MoE_minus_single=mean,episode_cluster_SE=se,low95=mean-delta,high95=mean+delta,independent_diagnostic_episodes=4,assumption='Exploratory Student-t CI over4episode mean differences; states/candidates are not treated as independent; no multiplicity correction'))
with (C/'reports/diagnostic_paired_episode_uncertainty.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
(C/'readiness/replication_hold.json').write_text(json.dumps(dict(status='HOLD',training_seeds=[102,103,104,105],reason='New held-out on-policy PIMoE EIG RMSE0.17427 exceeds original0.15nat validation tolerance; initial controlled official EIG comparison pending. No model tuning/removal/failed-seed substitution.',core_functional_tests_passed=True,heldout_information_validation_passed=False,single_control_evaluation_permitted=True),indent=2))
print(json.dumps(audit,indent=2));print(json.dumps(paired,indent=2))
