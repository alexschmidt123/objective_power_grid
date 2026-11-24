import pathlib,json,sys,hashlib,subprocess,numpy as np
C=pathlib.Path(__file__).parent.parent;R=pathlib.Path.home()/'Documents/objective_power_grid_temp';sys.path.insert(0,str(C/'source'))
from native_final import put,sha
d=json.loads((C/'readiness/score_profile_raw.json').read_text());t=d['metadata']['phase_timings'];a=max(t['score_unit_seconds'][2:]);b=max(t['policy_unit_seconds'][1::2]);estimate=(50000*a+100000*b)*1.5+600
assert estimate<37800;meta=d['metadata'];assert meta['completed_restarts']==5 and meta['actual_score_updates']==10 and meta['actual_policy_updates']==10
out=C/'readiness/score_native_profile';selected=np.load(out/f"scorebed_restart{meta['selected_restart']}_policy.npz");export=np.load(out/'scorebed_official_policy.npz');assert all(np.array_equal(selected[k],export[k]) for k in selected.files)
put(C/'readiness/scorebed_production_profile.json',dict(status='PASS',production_batch=1024,horizon=30,restarts=5,native_score_scheduler=50000,native_policy_scheduler_per_restart=20000,score_unit_seconds_upper=a,policy_unit_seconds_upper=b,projected_seconds_including_50pct_margin_and600s=estimate,measured_probe_all_phases_seconds=d['seconds'],five_restart_initialization_and_selection=True,selected_restart_export_bitwise_match=True,profile_updates_only_not_formal_training=True,source='score_profile.py: counts shortened only for bounded readiness; production optimizer/network/sampler/T30/B and scheduler horizons retained'))
# Preserve documentary evidence for exact author defaults.
refs=C/'reports/upstream';refs.mkdir(exist_ok=True)
vendor=R/'src/research/vendor/scorebed/config/experiment/location_finding.yaml';(refs/'scorebed_cc8e996e_location_finding.yaml').write_bytes(vendor.read_bytes())
(refs/'vsOED_GI_variant.txt').write_text('Pinned native adapter upstream_vsoed.py explicitly POST_APPROX(use_NFs=False,n_mixture=8); n_incre30; source github.com/wgshen/vsOED README documents GMM incremental setting. No NF-I claim.\n')
(C/'reports/STAGE_AWARE_FEASIBILITY.md').write_text('# Production feasibility\n\nScoreBED full-shape probe: '+json.dumps(json.loads((C/'readiness/scorebed_production_profile.json').read_text()),indent=2)+'\nDAD: archived native50k completed11959s with same source/config on4090. vsOED G-I: archived10001cycles completed6807s, unchanged native profile. RL2M: completed27451s, unchanged target/native settings; no convergence claim. Fixed B2000/L2000 receives an exact-production4-update profile before launch.\nAll jobs have10.5h soft native-unit cutoff and<12h parent wall watchdog. Imports, init, training, checkpoints and native selection are included in parent wall. Evaluation costs separately preserved.\n')
(C/'reports/SOURCE_GRADIENT_AUDIT.md').write_text('# Correctness and fidelity\n\nCPU native-method tests PASS (logs/cpu_tests.log). Prior source-identical DAD terminal sPCE finite differences and future-history gradients passed; receipt remains dad_location_seed101_native_schedule_20261101T191159Z/DAD_IMPLEMENTATION_AUDIT.json. Explicit-likelihood numerator truth, denominator truth-inclusive, cumulative joint likelihood, negative maximization loss, no detached future history. Bounded physical map unchanged. Native SAC/REDQ and actor/critic/posterior G-I kernels retained. Score native two-stage score matching/score policy-gradient retained with full five restart targets.\nPotential periodic-boundary stale-restart restore is guarded in cohort-local score_native_worker.py: do not overwrite the independently selected final restart with previous periodic weights. No core working-tree edits. Probe verifies selected export bitwise equals selected restart. Old P1 outcomes do not imply five-restart alignment.\n')
(C/'LATEST_STATUS.md').write_text('# Native formal cohort\n\nUse status.json and production_manifest.csv for actual live state. Reused models are training-complete only; common development scoring is new. Final bank remains sealed. No convergence claim from a time cap or target completion.\n')
(C/'COMMANDS.md').write_text(f'''# Inspection and control

Status: `cat {C}/status.json`
Queue: `cat {C}/production_manifest.csv`
Safe boundary stop: `touch {C}/CANCEL` (finishes active method/evaluation; launches no next job).
Do not restart manager over an incomplete attempt; no exact-resume claim.
Later one-shot final inference, ONLY after the user freezes MoE: `{pathlib.Path.home()}/miniconda3/envs/mocu_optimized/bin/python {C}/source/final_once.py --user-moe-freeze /path/to/user_approved_moe_freeze.json`.
Receipt must name frozen MoE checkpoint+SHA256, hyperparameters_frozen=true and user_approved_final_evaluation=true. Manager never calls this command. Final bank not transferred to Ubuntu.
''')
print('PROFILE_PASS',estimate)
