from pathlib import Path
import ast,json,yaml,hashlib
from tools.sir_journal_report import stats
from src.research import project_scope
R=Path(__file__).resolve().parents[2]
def test_protocol_exact_scope_native_profiles_and_budget():
 c=yaml.safe_load((R/'configs/sir_journal_corrected.yaml').read_text());p=c['research']['journal_protocol']
 assert tuple(c['experiment']['methods'])==project_scope.BASELINES
 assert p['stopping_rule']['minimum_episodes']==p['stopping_rule']['maximum_episodes']==1000000
 assert p['checkpoint_selection_mode']=='final_fixed_budget' and 'compute_budget' not in p
 assert p['method_settings']['dad']['batch_size']==p['method_settings']['dad']['contrasts']==2000
 assert p['method_settings']['rl_sboed']['rl_native_contrasts']==100000
 assert p['method_settings']['scorebed']['scorebed_pretraining_episodes']==897736
 assert p['native_vsoed_profile']['initial_replay']+p['native_vsoed_profile']['updates']*p['native_vsoed_profile']['new_trajectories']==1000000
 assert p['confirmation_seed'] not in [1001,1002,1003,97402611,97402621,97402622]

def test_statistics_and_identifier_no_legacy_alias():
 import numpy as np
 x=np.array([1.,2.,3.,4.]);m,se,lo,hi=stats(x)
 assert m==2.5 and se==np.std(x,ddof=1)/2 and lo<m<hi
 for alias in ['moe_sboed','DeepSeekMoE-sBOED-v1','fh_moe_v3']:
  import pytest
  with pytest.raises(ValueError):project_scope.resolve(alias)
 assert project_scope.resolve('moe_sboed_v1')=='moe_sboed_v1'

def test_train_only_dispatch_and_source_compile():
 source=(R/'src/research/journal_run.py').read_text();assert "if getattr(args,'train_only',False):" in source
 assert "if not getattr(args,'train_only',False):plot_results" in source
 for p in [R/'tools/sir_journal_campaign.py',R/'tools/sir_journal_report.py',R/'src/research/runner.py',R/'src/research/journal_run.py']:ast.parse(p.read_text())
