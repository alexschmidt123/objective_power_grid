import copy,hashlib,json,os,subprocess,sys
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import pytest,torch,yaml
from src.research import project_scope as scope
from src.research.deepseek_sboed import DeepSeekPolicy,load_deepseek
from src.research.benchmarks import LocationFinding
from tools.tests.test_deepseek_sboed import history
ROOT=Path(__file__).resolve().parents[2]

def test_registry_and_no_automatic_future_registration():
 assert scope.ACTIVE==('random','fixed','dad','rl_sboed','vsoed','scorebed','moe_sboed_v1')
 for s in ['MoE-sBOED','MoE-sBOED-v1','moe_sboed']:assert scope.resolve(s)==scope.MOE
 for s in ['MoE-sBOED-v2','MoE-sBOED-v3','MoE-sBOED-v4','FH-MoE-v3']:
  with pytest.raises(ValueError):scope.resolve(s)
 for s in scope.LEGACY:
  with pytest.raises(ValueError):scope.resolve(s)

def test_current_configs_and_frozen_native_recipe():
 for name in ['location','sir']:
  p=ROOT/f'configs/current_{name}.yaml';c=yaml.safe_load(p.read_text())
  assert tuple(c['experiment']['methods'])==scope.BASELINES
  assert tuple(c['current_method_registry'])==scope.ACTIVE
  def check_keys(x):
   if isinstance(x,dict):
    for k,v in x.items():
     assert k not in scope.LEGACY and not str(k).startswith(('moe_','strategy_','residual_'))
     check_keys(v)
   elif isinstance(x,list):
    for v in x:check_keys(v)
  check_keys(c['research'])
  assert scope.guard(['--config',str(p)])
 assert scope.guard(['--moe-sboed-v1','--config',str(ROOT/'configs/deepseek_sboed_location.yaml')])
 with pytest.raises(ValueError):scope.guard(['--moe-sboed-v1','--config',str(ROOT/'configs/current_sir.yaml')])

def test_no_alias_to_legacy_dispatch(tmp_path):
 for args in [['--methods','moe'],['--methods','resmoe'],['--methods','moe_sboed_v1'],['--variant','full'],['--base-checkpoint','x'],['--config',str(ROOT/'configs/e1_sir_native.yaml')]]:
  with pytest.raises(ValueError):scope.guard(args)
 cfg=yaml.safe_load((ROOT/'configs/deepseek_sboed_location.yaml').read_text());cfg['contrasts']=1;p=tmp_path/'changed.yaml';p.write_text(yaml.safe_dump(cfg))
 with pytest.raises(ValueError):scope.guard(['--moe-sboed-v1','--config',str(p)])

def test_checkpoint_rejects_same_shapes_but_unregistered_source():
 m=DeepSeekPolicy();x=dict(method='DeepSeekMoE-sBOED',variant='full',horizon=30,source_commit='13681d7091189a4e68056e6a2864dc58f9d5700b',state_dict=m.state_dict())
 assert scope.assert_policy_checkpoint(x)
 for key,val in [('variant','shared_only'),('source_commit','042915bfe8312faa333b1b825b94766c00c68dbf'),('method','moe_sboed_family_v1')]:
  y=dict(x);y[key]=val
  with pytest.raises(ValueError):scope.assert_policy_checkpoint(y)
 with pytest.raises(ValueError):scope.assert_policy_checkpoint(x,'sir_ode')

def test_unregistered_digest_never_loads(tmp_path):
 p=tmp_path/'legacy.pt';torch.save({'method':'DeepSeekMoE-sBOED'},p)
 with patch('src.research.deepseek_sboed.load_deepseek',side_effect=AssertionError('must not load')):
  with pytest.raises(ValueError):scope.load_current_policy(p)

def test_deployment_only_no_optimizer_simulator_likelihood_or_updates():
 torch.manual_seed(101);m=DeepSeekPolicy().eval();before={k:v.clone() for k,v in m.state_dict().items()};b=LocationFinding({'location_finding':{'design_bounds':[-4,4]}})
 hooks=[];calls=[0]*7
 for i,e in enumerate(m.routed):hooks.append(e.register_forward_hook(lambda a,b,c,i=i:calls.__setitem__(i,calls[i]+1)))
 with ExitStack() as stack:
  for target in ['torch.optim.Adam.step','torch.optim.SGD.step','torch.optim.LBFGS.step','torch.autograd.grad','torch.Tensor.backward','src.research.engine.rollout','src.research.journal_validation.CommonValidation.__call__','src.research.benchmarks.Benchmark.forward','src.research.benchmarks.Benchmark.log_likelihood','src.research.benchmarks.LocationFinding.propagate','scipy.optimize.minimize','scipy.optimize.differential_evolution']:
   stack.enter_context(patch(target,side_effect=AssertionError('forbidden deployment call '+target)))
  with torch.inference_mode():
   # A forward call at each possible deployment stage; no hidden-theta inputs.
   for stage in range(30):
    raw=m(history(stage,batch=1),stage);action=b.feasible(raw,[],30)
    assert action.shape==(1,2) and torch.isfinite(action).all() and (action.abs()<=4).all()
    assert m.last_dispatch['indices'].unique().numel()==3
  assert sum(calls)==30*3
 for h in hooks:h.remove()
 for k,v in m.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
 assert not m._gate_cache

def test_foundation_files_and_archival_checksums():
 files=['deepseek_sboed.py','deepseek_sboed_experiment.py','engine.py','dad_location_policy.py','benchmarks.py','journal_validation.py']
 for f in files:
  rel='src/research/'+f;old=subprocess.check_output(['git','show','13681d7091189a4e68056e6a2864dc58f9d5700b:'+rel],cwd=ROOT)
  assert (ROOT/rel).read_bytes()==old
 archive=ROOT/'historical/project_scope_20261026';manifest=json.loads((archive/'manifest.json').read_text())
 for rel,digest in manifest.items():assert hashlib.sha256((archive/rel).read_bytes()).hexdigest()==digest
 # Historical experiment-facing classes remain importable, without activation.
 from src.research.moe_family import load_policy
 assert callable(load_policy)

def test_archival_metadata_reader_does_not_deploy(tmp_path):
 p=tmp_path/'old.pt';torch.save({'kind':'moe_sboed_family_v1','state_dict':{'x':torch.ones(2)}},p)
 before=p.read_bytes();x=scope.historical_checkpoint_metadata(p)
 assert x['status']=='LEGACY_EXCLUDED_FROM_CURRENT_PROJECT' and x['state_shapes']=={'x':[2]} and p.read_bytes()==before

def test_real_registered_v1_checkpoint_compatible():
 p=os.environ.get('MOE_V1_CHECKPOINT')
 if not p:pytest.skip('set MOE_V1_CHECKPOINT to preserved official checkpoint')
 saved=torch.load(p,map_location='cpu',weights_only=False);assert scope.assert_policy_checkpoint(saved)
 m=scope.load_current_policy(p);original=load_deepseek(p)
 from src.research.journal_checkpoints import load_checkpoint
 q=load_checkpoint(p,LocationFinding({'location_finding':{'design_bounds':[-4,4]}}))
 with torch.inference_mode():
  for stage in [0,1,10,20,29]:
   f=history(stage,batch=4);torch.testing.assert_close(m(f,stage),original(f,stage),rtol=0,atol=0);torch.testing.assert_close(q(f,stage),original(f,stage),rtol=0,atol=0)


def test_benchmark_identity_does_not_accept_wrong_physics():
 raw=yaml.safe_load((ROOT/'configs/deepseek_sboed_location.yaml').read_text())['benchmark']
 from src.research.benchmarks import make_benchmark
 assert scope.validate_current_benchmark(make_benchmark(raw))
 for section,key,value in [('observation','noise_sigma',.8),('location_finding','design_bounds',[-5,5])]:
  changed=copy.deepcopy(raw);changed[section][key]=value
  with pytest.raises(ValueError):scope.validate_current_benchmark(make_benchmark(changed))

def test_general_registry_and_plot_do_not_relabel_legacy():
 from src.context import normalize_method_key,ALL_METHOD_KEYS,TRAINABLE_METHOD_KEYS
 assert 'moe_sboed' not in ALL_METHOD_KEYS and 'moe_sboed' not in TRAINABLE_METHOD_KEYS
 for s in ['dad','rl_sboed','random','fixed']:assert normalize_method_key(s)==s
 for s in ['moe_sboed','MoE-sBOED','resmoe','switchmoe']:
  with pytest.raises(ValueError):normalize_method_key(s)
 from src.results.plots import _poster_label
 assert _poster_label('moe_sboed') is None and _poster_label('ResMoE-sBOED') is None
 assert _poster_label('moe_sboed_v1')=='MoE-sBOED-v1'

def test_legacy_launchers_fail_before_work(tmp_path):
 checks=[['bash','run.sh','--methods','moe'],[sys.executable,'-m','src.research.runner','--methods','resmoe'],[sys.executable,'-m','tools.e1_preserve_moe_queue','--host','ubuntu-pc'],[sys.executable,'-m','tools.residual_1m_campaign_v2']]
 for args in checks:
  p=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  assert p.returncode!=0 and ('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT' in p.stdout+p.stderr)
 assert list(tmp_path.iterdir())==[]


def test_shell_helpers_preserve_conventional_all_selection():
 assert scope.guard(['--method','all'])
 # Source helpers with unrelated caller arguments: no dispatcher may execute.
 p=subprocess.run(['bash','-c','set -- --method all; source ./run.sh; test -n "$DEFAULT_STEP_NUMBER"'],cwd=ROOT,capture_output=True,text=True)
 assert p.returncode==0,p.stdout+p.stderr
