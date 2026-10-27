"""Current MoE-sBOED method registry and fail-closed project dispatch boundaries.
Historical classes remain importable for explicit archival inspection only.
"""
import argparse,json,sys,hashlib
from pathlib import Path
import yaml
BASELINES=('random','fixed','dad','rl_sboed','vsoed','scorebed')
MOE='moe_sboed_v1';ACTIVE=BASELINES+(MOE,)
LABELS=dict(random='Random',fixed='Optimized Fixed',dad='Corrected DAD',rl_sboed='Corrected RL-sBOED',vsoed='Corrected vsOED',scorebed='Corrected ScoreBED',moe_sboed_v1='MoE-sBOED-v1')
ALIASES={'MoE-sBOED':MOE,'MoE-sBOED-v1':MOE,'moe-sboed-v1':MOE}
LEGACY={'moe_sboed','DeepSeekMoE-sBOED-v1','moe','resmoe','switchmoe','switchresmoe','strategy_moe_sboed','Strategy-MoE','pimoe','pimoe_ic','sa_moe','ir_moe','rc_moe','fh_moe','sum_single','shared_only','routed_only','attention_dense_moe','attention_sparse_moe','existing_moe_hard','existing_moe_relaxed'}
LEGACY_FLAGS={'--dad-mlp-continue-from','--relaxed-routing','--variant','--family-evaluate-from','--moe-diagnostic-from','--base-checkpoint','--base-finetune','--residual-scale','--residual-initial-std','--ablations','--pimoe','--pimoe-ic','--pimoe-validation','--sa-moe','--fh-moe-v3-discovery'}
class LegacyExcluded(ValueError):pass

def resolve(method):
 m=ALIASES.get(method,method)
 if m in LEGACY or any(x in str(m).lower() for x in ['pimoe','resmoe','switchmoe','switchresmoe','fh-moe','fh_moe','ir-moe','ir_moe','rc-moe']):raise LegacyExcluded('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: '+str(method))
 if m not in ACTIVE:raise ValueError('Unregistered current method/version: '+str(method))
 return m

def label(method):return LABELS[resolve(method)]
def describe():
 return [dict(id=m,name=LABELS[m],family='conventional_baseline' if m in BASELINES else 'DeepSeekMoE',location='registered' if m in BASELINES else 'validated_v1',sir='corrected_baseline_protocol' if m in BASELINES else 'PENDING_INDEPENDENT_SIR_VALIDATION') for m in ACTIVE]

def value(argv,flag):
 for i,x in enumerate(argv):
  if x==flag:return argv[i+1] if i+1<len(argv) else None
  if x.startswith(flag+'='):return x.split('=',1)[1]
 return None

def guard(argv):
 for token in argv:
  if token.split('=',1)[0] in LEGACY_FLAGS:raise LegacyExcluded('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: '+token)
 cfgpath=value(argv,'--config');cfg={}
 if cfgpath and Path(cfgpath).is_file():
  cfg=yaml.safe_load(Path(cfgpath).read_text()) or {}
  if cfg.get('project_scope')=='LEGACY_EXCLUDED_FROM_CURRENT_PROJECT':raise LegacyExcluded('Historical configuration is archived, not an active project workflow: '+cfgpath)
 requested=value(argv,'--methods') or value(argv,'--method')
 if requested:
  for m in requested.split(','):
   if m!='all':resolve(m)
 else:
  for m in cfg.get('experiment',{}).get('methods',[]):resolve(m)
 # The immutable validated native v1 recipe is the only accepted flat MoE recipe.
 native='--moe-sboed-v1' in argv or '--deepseek-sboed' in argv
 if requested and any(resolve(m)==MOE for m in requested.split(',') if m!='all') and not native:
  raise ValueError('Use --moe-sboed-v1 with the validated native recipe; legacy generic dispatch is forbidden')
 if native:
  raw=cfg.get('benchmark',{});system=raw.get('system',{}).get('name')
  if system!='location_finding':raise ValueError('SIR MoE-sBOED-v1 is PENDING independent system validation; no legacy fallback')
  if cfg.get('method') not in ['DeepSeekMoE-sBOED','MoE-sBOED-v1']:raise ValueError('Use the verified v1 native recipe, not an exploratory/FH configuration')
  if hashlib.sha256(Path(cfgpath).read_bytes()).hexdigest()!='b2c920b9830babf3cec75a6fc6c0c305e5ae1a1104744d3c9f368101f360d8b0':raise ValueError('Native v1 recipe differs from its validated frozen configuration')
 return True

def assert_policy_checkpoint(saved,benchmark='location_finding'):
 if benchmark!='location_finding':raise ValueError('SIR MoE-sBOED-v1 is PENDING independent system validation')
 if saved.get('method')!='DeepSeekMoE-sBOED' or saved.get('variant')!='full' or saved.get('horizon')!=30:raise LegacyExcluded('Checkpoint is not the full validated location MoE-sBOED-v1; no alias fallback')
 if saved.get('source_commit')!='13681d7091189a4e68056e6a2864dc58f9d5700b':raise LegacyExcluded('Unregistered training source: an FH/RC/IR checkpoint is not v1 merely because its tensor shapes match')
 state=saved.get('state_dict',{})
 required={'router.weight':(7,16),'shared.up.weight':(8,16),'emitter.1.weight':(2,16),'encoder.0.weight':(256,3)}
 for k,shape in required.items():
  if k not in state or tuple(state[k].shape)!=shape:raise ValueError('Invalid v1 architecture at '+k)
 for e in range(7):
  for part,shape in [('up',(8,16)),('gate',(8,16)),('down',(16,8))]:
   if tuple(state[f'routed.{e}.{part}.weight'].shape)!=shape:raise ValueError('Invalid routed expert computation')
 if sum(v.numel() for v in state.values())!=8434:raise ValueError('Unexpected v1 parameter/state count')
 return True

def validate_current_benchmark(benchmark):
 from src.research.benchmarks import make_benchmark
 from src.research.moe_family import benchmark_signature
 recipe=Path(__file__).resolve().parents[2]/'configs/deepseek_sboed_location.yaml'
 reference=make_benchmark(yaml.safe_load(recipe.read_text())['benchmark'])
 if benchmark.name!='location_finding' or benchmark_signature(benchmark)!=benchmark_signature(reference):
  raise ValueError('MoE-sBOED-v1 benchmark physics/prior/likelihood/constraints differ from the validated reference')
 return True

def load_current_policy(path,benchmark='location_finding'):
 import torch
 if hashlib.sha256(Path(path).read_bytes()).hexdigest()!='f110766019e3628bb02c93815becd7e5795b3abc05626eabd91b7dbeb0381ee7':raise LegacyExcluded('Checkpoint digest is not registered MoE-sBOED-v1; no legacy substitution')
 saved=torch.load(path,map_location='cpu',weights_only=False);assert_policy_checkpoint(saved,benchmark)
 from src.research.deepseek_sboed import load_deepseek
 return load_deepseek(path,'cpu')

def historical_checkpoint_metadata(path):
 """Explicit read-only archive reader: never dispatches a policy or simulator."""
 import torch
 p=Path(path);x=torch.load(p,map_location='cpu',weights_only=False)
 return dict(status='LEGACY_EXCLUDED_FROM_CURRENT_PROJECT',path=str(p),keys=list(x),method=x.get('method',x.get('kind',x.get('model'))),architecture=x.get('architecture'),state_shapes={k:list(v.shape) for k,v in x.get('state_dict',{}).items() if hasattr(v,'shape')})

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['guard','list','inspect-archive']);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
 if a.action=='list':print(json.dumps(describe(),indent=2));return
 if a.action=='inspect-archive':print(json.dumps(historical_checkpoint_metadata(a.args[0]),indent=2));return
 try:guard(a.args[1:] if a.args and a.args[0]=='--' else a.args)
 except ValueError as e:print(str(e),file=sys.stderr);raise SystemExit(2)
if __name__=='__main__':main()
