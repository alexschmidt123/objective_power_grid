"""Explicit Ubuntu-only V5 diagnostic entrypoint; no formal training action."""
import argparse, copy, hashlib, importlib.util, json, time
from dataclasses import asdict
from pathlib import Path
import torch,yaml
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.benchmarks import make_benchmark
from src.research.moe_v5_diagnostics import diagnose,Work
from src.research.moe_v5_credit import replicated_credit

ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def verified_reference(path):
    path=Path(path).resolve();manifest=json.loads((path/'remote_manifest.json').read_text())
    receipt=json.loads((path.parent/'v4_parity_receipt.json').read_text())
    if receipt['status']!='PASS' or not manifest['pre_post_transfer_equal']:raise ValueError('V4 parity/transfer gate failed')
    for row in manifest['remote_files']:
        if digest(row['local_destination'])!=row['sha256']:raise ValueError('Reference bytes changed')
    for name,h in manifest['extracted_allowlist'].items():
        if digest(path/'source'/name)!=h:raise ValueError('Reference source changed: '+name)
    for name in ['deepseek_sboed.py','dad_location_policy.py']:
        if digest(ROOT/'src/research'/name)!=manifest['extracted_allowlist']['src/research/'+name]:raise ValueError('V4 architecture differs')
    return path,receipt

def load_policy(reference,seed):
    folder=reference/f'formal/seed{seed}';config=json.loads((folder/'configuration.json').read_text())
    checkpoint=next(folder.glob('*.pt'));saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['configuration_sha256']!=digest(folder/'configuration.json'):raise ValueError('Configuration lineage mismatch')
    p=DeepSeekPolicy().eval();p.load_state_dict(saved['state_dict'],strict=True)
    if p.counts()['total_parameters']!=8434 or p.top_k!=3 or len(p.routed)!=7 or p.shared is None:raise ValueError('Wrong V4 architecture')
    return p,config,checkpoint

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--seed',type=int,choices=[101,102],default=101)
    parser.add_argument('--action',choices=['diagnostics','credit-smoke'],default='diagnostics')
    args=parser.parse_args(argv);cfg=yaml.safe_load(Path(args.config).read_text());v=cfg['v5']
    if cfg['method']!='MoE-sBOED-v5' or v['mode'] not in {'diagnostics','conditional','expert_credit','combined'}:raise ValueError('Unknown version/mode')
    if v['conditional']['enabled'] or v['conditional']['alpha']!=0:raise ValueError('Conditional training remains disabled: no L2000 empirical saturation or validated location posterior')
    if v['expert_credit']['enabled']:raise ValueError('Formal V5 training is not authorized; bounded credit diagnostics only')
    if v['mode']=='combined':raise ValueError('Combined mode blocked until single-change scientific gates pass')
    caps=cfg['smoke'];batch=int(caps['trajectories']);contrasts=int(caps['contrasts'])
    if batch>16 or contrasts>32 or caps['optimizer_updates']!=0:raise ValueError('Tiny diagnostic cap exceeded')
    out=Path(args.output).resolve();out.relative_to(ROOT/'scratch_runs')
    if not out.relative_to(ROOT/'scratch_runs').parts[0][:8].isdigit():raise ValueError('Use timestamped scratch campaign')
    ref,receipt=verified_reference(ROOT/cfg['reference_bundle'])
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);started=time.monotonic()
    p,physical,checkpoint=load_policy(ref,args.seed)
    with torch.device('cpu'):
        benchmark=make_benchmark(physical['benchmark'])
        if args.action=='diagnostics':result=diagnose(benchmark,p,batch,contrasts,596000+args.seed)
        else:
            work=Work();record=replicated_credit(benchmark,p,597000+args.seed,batch=1,negatives=contrasts,repeats=4,work=work)
            a=record['advantages'];result=dict(status='PASS',paired_advantages=a.tolist(),mean=float(a.mean()),mcse=float(a.std(unbiased=True)/a.numel()**.5),
                same_prefix=record['same_prefix'],work=asdict(work),not_a_performance_or_specialization_result=True,
                interpretation=record['interpretation'])
    dump(out/'diagnostics.json',result)
    dump(out/'run_manifest.json',dict(method='MoE-sBOED-v5',status='DEVELOPMENT_UNVALIDATED',configuration=cfg,
        reference_checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),reference_parity_status=receipt['status'],
        source_sha256={str(f.relative_to(ROOT)):digest(f) for f in (ROOT/'src/research').glob('moe_v5_*.py')},
        seed=args.seed,wall_seconds=time.monotonic()-started,charged_gpu_training_seconds=0,optimizer_updates=0,
        benchmark='location_finding_T30',conditional_training_enabled=False,credit_training_enabled=False,
        checkpoint_selection='none: diagnostics do not select or alter a checkpoint'))
    print(json.dumps(dict(status='COMPLETE_DIAGNOSTICS_ONLY',output=str(out),work=result['work'])),flush=True)

if __name__=='__main__':main()
