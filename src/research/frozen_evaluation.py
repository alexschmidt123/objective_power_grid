"""Evaluation-only native checkpoint reuse via the canonical experiment launcher.

No trainer, optimizer or new estimator. Confirmation requires a sealed campaign.
"""
import hashlib,json,time,sys
from pathlib import Path
import numpy as np
import torch

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def check_gate(path,checkpoint,method,train_seed,source_manifest):
    gate=read(path)
    if gate.get('status')!='frozen_ready_for_final_evaluation':raise ValueError('Confirmation gate not frozen')
    for name,expected in gate['sealed_files'].items():
        if digest(name)!=expected:raise ValueError('Sealed file changed: '+name)
    if digest(source_manifest)!=gate['source_manifest_sha256']:raise ValueError('Confirmation source manifest mismatch')
    record=next((e for e in gate['policies'] if e['method']==method and e['seed']==train_seed),None)
    if record is None:raise ValueError('Policy not registered for confirmation')
    if method!='random' and (str(Path(checkpoint).resolve())!=record['checkpoint'] or digest(checkpoint)!=record['checkpoint_sha256']):raise ValueError('Confirmation checkpoint identity mismatch')
    return gate

def main(args,raw,root):
    from src.research.runner import DEFAULTS,evaluate,write_json,freeze_source,Tee
    from src.research.benchmarks import make_benchmark
    from src.research.moe_family import benchmark_signature
    from src.research.journal_checkpoints import load_checkpoint
    source=Path(args.frozen_evaluate_from).resolve();cfg=read(source/'run_config.json')
    methods=[args.variant] if args.variant else (args.methods or '').split(',')
    if len(methods)!=1 or not methods[0]:raise ValueError('Evaluate exactly one explicit method')
    method=methods[0];native=args.frozen_native_method or method
    seeds=[int(x) for x in (args.eval_seeds or '').split(',') if x]
    if not seeds:raise ValueError('Explicit evaluation seeds required')
    if args.seed!=cfg['train_seed']:raise ValueError('Training seed provenance mismatch')
    if args.smoke or args.training_episodes or args.base_finetune:raise ValueError('Frozen evaluation cannot train')
    h=args.T or int(raw['experiment']['step_number'])
    if cfg['horizon']!=h:raise ValueError('Horizon mismatch')
    b=make_benchmark(raw)
    if benchmark_signature(b)!=benchmark_signature(make_benchmark(cfg['benchmark'])):raise ValueError('Benchmark mismatch')
    cp=source/'models'/(native+'.pt')
    settings=DEFAULTS|raw['research'].get('settings',{})
    for key in DEFAULTS:
        value=getattr(args,key,None)
        if value is not None:settings[key]=value
    torch.set_num_threads(int(settings.get('threads',4)))
    if 8831001 in seeds:
        if args.fixed_expert is not None or args.relaxed_routing:raise ValueError('No diagnostic intervention on reserved final data')
        if not args.confirmation_gate or not args.source_check_manifest:raise ValueError('Final seed requires frozen gate and source manifest')
        check_gate(args.confirmation_gate,cp,method,args.seed,args.source_check_manifest)
        if (settings['eval_systems'],settings['eval_contrasts'],h)!=(2000,1000000,30):raise ValueError('Final location resolution mismatch')
    p=None if method=='random' else load_checkpoint(cp,b)
    if hasattr(p,'set_routing'):p.set_routing('learned_switch',args.relaxed_routing)
    hook=None
    if args.fixed_expert is not None:
        if method!='resmoe' or not 0<=args.fixed_expert<4:raise ValueError('Fixed residual expert diagnostic requires resmoe')
        def force(module,inputs,output):
            weights,selectors,bits=output
            hard=torch.zeros_like(weights);hard[:,args.fixed_expert]=1.
            return hard,selectors,bits
        hook=p.router.register_forward_hook(force)
    out=Path(args.output).resolve();rel=out.relative_to(root/'experiments')
    if len(rel.parts)<2 or len(rel.parts[0])!=8 or not rel.parts[1].startswith(rel.parts[0]+'_'):raise ValueError('Use dated campaign')
    out.mkdir(parents=True,exist_ok=False)
    for d in ['models','logs','eval']:(out/d).mkdir()
    handle=(out/'logs/run_log.log').open('w');sys.stdout=Tee(sys.stdout,handle);sys.stderr=Tee(sys.stderr,handle)
    metadata=dict(evaluation_only=True,benchmark=raw,horizon=h,train_seed=args.seed,methods=methods,eval_seeds=seeds,source_run=str(source),source_checkpoint=None if p is None else str(cp),source_checkpoint_sha256=None if p is None else digest(cp),source_training_configuration_sha256=digest(source/'run_config.json'),settings=settings,generated_training_episodes=0,not_an_independent_training=True,fixed_expert=args.fixed_expert,final_confirmation=8831001 in seeds,confirmation_gate=args.confirmation_gate)
    write_json(out/'run_config.json',metadata);freeze_source(root,out)
    start=time.monotonic();summaries=[]
    try:
        for seed in seeds:
            rows,summary=evaluate(b,p,method,h,settings,seed,dict(training_seconds=0.),out,paper=True,infer_parameters=False)
            write_json(out/'eval'/f'{method}_seed{seed}.json',rows);np.save(out/'eval'/f'{method}_seed{seed}_scores.npy',np.asarray([r['spce'] for r in rows]));summaries.append(summary)
        write_json(out/'summary.json',summaries);write_json(out/'completion.json',dict(status='complete',evaluation_only=True,generated_training_episodes=0,evaluation_seconds=time.monotonic()-start,checkpoint_sha256=metadata['source_checkpoint_sha256']))
    finally:
        if hook is not None:hook.remove()
    return out
