"""One shared 500k base + two 500k residual learners; root launchers only."""
import argparse,hashlib,json,math,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'experiments/10152026/10152026_location_residual_1m_total'
VARIANTS=('resmoe','switchresmoe')
NAMES={'random':'Random','fixed':'Optimized Fixed','dad':'DAD','rl_sboed':'RL-sBOED','vsoed':'vsOED','scorebed':'ScoreBED','moe':'MoE-sBOED','resmoe':'ResMoE-sBOED','switchmoe':'SwitchMoE-sBOED','switchresmoe':'SwitchResMoE-sBOED'}
ORDER=('random','fixed','dad','rl_sboed','vsoed','scorebed','moe','resmoe','switchmoe','switchresmoe')

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x);m=float(x.mean());se=float(x.std(ddof=1)/math.sqrt(len(x)))
    return dict(mean=m,mc_standard_error=se,conditional_95_interval=[m-1.96*se,m+1.96*se])
def identity(rows):return [[r['truth'],r['measurement_noise_standard']] for r in rows]

def prepare():
    assert not C.exists(),'Do not overwrite a campaign'
    test=Path('/tmp/residual_1m_preflight.log').read_text();assert '42 passed' in test and 'FAILED ' not in test
    C.mkdir(parents=True);(C/'configs').mkdir();(C/'audit').mkdir()
    shutil.copy('/tmp/residual_1m_preflight.log',C/'audit/preflight.log')
    raw=yaml.safe_load((ROOT/'configs/location_finding_eig.yaml').read_text())
    p=raw['research']['journal_protocol']
    p['status']='fixed_budget_development_predeclared'
    p['comparison']='one_million_total_primary_training_episodes'
    p['checkpoint_selection_mode']='final_fixed_budget'
    p['checkpoint_selection']='final_predeclared_fixed_budget'
    p['claim_limit']='Development only; historical final-checkpoint references; no across-seed superiority claim'
    p['development_training_seeds']=[101]
    p['validation'].update(seed=771001,systems=512,contrasts=1000000,batch_size=16,chunk=16384)
    p['stopping_rule'].update(minimum_episodes=500000,maximum_episodes=500000,validation_every_episodes=125000,maximum_training_seconds=7200)
    p['method_stopping']={}
    p['common_pathwise_settings']['router_schedule_episodes']=500000
    raw['research']['paper_protocol'].update(nominal_training_trajectories=500000,regime='half_million_phase_with_one_million_total_model',checkpoint_selection='final',training_budget_unit='primary_episodes')
    raw['research']['settings']['strategy_diagnostics_every']=1
    raw['experiment']['methods']=['resmoe','switchresmoe']
    config=C/'configs/location_500k_phase.yaml';config.write_text(yaml.safe_dump(raw,sort_keys=False))
    old=read(ROOT/'experiments/10152026/10152026_journal_location_sir/audit/historical_development_reference.json')
    refs={}
    for item in old['items']:
        method=item['method']
        if method=='moe_sboed':method='moe'
        if method=='strategy_moe_sboed':method='switchmoe'
        if method in ORDER and method not in VARIANTS:
            refs[method]=item
    assert set(refs)==set(ORDER)-set(VARIANTS),set(refs)
    write(C/'historical_references.json',refs)
    historical={}
    for item in old['items']:
        if item['method'] in VARIANTS:
            path=Path(item['run'])/'models'/(item['method']+'.pt')
            historical[str(path)]=digest(path)
    write(C/'audit/historical_checkpoints.json',historical)
    source={str(f.relative_to(ROOT)):digest(f) for f in sorted((ROOT/'src').rglob('*.py'))}
    source[str(config.relative_to(ROOT))]=digest(config)
    write(C/'audit/source_guard.json',dict(sha256=source))
    write(C/'protocol.json',dict(status='development_frozen_before_training',methods=list(ORDER),benchmarks=['location_finding','sir_ode'],current_execution='location_only',training_seed=101,development_evaluation_seed=1001,reserved_final_seed=8831001,reserved_final_seed_used=False,phase_episodes=500000,total_per_model=1000000,physical_training_episodes_for_shared_base_and_two_variants=1500000,training_contrasts=1024,batch_size=64,base='Existing compatible sum-encoder64/64tanh internal base, not a benchmark baseline or architecture ablation',base_freeze=True,selection='Final exact-budget checkpoint, matching historical fixed-budget runs; validation diagnostic only',validation=dict(p['validation']),router_schedule='Gamma1 through30percent of residual500k phase, cosine to0.001 at95percent; switching gate unchanged',train_seed_replication_candidates=[70101,70102,70103,70104,70105],replication='Not launched until development verification and explicit protocol seal; never use8831001 during development',historical_preservation=historical,minimum_required_tests=42))
    protocol_text='''# One-million-total residual MoE campaign

Exactly ten primary methods: Random, Optimized Fixed, DAD, RL-sBOED, vsOED, ScoreBED and the four canonical MoE variants. DAD-MLP is not a primary baseline or architecture ablation. Only the internal compatible base and the two residual variants are trained in this campaign.

Per seed: train the existing compatible sum-encoder base for exactly500000 primary episodes, freeze it and reuse the exact checkpoint for both residual variants; each then trains exactly500000 additional episodes. The full pipeline uses1500000 physical training episodes because base computation is shared; end-to-end attribution is1000000 episodes for each variant. L1024, batch64,64/64tanh residual experts, component clipping, optimizer, regularizers, likelihood, first action and feasibility map are unchanged. The30/95percent router schedule is relative to the500000-episode residual phase. Switching relaxation is unchanged.

Checkpoint selection is the final allocated-budget checkpoint, matching the historical fixed-budget study. Validation every125000 phase episodes uses512 fixed paired trajectories, L1000000, seed771001 for diagnostics only; there is no plateau early stop or best-validation selection. Train both phases fully, including a final partial batch32. Any safety interruption is incomplete, never a valid1M model. Base final weights, encoder and action transformation are reused intact. Evaluation uses2000 paired cases/L1000000, development1001. Validation and evaluation work is separately counted and never called training episodes.

Development seed101 comes first. Candidate independent replication seeds70101–70105 must be checked for prior use and the source/config protocol sealed after development verification, before launching confirmation. Reserved final evaluation8831001 remains untouched until confirmation. Historical1M references have limited seed evidence and final-checkpoint selection; optimization schedules differ and are disclosed rather than relabelled controlled mechanism ablations. Historical2M residual and5M vsOED results are excluded from the main ten-method table, preserved on disk. There is no DAD-MLP baseline or tanh-architecture ablation in the planned location/SIR journal study.
'''
    (C/'protocol.md').write_text(protocol_text)
    (ROOT/'docs/residual_1m_protocol.md').write_text(protocol_text)
    report()
    print('Prepared '+str(C),flush=True)

def check_guard():
    for rel,expected in read(C/'audit/source_guard.json')['sha256'].items():
        if digest(ROOT/rel)!=expected:raise RuntimeError('Source/config changed: '+rel)
    for path,expected in read(C/'audit/historical_checkpoints.json').items():
        assert digest(path)==expected,'Historical checkpoint changed'

def run_root(seed,method,folder,base=None):
    check_guard()
    cmd=['bash','run.sh','--config',str(C/'configs/location_500k_phase.yaml'),'--journal','--seed',str(seed),'--eval-seeds','1001','--eig-only','--source-check-manifest',str(C/'audit/source_guard.json'),'--output',str(folder)]
    cmd+=['--methods','sum_single'] if method=='base' else ['--variant',method,'--base-checkpoint',str(base)]
    write(folder.parent/(method+'_command.json'),dict(command=cmd))
    print('Starting',method,'seed',seed,flush=True)
    start=time.perf_counter();subprocess.run(cmd,cwd=ROOT,check=True)
    write(folder.parent/(method+'_walltime.json'),dict(seconds=time.perf_counter()-start))
    name='sum_single' if method=='base' else method
    t=read(folder/'models'/f'{name}_training.json')
    assert t['generated_training_trajectories']==500000
    assert t['checkpoint_selection']=='final_predeclared_fixed_budget'
    assert t['selected_checkpoint']['episodes']==500000
    assert t['convergence_status']=='fixed_budget_complete'
    if method!='base':
        assert t['total_training_episodes']==1000000
        assert t['base_checkpoint']['sha256']==digest(base)
    return t

def run_seed(seed):
    torch.set_num_threads(4)
    out=C/f'development_seed{seed}';out.mkdir(exist_ok=False)
    run_root(seed,'base',out/'base')
    base=out/'base/models/sum_single.pt';base_hash=digest(base)
    # Check the real freshly trained base before residual training.
    from src.research.benchmarks import make_benchmark
    from src.research.moe_family import BasePolicy,MoESBOEDPolicy
    from src.research.engine import rollout
    b=make_benchmark(yaml.safe_load((C/'configs/location_500k_phase.yaml').read_text()))
    for variant in VARIANTS:
        reference=BasePolicy.from_checkpoint(base,b,30).policy.eval()
        p=MoESBOEDPolicy(variant,b,30,base_checkpoint=base).eval()
        p.core.alpha=0.
        with torch.no_grad():
            a=rollout(b,reference,30,4,998101,8);r=rollout(b,p,30,4,998101,8)
        for x,y in zip(a['actions'],r['actions']):torch.testing.assert_close(x,y,rtol=0,atol=0)
        for x,y in zip(a['observations'],r['observations']):torch.testing.assert_close(x,y,rtol=0,atol=0)
    write(out/'base_equivalence.json',dict(passed=True,variants=list(VARIANTS),actions_and_observations_exact=True,base_sha256=base_hash))
    for variant in VARIANTS:
        run_root(seed,variant,out/variant,base)
        assert digest(base)==base_hash
        report()
    check_guard();write(out/'exit_code.json',dict(exit_code=0));report()

def training_diagnostics(path,budget=500000):
    groups={};mass={};updates=0
    for line in path.open():
        row=json.loads(line);updates+=1;phase=min(4,(row['primary_episodes']-1)*5//budget)
        mass.setdefault(phase,[]).append(row['router_probabilities'])
        for group,value in row['gradient_before'].items():
            g=groups.setdefault(group,dict(raw=[],post=[],late=[]));g['raw'].append(value);g['post'].append(row['gradient_after'][group])
            if row['primary_episodes']>budget*.8:g['late'].append(row['gradient_after'][group])
    result={}
    for key,g in groups.items():
        a=np.asarray(g['post']);late=np.asarray(g['late'])
        result[key]=dict(raw_quantiles_p50_p95_p99_max=np.quantile(g['raw'],[.5,.95,.99,1]).tolist(),fraction_updates_above_1e_8=float((a>1e-8).mean()),final20_fraction_above_1e_8=float((late>1e-8).mean()))
    return dict(updates=updates,gradient_exposure=result,router_mass_by_training_fifth={str(k):np.mean(v,axis=0).tolist() for k,v in mass.items()},gradient_threshold=1e-8)

def report():
    refs=read(C/'historical_references.json');items={};rows_by={};paired={};costs={};diagnostics={}
    for method,item in refs.items():
        items[method]=dict(method=method,name=NAMES[method],eig=item['eig'],status='historical_development',training_episodes=0 if method=='random' else 1000000,checkpoint_selection='historical_final',run=item['run'])
    seed=read(C/'protocol.json')['training_seed']
    seed_dir=C/f'development_seed{seed}'
    base_path=seed_dir/'base/models/sum_single_training.json';base=read(base_path) if base_path.exists() else None
    for variant in VARIANTS:
        folder=seed_dir/variant
        if not (folder/'exit_code').exists() or (folder/'exit_code').read_text().strip()!='0':
            items[variant]=dict(method=variant,name=NAMES[variant],eig=None,status='running_or_pending',training_episodes=1000000,checkpoint_selection='final_predeclared_fixed_budget');continue
        t=read(folder/'models'/f'{variant}_training.json');r=read(folder/'eval'/f'{variant}_seed1001.json')
        assert len(r)==2000 and t['total_training_episodes']==1000000
        items[variant]=dict(method=variant,name=NAMES[variant],eig=stats([x['spce'] for x in r]),status='new_1M_total_development',training_episodes=t['total_training_episodes'],checkpoint_selection=t['checkpoint_selection'],run=str(folder))
        rows_by[variant]=r
        from tools.report_moe_family import summarize_rows
        diagnostics[variant]=dict(hard_inference=summarize_rows(r),training=training_diagnostics(folder/'models'/f'{variant}_diagnostics.jsonl'))
        valid_stage_work=sum(z['validation']['forward_particle_stages'] for phase in [base,t] for z in phase['journal_validation_history'])
        costs[variant]=dict(base_training_episodes=500000,residual_training_episodes=500000,total_training_episodes=1000000,training_truth_trajectory_simulations=1000000,training_contrast_trajectory_simulations=1024000000,training_particle_stage_likelihood_evaluations=30750000000,training_truth_and_contrast_trajectories=1025000000,base_optimization_seconds=base['training_seconds'],residual_optimization_seconds=t['training_seconds'],total_optimization_seconds=base['training_seconds']+t['training_seconds'],validation_seconds=base['validation_seconds']+t['validation_seconds'],validation_forward_particle_stages=valid_stage_work,heldout_forward_particle_stages=read(folder/'summary.json')[0]['forward_particle_stages'],base_computation_shared=True,base_sha256=t['base_checkpoint']['sha256'])
        for method,item in refs.items():
            refpath=Path(item['run'])/'eval'/f"{item['method']}_seed1001.json"
            other=read(refpath)
            assert identity(r)==identity(other),'Evaluation cases are not paired'
            assert digest(folder/'eval/contrast_bank_seed1001.npy')==digest(Path(item['run'])/'eval/contrast_bank_seed1001.npy'),'Contrast bank differs'
            paired[variant+'_minus_'+method]=stats(np.asarray([x['spce'] for x in r])-np.asarray([x['spce'] for x in other]))
    if all(v in rows_by for v in VARIANTS):
        paired['switchresmoe_minus_resmoe']=stats(np.asarray([x['spce'] for x in rows_by['switchresmoe']])-np.asarray([x['spce'] for x in rows_by['resmoe']]))
    write(C/'report.json',dict(phase='development',training_seed=seed,evaluation_seed=1001,final_seed_used=False,methods=[items[k] for k in ORDER],paired_differences=paired,costs=costs,diagnostics=diagnostics,replicated_results_available=False,claim_limit='Trajectory MC intervals conditional on one training; historical optimizers/schedules are not newly controlled; no across-seed superiority claim'))
    md='# Ten-method location comparison:1M total training episodes\n\nDevelopment results only. No DAD-MLP baseline. No five-million vsOED or two-million residual result is included.\n\n| Method | Held-out EIG | Status |\n|---|---:|---|\n'
    for method in ORDER:
        item=items[method];value='Pending' if item['eig'] is None else f"{item['eig']['mean']:.5f}"
        md+=f"| {NAMES[method]} | {value} | {item['status']} |\n"
    md+='\nAll methods use common2000/L1million development evaluation1001. New residual methods include500k base+500k residual. Historical references remain single-seed final-checkpoint evidence. Across-seed final replicated results are not yet available. Machine-readable paired differences, simulation/likelihood costs and expert/switching diagnostics are in report.json.\n'
    (C/'report.md').write_text(md)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','report']);parser.add_argument('--seed',type=int,default=101);args=parser.parse_args()
    try:
        if args.action=='prepare':prepare()
        elif args.action=='run':run_seed(args.seed)
        else:report()
    except BaseException as exc:
        if args.action=='run':write(C/'pipeline_failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise
