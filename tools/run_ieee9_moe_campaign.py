"""Freeze and run the authorized matched IEEE9 MoE comparison on LabPC."""
from pathlib import Path
import argparse,csv,datetime,hashlib,json,os,shutil,subprocess,sys,traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CAMPAIGN=None
OPTIONS=None
BASELINES={3:'experiments/09142026/09142026_ieee9_eig_endpoint_rocof_T3_train101_eval1001/T3/run',
4:'experiments/09152026/09152026_endpoint_rocof_T4-5_train101_eval1001_rerun12h/T4/run',
5:'experiments/09152026/09152026_endpoint_rocof_T4-5_train101_eval1001_rerun12h/T5/run'}

def collect():
    scores={};paired=[];budgets=[]
    for horizon,baseline in BASELINES.items():
        run=CAMPAIGN/f'T{horizon}'/'run';old=ROOT/baseline
        assert (run/'completion.json').exists()
        new_cfg=json.loads((run/'run_config.json').read_text())
        old_cfg=json.loads((old/'run_config.json').read_text())
        assert new_cfg['physical_config']['swing_equation']==old_cfg['physical_config']['swing_equation']
        assert new_cfg['observation_kind']==old_cfg['observation_kind']=='endpoint_rocof'
        for key in ['T','seed','eval_seed','eval_systems','contrasts','noise_sigma','window','validation_systems','duration_min','duration_max','min_duration_separation','amplitude','bus','planner_particles']:
            assert new_cfg['settings'][key]==old_cfg['settings'][key],key
        for relative,digest in old_cfg['source_hashes'].items():
            if relative.startswith('src/domains/swing/'):
                assert new_cfg['source_hashes'][relative]==digest,relative
        assert (run/'exit_code').read_text().strip()=='0'
        rows=json.loads((run/'rollouts.json').read_text())
        original=json.loads((old/'rollouts.json').read_text())
        assert len(rows)==128 and all(r['method']=='moe_sboed' for r in rows)
        assert all(np.isfinite(r['terminal_spce_nats']) for r in rows)
        assert all(np.all(np.diff(r['duration_sequence_s'])>=.01) for r in rows)
        by_id={r['system']:r for r in rows}
        assert len(by_id)==128
        new_summary=json.loads((run/'summary.json').read_text())[0]
        assert abs(new_summary['mean_spce_nats']-np.mean([r['terminal_spce_nats'] for r in rows]))<1e-10
        diag=json.loads((run/'models/moe_sboed_training_diagnostics.json').read_text())
        budgets.append(f"T={horizon}: {diag.get('behavior_trajectories','unreported')} behavior trajectories; {diag.get('counterfactual_trajectories','unreported')} posterior-predictive trajectories; {diag.get('completed_updates','unreported')} completed updates.")
        (CAMPAIGN/f'T{horizon}'/'comparison_verification.json').write_text(json.dumps({
            'matched_physics_source':True,'matched_evaluation_settings':True,'training':diag},indent=2)+'\n')
        for row in original:assert row['true_MK']==by_id[row['system']]['true_MK']
        for row in json.loads((old/'summary.json').read_text())+json.loads((run/'summary.json').read_text()):
            scores.setdefault(row['method'],{})[horizon]=row['mean_spce_nats']
        for method in ['dad','rl_sboed','step_dad','myopic','fixed','random']:
            baseline_rows=sorted((r for r in original if r['method']==method),key=lambda r:r['system'])
            differences=np.array([by_id[r['system']]['terminal_spce_nats']-r['terminal_spce_nats'] for r in baseline_rows])
            paired.append({'T':horizon,'comparison':'moe_sboed - '+method,'mean_difference':float(differences.mean()),
                'paired_system_standard_error':float(differences.std(ddof=1)/np.sqrt(len(differences))),
                'note':'Across held-out systems for one trained policy and one evaluation seed; not training-seed uncertainty'})
    (CAMPAIGN/'paired_comparisons.json').write_text(json.dumps(paired,indent=2)+'\n')
    with (CAMPAIGN/'comparison.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['Method','T=3','T=4','T=5'])
        for name in ['moe_sboed','dad','rl_sboed','step_dad','myopic','fixed','random']:
            writer.writerow([name,*[scores[name][t] for t in (3,4,5)]])
    lines=['# IEEE9 signed endpoint RoCoF, 3.5 seconds','',
        'One training seed (101), one evaluation seed (1001), 128 test systems, 128 contrasts.',
        'Mean sPCE in nats; no across-seed SD is available. Training algorithms and compute costs differ.',
        ' '.join(budgets)+' Baseline training budgets are preserved in their source records. This is not a matched-compute comparison.','',
        '| Method | T=3 | T=4 | T=5 |','|---|---:|---:|---:|']
    for name in ['moe_sboed','dad','rl_sboed','step_dad','myopic','fixed','random']:
        lines.append('| '+name+' | '+' | '.join(f'{scores[name][t]:.4f}' for t in (3,4,5))+' |')
    (CAMPAIGN/'comparison.md').write_text('\n'.join(lines)+'\n')

def worker():
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1',BOED_EXECUTION_SITE='labpc')
    env['PATH']='/home/grads/g/g.lin/miniconda3/envs/mocu_optimized/bin:'+env['PATH']
    commands=json.loads((CAMPAIGN/'commands.json').read_text())
    complete=[]
    for horizon,command in commands.items():
        (CAMPAIGN/'status.json').write_text(json.dumps({'state':'running','horizon':int(horizon),'completed_horizons':complete,'pid':os.getpid()},indent=2)+'\n')
        with (CAMPAIGN/f'T{horizon}'/'run.log').open('w') as log:
            result=subprocess.run(command,cwd=CAMPAIGN/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            (CAMPAIGN/'status.json').write_text(json.dumps({'state':'failed','horizon':int(horizon),'exit_code':result.returncode,'completed_horizons':complete},indent=2)+'\n')
            return result.returncode
        complete.append(int(horizon))
    try:
        collect()
    except Exception:
        (CAMPAIGN/'status.json').write_text(json.dumps({'state':'comparison_failed','completed_horizons':complete,
            'traceback':traceback.format_exc()},indent=2)+'\n')
        raise
    (CAMPAIGN/'status.json').write_text(json.dumps({'state':'complete','completed_horizons':complete},indent=2)+'\n')
    return 0

def launch():
    campaign=CAMPAIGN
    campaign.mkdir(parents=True,exist_ok=False)
    source=campaign/'source';source.mkdir()
    for name in ['src','scripts','configs']:
        shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','AGENTS.md'))
    for name in ['run.sh','sweep_run.sh','requirements.txt']:
        shutil.copy2(ROOT/name,source/name)
    manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    (campaign/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (campaign/'training_recipe.json').write_text(json.dumps({
        'mode':OPTIONS.moe_training_mode,'updates':OPTIONS.updates,'batch_size':OPTIONS.batch_size,
        'learning_rate':OPTIONS.learning_rate,'validation_every':OPTIONS.validate_every,'moe_fantasies':4,
        'evaluation_selection':False,'planner_particles':128,'n_experts':2,
        'baseline_paths':BASELINES,'note':'New architecture, no teacher; comparison is matched evaluation, not matched compute.'},indent=2)+'\n')
    commands={}
    for horizon in (3,4,5):
        (campaign/f'T{horizon}').mkdir()
        commands[str(horizon)]=['bash','run.sh','--config','configs/ieee9_eig.yaml','--method','moe_sboed',
            '--T',str(horizon),'--observation-kind','endpoint_rocof','--window','3.5','--N_obs','0',
            '--noise_sigma','0.005','--seed','101','--eval-seeds','1001','--updates',str(OPTIONS.updates),'--batch-size',str(OPTIONS.batch_size),
            '--contrasts','128','--validation-systems','128','--validate-every',str(OPTIONS.validate_every),'--eval-systems','128',
            '--learning-rate',str(OPTIONS.learning_rate),'--moe-training-mode',OPTIONS.moe_training_mode,'--output',str(campaign/f'T{horizon}'/'run')]
    (campaign/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    shutil.copy2(__file__,campaign/'campaign_driver.py')
    # Execute the reusable tool from its original root; scientific modules load
    # exclusively from the frozen source through each root run.sh invocation.
    with (campaign/'driver.log').open('w') as log:
        process=subprocess.Popen([sys.executable,__file__,'--worker','--campaign',str(CAMPAIGN)],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
    (campaign/'launch.json').write_text(json.dumps({'pid':process.pid,'started':datetime.datetime.now().isoformat(),
        'expected_total_minutes':([240,480] if OPTIONS.moe_training_mode=='belief_specialist' else [90,150]),'hardware':'LabPC RTX4090','baseline_paths':BASELINES,
        'one_training_seed':101,'one_evaluation_seed':1001,'source_manifest_sha256':hashlib.sha256((campaign/'source_manifest.json').read_bytes()).hexdigest()},indent=2)+'\n')
    print('LAUNCHED',process.pid,campaign)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--campaign',required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--collect-only',action='store_true')
    parser.add_argument('--moe-training-mode',choices=['legacy','stable','belief_value','belief_specialist'],default='belief_value')
    parser.add_argument('--updates',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--validate-every',type=int,default=50)
    parser.add_argument('--learning-rate',type=float,default=.0003)
    OPTIONS=parser.parse_args()
    CAMPAIGN=(ROOT/OPTIONS.campaign).resolve()
    relative=CAMPAIGN.relative_to(ROOT/'experiments')
    if len(relative.parts)!=2 or len(relative.parts[0])!=8 or not relative.parts[1].startswith(relative.parts[0]+'_'):
        parser.error('Use experiments/MMDDYYYY/MMDDYYYY_campaign')
    if OPTIONS.worker:sys.exit(worker())
    if OPTIONS.collect_only:collect()
    else:launch()
