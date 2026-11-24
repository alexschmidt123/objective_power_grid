"""CPU-only independent verification and scoped publication of completed V4 results."""
from pathlib import Path
import csv,hashlib,json,math,subprocess
import numpy as np
from scipy.stats import t
M=Path.home()/'Documents/objective_power_grid'
R=M/'reports/moe_sboed_v4_allmethods_20261117'
O=M/'experiments/location_finding/moe_sboed_v4_allmethods_20261117'
B='research/hec-v3-routing-validation-20261115'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
    return h.hexdigest()
def put(name,value):(R/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
    assert read(O/'STATUS.json')['status']=='COMPLETE'
    d=read(R/'RESULTS.json');assert d['status']=='COMPLETE' and len(d['records'])==50
    protocol=read(R/'PROTOCOL.json');assert d['protocol']==protocol['evaluation']
    for path,value in protocol['source_sha256'].items():assert sha(M/path)==value,path
    vectors={};checkpoints=set()
    for row in d['records']:
        v=np.load(row['scores']);assert v.shape==(2000,) and np.isfinite(v).all()
        assert sha(row['scores'])==row['scores_sha256']
        assert sha(row['checkpoint'])==row['checkpoint_sha256'];checkpoints.add(row['checkpoint'])
        assert abs(float(v.mean())-row['EIG'])<1e-12
        assert abs(float(v.std(ddof=1)/math.sqrt(2000))-row['MCSE'])<1e-12
        vectors[row['method'],row['seed']]=v
    for name,c in d['paired_seed_comparisons'].items():
        delta=np.array([(vectors['moe-sboed',s]-vectors[name,s]).mean() for s in range(101,106)])
        se=float(delta.std(ddof=1)/math.sqrt(5));mu=float(delta.mean());width=float(t.ppf(.975,4))*se
        assert abs(mu-c['mean_delta'])<1e-12
        np.testing.assert_allclose(c['CI95'],[mu-width,mu+width],rtol=0,atol=1e-12)
    assert len(checkpoints)==30
    bank_uncertainty={}
    for name in d['paired_seed_comparisons']:
        per_case=np.mean(np.stack([vectors['moe-sboed',seed]-vectors[name,seed] for seed in range(101,106)]),axis=0)
        se=float(per_case.std(ddof=1)/math.sqrt(2000));mu=float(per_case.mean());width=float(t.ppf(.975,1999))*se
        bank_uncertainty[name]=dict(mean_delta=mu,MCSE_of_seed_average=se,case_CI95=[mu-width,mu+width],
          scope='2000 common cases averaged across five fixed checkpoints;not 10000 independent cases;separate from training-seed variability and contrast-bank bias')
    put('BANK_CONDITIONAL_MC_UNCERTAINTY.json',bank_uncertainty)
    for seed in range(101,106):assert read(O/f'seed{seed}/STANDALONE_AUDIT.json')['status']=='PASS'
    training=read(M/'reports/moe_sboed_v4_replication_20261117/COMPUTE.json')
    costrows=[]
    for row in d['records']:
        name=row['method'];seed=row['seed'];root=O/f'seed{seed}'
        if name.startswith('moe-sboed'):
            inherited=next(r for r in training if r['seed']==seed and r['method']==('MoE-sBOED-V4-uniform' if name=='moe-sboed-uniform' else 'MoE-sBOED-V4'))
            extracted=name.startswith('moe-sboed-e')
            cost=dict(method=name,seed=seed,new_training_seconds=0 if extracted else inherited['training_seconds'],
              inherited_parent_training=inherited,expert_training_scope='Inherited matching learned checkpoint;not additional training' if extracted else 'Continuation from shared frozen V2 parent',
              evaluation_seconds=row['evaluation_seconds'],online_amortized_seconds_per_episode=row['online_amortized_seconds_per_episode'],
              canonical_evaluation_worker_resources=read(root/'RESOURCES.json'),
              memory_scope='Worker retains full learned/uniform models;memory is not isolated standalone-policy footprint')
        else:
            old=read(row['result'])
            cost=dict(method=name,seed=seed,new_training_seconds=row['training']['offline_seconds'],training_receipt=row['training'],
              evaluation_seconds=row['evaluation_seconds'],online_amortized_seconds_per_episode=row['online_amortized_seconds_per_episode'],
              original_evaluation_resources={k:v for k,v in old.items() if k.startswith(('cpu_','torch_','gpu_'))},
              missing_metrics='Historical training CPU/GPU utilization unavailable unless present in original training receipts')
        costrows.append(cost)
    put('COMPUTE_FAIRNESS.json',costrows)
    overhead=[]
    for seed in range(101,106):
        resources=read(O/f'seed{seed}/RESOURCES.json')
        measured=sum(r['evaluation_seconds'] for r in costrows if r['seed']==seed and r['method'].startswith('moe-sboed'))
        overhead.append(dict(seed=seed,worker_wall_seconds=resources['wall_seconds'],summed_policy_evaluation_seconds=measured,
          measured_non_evaluation_worker_seconds=resources['wall_seconds']-measured,
          scope='Includes model loading,audit,report construction and hashing within worker resource boundary;Python import/setup before boundary is unavailable'))
    failure=O/'seed101_attempt001_device_load_failure/RESOURCES.json'
    put('EXECUTION_OVERHEAD.json',dict(successful_workers=overhead,
      failed_device_load_attempt_resources=read(failure) if failure.exists() else None,
      CPU_smoke_wall_seconds=None,CPU_smoke_timing_status='Not separately measured;unavailable,not zero',
      inherited_parent_cost_scope='Original V2 pretraining/providers are shared and must be counted separately from V4 continuation;do not sum duplicated inherited receipts once per extracted expert'))
    with (R/'COMPUTE_FAIRNESS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Method','Seed','Additional_training_seconds_for_this_policy','Evaluation_seconds','Batch16_amortized_episode_seconds','Training_scope'])
        for r in costrows:w.writerow([r['method'],r['seed'],r['new_training_seconds'],r['evaluation_seconds'],r['online_amortized_seconds_per_episode'],r.get('expert_training_scope','Reused native external training')])
    put('INDEPENDENT_VERIFICATION.json',dict(status='PASS',rows=50,unique_checkpoints=30,training_updates_this_campaign=0,
      five_seed_pairs_verified=9,checkpoint_and_score_hashes_verified=True,canonical_protocol=d['protocol'],all_five_standalone_audits='PASS',
      scope='CPU-only independent recomputation of score means,MCSE and paired seed-level t intervals;not independent simulator reproduction'))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for filename,names,title in [('learned_vs_uniform.png',['moe-sboed','moe-sboed-uniform'],'Learned versus independently trained uniform'),
      ('expert_comparison.png',['moe-sboed',*[f'moe-sboed-e{k}' for k in range(1,5)]],'Full V4 versus extracted specialist paths')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for i,name in enumerate(names):
            values=[vectors[name,s].mean() for s in range(101,106)]
            ax.errorbar(i,np.mean(values),yerr=np.std(values,ddof=1),fmt='o',capsize=5)
            ax.scatter(i+np.linspace(-.07,.07,5),values,s=20)
        ax.set_xticks(range(len(names)),names,rotation=20,ha='right');ax.set_ylabel('EIG (nats); error bars: seed SD');ax.set_title(title)
        fig.tight_layout();fig.savefig(R/filename,dpi=180);plt.close(fig)
    (R/'REPRODUCIBILITY.md').write_text('''# Reproducing the V4 all-method comparison

V4 is the current moe-sboed. All ten method rows use the original canonical bank and frozen scorer, T=30, 2,000 cases and 1M contrasts. Original V4 pilot and replication banks remain separate. See PROTOCOL.json for exact hashes and the historical zero-error reproduction certificate.

The terminal V4 checkpoints are frozen. No training, tuning, reinitialization or checkpoint selection occurs. E1-E4 are executable single-specialist policies extracted from the corresponding learned checkpoint, with inherited common query provider, action head and weak correction. Neither the router nor the other specialists is evaluated. The decoder was trained on mixtures, so these are extracted-policy diagnostics, not independently optimized standalone methods.

CPU smoke: `CUDA_VISIBLE_DEVICES= python tools/moe_sboed_v4_allmethods_20261117.py --task smoke`.
GPU evaluation: `python tools/moe_sboed_v4_allmethods_20261117.py --task controller`, with existing resource locks. Completed jobs are skipped; preserved partial jobs require review. Use the original conda dellma environment. To regenerate tables without GPU evaluation use `--task report`. For independent CPU verification and scoped publication use `python tools/moe_sboed_v4_allmethods_publish.py`.

Raw score vectors, extracted state dictionaries, runtime logs and resource samples remain outside Git under experiments/location_finding/moe_sboed_v4_allmethods_20261117. CHECKPOINT_MANIFEST.json records original terminal checkpoint hashes and exact per-case score paths. External baselines reuse the verified historical native checkpoints and canonical score vectors.

Timing is CUDA-synchronized forward time including shared belief/model queries, at batch size 16. Amortized design and episode times are not single-case latency. Scoring and complete evaluation wall time are separate. GPU/CPU resource samples cover each six-policy worker; peak memory includes resident learned/uniform models. Historical baseline training resource measurements are not invented. COMPUTE_FAIRNESS.json preserves complete available training receipts and original V4 initialization/pretraining lineage. Extracted training cost is inherited, not zero total cost.

Five-seed t intervals use five paired seed differences. Case MCSE is separate and conditional on the common contrast bank. Multiple primary comparisons include Holm p values and simultaneous Bonferroni intervals. The strongest extracted expert must not be reported using an unadjusted selected-winner test. Shared pretraining and previous use of the canonical bank make this an exploratory continuation-seed comparison, not independent from-scratch or untouched-bank confirmation. Native baseline budgets differ; no perfect compute-matching claim.
''')
    assert subprocess.check_output(['git','branch','--show-current'],cwd=M,text=True).strip()==B
    files=['tools/moe_sboed_v4_allmethods_publish.py']+[str(x.relative_to(M)) for x in R.iterdir() if x.suffix in {'.md','.json','.csv','.png'}]
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).splitlines())
    assert staged.issubset(set(files)),'Preserve unrelated staged work'
    subprocess.run(['git','diff','--check','--',*files],cwd=M,check=True)
    figures=[f for f in files if f.endswith('.png')]
    assert all((M/f).stat().st_size<2000000 for f in figures)
    subprocess.run(['git','add','--',*[f for f in files if f not in figures]],cwd=M,check=True)
    subprocess.run(['git','add','--force','--',*figures],cwd=M,check=True)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).strip():
        subprocess.run(['git','diff','--cached','--stat'],cwd=M,check=True)
        subprocess.run(['git','commit','-m','Report complete V4 five-seed canonical all-method comparison'],cwd=M,check=True)
    subprocess.run(['git','push','origin',B],cwd=M,check=True)
    print(subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip())
if __name__=='__main__':main()
