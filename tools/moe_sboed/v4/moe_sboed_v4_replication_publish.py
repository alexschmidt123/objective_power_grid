"""Validate and publish ONLY completed/partial exploratory replication reports."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
M=Path.home()/'Documents/objective_power_grid'
O=M/'experiments/location_finding/moe_sboed_v4_replication_20261117'
R=M/'reports/moe_sboed_v4_replication_20261117'
BRANCH='research/hec-v3-routing-validation-20261115'


def read(p):return json.loads(p.read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()


def main():
    status=read(O/'STATUS.json')
    assert status['status'] in ['COMPLETE','FINISHED_WITH_FAILURES','TERMINATED_BY_USER']
    subprocess.run(['/home/grads/g/g.lin/miniconda3/envs/dellma/bin/python',str(M/'tools/moe_sboed_v4_replication_20261117.py'),'--task','report'],check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
    cohort=read(M/'configs/moe_sboed_v4_replication_20261117.json')
    for name,digest in cohort['frozen_source_sha256'].items():assert sha(M/name)==digest,name
    old=read(O/'PILOT_PRESERVATION.json')
    for name,digest in old['files'].items():assert sha(M/name)==digest,name
    result=read(R/'RESULTS.json');new_completed=0
    for seed in range(102,106):
        for arm in ['learned','uniform']:
            folder=O/f'seed{seed}'/arm
            if not (folder/'REPLICATION_VALIDATION.json').exists():continue
            d=read(folder/'RESULTS.json');init=read(folder/'INITIALIZATION.json')
            assert d['seed']==seed and init['seed']==seed and d['updates']==2000
            assert [v['updates'] for v in d['phases']]==[200,1800]
            assert [sum(v['counts'][k] for v in d['phases']) for k in ['actor','critic','posterior']]==[2000,10000,300000]
            assert d['new_training_episodes']==2000000 and d['total_seconds']<=14400
            assert d['evaluation']['bank_sha256']==read(O/'BANK_IDENTITY.json')['sha256']
            new_completed+=1
    if status['status']=='COMPLETE':assert new_completed==8 and result['completed_V4_evaluations']==10 and result['primary_seed_level']['n_training_seeds']==5
    receipt=dict(study_type=cohort['study_type'],controller_status=status['status'],new_completed_training_runs=new_completed,
        completed_V4_evaluations=result['completed_V4_evaluations'],seed101_retrained=False,
        original_pilot_artifact_hashes_verified=len(old['files']),architecture_sources_unchanged=True,
        reports_status=result['status'],failures=status.get('failed',[]))
    (R/'COMPLETION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert subprocess.check_output(['git','branch','--show-current'],cwd=M,text=True).strip()==BRANCH
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).strip(),'Preserve unrelated staged work'
    files=[str((R/n).relative_to(M)) for n in ['RESULTS.md','RESULTS.json','FIVE_SEED_RESULTS.csv','CHECKPOINT_MANIFEST.json','COMPUTE.json','DIAGNOSTICS.json','COMPLETION_RECEIPT.json']]
    subprocess.run(['git','diff','--check','--',*files],cwd=M,check=True)
    subprocess.run(['git','add','--',*files],cwd=M,check=True)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).strip():
        subprocess.run(['git','diff','--cached','--stat'],cwd=M,check=True)
        subprocess.run(['git','commit','-m','Report exploratory five-seed V4 routing replication: '+result['status']],cwd=M,check=True)
    subprocess.run(['git','push','origin','HEAD:'+BRANCH],cwd=M,check=True)
    print(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip(),**receipt)))


if __name__=='__main__':main()
