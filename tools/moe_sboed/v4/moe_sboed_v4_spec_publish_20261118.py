"""CPU-only V4-Spec completion verification,reporting and scoped Git publication."""
from pathlib import Path
import hashlib,json,math,subprocess,sys
import numpy as np
from scipy.stats import t
M=Path.home()/'Documents/objective_power_grid';O=M/'experiments/location_finding/moe_sboed_v4_spec_20261118';R=M/'reports/moe_sboed_v4_spec_20261118'
BRANCH='research/hec-v3-routing-validation-20261115'
def read(p):return json.loads(Path(p).read_text())
def put(name,value):(R/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):h.update(b)
    return h.hexdigest()
def publish():
    status=read(O/'STATUS.json');assert status['status'] in ['COMPLETE','FAILED_PRESERVED']
    subprocess.run([sys.executable,'-B',str(M/'tools/moe_sboed_v4_spec_20261118.py'),'--task','report'],cwd=M,check=True)
    config=read(M/'configs/moe_sboed_v4_spec_20261118.json')
    for path,value in config['source_sha256'].items():assert sha(M/path)==value,path
    d=read(R/'RESULTS.json');costs=[];manifest=[];physical=dict(actor_updates=0,critic_updates=0,posterior_updates=0,new_episodes=0)
    warm_path=O/'shared/RESULTS.json';warm=read(warm_path) if warm_path.exists() else None
    if warm:
        assert sha(O/'shared/runtime.pkl')==warm['runtime_sha256'];assert warm['phase']['counts']==dict(actor=200,critic=1000,posterior=30000)
        physical.update(actor_updates=200,critic_updates=1000,posterior_updates=30000,new_episodes=200000)
        exposure=warm['exposure'];stage=np.asarray(exposure['stage_selected'],dtype=np.float64);total=np.asarray(exposure['stage_total'],dtype=np.float64)
        assert total.sum()==60000000
        put('EXPERT_EXPOSURE.json',dict(stage_total=total.tolist(),stage_selected=stage.tolist(),expert_selected=stage.sum(0).tolist(),
          frequency=(stage.sum(0)/total.sum()).tolist(),physical_warm_actor_history_positions=int(total.sum()),
          meaning='Replay-prefix exposures,not distinct new episodes. Stagewise float32 integer counters<=2M are exact;sum in float64 avoids aggregate telemetry rounding.',
          shared_identically_between_arms=True,joint='All four experts receive normal unmasked native gradients for1800 updates'))
    for arm in ['learned','uniform']:
        folder=O/arm;path=folder/'RESULTS.json'
        if not path.exists():continue
        row=read(path);assert row['status']=='COMPLETE' and row['actor_updates']==2000 and row['new_episodes_logical']==2000000
        assert row['critic_updates']==10000 and row['posterior_updates']==300000 and row['logical_total_seconds']<=14400
        assert read(folder/'CLONED_WARMUP_IDENTITY.json')['identity']==warm['identity']
        assert sha(folder/'joint_actor.pt')==row['checkpoint_sha256']
        v=np.load(folder/'evaluation/scores.npy');assert len(v)==2000 and np.isfinite(v).all();assert abs(v.mean()-row['evaluation']['mean'])<1e-12
        manifest.append(dict(method=row['method'],checkpoint=str(folder/'joint_actor.pt'),checkpoint_sha256=row['checkpoint_sha256'],runtime=str(folder/'runtime.pkl'),runtime_sha256=sha(folder/'runtime.pkl'),configuration=str(M/'configs/moe_sboed_v4_spec_20261118.json')))
        phase=row['phase'];physical['actor_updates']+=phase['counts']['actor'];physical['critic_updates']+=phase['counts']['critic'];physical['posterior_updates']+=phase['counts']['posterior'];physical['new_episodes']+=phase['new_episodes']
        costs.append(dict(arm=arm,logical_total_seconds=row['logical_total_seconds'],logical_new_episodes=row['new_episodes_logical'],joint_training_seconds=row['joint_training_seconds'],shared_charged_seconds=row['shared_preparation_and_warmup_seconds_charged'],evaluation=row['evaluation'],resources=read(folder/'RESOURCES.json'),model_queries=row['model_query_counts']))
    if status['status']=='COMPLETE':
        assert len(d['rows'])==5 and len(manifest)==2 and physical==dict(actor_updates=3800,critic_updates=19000,posterior_updates=570000,new_episodes=3800000)
        for key in ['original_learned','original_uniform','vsoed']:
            cp=read(O/'references'/key/'CHECKPOINT.json');assert sha(cp['path'])==cp['sha256'];manifest.append(dict(method=key,checkpoint=cp['path'],checkpoint_sha256=cp['sha256'],retrained=False))
    prep={}
    for name in ['calibration','audit','saturation','shared','references']:
        path=O/name/'RESOURCES.json'
        if path.exists():prep[name]=read(path)
    clone=read(R/'CLONE_SERIALIZATION_AUDIT.json') if (R/'CLONE_SERIALIZATION_AUDIT.json').exists() else None
    put('COMPUTE.json',dict(logical_methods=costs,physical_unique_training=physical,shared_and_reference_resources=prep,clone_audit=clone,
      native_audit_discarded_episodes=1000,notes=['Shared warmup charged to both logical ceilings,but physical execution counted once.','Original frozen V2/provider pretraining is inherited and common to original V4 and both Spec arms;not free from-scratch training.','Sampled GPU busy time is an estimate,not exact kernel occupancy.','Additional regime opportunity-provider rows lack an independent counter;not imputed as zero.']))
    put('CHECKPOINT_MANIFEST.json',manifest)
    diagnostics={}
    for key,row in d['rows'].items():diagnostics[key]=row.get('diagnostics')
    put('ROUTING_AND_FEATURE_DIAGNOSTICS.json',diagnostics)
    text='# Scientific conclusion: V4-Spec seed101\n\n'
    if d['status']=='COMPLETE':
        decision=d['decision'];text+='The prespecified same-bank paired comparisons are complete.\n\n'
        for name,row in d['comparisons'].items():text+=f"- {name}: {row['mean']:+.5f} nats;95% paired evaluation CI [{row['CI95'][0]:+.5f},{row['CI95'][1]:+.5f}].\n"
        text+='\n'+('Both primary improvement checks and the uniform point-estimate guard passed. Matched seeds102-105 may be recommended for a separately authorized exploratory replication.' if decision['recommend_replication'] else 'The complete improvement criteria were not met. Report this pilot as negative/inconclusive and stop;do not expand or tune.')+'\n'
    else:text+='The pilot did not finish. Preserve partial artifacts and the exact failure;no performance or success claim is supported.\n\n'+json.dumps(status,indent=2)+'\n'
    text+='\nThis is one continuation seed with shared parent/provider training,not a robustness study. The new bank is independent of previous architecture-selection banks. Regimes are exposure heuristics,not optimal-expert labels. Nonuniform routing and feature differences alone do not demonstrate information value.\n\n'
    text+='The canonical1M-contrast estimator has material saturation. A prespecified64-case diagnostic found substantial1M->4M shifts for original frozen policies;see SATURATION_SENSITIVITY.json. Primary estimates remain matched1M finite-contrast lower bounds,not exact EIG,and case MCSE does not cover finite-contrast bias. A positive pilot result is conditional on this estimator and needs independent estimator-resolution validation before a stronger scientific claim.\n'
    (R/'SCIENTIFIC_CONCLUSION.md').write_text(text)
    put('COMPLETION_VERIFICATION.json',dict(status='COMPLETE' if d['status']=='COMPLETE' else 'INCOMPLETE_PRESERVED',controller=status,matched_warmup_verified=bool(warm),completed_models=len(costs),new_bank_sha256=read(O/'BANK_IDENTITY.json')['sha256'],no_automatic_expansion=True))
    if d['status']=='COMPLETE':
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        names=list(d['rows']);fig,ax=plt.subplots(figsize=(10,4))
        ax.bar(range(len(names)),[d['rows'][key]['mean'] for key in names],yerr=[1.96*d['rows'][key]['MCSE'] for key in names],capsize=4)
        ax.set_xticks(range(len(names)),[d['rows'][key]['method'] for key in names],rotation=20,ha='right');ax.set_ylabel('EIG lower bound (nats);95% case MC intervals');fig.tight_layout();fig.savefig(R/'EIG_COMPARISON.png',dpi=180);plt.close(fig)
    assert subprocess.check_output(['git','branch','--show-current'],cwd=M,text=True).strip()==BRANCH
    files=[str(path.relative_to(M)) for path in R.iterdir() if path.suffix in ['.json','.md','.csv','.png']]+['tools/moe_sboed_v4_spec_publish_20261118.py']
    staged=set(subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).splitlines());assert staged<=set(files),'Unrelated staged work preserved'
    subprocess.run(['git','diff','--check','--',*files],cwd=M,check=True)
    normal=[x for x in files if not x.endswith('.png')];images=[x for x in files if x.endswith('.png')]
    subprocess.run(['git','add','--',*normal],cwd=M,check=True)
    if images:
        assert all((M/path).stat().st_size<2000000 for path in images);subprocess.run(['git','add','--force','--',*images],cwd=M,check=True)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=M,text=True).strip():
        subprocess.run(['git','diff','--cached','--stat'],cwd=M,check=True);subprocess.run(['git','commit','-m','Report matched V4-Spec pilot and paired scientific conclusions'],cwd=M,check=True)
    subprocess.run(['git','push','origin',BRANCH],cwd=M,check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=M,text=True).strip();(O/'PUBLISHED.json').write_text(json.dumps(dict(commit=commit,status=d['status']),indent=2)+'\n');print(commit,flush=True)
if __name__=='__main__':publish()
