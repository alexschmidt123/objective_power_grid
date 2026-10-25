"""Read-only source/artifact inventory and verified paired comparisons."""
from pathlib import Path
import hashlib,json,math,os,time
import numpy as np
import torch,yaml

R=Path(__file__).resolve().parents[1]
S=Path.home()/'Documents/objective_power_grid_integrated_information_moe'
C=S/'experiments/10242026/10242026_ir_moe_seed101'
D=R/'experiments/10242026/10242026_ir_moe_labpc_diagnostics'
D.mkdir(parents=True,exist_ok=True)
for name in ['reports','data','figures','costs','metadata','logs']:(D/name).mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
read=lambda p:json.loads(Path(p).read_text())
start=time.perf_counter()
cfg=yaml.safe_load((C/'configs/seed101.yaml').read_text())
cost=read(C/'costs/full_final.json');progress=read(C/'costs/full_progress.json')
state=torch.load(C/'models/full/selected.pt',map_location='cpu',weights_only=False)
assert all(torch.isfinite(x).all() for x in state['state_dict'].values())
freeze=read(C/'reports/checkpoint_freeze.json')
assert sha(C/'models/full/selected.pt')==freeze['checkpoint_sha256']==cost['checkpoint_sha256']
sourcehash=read(C/'source_manifest/frozen_source.json')['sha256']
changed=[name for name,digest in sourcehash.items() if sha(S/name)!=digest]
assert not changed,changed
official=read(C/'reports/ir_seed1001_result.json')
scores=np.load(C/'evaluations/ir_seed1001_scores.npy')
assert scores.shape==(2000,) and np.isfinite(scores).all()
assert abs(scores.mean()-official['EIG'])<1e-12
assert abs(scores.std(ddof=1)/np.sqrt(2000)-official['MC_SE'])<1e-12
bank=Path(cfg['canonical_evaluation_directory'])
assert sha(bank/'location_cases.npz')==cfg['cases_sha256']
assert sha(bank/'location_contrasts.npy')==cfg['contrast_sha256']
cases=np.load(bank/'location_cases.npz')
identity=hashlib.sha256(cases['truth'].tobytes()+cases['noise'].tobytes()+np.load(bank/'location_contrasts.npy').tobytes()).hexdigest()
assert identity==official['metadata']['paired_draws_sha256']
pairs=[]
def add_pair(label,values,proof):
    delta=scores-np.asarray(values);se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    pairs.append(dict(method=label,paired=True,N=len(delta),reference_mean=float(np.mean(values)),difference=float(delta.mean()),paired_MC_SE=se,CI95=[float(delta.mean()-1.96*se),float(delta.mean()+1.96*se)],alignment=proof))
v1=Path(cfg['reference_campaign'])
old=np.load(v1/'evaluations/official_seed1001_scores.npy');oldmeta=read(v1/'reports/official_seed101_result.json')
assert oldmeta['metadata']['paired_draws_sha256']==identity
add_pair('DeepSeekMoE-v1',old,dict(paired_draws_sha256=identity,source=str(v1/'evaluations/official_seed1001_scores.npy'),source_sha256=sha(v1/'evaluations/official_seed1001_scores.npy')))
base=bank.parent/'location/seed101'
refs=[('vsOED',base/'vsoed/eval','vsoed'),('Corrected DAD',Path(cfg['corrected_DAD_checkpoint']).parent.parent/'eval','dad')]
for label,folder,method in refs:
    p=folder/(method+'_seed1001.json');data=read(p);values=np.asarray([x['spce'] for x in data])
    assert len(data)==2000 and [x['trajectory_id'] for x in data]==list(range(2000))
    assert np.array_equal(np.asarray([x['truth'] for x in data]),cases['truth'])
    assert np.array_equal(np.asarray([x['measurement_noise_standard'] for x in data]),cases['noise'])
    assert sha(folder/'contrast_bank_seed1001.npy')==cfg['contrast_sha256']
    add_pair(label,values,dict(truth_exact=True,noise_exact=True,trajectory_order_exact=True,contrast_sha256=cfg['contrast_sha256'],source=str(p),source_sha256=sha(p)))
    np.save(D/'data'/(method+'_seed1001_scores.npy'),values)
np.save(D/'data/ir_seed1001_scores.npy',scores)
records=progress['records'];last=records[-1]
stats=dict(last_logged_update=last['update'],last_logged_spce=last['train_spce'],ceiling=math.log(cfg['contrasts']+1),late_logged_mean_spce=float(np.mean([x['train_spce'] for x in records[-20:]])),late_logged_gradient_max=max(x['preclip_gradient_norm'] for x in records[-20:]),logged_gradient_max=max(x['preclip_gradient_norm'] for x in records),logged_gradient_median=float(np.median([x['preclip_gradient_norm'] for x in records])),logged_fraction_clipped=float(np.mean([x['preclip_gradient_norm']>5 for x in records])),last_logged_diagnostics=last)
summary=dict(source_campaign=str(C),source_commit=state['source_commit'],checkpoint_sha256=sha(C/'models/full/selected.pt'),checkpoint_numerically_valid=True,all_frozen_hashes_verified=len(sourcehash),original_controller=read(C/'registry.json'),reporting_failure=read(C/'reports/failure.json'),configuration=cfg,training=dict(executed_updates=cost['executed_updates'],selected_update=cost['selected_update'],charged_seconds=cost['clock']['charged_seconds'],stop_reason=cost['stop_reason'],counts=cost['clock']['counts'],architecture=cost['counts']),official=official,confirmation=read(C/'reports/ir_seed1079_result.json'),statistics=stats,development=read(C/'reports/development_curve.json'),interventions=read(C/'reports/development_comparisons.json'),latency=read(C/'reports/online_latency.json'),pairing=pairs)
write(D/'metadata/source_inventory.json',summary)
write(D/'data/paired_comparisons.json',pairs)
write(D/'costs/CPU_inventory.json',dict(wall_seconds=time.perf_counter()-start,optimizer_updates=0,new_simulator_calls=0,GPU_seconds=0))
print(json.dumps(dict(training=summary['training'],last_logged_update=last['update'],last_logged_spce=last['train_spce'],official=official,pairing=pairs,source_valid=True),indent=2))

