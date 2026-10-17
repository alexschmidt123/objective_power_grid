import os,sys,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
C=Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_moe_st_routing_dev_103_105');S=C/'source';R=C.parents[2];sys.path.insert(0,str(S))
import numpy as np,torch,yaml
torch.set_num_threads(2)
from src.research.journal_validation import CommonValidation
raw=yaml.safe_load((C/'original.yaml').read_text());v=CommonValidation(raw,30,1001,2000,1000000,16,16384)
results=[]
for seed in [103,105]:
 run=R/f'experiments/10152026/10152026_journal_five_seed_fixed_budget/location/seed{seed}/moe'
 cases=json.loads((run/'eval/moe_seed1001.json').read_text());assert len(cases)==2000
 assert np.array_equal(np.array([r['truth'] for r in cases]),v.truth.numpy())
 assert np.array_equal(np.array([r['measurement_noise_standard'] for r in cases]),v.noise.numpy())
 assert np.array_equal(np.load(run/'eval/contrast_bank_seed1001.npy'),v.bank.numpy())
 selected=[r['selected_expert'] for c in cases for r in c['routing']];assert len(selected)==58000
 mean=float(np.mean([r['spce'] for r in cases]));expected=json.loads((run/'summary.json').read_text())[0]['mean_spce'];assert mean==expected
 results.append(dict(seed=seed,original_eig=mean,postinitial_expert_selections=len(selected)))
manifest=json.loads((C/'source_manifest.json').read_text())['sha256'];assert all(hashlib.sha256((S/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
(C/'evaluation_readiness.json').write_text(json.dumps(dict(status='passed',paired_draws_sha256=v.identity,originals=results,source_manifest_unchanged=True,extra_training=False),indent=2)+'\n');print('EVALUATION_READINESS_PASSED',results)
