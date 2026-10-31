import copy,json,time
from pathlib import Path
import numpy as np
import torch,yaml
from src.research.sir_dad import SIRSumMLP
from src.research.journal_validation import CommonValidation

torch.set_num_threads(2);torch.manual_seed(101)
raw=yaml.safe_load(Path('configs/sir_ode_research.yaml').read_text())
p=SIRSumMLP(5).cpu().eval()
ref=CommonValidation(raw,5,771203,8,128,4,128)
cfg=copy.deepcopy(raw);cfg['sir_ode']['execution_backend']='fused_rk4'
candidate=CommonValidation(cfg,5,771203,8,128,4,128,device='cuda:0')
assert ref.identity==candidate.identity
a,ca=ref(p);b,cb=candidate(p)
np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-7)
out=Path('experiments/10162026/10162026_ubuntu_sir_execution')
(out/'validation_equivalence.json').write_text(json.dumps(dict(status='passed',
    maximum_absolute_score_difference=float(abs(a-b).max()),
    reference=ca,accelerated=cb,reference_scores=a.tolist(),
    accelerated_scores=b.tolist(),reserved_seed_used=False),indent=2))
print('VALIDATION_EQUIVALENCE_PASSED',float(abs(a-b).max()),ca['seconds'],cb['seconds'],flush=True)
