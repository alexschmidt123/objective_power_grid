"""Read-only verification of transferred context artifacts and paired scores."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from src.research.fh_moe_v3_context import discovery_metrics
from src.research.deepseek_sboed_experiment import write

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--evidence',required=True);p.add_argument('--receipt',required=True);a=p.parse_args();c=Path(a.campaign);e=Path(a.evidence);started=time.perf_counter();torch.set_num_threads(2)
 manifest=json.loads((e/'artifact_manifest.json').read_text());verified=[]
 for item in manifest:
  f=c/item['name'];assert f.stat().st_size==item['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],str(f);verified.append(item['name'])
 freeze=json.loads((c/'source_freeze.json').read_text())
 for f,h in freeze.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h,f
 splits=[torch.load(c/f'gradient_dataset_split{i}.pt',map_location='cpu',weights_only=False) for i in range(3)];saved=json.loads((c/'context_gate.json').read_text());reproduced={}
 for t in [1,10,20,29]:
  r,_=discovery_metrics(*[(s[t][0],s[t][1]) for s in splits]);s=saved['stages'][str(t)]
  assert r['PASS']==s['PASS'] and r['checks']==s['checks']
  errors=[abs(r[k]-s[k]) for k in ['between_context_cosine','assignment_kappa']]
  assert max(errors)<1e-8
  reproduced[t]=dict(PASS=r['PASS'],checks_equal=True,max_summary_error=max(errors))
 reference=np.load(c/'shared_reference_development_scores.npy');native=np.load(c/'native_control_development_scores.npy');d=reference-native
 reported=json.loads((c/'development.json').read_text())['shared_reference_minus_native'];assert abs(d.mean()-reported['mean'])<1e-12 and abs(d.std(ddof=1)/len(d)**.5-reported['SE'])<1e-12
 write(a.receipt,dict(PASS=True,verified_artifacts=verified,source_hashes_equal=True,reproduced_gate=reproduced,paired_development_scores_reproduced=True,CPU_wall_seconds=time.perf_counter()-started,device='CPU only',torch_version=torch.__version__))
 print('ARTIFACT_AUDIT_PASS',time.perf_counter()-started)
if __name__=='__main__':main()
