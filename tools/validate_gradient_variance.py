"""CPU-only independent integrity/reproduction. Never invokes GPU collection."""
import argparse,hashlib,json,math,time
from pathlib import Path
from src.research.gradient_variance_analysis import new_audit,saved_audit
from src.research.fh_moe_v3_discovery import guard
from src.research.deepseek_sboed_experiment import write

def compare(a,b,path='root'):
 if isinstance(a,dict):
  for key in a:
   if key in ['CPU_seconds','CPU_wall_seconds']:continue
   compare(a[key],b[key],path+'.'+key)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+f'[{i}]')
 elif isinstance(a,(int,float)) and not isinstance(a,bool):
  tol=2e-5 if 'float32_max_responsibility_error' in path else 1e-9
  assert math.isclose(a,b,rel_tol=tol,abs_tol=tol),(path,a,b)
 else:assert a==b,(path,a,b)

def main():
 p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--v3-campaign',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();C=Path(a.campaign);V=Path(a.v3_campaign);O=Path(a.output);O.mkdir(parents=True,exist_ok=False);started=time.perf_counter();guard()
 manifest=json.loads(Path(a.manifest).read_text());names=[]
 for item in manifest:
  f=C/item['name'];assert f.stat().st_size==item['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==item['sha256'],str(f);names.append(item['name'])
 # Reproduce summaries in a new output folder; transferred records unchanged.
 for filename in ['new_split0.pt','new_split1.pt','new_split2.pt','contrast_sensitivity.pt','observation_contrast.pt','reference_sensitivity.pt']:(O/filename).symlink_to((C/filename).resolve())
 saved_audit(V,O);new_audit(O)
 compare(json.loads((C/'saved_signal_audit.json').read_text()),json.loads((O/'saved_signal_audit.json').read_text()))
 compare(json.loads((C/'variance_results.json').read_text()),json.loads((O/'variance_results.json').read_text()))
 historical=json.loads((V/'context_gate.json').read_text())['stages'];current=json.loads((O/'saved_signal_audit.json').read_text())['stages']
 for t in historical:compare(historical[t],current[t]['historical_metrics'])
 write(O/'receipt.json',dict(PASS=True,CPU_only=True,verified_artifacts=names,all_saved_and_new_metrics_reproduced=True,six_foundation_files_unchanged=True,CPU_wall_seconds=time.perf_counter()-started,precision_tolerance='float64 summaries1e-9; float32 roundoff diagnostic2e-5; timing fields excluded'))
 print('VALIDATION_COMPLETE',time.perf_counter()-started)
if __name__=='__main__':main()
