"""Guarded manual reevaluation; no training and no concurrent GPU job."""
from pathlib import Path
import json,sys,subprocess,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from src.research.moe_v5a_training import acquire
C=R/'scratch_runs/10312026_073000_moe_sboed_v5_train'
s=json.loads((C/'seed101/run_status.json').read_text())
if s['state']!='COMPLETE_TRAINING' or not s['latest_checkpoint']['eligible'] or not s['latest_checkpoint']['path'].endswith('_final.pt'):
 raise SystemExit('Wait for the final budget checkpoint; current state: '+s['state'])
leases=acquire()
try:
 out=C/('reevaluation_seed101_'+time.strftime('%m%d%Y_%H%M%S',time.gmtime()))
 command=['bash',str(R/'run.sh'),'--moe-sboed-v5a-training','--config',str(R/'configs/moe_sboed_v5a_train_20261031.yaml'),'--action','evaluate','--evaluate-checkpoint',s['latest_checkpoint']['path'],'--output',str(out)]
 subprocess.run(command,cwd=R,check=True)
finally:
 for f in leases:f.close()
