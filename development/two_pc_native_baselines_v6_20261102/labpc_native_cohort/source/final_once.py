"""Explicit post-selection one-shot final inference; never called by manager."""
import pathlib,sys,json,os,fcntl
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from native_final import C,read,put,sha,verify,locks,evaluate
if len(sys.argv)!=3 or sys.argv[1]!='--user-moe-freeze':raise SystemExit('Usage: final_once.py --user-moe-freeze /path/to/user_approved_moe_freeze.json')
receipt=pathlib.Path(sys.argv[2]);freeze=read(receipt)
assert freeze.get('user_approved_final_evaluation') is True
assert freeze['selected_moe_checkpoint'] and sha(freeze['selected_moe_checkpoint'])==freeze['checkpoint_sha256']
assert freeze.get('hyperparameters_frozen') is True
verify();jobs=read(C/'jobs.json');assert not any(x['status'] in ['RUNNING','QUEUED'] for x in jobs)
held=locks()
try:
 marker=C/'FINAL_EVALUATION_STARTED.json'
 fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
 with os.fdopen(fd,'w') as f:json.dump(dict(user_freeze=str(receipt),receipt_sha256=sha(receipt),pid=os.getpid()),f,indent=2)
 for row in jobs:
  if row['method']=='random' or row.get('checkpoint'):evaluate(row,'sealed_final')
 put(C/'FINAL_EVALUATION_COMPLETED.json',dict(status='COMPLETE',receipt_sha256=sha(receipt)))
finally:
 for f in held:f.close()
