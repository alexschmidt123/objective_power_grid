"""Mac administrative watcher: synchronize only the sealed aggregate report."""
import argparse,pathlib,subprocess,json,time,sys,fcntl,os
REMOTE=r'''
from pathlib import Path
import json,hashlib,subprocess
b=Path.home()/'Documents/objective_power_grid_temp';q=b/'scratch_runs/moe_v3_uniform_router_queue_20261111';s=q/'STATUS.json'
if not s.exists():print(json.dumps(dict(status='WAIT')));raise SystemExit
status=json.loads(s.read_text())
if status['status']=='BLOCKED':print(json.dumps(status));raise SystemExit
if status['status']!='COMPLETED':print(json.dumps(dict(status='WAIT',evaluation=status['status'])));raise SystemExit
root=Path(status['results']);m=root/'ARTIFACT_SHA256.json'
if not m.exists():print(json.dumps(dict(status='WAIT',reason='awaiting result seal')));raise SystemExit
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();files=json.loads(m.read_text())
for name in ['RESULTS.json','UNIFORM_ROUTER_EIG.md']:assert sha(root/name)==files[name],name
r=json.loads((root/'RESULTS.json').read_text());report=(root/'UNIFORM_ROUTER_EIG.md').read_text();report+='\nAggregate selection and resource summary:\n\n```json\n'+json.dumps(dict(selection={k:dict(counts=r[k]['selection_counts'],frequencies=r[k]['selection_frequencies'],evaluation_seconds=r[k]['evaluation_seconds'],CPU_seconds=r[k]['process_CPU_seconds'],GPU_peak_bytes=r[k]['GPU_peak_allocated_bytes']) for k in ['learned','uniform']},resources=r['sampled_resources'],provenance=r['provenance']),indent=2)+'\n```\n'
g=b/'development/hec_git_sync';relative='development/moe_sboed/v3_uniform_router_20261111/RESULTS_SUMMARY.md';target=g/relative;changed=subprocess.check_output(['git','-C',str(g),'status','--porcelain'],text=True).splitlines();assert all(row[3:]==relative for row in changed),'Other dirty work preserved;report sync blocked'
if target.exists():assert target.read_text()==report,'Existing different report preserved'
else:target.write_text(report)
print(json.dumps(dict(status='READY',file=relative)))
'''

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');parser.add_argument('--state-dir',type=pathlib.Path);args=parser.parse_args();compile(REMOTE,'<remote_sealed_report>','exec')
 if args.validate:print('REPORT_SYNC_VALIDATED');return
 assert sys.platform=='darwin' and args.state_dir is not None;d=args.state_dir;d.mkdir(parents=True,exist_ok=True);guard=(d/'.watcher.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);state=d/'SYNC_STATUS.json'
 def record(status,**kw):state.write_text(json.dumps(dict(status=status,time=time.time(),pid=os.getpid(),**kw),indent=2)+'\n')
 record('WAITING_FOR_EVALUATION')
 try:
  while True:
   p=subprocess.run(['ssh','-q','-o','BatchMode=yes','labpc','python3 -'],input=REMOTE,text=True,capture_output=True,timeout=60)
   if p.returncode:raise RuntimeError(p.stderr[-2000:] or 'Remote report extraction failed')
   row=json.loads(p.stdout.strip().splitlines()[-1])
   if row['status']=='BLOCKED':record('BLOCKED',evaluation=row);return
   if row['status']=='WAIT':time.sleep(30);continue
   assert row['status']=='READY';record('SYNCHRONIZING');tool=pathlib.Path.home()/'scratch_runs/moe_workflow_sync_20261109/moe_sync.py';receipt=d/'REPORT_SYNC_RECEIPT.json'
   p=subprocess.run([sys.executable,str(tool),'--message','Report frozen V3 learned-versus-uniform EIG and resource results','--files',row['file'],'--receipt',str(receipt)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=300);(d/'sync.log').write_text(p.stdout)
   if p.returncode:raise RuntimeError('Safe synchronization failed;see sync.log;work preserved')
   r=json.loads(receipt.read_text());assert r['status']=='VERIFIED';record('COMPLETED',commit=r['github_commit'],receipt=str(receipt));return
 except Exception as exc:record('BLOCKED',reason=str(exc))
if __name__=='__main__':main()
