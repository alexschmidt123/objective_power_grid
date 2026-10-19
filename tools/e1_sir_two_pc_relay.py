"""SSH artifact relay only; scientific decisions and final aggregation stay on LabPC.

Run on the existing SSH client. All research source is authored/approved on LabPC.
Canonical run.sh execution is delegated to tools.e1_sir_replication/journal_campaign.
"""
import hashlib,json,subprocess,sys,time
from pathlib import Path
HOME={'labpc':'/home/grads/g/g.lin','ubuntu-pc':'/home/qiqi'}
W={h:HOME[h]+'/Documents/objective_power_grid_e1_sir_replication_20261019' for h in HOME}
REL='experiments/10192026/10192026_sir_e1_five_seed_verified'
COR='experiments/10192026/10192026_sir_e1_corrected_implementations'
OLDW={h:HOME[h]+'/Documents/objective_power_grid_e1_sir_verified_20261019' for h in HOME}
LOCAL=Path.home()/'e1_execution_20261018/sir_relay_20261019';LOCAL.mkdir(parents=True,exist_ok=True)
STATE=LOCAL/'state.json'
def note(state,**kw):
 data=dict(state=state,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw);STATE.write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
def remote(host,code,timeout=180):
 p=subprocess.run(['ssh',host,'python3','-'],input=code,text=True,capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError(host+': '+p.stderr[-2000:]+p.stdout[-2000:])
 lines=[x[7:] for x in p.stdout.splitlines() if x.startswith('E1JSON=')];return json.loads(lines[-1]) if lines else None
def read(host,path):return remote(host,f'from pathlib import Path\nimport json\np=Path({path!r});print("E1JSON="+json.dumps(json.loads(p.read_text()) if p.exists() else None))')
def wait(host,path,max_hours=12):
 start=time.monotonic()
 while True:
  data=read(host,path)
  if data is not None:return data
  if time.monotonic()-start>max_hours*3600:raise TimeoutError('Waiting for '+host+':'+path)
  note('waiting',host=host,path=path);time.sleep(60)
def package(host,root,tag,paths=None):
 return remote(host,f'''from pathlib import Path
import json,hashlib,tarfile
r=Path({root!r}); names={paths!r};files=[p for p in r.rglob('*') if p.is_file()] if names is None else [r/p for p in names]
a=Path('/tmp/{tag}.tar.gz');m={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
with tarfile.open(a,'w:gz',dereference=True) as t:
 for p in files:t.add(p,arcname=str(p.relative_to(r)),recursive=False)
mp=Path('/tmp/{tag}.json');mp.write_text(json.dumps(m))
print('E1JSON='+json.dumps(dict(archive=str(a),manifest=str(mp),sha256=hashlib.sha256(a.read_bytes()).hexdigest(),files=len(files))))''',timeout=600)
def transfer(source,dest,root,tag,paths=None,merge=False):
 info=package(source,root,tag,paths);a=LOCAL/(tag+'.tar.gz');m=LOCAL/(tag+'.json')
 for src,out in [(info['archive'],a),(info['manifest'],m)]:subprocess.run(['scp',source+':'+src,str(out)],check=True,timeout=600)
 assert hashlib.sha256(a.read_bytes()).hexdigest()==info['sha256']
 for p in [a,m]:subprocess.run(['scp',str(p),dest+':/tmp/'+p.name],check=True,timeout=600)
 return info

def extract(host,target,tag,info,merge=False):
 return remote(host,f'''from pathlib import Path
import tarfile,json,hashlib
r=Path({target!r});r.mkdir(parents=True,exist_ok={merge!r});a=Path('/tmp/{tag}.tar.gz');assert hashlib.sha256(a.read_bytes()).hexdigest()=={info['sha256']!r};m=json.loads(Path('/tmp/{tag}.json').read_text())
with tarfile.open(a) as t:
 for entry in t.getmembers():
  assert not entry.name.startswith('/') and '..' not in Path(entry.name).parts
  p=r/entry.name
  if p.exists():assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==m[entry.name],str(p)
  else:t.extract(entry,r)
assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==v for p,v in m.items())
print('E1JSON='+json.dumps(dict(status='PASS',files=len(m),archive_sha256={info['sha256']!r})))''',timeout=600)
def launch(host,action,stem):
 return remote(host,f'''from pathlib import Path
import subprocess,json
r=Path({W[host]!r});c=r/{REL!r};log=c/'logs/{stem}.log';assert not log.exists(),str(log)
cmd=[str(Path.home()/'miniconda3/envs/mocu_optimized/bin/python'),'-u','tools/e1_sir_replication.py',{action!r}]+(['--host',{host!r}] if {action!r}=='run' else [])
wrapper='import subprocess,pathlib; p=subprocess.run('+repr(cmd)+'); pathlib.Path('+repr(str(c/'metadata/{stem}_exit.json'))+').write_text(__import__("json").dumps(dict(exit_code=p.returncode)))'
f=log.open('x');p=subprocess.Popen([str(Path.home()/'miniconda3/envs/mocu_optimized/bin/python'),'-u','-c',wrapper],cwd=r,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(c/'metadata/{stem}_pid.json').write_text(json.dumps(dict(pid=p.pid,command=cmd)));print('E1JSON='+json.dumps(dict(pid=p.pid,command=cmd)))''')
def main():
 note('waiting_corrected_ScoreBED101')
 score=wait('ubuntu-pc',OLDW['ubuntu-pc']+'/'+COR+'/metadata/score101_complete.json')
 assert score['status']=='complete',score
 info=transfer('ubuntu-pc','labpc',OLDW['ubuntu-pc']+'/'+COR,'sir_corrected101_worker_complete')
 result=extract('labpc',OLDW['labpc']+'/'+COR+'/worker_ubuntu','sir_corrected101_worker_complete',info)
 remote('labpc',f'''from pathlib import Path
import json
c=Path({W['labpc']+'/'+REL!r});s=Path({OLDW['labpc']+'/'+COR+'/worker_ubuntu/sir/seed101/scorebed'!r});t=json.loads((s/'models/scorebed_training.json').read_text());assert t['generated_training_trajectories']==1000000 and t['sir_score_adapter_revision']==2
(c/'metadata/corrected_score101_import_verified.json').write_text(json.dumps(dict(status='PASS',run=str(s),transfer={result!r})))''')
 # Worker uses a clean checkout of the exact source already approved on LabPC.
 note('launching_worker_precision_check');launch('ubuntu-pc','worker-resolution','worker_precision')
 check=wait('ubuntu-pc',W['ubuntu-pc']+'/'+REL+'/metadata/worker_precision_exit.json',max_hours=12);assert check['exit_code']==0,check
 paths=['metadata/worker_precision_complete.json','reports/sir_1m_4m_resolution.csv','reports/sir_1m_4m_pairs.csv']+[f'evaluations/{m}_N128_1m_4m.npz' for m in ['random','fixed','dad','rl_sboed','vsoed','scorebed','moe','resmoe','switchmoe','switchresmoe']]
 info=transfer('ubuntu-pc','labpc',W['ubuntu-pc']+'/'+REL,'sir_worker_precision',paths)
 extract('labpc',W['labpc']+'/'+REL,'sir_worker_precision',info,merge=True)
 note('waiting_full2048_precision')
 p=HOME['labpc']+'/Documents/objective_power_grid_e1_sir_verified_20261019/experiments/10192026/10192026_e1_sir_verification/metadata/resolution_N2048_complete.json'
 wait('labpc',p,max_hours=12)
 note('applying_LabPC_numerical_gate')
 remote('labpc',f'''import subprocess
from pathlib import Path
p=subprocess.run([str(Path.home()/'miniconda3/envs/mocu_optimized/bin/python'),'tools/e1_sir_replication.py','freeze'],cwd={W['labpc']!r});assert p.returncode==0''')
 assert read('labpc',W['labpc']+'/'+REL+'/metadata/replication_gate.json')['status']=='PASS'
 info=transfer('labpc','ubuntu-pc',W['labpc']+'/'+REL,'sir_frozen_replication')
 # Existing worker precision artifacts must match; differing registry metadata is an error.
 extract('ubuntu-pc',W['ubuntu-pc']+'/'+REL,'sir_frozen_replication',info,merge=True)
 for host in HOME:
  remote(host,f'''import sys
from pathlib import Path
sys.path.insert(0,{W[host]!r})
from tools import journal_campaign as j
j.C=Path({W[host]+'/'+REL!r});j.guard()''')
 note('launching_held_seeds102_105_after_PASS')
 jobs={h:launch(h,'run',h+'_replication') for h in HOME};note('five_seed_training_running',jobs=jobs)
 # Jobs are independent of this relay connection; final outputs remain on each host.
 for host in HOME:
  code=wait(host,W[host]+'/'+REL+'/metadata/'+host+'_replication_exit.json',max_hours=120)
  if code['exit_code']!=0:note('training_controller_failed',host=host,exit=code)
 info=transfer('ubuntu-pc','labpc',W['ubuntu-pc']+'/'+REL,'sir_replication_worker_complete')
 imported=extract('labpc',W['labpc']+'/'+REL+'/worker_ubuntu','sir_replication_worker_complete',info)
 remote('labpc',f'''from pathlib import Path
import json
c=Path({W['labpc']+'/'+REL!r});master=json.loads((c/'registry.json').read_text());worker=json.loads((c/'worker_ubuntu/registry.json').read_text())
for e in worker:
 original=e['run'];e['worker_original_run']=original;e['run']=str(c/'worker_ubuntu'/Path(original).relative_to(Path({W['ubuntu-pc']+'/'+REL!r})));e['verified_import']=True;assert not any(x['seed']==e['seed'] and x['method']==e['method'] for x in master);master.append(e)
(c/'registry.json').write_text(json.dumps(master,indent=2));(c/'metadata/worker_complete_import_verified.json').write_text(json.dumps({imported!r}))''')
 note('remaining_training_finished_imported',next='Verify seed101 high-resolution reevaluation, final statistics and independent confirmation; do not claim E1 complete yet')
if __name__=='__main__':
 try:main()
 except Exception as exc:note('STOPPED_REQUIRES_REVIEW',error=repr(exc));raise
