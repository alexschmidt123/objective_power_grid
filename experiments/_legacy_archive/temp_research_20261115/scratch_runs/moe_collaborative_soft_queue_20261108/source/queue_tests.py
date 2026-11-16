"""Dependency/resource/dedup fixture;never starts a scientific process."""
import sys,pathlib,json,types,os,fcntl,time
sys.dont_write_bytecode=True
import watch_queue as w
fixture=w.O/f'evidence/QUEUE_TEST_ONLY_{time.time_ns()}';fixture.mkdir(exist_ok=True);w.UP=fixture
state,_=w.upstream_gate();assert state=='WAIT'
for name,data in {'evidence/EXPERIMENT_COMPLETION.json':{'status':'COMPLETE'},'arms/control/RESULT.json':{'status':'COMPLETE','native_iterations':10001},'arms/candidate/RESULT.json':{'status':'COMPLETE','native_iterations':10001},'evaluation/COMPLETION.json':{'status':'COMPLETE'},'evaluation/NATIVE_COMPETENCE_PANEL.json':{'status':'COMPLETE'},'evaluation/SPECIALIZATION_RESULTS.json':{'specialization_criterion_met':False},'evaluation/EIG_RESULTS.json':{'fixture_only':True},'evidence/FINAL_PRESERVATION.json':{'status':'VERIFIED'},'ARTIFACT_SHA256.json':{'status':'VERIFIED'}}.items():
 p=fixture/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data))
state,_=w.upstream_gate();assert state=='READY','Unfavorable scientific result must not block'
(fixture/'arms/candidate/RESULT.json').write_text(json.dumps({'status':'FAILED','native_iterations':2}));state,_=w.upstream_gate();assert state=='FAILED'
original=w.subprocess.run
w.subprocess.run=lambda args,**kw:types.SimpleNamespace(returncode=0,stdout='147, 0\n' if '--query-gpu=memory.used,utilization.gpu' in args else '',stderr='');assert w.gpu_idle()[0]
w.subprocess.run=lambda args,**kw:types.SimpleNamespace(returncode=0,stdout='1500, 50\n' if '--query-gpu=memory.used,utilization.gpu' in args else '12345',stderr='');assert not w.gpu_idle()[0];w.subprocess.run=original
claim=fixture/'EXCLUSIVE_CLAIM_TEST';fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.close(fd)
try:os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);raise AssertionError('Duplicate permitted')
except FileExistsError:pass
other=(w.O/'QUEUE_PROCESS.lock').open('a')
try:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB);raise AssertionError('Duplicate watcher permitted')
except BlockingIOError:pass
(w.O/'evidence/QUEUE_TEST_RESULTS.json').write_text(json.dumps(dict(status='PASS',upstream_failure_stops=True,unfavorable_results_do_not_stop=True,idle_gpu_required=True,persistent_claim_prevents_duplicates=True,flock_prevents_duplicate_watchers=True,training_processes_launched=0,GPU_used=False),indent=2));print('QUEUE_DEPENDENCY_TESTS_PASS')
