import json,time,fcntl,subprocess,sys
import pytest
from src.research import sir_t3_dad_fixed as campaign

def test_per_method_and_aggregate_budget_do_not_reset(tmp_path,monkeypatch):
 monkeypatch.setattr(campaign,'C',tmp_path);(tmp_path/'manifests').mkdir();p=tmp_path/'manifests/aggregate_budget.json';p.write_text(json.dumps(dict(cap_seconds=21600,caps=dict(fixed=7200,dad=14400),spent=dict(fixed=120.,dad=0.),active=None)));b=campaign.Budget('fixed');assert b.spent()>=120
 with pytest.raises(AssertionError):campaign.Budget('dad')
 b.start=time.monotonic()-7075;assert not b.can();spent=b.finish('TEST');x=json.loads(p.read_text());assert 7195<=spent<7200 and x['active'] is None and sum(x['spent'].values())<=21600

def test_inherited_flock_keeps_exclusive_lease_until_child_exit(tmp_path):
 path=tmp_path/'device.lock';f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);code='import os,sys;fd=int(sys.argv[1]);os.fstat(fd);print("READY",flush=True);sys.stdin.read(1)';p=subprocess.Popen([sys.executable,'-c',code,str(f.fileno())],pass_fds=(f.fileno(),),stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True);assert p.stdout.readline().strip()=='READY';f.close()
 with path.open('a+') as probe:
  with pytest.raises(BlockingIOError):fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
  p.communicate('q',timeout=5);assert p.returncode==0;fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
