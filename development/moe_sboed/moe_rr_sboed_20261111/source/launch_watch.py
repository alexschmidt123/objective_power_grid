"""Mac-only duplicate-safe completion watcher launcher; no experiment launch."""
from pathlib import Path
import subprocess,sys,json,time
root=Path.home()/'scratch_runs/moe_rr_sboed_20261111'
assert not (root/'WATCH_LAUNCH.json').exists(),'Watcher already launched'
with (root/'watch.log').open('xb') as log:
    child=subprocess.Popen([sys.executable,'-u',str(root/'source/watch_sync.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=child.pid,time=time.time(),purpose='Observe owned RR experiment;publish aggregate reports and safely synchronize;no training restart')
(root/'WATCH_LAUNCH.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
