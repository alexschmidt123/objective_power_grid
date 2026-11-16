"""One original-V2 native update validates the matched control logging path."""
import time
from v3_native import n,ROOT
from v3_training import phase
out=ROOT/'scratch_runs/hec_v3_feature_20261114_control_smoke01';assert not out.exists();out.mkdir()
n.original.setup();lease=n.original.acquire();start=time.monotonic()
try:
    state=n.load_parent();a=state['algorithm'];n.original.restore(state['rng'])
    result=phase(a,'control',1,out,start,600);assert result['counts']==dict(actor=1,critic=5,posterior=150)
    n.put(out/'RESULTS.json',dict(status='COMPLETE',counts=result['counts'],purpose='Control integration only;not a comparison result'))
finally:
    for f in lease:f.close()
