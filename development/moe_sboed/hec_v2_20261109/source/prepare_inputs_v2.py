from native_common import *
from restore_v1 import V1,SG,COL,CFG
import shutil
assert not (O/'evidence/INPUT_PROVENANCE.json').exists();copies={}
for name,src in {**{n:V1/f'inputs/{n}.pt' for n in ['sg','parent','historical_control','reference','original','ordinary','vsoed']},'v1':V1/'arms/candidate/actor.pt','v1_control':V1/'arms/control/actor.pt','sg_control':SG/'arms/control/actor.pt'}.items():
 dst=O/f'inputs/{name}.pt';shutil.copyfile(src,dst);assert sha(src)==sha(dst);copies[name]=dict(source=str(src),sha256=sha(src))
for n in ['COL_CONFIG.json','HISTORICAL_BASELINES.json']:shutil.copyfile(V1/'inputs'/n,O/'inputs'/n)
assert copies['v1']['sha256']==CFG['parent_actor_sha256'];put(O/'evidence/INPUT_PROVENANCE.json',dict(status='VERIFIED',copies=copies,parent_runtime_sha256=CFG['parent_runtime_sha256'],training_updates=0));print('V2_INPUTS_VERIFIED')
