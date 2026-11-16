from native_common import *
from restore_sg import SG,COL,CFG
import shutil
assert not (O/'evidence/INPUT_PROVENANCE.json').exists();copies={}
for name,source in {'sg':SG/'arms/candidate/actor.pt',**{n:SG/f'inputs/{n}.pt' for n in ['parent','historical_control','reference','original','ordinary','vsoed']}}.items():
 dst=O/f'inputs/{name}.pt';shutil.copyfile(source,dst);assert sha(source)==sha(dst);copies[name]=dict(source=str(source),sha256=sha(source))
assert copies['sg']['sha256']==CFG['parent_actor_sha256'];assert sha(SG/'arms/candidate/runtime.pkl')==CFG['parent_runtime_sha256']
for source,name in [(SG/'inputs/HISTORICAL_BASELINES.json','HISTORICAL_BASELINES.json'),(COL/'CONFIG_LOCKED.json','COL_CONFIG.json')]:shutil.copyfile(source,O/'inputs'/name)
put(O/'evidence/INPUT_PROVENANCE.json',dict(status='VERIFIED',copies=copies,SG_runtime_sha256=CFG['parent_runtime_sha256'],single_seed=101,simulator='two sources R4 standard normal prior;log(.1+sum 1/(1e-4+distance^2));normal noise .5;design square [-4,4]^2',training_updates=0));print('HEC_INPUTS_VERIFIED')
