from common import *
assert not any((O/p).exists() for p in ['PRETRAIN_MANIFEST.json','arms','initial','evaluation','evidence/TRAINING_STARTED.json','evidence/LAUNCH_RECEIPT.json']);assert CFG['status']=='VALIDATED_FIXED_PROTOCOL' and CFG['updates_per_arm']==2000 and CFG['discount']==1 and CFG['main_auxiliary_loss']==0 and not CFG['new_rewards']
for n in ['GRADIENT_VALIDATION.json','EVALUATION_CHECKS.json','CONDITIONAL_CHECKS.json','dad_preflight.json','rl_sboed_preflight.json','scorebed_preflight.json']:assert json.loads((O/'evidence'/n).read_text())['status']=='PASS',n
sync=json.loads((O/'evidence/GITHUB_SYNC_RECEIPT.json').read_text());assert sync['status']=='VERIFIED' and sync['labpc']['commit']==sync['ubuntu']['commit']==sync['github_commit'];refs=json.loads((OLD/'evidence/REFERENCE_SOURCE_INPUTS.json').read_text());sealed=json.loads((OLD/'ARTIFACT_SHA256.json').read_text())['files']
for rel,h in sealed.items():
 if rel.startswith(('source/','inputs/')) or rel in ['arms/candidate/actor.pt','arms/candidate/runtime.pkl','arms/candidate/RESULT.json','CONFIG_LOCKED.json','EIG_COMPARISON.md','NEXT_RESEARCH_DECISION.md']:refs[str(OLD/rel)]=h
refs[str(OLD/'ARTIFACT_SHA256.json')]=sha(OLD/'ARTIFACT_SHA256.json');audit=R/'moe_hec_v2_internal_ablation'
for p in audit.rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts:refs[str(p)]=sha(p)
for p,h in refs.items():assert sha(p)==h,p
put(O/'evidence/PROTECTED_INPUTS_BEFORE.json',refs);files={str(p.relative_to(O)):sha(p) for folder in ['source','evidence','inputs'] for p in (O/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
for n in ['CONFIG_LOCKED.json','PROTOCOL_LOCK.md','JOINT_LEARNING_ALGORITHM.md','HEC_V2_ROUTING_DIAGNOSIS.md','GRADIENT_VALIDATION.md']:files[n]=sha(O/n)
put(O/'PRETRAIN_MANIFEST.json',dict(status='LOCKED_BEFORE_TRAINING',time=time.time(),git_commit=sync['github_commit'],files=files,protected_files=len(refs),scientific_optimizer_updates=0,two_fixed_arms=True));print('HECV21_PROTOCOL_LOCKED',sync['github_commit'])
