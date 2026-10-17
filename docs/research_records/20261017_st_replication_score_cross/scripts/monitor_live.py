import json,time
from pathlib import Path
C=Path('/home/grads/g/g.lin/Documents/objective_power_grid/experiments/10172026/10172026_st_replication_score_reverse_cross')
s=json.loads((C/'status.json').read_text());out={'status':s,'checked_unix':time.time(),'completed':sorted(p.stem for p in (C/'completion').glob('*.json'))}
name=s.get('phase','')
if name.startswith('moe_seed'):
 p=C/'runs'/name/'models/late_diagnostics/training_trace.jsonl'
 if p.exists():
  with p.open('rb') as f:
   f.seek(0,2);n=f.tell();f.seek(max(0,n-131072));lines=f.read().splitlines()
  for line in reversed(lines):
   try:d=json.loads(line);break
   except json.JSONDecodeError:continue
  out['primary_episodes']=d['primary_episodes'];out['gamma']=d['gamma'];out['last_batch_expert_usage']=d['training_expert_usage'];out['selector_saturation']=d['saturated_bit_fraction']
  elapsed=time.time()-s['started_utc'];out['phase_elapsed_minutes']=elapsed/60;out['rough_training_remaining_minutes']=elapsed*(1000000/d['primary_episodes']-1)/60
  val=p.parent/'checkpoints.json'
  if val.exists():
   recs=json.loads(val.read_text())
   if recs:out['latest_diagnostic_validation']={k:recs[-1][k] for k in ['primary_episodes','validation_eig','mc_se']}
print(json.dumps(out,indent=2))
