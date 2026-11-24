"""CPU-only common-module equality and forward timing;no training or test-bank access."""
from pathlib import Path
import json,runpy,time,torch
M=Path.home()/'Documents/objective_power_grid';s=runpy.run_path(str(M/'src/research/moe_sboed_v5_history_20261119/run.py'),run_name='v5_preflight')
torch.set_num_threads(2);start=time.monotonic();state=s['load_parent']('cpu');parent=state['algorithm'].actor_net
new=s['new_actor'](parent,'learned');template=s['StudyMoE'](parent,202611150101,'uniform');s['engine'].seed_all(s['cfg']()['initialization_seed']%(2**32));old=s['OriginalV4Actor'](template,'learned')
a,b=new.state_dict(),old.state_dict();common=[k for k in b if not k.startswith('action_head.')];assert all(torch.equal(a[k],b[k]) for k in common)
x=torch.zeros(2,117);x[:,0]=1
new.configure('frozen');new.eval()
with torch.no_grad():
 new(x);tick=time.monotonic()
 for _ in range(20):new(x)
 seconds=time.monotonic()-tick
result=dict(status='PASS',non_head_states_bitwise_equal_to_original_V4_initialization=True,common_state_tensors=len(common),
 V4_head_parameters=sum(p.numel() for p in old.action_head.parameters()),V5_head_parameters=sum(p.numel() for p in new.action_head.parameters()),
 new_history_parameters=sum(v.numel() for v in new.history_context.parameters()),expert_conditioning_parameters=[sum(v.numel() for v in e.condition.parameters()) for e in new.experts],CPU_batch2_forward_seconds=seconds/20,CPU_amortized_design_seconds=seconds/40,CPU_threads=2,cases='Two stage0 zero-history inputs',
 timing_scope='CPU wall,20 warmed forwards,no simulator/scorer;not GPU or general episode latency',seconds=time.monotonic()-start)
print(json.dumps(result,indent=2));s['put'](s['O']/'COMMON_MODULE_AND_COST_AUDIT.json',result)
