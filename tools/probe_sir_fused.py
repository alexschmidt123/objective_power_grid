"""Bounded T=5 execution comparison, not a training experiment."""
import copy, json, time
from pathlib import Path
import torch, yaml
from src.research.benchmarks import SIR
from src.research.engine import rollout
from src.research.sir_dad import SIRSumMLP

OUT=Path('experiments/10162026/10162026_ubuntu_sir_fused_probe')
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
raw=yaml.safe_load(Path('configs/sir_ode_research.yaml').read_text())
records=[]
def save():
    (OUT/'probe.json').write_text(json.dumps(records,indent=2)+'\n')
with torch.device('cuda:0'):
    theta=torch.tensor([[.5,.1],[1.7,.4],[.08,.03]],dtype=torch.float64)
    initial=SIR(raw).initial((3,))
    base_action=torch.tensor([[25.],[98.],[.11]],dtype=torch.float64)
    reference=None
    for backend in ['eager','fused_rk4','fused_rk4']:
        cfg=copy.deepcopy(raw);cfg['sir_ode']['execution_backend']=backend
        b=SIR(cfg);a=base_action.clone().requires_grad_()
        th=theta.clone().requires_grad_()
        torch.cuda.synchronize();t=time.perf_counter()
        mean,state=b.propagate(th,initial,a)
        grads=torch.autograd.grad(mean.sum(),(a,th))
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        values=[state.detach(),*[g.detach() for g in grads]]
        if reference is None:reference=values
        else:
            for actual,expected in zip(values,reference):
                torch.testing.assert_close(actual,expected,rtol=1e-7,atol=1e-8)
        row=dict(test='state_and_action_parameter_gradients',backend=backend,
                 seconds=seconds,max_abs_errors=[float((v-r).abs().max()) for v,r in zip(values,reference)])
        records.append(row);print(json.dumps(row),flush=True);save()
    # Full T=5 rollout checks use identical policy, parameters and random stream.
    torch.manual_seed(101);policy=SIRSumMLP(5)
    reference=None
    for backend in ['eager','fused_rk4','fused_rk4']:
        cfg=copy.deepcopy(raw);cfg['sir_ode']['execution_backend']=backend
        b=SIR(cfg);policy.zero_grad(set_to_none=True)
        torch.cuda.synchronize();t=time.perf_counter()
        result=rollout(b,policy,5,4,101901,8)
        loss=-result['info'][:,-1].mean();loss.backward()
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        values=[torch.stack(result['actions']),torch.stack(result['observations']),
                result['info'],*[p.grad for p in policy.parameters() if p.grad is not None]]
        values=[v.detach().clone() for v in values]
        if reference is None:reference=values
        else:
            for actual,expected in zip(values,reference):
                torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
        row=dict(test='T5_rollout_loss_and_policy_gradients',backend=backend,
                 batch=4,contrasts=8,seconds=seconds,loss=float(loss.detach()),
                 max_abs_error=max(float((v-r).abs().max()) for v,r in zip(values,reference)))
        records.append(row);print(json.dumps(row),flush=True);save()
    for backend in ['eager','fused_rk4','fused_rk4']:
        cfg=copy.deepcopy(raw);cfg['sir_ode']['execution_backend']=backend
        b=SIR(cfg);policy.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
        result=rollout(b,policy,5,32,101902,1024)
        loss=-result['info'][:,-1].mean();loss.backward()
        torch.cuda.synchronize();seconds=time.perf_counter()-t
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
        row=dict(test='training_shape_T5_forward_backward',backend=backend,batch=32,
                 contrasts=1024,seconds=seconds,loss=float(loss.detach()),
                 peak_memory_bytes=torch.cuda.max_memory_allocated(),
                 extrapolated_1m_optimization_hours=seconds*1000000/32/3600)
        records.append(row);print(json.dumps(row),flush=True);save()
(OUT/'probe_complete.json').write_text(json.dumps(dict(status='passed',reserved_test_seed_used=False)))
