from pathlib import Path
import json,hashlib
import pytest
import copy,torch
from src.research.project_scope import resolve,label,describe,guard
from src.research.moe_v5_experiment import verified_reference,load_policy

ROOT=Path(__file__).resolve().parents[2]
C=ROOT/'scratch_runs/10312026_054200_moe_sboed_v5_ubuntu'

def test_version_registration_is_development_only_and_generic_launch_rejected():
    assert resolve('MoE-sBOED-v5')=='moe_sboed_v5' and label('moe_sboed_v5')=='MoE-sBOED-v5'
    row=next(x for x in describe() if x['id']=='moe_sboed_v5')
    assert row['location']=='DEVELOPMENT_UNVALIDATED_DIAGNOSTICS_ONLY'
    with pytest.raises(ValueError):guard(['--methods','moe_sboed_v5'])
    assert guard(['--moe-sboed-v5-development','--methods','moe_sboed_v5'])

def test_verified_frozen_reference_and_original_checkpoint_loading():
    ref,receipt=verified_reference(C/'labpc_v4_reference')
    p,cfg,path=load_policy(ref,101)
    assert p.counts()['total_parameters']==8434 and cfg['horizon']==30
    assert receipt['status']=='PASS' and path.is_file()

def test_baseline_and_old_model_files_are_unchanged():
    before=json.loads((C/'source_before_v5.json').read_text())
    allowed={'src/online_cli.py','src/research/project_scope.py'}
    changed=[]
    for name,h in before.items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=h:changed.append(name)
    assert set(changed)==allowed

def test_zero_switch_native_optimizer_update_and_checkpoint_roundtrip(tmp_path):
    from src.research.deepseek_sboed import DeepSeekPolicy
    from src.research.benchmarks import make_benchmark
    from src.research.engine import rollout
    from src.research.moe_v5_conditional import combined_objective
    ref,_=verified_reference(C/'labpc_v4_reference');a,cfg,path=load_policy(ref,101)
    saved=torch.load(path,map_location='cpu',weights_only=False);b=copy.deepcopy(a)
    pairs=[]
    for policy in [a,b]:
        optimizer=torch.optim.Adam(policy.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']))
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        policy.train();pairs.append((policy,optimizer))
    for index,(policy,optimizer) in enumerate(pairs):
        r=rollout(make_benchmark(cfg['benchmark']),policy,30,2,596774,16)
        main=-r['info'][:,-1].mean()+policy.balance_loss()
        loss=combined_objective(main,0,lambda: (_ for _ in ()).throw(AssertionError('Unexpected auxiliary draw'))) if index else main
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),5,error_if_nonfinite=True);optimizer.step()
    for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    cp=tmp_path/'zero_switch_test_only.pt';torch.save(dict(state_dict=b.state_dict(),WORKABILITY_ONLY=True),cp)
    clone=DeepSeekPolicy();clone.load_state_dict(torch.load(cp,weights_only=False)['state_dict'],strict=True)
    for p,q in zip(b.parameters(),clone.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
