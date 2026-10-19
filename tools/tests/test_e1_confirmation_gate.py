import json
from pathlib import Path
import pytest
from src.research.e1_confirmation import sha,verify_gate

def fixture(tmp_path):
 code=tmp_path/'source.py';code.write_text('frozen code')
 cp=tmp_path/'policy.pt';cp.write_bytes(b'frozen parameters')
 bank=tmp_path/'bank.npy';bank.write_bytes(b'untouched bank')
 gate=tmp_path/'gate.json';g=dict(status='frozen_ready_for_confirmation',evaluation_seed=172983421,sealed_files={str(bank):sha(bank)},source_sha256={'source.py':sha(code)},policies=[dict(method='dad',seed=101,source_run=str(tmp_path),checkpoint=str(cp),checkpoint_sha256=sha(cp))]);gate.write_text(json.dumps(g))
 return gate,g,cp,bank,code

def test_registered_unchanged_policy_passes(tmp_path):
 gate,g,*_=fixture(tmp_path);actual,e=verify_gate(gate,'dad',101,tmp_path,tmp_path);assert actual==g

@pytest.mark.parametrize('part',['policy','bank','source'])
def test_mutation_rejected_before_scoring(tmp_path,part):
 gate,g,cp,bank,code=fixture(tmp_path);{'policy':cp,'bank':bank,'source':code}[part].write_bytes(b'changed')
 with pytest.raises(ValueError):verify_gate(gate,'dad',101,tmp_path,tmp_path)

@pytest.mark.parametrize('seed',[1001,8831001])
def test_consumed_seed_is_rejected(tmp_path,seed):
 gate,g,*_=fixture(tmp_path);g['evaluation_seed']=seed;gate.write_text(json.dumps(g))
 with pytest.raises(ValueError):verify_gate(gate,'dad',101,tmp_path,tmp_path)
