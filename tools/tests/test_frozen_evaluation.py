import json,hashlib
from pathlib import Path
import pytest
from src.research.frozen_evaluation import check_gate

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fixture(tmp_path):
 cp=tmp_path/'model.pt';cp.write_bytes(b'fixed policy');source=tmp_path/'source.json';source.write_text('{}');protocol=tmp_path/'protocol.json';protocol.write_text('{}');gate=tmp_path/'gate.json';data=dict(status='frozen_ready_for_final_evaluation',sealed_files={str(protocol):sha(protocol)},source_manifest_sha256=sha(source),policies=[dict(method='moe',seed=103,checkpoint=str(cp),checkpoint_sha256=sha(cp))]);gate.write_text(json.dumps(data));return cp,source,protocol,gate

def test_frozen_gate_accepts_exact_policy(tmp_path):
 cp,source,protocol,gate=fixture(tmp_path);assert check_gate(gate,cp,'moe',103,source)['status']=='frozen_ready_for_final_evaluation'
@pytest.mark.parametrize('change',['checkpoint','protocol','manifest','seed','method','status'])
def test_frozen_gate_rejects_drift(tmp_path,change):
 cp,source,protocol,gate=fixture(tmp_path);seed=103;method='moe'
 if change in ['checkpoint','protocol','manifest']:{'checkpoint':cp,'protocol':protocol,'manifest':source}[change].write_bytes(b'changed')
 if change=='seed':seed=104
 if change=='method':method='resmoe'
 if change=='status':
  data=json.loads(gate.read_text());data['status']='pending';gate.write_text(json.dumps(data))
 with pytest.raises(ValueError):check_gate(gate,cp,method,seed,source)
