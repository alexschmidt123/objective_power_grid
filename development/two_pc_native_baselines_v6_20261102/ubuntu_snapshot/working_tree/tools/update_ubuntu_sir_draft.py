from pathlib import Path
import yaml
root=Path(__file__).resolve().parents[1]
c=root/'experiments/10162026/10162026_ubuntu_sir_t5_seed101'
assert not (c/'source_manifest.json').exists(), 'Cannot edit frozen configuration'
for name in ['sir_1m.yaml','sir_500k.yaml']:
    p=c/name;cfg=yaml.safe_load(p.read_text())
    cfg['research']['journal_protocol'].update(version='validation_convergence_v1',
        execution_protocol='sir_fixed_budget_execution_v1')
    p.write_text(yaml.safe_dump(cfg,sort_keys=False))
failed=c/'readiness/native_smoke'
if failed.exists():failed.rename(c/'readiness/native_smoke_dispatch_failure')
