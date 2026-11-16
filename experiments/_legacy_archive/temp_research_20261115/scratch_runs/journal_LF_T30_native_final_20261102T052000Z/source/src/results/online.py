"""Read completed online cost-utility runs without changing historical files."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def cost_run_identity(directory):
    directory=Path(directory)
    path=directory/'run_config.json'
    if not path.is_file():return None
    doc=json.loads(path.read_text())
    settings=doc.get('settings',{})
    if settings.get('objective')!='cost_utility':return None
    sigma=format(float(settings['noise_sigma']),'.12g').replace('.','p')
    return dict(config=doc['physical_config']['system']['name'],
        experiment_type='cost_utility',N_obs=int(settings['N_obs']),
        sigma_token=sigma,T=int(settings['T']),stamp=directory.name,
        seed=int(settings['seed']))


def cost_run_rows(directory):
    directory=Path(directory)
    if cost_run_identity(directory) is None:return []
    if not (directory/'completion.json').is_file() or not (directory/'exit_code').is_file():return []
    if (directory/'exit_code').read_text().strip()!='0':return []
    completion=json.loads((directory/'completion.json').read_text())
    rollouts=json.loads((directory/'rollouts.json').read_text())
    if len(rollouts)!=completion['records']:raise ValueError('Incomplete cost-utility rollout records')
    summaries={r['method']:r for r in json.loads((directory/'summary.json').read_text())}
    groups=defaultdict(list)
    for row in rollouts:groups[(row['method'],row['evaluation_seed'])].append(row)
    expected={(m,s) for m in completion['methods'] for s in completion['evaluation_seeds']}
    if set(groups)!=expected:raise ValueError('Missing method/seed cost-utility records')
    output=[]
    for (method,seed),rows in sorted(groups.items()):
        values=np.array([r['cost_utility'] for r in rows])
        if not np.isfinite(values).all():raise ValueError('Nonfinite cost utility')
        output.append(dict(method=method,eval_seed=seed,mean_cost_utility=float(values.mean()),
            mean_u_ctrl=float(np.mean([r['u_ctrl'] for r in rows])),
            safety_rate=float(np.mean([r['safe'] for r in rows])),
            training_time_seconds=summaries[method]['training_seconds'],
            online_seconds_per_rollout=float(np.mean([sum(r['decision_seconds_per_stage'])+
                r.get('terminal_controller_seconds',0.) for r in rows]))))
    return output
