"""Test-process-only rollout accounting; never edits baseline source."""
import json,time
from pathlib import Path
START=time.monotonic();CURRENT='collection';ROWS={};ORIGINAL=None;ORIGINAL_STEP=None
def row():return ROWS.setdefault(CURRENT,dict(rollout_calls=0,trajectories=0,maximum_negatives=0,simulator_particle_stages=0,likelihood_evaluations=0,optimizer_updates=0,gpu_training_seconds=0))
def pytest_addoption(parser):parser.addoption('--v5-accounting-output',default=None)
def pytest_configure(config):
    global ORIGINAL,ORIGINAL_STEP
    import torch
    from src.research import engine
    ORIGINAL=engine.rollout
    def counted(*args,**kwargs):
        horizon=args[2] if len(args)>2 else kwargs['horizon']
        batch=args[3] if len(args)>3 else kwargs['batch']
        contrasts=args[5] if len(args)>5 else kwargs['contrasts']
        r=row()
        if contrasts>32 or r['trajectories']+batch>16:raise AssertionError('Test exceeds per-mode tiny smoke cap')
        r['rollout_calls']+=1;r['trajectories']+=batch;r['maximum_negatives']=max(r['maximum_negatives'],contrasts)
        r['simulator_particle_stages']+=batch*horizon*(contrasts+1)
        if kwargs.get('track_likelihood',True):r['likelihood_evaluations']+=batch*horizon*(contrasts+1)
        return ORIGINAL(*args,**kwargs)
    engine.rollout=counted
    ORIGINAL_STEP=torch.optim.Adam.step
    def counted_step(*args,**kwargs):
        r=row();r['optimizer_updates']+=1
        if r['optimizer_updates']>3:raise AssertionError('Optimizer smoke cap exceeded')
        return ORIGINAL_STEP(*args,**kwargs)
    torch.optim.Adam.step=counted_step
def pytest_runtest_setup(item):
    global CURRENT
    CURRENT=item.nodeid
def pytest_sessionfinish(session,exitstatus):
    path=session.config.getoption('--v5-accounting-output')
    if path:Path(path).write_text(json.dumps(dict(status='PASS' if exitstatus==0 else 'FAIL',wall_seconds=time.monotonic()-START,by_test=ROWS,posterior_samples_in_gaussian_toy=64,optimizer_updates=sum(r['optimizer_updates'] for r in ROWS.values()),gpu_training_seconds=0),indent=2)+'\n')
def pytest_unconfigure(config):
    if ORIGINAL is not None:
        from src.research import engine
        engine.rollout=ORIGINAL
    if ORIGINAL_STEP is not None:
        import torch
        torch.optim.Adam.step=ORIGINAL_STEP
