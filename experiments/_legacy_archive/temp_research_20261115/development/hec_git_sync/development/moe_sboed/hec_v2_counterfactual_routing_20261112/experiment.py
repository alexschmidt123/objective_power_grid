"""One bounded frozen-V2 routing experiment. All outputs stay outside Git."""
import argparse
import copy
import gc
import hashlib
import io
import json
import os
import pathlib
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import torch
from routing import (gate_bank, fuse_options, routing_context, fit_router,
                     signal, terminal_scores, paired_stats, holm)

ROOT = pathlib.Path.home() / 'Documents/objective_power_grid_temp'
BASE = ROOT / 'scratch_runs/moe_hec_v2_icml_20261109'
ACTOR_SHA = '90797b277846dd1a5d7de3764f6fb021047530d1fbea7db859f938ef59b73108'
RUNTIME_SHA = 'd38c09167c4936a6a0660d16079ecc3d5e1108c32bdbc840cc8851c1ef7266b1'


def sha(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def load_frozen(device):
    os.environ['MOE_V2_RUN_ROOT'] = str(BASE)
    sys.path.insert(0, str(BASE / 'source'))
    from native_common import cloudpickle
    import restore_v1  # registers frozen module paths required by trusted local runtime
    import evaluate_v2
    actor_path = BASE / 'arms/candidate/actor.pt'
    runtime_path = BASE / 'arms/candidate/runtime.pkl'
    assert sha(actor_path) == ACTOR_SHA and sha(runtime_path) == RUNTIME_SHA
    old = torch.storage._load_from_bytes
    torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location=device, weights_only=False)
    try:
        with runtime_path.open('rb') as f:
            state = cloudpickle.load(f)
    finally:
        torch.storage._load_from_bytes = old
    actor = state['algorithm'].actor_net.eval().requires_grad_(False)
    checkpoint = torch.load(actor_path, map_location=device, weights_only=True)['actor_state']
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in checkpoint.items())
    del state, checkpoint
    gc.collect()
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return actor


def packed(actions, observations, n):
    """Observed prefix only. Variable-width histories are zero padded."""
    b = actions.shape[0]
    x = actions.new_zeros(b, 117, dtype=torch.float32)
    x[:, n] = 1
    if n:
        x[:, 30:30 + 2*n] = actions[:, :n].reshape(b, 2*n).float()
        x[:, 88:88+n] = observations[:, :n].float()
    return x


def random_bank(n, contrasts, seed, device):
    rng = np.random.default_rng(seed)
    return (torch.tensor(rng.standard_normal((n, 4)), device=device),
            torch.tensor(rng.standard_normal((n, 30)), device=device),
            torch.tensor(rng.standard_normal((contrasts, 4)), device=device))


def guard(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('Predeclared wall budget reached; partial artifacts preserved')


@torch.no_grad()
def rollout(actor, theta, noise, mode='original', router=None, deadline=float('inf')):
    b = len(theta)
    actions = torch.zeros(b, 30, 2, device=theta.device)
    observations = torch.zeros(b, 30, dtype=torch.float64, device=theta.device)
    counts = torch.zeros(8, dtype=torch.long, device=theta.device)
    for t in range(30):
        guard(deadline)
        x = packed(actions, observations, t)
        if mode in ['candidate', 'stage']:
            _, info = actor(x, collect=True)
            gates = gate_bank(info['gates'])
            choices = fuse_options(actor, x, info, gates)
            context = routing_context(actor, x, info, choices)
            chosen = router.select(context)
            action = choices[torch.arange(b, device=theta.device), chosen]
            counts += torch.bincount(chosen, minlength=8)
        elif mode == 'uniform':
            action = actor(x, routing_override=x.new_full((b, 4), .25))
        elif mode.startswith('expert'):
            k = int(mode[-1])-1
            gates = x.new_zeros(b, 4)
            gates[:, k] = 1
            action = actor(x, routing_override=gates)
        elif mode == 'original':
            action = actor(x)
        else:
            raise ValueError(mode)
        if not torch.isfinite(action).all() or not bool((action.abs() <= 4.000001).all()):
            raise ValueError('Nonfinite or infeasible action')
        actions[:, t] = action
        observations[:, t] = signal(theta.reshape(b, 2, 2), action.double()) + .5*noise[:, t]
    return actions, observations, counts


@torch.no_grad()
def make_teacher(actor, n, contrasts_count, seed, device, output, batch, deadline):
    """One independent complete episode/history per row; full terminal prior-sPCE labels.

    Eight gate interventions at one randomly chosen decision; then unchanged V2.
    All branches share prefix, latent draw, future noise and negative contrast bank.
    No recorded hidden parameter enters a routing context.
    """
    theta, noise, contrasts = random_bank(n, contrasts_count, seed, device)
    rng = np.random.default_rng(seed + 1)
    stages = torch.tensor(rng.integers(0, 30, size=n), device=device)
    contexts, targets = [], []
    for start in range(0, n, batch):
        guard(deadline)
        th, eps, ts = theta[start:start+batch], noise[start:start+batch], stages[start:start+batch]
        b = len(th)
        original_actions, original_observations, _ = rollout(actor, th, eps, deadline=deadline)
        xx = torch.cat([packed(original_actions[j:j+1], original_observations[j:j+1], int(ts[j]))
                        for j in range(b)])
        _, info = actor(xx, collect=True)
        options = gate_bank(info['gates'])
        designs = fuse_options(actor, xx, info, options)
        context = routing_context(actor, xx, info, designs)
        assert context.shape == (b, 182)
        # Every context has eight complete counterfactual branches, including a
        # branch that must reproduce the original trajectory exactly.
        actions = original_actions[:, None].expand(-1, 8, -1, -1).clone().reshape(b*8, 30, 2)
        observations = original_observations[:, None].expand(-1, 8, -1).clone().reshape(b*8, 30)
        branch_theta = th[:, None].expand(-1, 8, -1).reshape(b*8, 4)
        branch_noise = eps[:, None].expand(-1, 8, -1).reshape(b*8, 30)
        branch_stage = ts[:, None].expand(-1, 8).reshape(-1)
        branch_design = designs.reshape(b*8, 2)
        # Option zero is exactly the saved reference trajectory. Do not rerun
        # it in a different GEMM batch and let roundoff compound through history.
        nonreference = torch.arange(b*8, device=device) % 8 != 0
        torch.testing.assert_close(designs[:,0], original_actions[torch.arange(b,device=device),ts],
                                   atol=2e-5,rtol=1e-5)
        for t in range(30):
            guard(deadline)
            at, later = (branch_stage == t) & nonreference, (branch_stage < t) & nonreference
            actions[at, t] = branch_design[at]
            if later.any():
                state = packed(actions[later], observations[later], t)
                actions[later, t] = actor(state)
            changed = (branch_stage <= t) & nonreference
            observations[changed, t] = signal(branch_theta[changed].reshape(-1, 2, 2),
                                               actions[changed, t].double()) + .5*branch_noise[changed, t]
        assert torch.equal(actions.reshape(b,8,30,2)[:,0], original_actions)
        assert torch.equal(observations.reshape(b,8,30)[:,0], original_observations)
        scores = terminal_scores(actions, observations, branch_theta, contrasts).reshape(b, 8)
        contexts.append(context.cpu())
        targets.append(scores.float().cpu())
        print(json.dumps(dict(phase=output.stem, histories=min(start+batch,n), total=n,
                              paired_spread=float((scores-scores[:,1:2]).std()))), flush=True)
    data = dict(context=torch.cat(contexts), returns=torch.cat(targets), stages=stages.cpu(),
                seed=seed, contrasts=contrasts_count,
                target='single-decision gate intervention then frozen V2; complete terminal prior-sPCE',
                latent_policy_inputs=False)
    torch.save(data, output)
    return data


@torch.no_grad()
def evaluate(actor, router, stage_router, n, negatives, seed, device, batch, output, deadline):
    theta, noise, contrasts = random_bank(n, negatives, seed, device)
    names = ['candidate','original','uniform','stage','expert1','expert2','expert3','expert4']
    all_scores, records = {}, {}
    for name in names:
        tick, cpu = time.monotonic(), time.process_time()
        values, counts = [], torch.zeros(8, dtype=torch.long, device=device)
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        for start in range(0, n, batch):
            guard(deadline)
            th, eps = theta[start:start+batch], noise[start:start+batch]
            actions, observations, cc = rollout(actor, th, eps, name,
                                                stage_router if name == 'stage' else router, deadline)
            values.append(terminal_scores(actions, observations, th, contrasts).cpu())
            counts += cc
        score = torch.cat(values)
        all_scores[name] = score
        records[name] = dict(**paired_stats(score), wall_seconds=time.monotonic()-tick,
                             process_CPU_seconds=time.process_time()-cpu,
                             observations=30*n, negative_density_terms=n*30*negatives,
                             GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                             gate_pattern_counts=counts.cpu().tolist())
        np.save(output / (name+'_scores.npy'), score.numpy())
        print('EVALUATED', name, json.dumps(records[name]), flush=True)
        write(output/'EVALUATION_PROGRESS.json', records)
    comparisons = {name: paired_stats(all_scores['candidate']-all_scores[name])
                   for name in names if name != 'candidate'}
    holm(comparisons)
    return dict(methods=records, paired_candidate_minus=comparisons,
                all_comparisons_positive_corrected=all(v['mean']>0 and v['Holm_p']<.05 for v in comparisons.values()),
                one_frozen_lineage=True, final_bank_not_used_for_training_or_selection=True,
                finite_contrast_lower_bound=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--run', action='store_true', help='Explicitly execute the bounded pilot')
    parser.add_argument('--smoke', action='store_true', help='Tiny integration only; never scientific results')
    args = parser.parse_args()
    if not args.run:
        parser.error('--run is required; importing or inspection never launches computation')
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError('New isolated output required; no automatic resume or overwrite')
    if ROOT/'scratch_runs' not in output.parents:
        raise ValueError('Scientific output must live under LabPC TEMP/scratch_runs')
    output.mkdir(parents=True)
    config = dict(train_histories=512, validation_histories=128, final_trajectories=512,
                  contrasts=1000000, batch=16, ridge_penalty=10., wall_budget_seconds=1800,
                  seed=202611125001, option_count=8, changes='coordination selector only; frozen full V2',
                  smoke=args.smoke, no_budget_extension=True,
                  criterion='candidate beats original, uniform, stage and all four one-hot pathways with Holm7',
                  single_pathway_not_standalone_trained_expert=True,
                  validation='diagnostic only; no checkpoint selection or architecture search',
                  source_commit=subprocess.check_output(['git','-C',str(pathlib.Path(__file__).parent),
                                                       'rev-parse','HEAD'],text=True).strip())
    if args.smoke:
        config.update(train_histories=4, validation_histories=2, final_trajectories=2,
                      contrasts=31, batch=2, wall_budget_seconds=180)
    write(output/'CONFIG.json', config)
    protected = {str(p): sha(p) for p in [BASE/'arms/candidate/actor.pt', BASE/'arms/candidate/runtime.pkl',
                                           *sorted((BASE/'source').glob('*.py'))]}
    source = {str(p): sha(p) for p in pathlib.Path(__file__).parent.glob('*.py')}
    write(output/'PRE_RUN_MANIFEST.json',dict(protected=protected,source=source))
    os.environ['MOE_V2_RUN_ROOT'] = str(BASE)
    sys.path.insert(0, str(BASE/'source'))
    from native_common import original
    original.setup()
    while True:
        try:
            leases = original.acquire()
            break
        except BlockingIOError:
            write(output/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU',pid=os.getpid()))
            time.sleep(30)
    device = 'cuda'
    start = time.monotonic()
    deadline = start + config['wall_budget_seconds']
    write(output/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),time=time.time()))
    outcome = dict(status='FAILED')
    try:
        actor = load_frozen(device)
        before_state = {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
        train = make_teacher(actor,config['train_histories'],config['contrasts'],config['seed'],
                             device,output/'teacher_train.pt',config['batch'],deadline)
        validation = make_teacher(actor,config['validation_histories'],config['contrasts'],config['seed']+100,
                                  device,output/'teacher_validation.pt',config['batch'],deadline)
        router = fit_router(train['context'].to(device),train['returns'].to(device),ridge_penalty=config['ridge_penalty'])
        stage_router = fit_router(train['context'].to(device),train['returns'].to(device),stage_only=True,
                                  ridge_penalty=config['ridge_penalty'])
        torch.save(dict(router_state=router.state_dict(),stage_state=stage_router.state_dict(),config=config),
                   output/'router.pt')
        with torch.no_grad():
            context=validation['context'].to(device)
            choice=router.select(context).cpu()
            scores=validation['returns']
            selected=scores[torch.arange(len(scores)),choice]
            write(output/'VALIDATION.json',dict(candidate_minus_original=paired_stats(selected-scores[:,0]),
                                               candidate_minus_uniform=paired_stats(selected-scores[:,1]),
                                               selection_counts=torch.bincount(choice,minlength=8).tolist(),
                                               diagnostic_only=True))
        result=evaluate(actor,router,stage_router,config['final_trajectories'],config['contrasts'],
                        config['seed']+200,device,config['batch'],output,deadline)
        assert all(torch.equal(v,actor.state_dict()[k].cpu()) for k,v in before_state.items())
        result['frozen_V2_state_unchanged']=True
        write(output/'RESULTS.json',result)
        outcome=dict(status='SMOKE_COMPLETE' if args.smoke else 'COMPLETE',seconds=time.monotonic()-start,
                     routing_superiority_established=False if args.smoke else result['all_comparisons_positive_corrected'])
    except Exception as exc:
        import traceback
        traceback.print_exc()
        outcome=dict(status='FAILED',error=type(exc).__name__,message=str(exc),seconds=time.monotonic()-start)
    finally:
        unchanged=all(sha(pathlib.Path(p))==h for p,h in protected.items())
        sources_unchanged=all(sha(pathlib.Path(p))==h for p,h in source.items())
        outcome.update(historical_artifacts_preserved=unchanged,run_source_unchanged=sources_unchanged)
        if not unchanged or not sources_unchanged:
            outcome['status']='PRESERVATION_FAILED'
        write(output/'COMPLETION.json',outcome)
        write(output/'STATUS.json',outcome)
        print('EXPERIMENT_END',json.dumps(outcome),flush=True)
    if outcome['status'] not in ['COMPLETE','SMOKE_COMPLETE']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
