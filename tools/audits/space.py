"""Batched finite-loss planning screen with independent paired evaluation.

Lookahead is receding two-step planning, not a globally optimal T-step policy.
Fixed is calibrated greedy multistart, not an exhaustive optimal fixed design.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import torch
from src.objectives.mocu.train import _posterior_mocu_gpu


@dataclass
class AuditBudget:
    inner: int = 16
    outer: int = 8
    first_candidates: int = 6
    fixed_restarts: int = 4
    calibration: int = 128
    histories_per_batch: int = 8
    actions_per_batch: int = 6
    bootstrap: int = 2000


class SpacePlanner:
    def __init__(self, centres, required, grid, *, sigma, alpha=.05,
                 penalty=20., device='cpu', budget=None, objective='mocu'):
        if objective not in ('mocu', 'msc'): raise ValueError('Unknown objective')
        self.objective = objective
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.centres = self.tensor(centres)  # action x support x observation
        self.required = self.tensor(required)
        self.grid = self.tensor(grid)
        self.A, self.P, self.D = self.centres.shape
        self.sigma, self.alpha, self.penalty = float(sigma), float(alpha), float(penalty)
        self.budget = budget or AuditBudget()
        if sigma <= 0 or not np.isclose(penalty, 1/alpha):
            raise ValueError('Require positive Gaussian noise and penalty=1/alpha')
        if min(self.budget.inner, self.budget.outer, self.budget.first_candidates,
               self.budget.histories_per_batch, self.budget.actions_per_batch,
               self.budget.fixed_restarts, self.budget.calibration) < 1:
            raise ValueError('All audit budgets must be positive')
        # Include actions that differ in full vector response, not only immediate
        # information value. This reduces top-k pruning of complementary probes.
        features = self.centres - self.centres.mean(dim=1, keepdim=True)
        features = features.flatten(1)
        chosen = [int(features.square().sum(1).argmax())]
        distances = (features-features[chosen[0]]).square().sum(1)
        for _ in range(min(3, self.A-1)):
            distances[chosen] = -1
            action = int(distances.argmax())
            chosen.append(action)
            distances = torch.minimum(distances, (features-features[action]).square().sum(1))
        self.diverse = chosen

    def structure_summary(self):
        """Response redundancy diagnostics, never pass/fail criteria."""
        means = self.centres.cpu().numpy()
        features = (means-means.mean(axis=1, keepdims=True)).reshape(self.A, -1)/self.sigma
        energy = np.linalg.svd(features, compute_uv=False)**2
        rank = float(energy.sum()**2 / np.square(energy).sum()) if energy.sum()>0 else 0.
        norms = np.linalg.norm(features, axis=1)
        unit = features/np.maximum(norms[:, None], 1e-30)
        correlations = (unit@unit.T)[np.triu_indices(self.A, 1)]
        levels, counts = np.unique(self.required.cpu().numpy(), return_counts=True)
        return {'control_levels': levels.tolist(), 'control_mass': (counts/self.P).tolist(),
                'action_response_effective_rank': rank,
                'median_action_response_cosine': float(np.median(correlations)) if len(correlations) else None,
                'fraction_action_pairs_cosine_above_0995': float(np.mean(correlations>.995)) if len(correlations) else None,
                'informative_observation_coordinates': int(np.count_nonzero(means.var(axis=1).max(axis=0)>1e-20)),
                'rms_signal_to_noise_by_action': (norms/np.sqrt(self.P*self.D)).tolist(),
                'interpretation': 'correlation/rank diagnose redundancy; they do not bound decision complementarity'}

    def tensor(self, x):
        return torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def prior(self, n=1):
        return torch.full((n, self.P), -np.log(self.P), dtype=self.dtype, device=self.device)

    def risk(self, logw):
        shape = logw.shape[:-1]
        loss, control, _ = _posterior_mocu_gpu(
            logw.reshape(-1, self.P), self.required, self.grid, alpha=self.alpha,
            margin=0., undercontrol_penalty=self.penalty, violation_penalty=0.,
            robust_rule='quantile', snap_up=True, objective=self.objective)
        return loss.reshape(shape), control.reshape(shape)

    def update(self, logw, actions, observations):
        means = self.centres[actions]  # batch x support x observation
        return logw - ((observations[:, None, :]-means)/self.sigma).square().sum(-1)/2

    def fantasies(self, logw, actions, count, seed):
        """Common particle uniforms and noise across candidate actions/history rows."""
        rng = np.random.default_rng(seed)
        weights = logw.softmax(-1)
        uniforms = self.tensor(rng.random(count)).expand(len(logw), -1).contiguous()
        particles = torch.searchsorted(weights.cumsum(-1).contiguous(), uniforms).clamp(max=self.P-1)
        means = self.centres[actions[:, :, None], particles[:, None, :]]
        observations = means + self.tensor(rng.normal(0, self.sigma, (count, self.D)))[None, None]
        support_means = self.centres[actions]
        residual = (observations[:, :, :, None, :]-support_means[:, :, None, :, :])/self.sigma
        return logw[:, None, None, :] - residual.square().sum(-1)/2

    @torch.no_grad()
    def one_step(self, logw, used, seed):
        scores = torch.empty((len(logw), self.A), dtype=self.dtype, device=self.device)
        for start in range(0, len(logw), self.budget.histories_per_batch):
            stop = min(start+self.budget.histories_per_batch, len(logw))
            for action in range(0, self.A, self.budget.actions_per_batch):
                end = min(action+self.budget.actions_per_batch, self.A)
                ids = torch.arange(action, end, device=self.device).expand(stop-start, -1)
                posterior = self.fantasies(logw[start:stop], ids, self.budget.inner, seed)
                scores[start:stop, action:end] = self.risk(posterior)[0].mean(-1)
        return scores.masked_fill(used, torch.inf)

    @torch.no_grad()
    def choose(self, logw, used, *, depth, seed, fixed_next=None):
        immediate = self.one_step(logw, used, seed)
        if depth == 1:
            return immediate.argmin(-1)
        k = min(self.budget.first_candidates, int((~used).sum(1).min()))
        lists = []
        for scores, unavailable in zip(immediate.cpu().numpy(), used.cpu().numpy()):
            # Reserve two candidate slots for diverse responses. Always retain
            # Myopic, and the frozen Fixed next action when feasible.
            order = np.argsort(scores).tolist()
            selected = order[:max(1, k-2)]
            extras = ([] if fixed_next is None else [int(fixed_next)]) + self.diverse + order
            for action in extras:
                if len(selected) == k: break
                if not unavailable[action] and action not in selected:
                    selected.append(action)
            lists.append(selected)
        candidates = torch.as_tensor(lists, device=self.device)
        values = torch.empty_like(candidates, dtype=self.dtype)
        for column in range(k):
            action = candidates[:, column]
            first = self.fantasies(logw, action[:, None], self.budget.outer, seed+100003)[:, 0]
            future_used = used.repeat_interleave(self.budget.outer, 0).clone()
            future_used[torch.arange(len(future_used), device=self.device),
                        action.repeat_interleave(self.budget.outer)] = True
            second = self.one_step(first.reshape(-1, self.P), future_used, seed+200003)
            # Select the second action on one fantasy set and estimate its
            # value on another, avoiding min-of-noisy-estimates optimism.
            selected = second.argmin(-1)
            independent = self.fantasies(first.reshape(-1, self.P), selected[:, None],
                                         self.budget.inner, seed+300007)
            values[:, column] = self.risk(independent)[0].mean(-1).reshape(len(logw), -1).mean(-1)
        return candidates.gather(1, values.argmin(-1)[:, None]).squeeze(1)

    @torch.no_grad()
    def fixed_sequence(self, horizon, seed):
        """Greedy multistart calibration; no evaluation observations are used."""
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, self.P, self.budget.calibration)
        # Key calibration noise by action rather than sequence position: reset
        # experiments then have exactly order-invariant cached log likelihoods.
        observations = self.centres[:, ids].permute(1, 0, 2) + self.tensor(
            rng.normal(0, self.sigma, (len(ids), self.A, self.D)))
        increments = -(observations[:, :, None]-self.centres[None]).square().sum(-1)/(2*self.sigma**2)

        def scores(logw):
            result = []
            for a in range(0, self.A, self.budget.actions_per_batch):
                result.append(self.risk(logw[:, None]+increments[:, a:a+self.budget.actions_per_batch])[0].mean(0))
            return torch.cat(result)

        first_scores = scores(self.prior(len(ids)))
        starts = list(dict.fromkeys([int(first_scores.argmin())] + self.diverse +
                                    first_scores.argsort().tolist()))[:self.budget.fixed_restarts]
        best = (float('inf'), None)
        for first in starts:
            seq = [first]
            logw = self.prior(len(ids)) + increments[:, first]
            for _ in range(1, horizon):
                costs = scores(logw); costs[seq] = torch.inf
                action = int(costs.argmin()); seq.append(action)
                logw = logw + increments[:, action]
            loss = float(self.risk(logw)[0].mean())
            if loss < best[0]: best = (loss, sorted(seq))
        return best[1], best[0]

    @torch.no_grad()
    def evaluate(self, centres, required, horizon, fixed, seed):
        """Independent off-support validation; common noise for every method."""
        centres, required = self.tensor(centres), self.tensor(required)  # system x action x obs
        n = len(required)
        noise = self.tensor(np.random.default_rng(seed).normal(
            0, self.sigma, (n, horizon, self.A, self.D)))
        rows = {}
        methods = ('no_probe', 'random', 'fixed', 'myopic', 'lookahead') if self.objective == 'msc' else ('fixed', 'myopic', 'lookahead')
        for method in methods:
            started = time.perf_counter()
            logw = self.prior(n)
            used = torch.zeros((n, self.A), dtype=torch.bool, device=self.device)
            sequences = []
            stage_scores = [self.risk(logw)[0].cpu().numpy()]
            random_rng = np.random.default_rng(seed+700001)
            for step in range(0 if method == "no_probe" else horizon):
                if method == 'random':
                    actions = torch.as_tensor([random_rng.choice(np.flatnonzero(~row)) for row in used.cpu().numpy()], device=self.device)
                elif method == 'fixed':
                    actions = torch.full((n,), fixed[step], dtype=torch.long, device=self.device)
                else:
                    depth = 2 if method == 'lookahead' and step < horizon-1 else 1
                    decision_seed = seed + 10000019 + step*7919
                    if step == 0:
                        actions = self.choose(logw[:1], used[:1], depth=depth,
                            seed=decision_seed, fixed_next=fixed[step]).expand(n)
                    else:
                        actions = self.choose(logw, used, depth=depth,
                            seed=decision_seed, fixed_next=fixed[step])
                idx = torch.arange(n, device=self.device)
                observation = centres[idx, actions] + noise[idx, step, actions]
                logw = self.update(logw, actions, observation)
                used[idx, actions] = True
                sequences.append(actions.cpu().numpy())
                stage_scores.append(self.risk(logw)[0].cpu().numpy())
            risk, control = self.risk(logw)
            realized = control + self.penalty*(required-control).clamp_min(0) - required
            score = control if self.objective == 'msc' else realized
            rows[method] = {'loss': score.cpu().numpy(), 'posterior_risk': risk.cpu().numpy(),
                'control': control.cpu().numpy(),
                'posterior_ess': (1/logw.softmax(-1).square().sum(-1)).cpu().numpy(), 'bank_coverage': (control+1e-12 >= required).cpu().numpy(),
                'stage_scores': np.stack(stage_scores, axis=1),
                'sequence': np.stack(sequences, axis=1) if sequences else np.empty((n,0),dtype=int), 'seconds': time.perf_counter()-started}
            print(f"[audit] T={horizon} seed={seed} {method}: "
                  f"loss={float(score.mean()):.6f}, seconds={rows[method]['seconds']:.1f}", flush=True)
        return rows


def paired_interval(differences, *, bootstrap=2000, seed=493):
    """Cluster by theta after averaging repeated audit-noise seeds."""
    values = np.asarray(differences, dtype=float).mean(axis=0)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist(),
            'sample_std_across_theta': float(values.std(ddof=1)), 'n_theta': len(values),
            'uncertainty_scope': 'paired theta-cluster bootstrap, conditional on banks and calibrated Fixed'}


def summarize_runs(runs, prior_risk, budget):
    out = {'methods': {}}
    for method in runs[0]:
        losses = np.stack([r[method]['loss'] for r in runs])
        out['methods'][method] = {'mean_loss': float(losses.mean()),
            'std_across_seed_means': float(losses.mean(1).std(ddof=1)) if len(runs)>1 else None,
            'bank_coverage': float(np.mean([r[method]['bank_coverage'] for r in runs])),
            'mean_posterior_risk': float(np.mean([r[method]['posterior_risk'] for r in runs])),
            'median_posterior_ess': float(np.median([r[method]['posterior_ess'] for r in runs])),
            'fraction_posterior_ess_below_2': float(np.mean([r[method]['posterior_ess']<2 for r in runs])),
            'control_levels': sorted(set(np.concatenate([r[method]['control'] for r in runs]).tolist())),
            'mean_unique_sequences': float(np.mean([len(np.unique(r[method]['sequence'], axis=0)) for r in runs]))}
    for key, baseline, policy in (('adaptive_gain', 'fixed', 'lookahead'),
                                  ('myopic_adaptive_gain', 'fixed', 'myopic'),
                                  ('nonmyopic_gain', 'myopic', 'lookahead')):
        gap = np.stack([r[baseline]['loss']-r[policy]['loss'] for r in runs])
        out[key] = paired_interval(gap, bootstrap=budget.bootstrap)
        out[key]['fraction_of_prior_bayes_risk'] = out[key]['mean']/prior_risk if prior_risk>1e-12 else None
    return out
