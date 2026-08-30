#!/usr/bin/env python3
"""Search the 281-duration IEEE14 bank for six-duration MOCU spaces.

This is a method-independent structural audit.  It reuses the dense physical
bank, searches reproducibly (not exhaustively) over C(281, 6), and ranks sets
by adaptive and non-myopic reductions in terminal control loss.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.banks.audit as bank_audit
from src.banks.audit import screen_t2_adaptive_room
from src.control.posterior_ctrl import normalize_log_weights, posterior_safe_u_ctrl
from src.layout import make_experiment_dir_name
from src.observations.likelihood import evenly_spaced_indices
from tools.bank_sweeps.search_ieee9_eig_duration_sets_full import (
    candidate_key, proxy_scores, proxy_value,
)
from tools.bank_sweeps.sweep_ieee9_eig_duration_sets import (
    Catalog, load_catalog, mean_lcb, resolve_pool_actions, write_csv,
)


def install_safety_aware_mocu_terminal(*, undercontrol_penalty: float,
                                       violation_penalty: float) -> None:
    """Make the structural planner use the exact train/eval terminal cost."""
    def terminal_loss(U, log_w, *, alpha, margin, grid):
        w = normalize_log_weights(log_w)
        u = float(posterior_safe_u_ctrl(
            U, w, alpha, margin=margin, u_grid=grid, snap_up=True,
        ))
        shortfall = np.maximum(np.asarray(U, dtype=np.float64) - u, 0.0)
        realized = (u + float(undercontrol_penalty) * shortfall
                    + float(violation_penalty) * (shortfall > 0.0))
        return float(np.sum(w * (realized - np.asarray(U, dtype=np.float64))))
    # screen_t2_adaptive_room resolves this module global at runtime.
    bank_audit._terminal_u = terminal_loss


def load_support(bank_dir: Path, action_ids: np.ndarray, n_obs: int,
                 support_size: int, seed: int):
    full = np.load(bank_dir / "train" / "delta_f.npy", mmap_mode="r")
    u_full = np.load(bank_dir / "train" / "psi_star.npy", mmap_mode="r")
    obs = evenly_spaced_indices(full.shape[2], n_obs)
    rng = np.random.default_rng(seed)
    particles = np.sort(rng.choice(full.shape[0], min(support_size, full.shape[0]), replace=False))
    selected = np.asarray(full[np.ix_(particles, action_ids, obs)], dtype=np.float32)
    return np.transpose(selected, (1, 0, 2)), np.asarray(u_full[particles], dtype=np.float64), obs


def audit_one(key, *, durations, centres, U, n_buses, sigma, seed, outer, inner,
              alpha, margin, u_grid, min_branch_share):
    ids = np.concatenate([np.arange(i*n_buses, (i+1)*n_buses) for i in key])
    Y = np.transpose(centres[ids], (1, 0, 2))
    result = screen_t2_adaptive_room(
        Y=Y, U=U, sigma=sigma, alpha=alpha, margin=margin, u_grid=u_grid,
        n_outer=outer, n_inner=inner, top_k=min(12, len(ids)), seed=seed,
    )
    masses = result.get("second_action_mass", {})
    duration_mass = np.zeros(6, dtype=np.float64)
    for action, mass in masses.items():
        duration_mass[int(action) // n_buses] += float(mass)
    meaningful = int(np.sum(duration_mass >= min_branch_share))
    myopic_duration = int(result["myopic_first_design"]) // n_buses
    planning_duration = int(result["planning_first_design"]) // n_buses
    return {
        "adaptive_gap": -float(result["planning_minus_fixed"]),
        "nonmyopic_gap": -float(result["planning_minus_myopic"]),
        "mean_branch_value": float(result["mean_branch_value"]),
        "meaningful_duration_branches": meaningful,
        "dominant_duration_branch_share": float(duration_mass.max()),
        "first_duration_differs": myopic_duration != planning_duration,
    }


def exact_audit(key, *, durations, centres, U, n_buses, sigma, seeds, outer, inner,
                alpha, margin, u_grid, min_branch_share, max_dominance,
                min_adaptive_advantage=.01, min_mean_branch_value=.01):
    rows = [audit_one(
        key, durations=durations, centres=centres, U=U, n_buses=n_buses,
        sigma=sigma, seed=s, outer=outer, inner=inner, alpha=alpha,
        margin=margin, u_grid=u_grid, min_branch_share=min_branch_share,
    ) for s in seeds]
    ad_mean, ad_lcb = mean_lcb([r["adaptive_gap"] for r in rows])
    nm_mean, nm_lcb = mean_lcb([r["nonmyopic_gap"] for r in rows])
    bv_mean, bv_lcb = mean_lcb([r["mean_branch_value"] for r in rows])
    min_branches = min(r["meaningful_duration_branches"] for r in rows)
    max_dom = max(r["dominant_duration_branch_share"] for r in rows)
    differs = all(r["first_duration_differs"] for r in rows)
    passed = (ad_mean >= min_adaptive_advantage and ad_lcb > 0 and
              nm_lcb > 0 and bv_mean >= min_mean_branch_value and bv_lcb > 0 and
              min_branches >= 2 and max_dom <= max_dominance and differs)
    return {
        "indices": key, "durations_s": ";".join(f"{durations[i]:.2f}" for i in key),
        "adaptive_gap_mean": ad_mean, "adaptive_gap_lcb95": ad_lcb,
        "nonmyopic_gap_mean": nm_mean, "nonmyopic_gap_lcb95": nm_lcb,
        "mean_branch_value_mean": bv_mean, "mean_branch_value_lcb95": bv_lcb,
        "min_meaningful_duration_branches": min_branches,
        "max_dominant_duration_branch_share": max_dom,
        "lookahead_duration_differs_all_seeds": int(differs), "passes_gates": int(passed),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank-dir", default="data/ieee14_duration_dense_0p01")
    p.add_argument("--global-samples", type=int, default=20000)
    p.add_argument("--global-exact", type=int, default=120)
    p.add_argument("--local-parents", type=int, default=12)
    p.add_argument("--local-exact", type=int, default=120)
    p.add_argument("--finalists", type=int, default=20)
    p.add_argument("--search-seed", type=int, default=8675309)
    p.add_argument("--audit-seeds", default="101,202,303")
    p.add_argument("--N-obs", dest="n_obs", type=int, default=5)
    p.add_argument("--noise-sigma", type=float, default=0.005)
    p.add_argument("--support-size", type=int, default=96)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--u-grid", default="0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--undercontrol-penalty", type=float, default=20.0)
    p.add_argument("--violation-penalty", type=float, default=0.0)
    args = p.parse_args()
    bank = (ROOT / args.bank_dir).resolve(); catalog: Catalog = load_catalog(bank)
    durations = tuple(round(.20 + .01*i, 2) for i in range(281))
    pool, _ = resolve_pool_actions(catalog, durations); n_buses = len(catalog.buses)
    centres, U, obs = load_support(bank, pool, args.n_obs, args.support_size, 7919)
    seeds = tuple(int(x) for x in args.audit_seeds.split(",")); u_grid = np.asarray([float(x) for x in args.u_grid.split(",")])
    install_safety_aware_mocu_terminal(
        undercontrol_penalty=args.undercontrol_penalty,
        violation_penalty=args.violation_penalty,
    )
    folder = make_experiment_dir_name("ieee14_mocu_duration_fullgrid_audit", "objective_based", 2, n_obs=args.n_obs, noise_sigma=args.noise_sigma)
    out = ROOT / "experiments" / folder; diag = out / "diagnostics"; diag.mkdir(parents=True, exist_ok=True)
    started = time.time(); rng = np.random.default_rng(args.search_seed)
    run = {"system":"ieee14", "physical_bus_count":n_buses, "dynamic_machine_buses":[1,2,3,6,8], "latent_dimension":10, "study_type":"MOCU_full_281_duration_global_local_audit_v2", "objective":"safety_aware_mocu_identical_to_training_evaluation", "undercontrol_penalty":args.undercontrol_penalty, "violation_penalty":args.violation_penalty, "duration_grid_s":[.20,3.00,.01], "n_duration_options":281, "combination_space":math.comb(281,6), "choose":6, "action_space_per_candidate":6*n_buses, "global_samples":args.global_samples, "global_exact":args.global_exact, "local_parents":args.local_parents, "local_exact":args.local_exact, "finalists":args.finalists, "audit_seeds":list(seeds), "N_obs":args.n_obs, "noise_sigma":args.noise_sigma, "support_size":len(U), "status":"running"}
    (out/"run_config.json").write_text(json.dumps(run,indent=2)+"\n")
    print(f"MOCU full grid=281, C(281,6)={math.comb(281,6):,}; reusing {bank}", flush=True)
    info, distance = proxy_scores(centres, 281, n_buses)
    candidates=set()
    for _ in range(args.global_samples): candidates.add(candidate_key(rng.choice(281,6,replace=False)))
    for offset in range(281): candidates.add(candidate_key(np.r_[offset, rng.choice(np.delete(np.arange(281),offset),5,replace=False)]))
    proxy_order=sorted(candidates,key=lambda k:proxy_value(k,info,distance),reverse=True)
    # Split the exact budget between proxy-strong and uniform candidates.  A
    # pure response-dissimilarity proxy over-selected tightly packed long
    # durations and could miss control-loss / branching structure.
    n_proxy=max(1,args.global_exact//2)
    global_keys=proxy_order[:n_proxy]
    remainder=[k for k in candidates if k not in set(global_keys)]
    if remainder and len(global_keys)<args.global_exact:
        pick=rng.choice(len(remainder),min(args.global_exact-len(global_keys),len(remainder)),replace=False)
        global_keys += [remainder[int(i)] for i in np.atleast_1d(pick)]
    low=[]
    def score(keys, label):
        for i,key in enumerate(keys,1):
            row=exact_audit(key,durations=durations,centres=centres,U=U,n_buses=n_buses,sigma=args.noise_sigma,seeds=(seeds[0],),outer=8,inner=4,alpha=args.alpha,margin=args.margin,u_grid=u_grid,min_branch_share=.10,max_dominance=.75)
            row["proxy_score"]=proxy_value(key,info,distance); low.append(row)
            if i%20==0 or i==len(keys): print(f"{label} exact {i}/{len(keys)}",flush=True)
    score(global_keys,"global")
    low.sort(key=lambda r:min(r["adaptive_gap_mean"],r["nonmyopic_gap_mean"],r["mean_branch_value_mean"]),reverse=True)
    local=set()
    for parent in low[:args.local_parents]:
        base=tuple(parent["indices"])
        for pos in range(6):
            for replacement in range(281):
                if replacement not in base:
                    trial=list(base); trial[pos]=replacement; local.add(candidate_key(trial))
    local_order=sorted(local,key=lambda k:proxy_value(k,info,distance),reverse=True)
    n_proxy=max(1,args.local_exact//2)
    local_keys=local_order[:n_proxy]
    remainder=[k for k in local if k not in set(local_keys)]
    if remainder and len(local_keys)<args.local_exact:
        pick=rng.choice(len(remainder),min(args.local_exact-len(local_keys),len(remainder)),replace=False)
        local_keys += [remainder[int(i)] for i in np.atleast_1d(pick)]
    score(local_keys,"local")
    best={}
    for row in low:
        key=tuple(row["indices"]); value=min(row["adaptive_gap_mean"],row["nonmyopic_gap_mean"],row["mean_branch_value_mean"])
        if key not in best or value>best[key][0]: best[key]=(value,row)
    # Build a multi-objective finalist slate instead of filling it with many
    # near-identical proxy neighbors.  Round-robin ranks retain candidates that
    # are strongest in adaptive value, non-myopic value, branch value, or
    # duration branching, plus balanced candidates.
    rows=[x[1] for x in best.values()]
    rankings=[
        sorted(rows,key=lambda r:min(r["adaptive_gap_mean"],r["nonmyopic_gap_mean"],r["mean_branch_value_mean"]),reverse=True),
        sorted(rows,key=lambda r:r["adaptive_gap_mean"],reverse=True),
        sorted(rows,key=lambda r:r["nonmyopic_gap_mean"],reverse=True),
        sorted(rows,key=lambda r:r["mean_branch_value_mean"],reverse=True),
        sorted(rows,key=lambda r:(r["min_meaningful_duration_branches"],-r["max_dominant_duration_branch_share"]),reverse=True),
    ]
    finalists=[]; seen=set(); depth=0
    while len(finalists)<args.finalists and depth<max(map(len,rankings)):
        for ranking in rankings:
            if depth<len(ranking):
                row=ranking[depth]; key=tuple(row["indices"])
                if key not in seen:
                    finalists.append(row); seen.add(key)
                    if len(finalists)>=args.finalists: break
        depth+=1
    final=[]
    for i,prior in enumerate(finalists,1):
        row=exact_audit(tuple(prior["indices"]),durations=durations,centres=centres,U=U,n_buses=n_buses,sigma=args.noise_sigma,seeds=seeds,outer=24,inner=16,alpha=args.alpha,margin=args.margin,u_grid=u_grid,min_branch_share=.10,max_dominance=.75)
        row["proxy_score"]=proxy_value(tuple(prior["indices"]),info,distance); row.pop("indices",None); final.append(row)
        print(f"strict {i}/{len(finalists)} {row['durations_s']} pass={row['passes_gates']}",flush=True)
    for row in low: row.pop("indices",None)
    final.sort(key=lambda r:(r["passes_gates"],min(r["adaptive_gap_lcb95"],r["nonmyopic_gap_lcb95"],r["mean_branch_value_lcb95"])),reverse=True)
    passing=[r for r in final if r["passes_gates"]]
    write_csv(diag/"search_candidates.csv",low); write_csv(diag/"strict_finalists.csv",final); write_csv(diag/"passing_candidates.csv",passing)
    summary={**run,"status":"pass" if passing else "no_duration_set_passed","n_global_candidates":len(candidates),"n_local_neighbors":len(local),"n_strict_finalists":len(final),"n_passing":len(passing),"search_is_exhaustive":False,"search_note":f"All 281 durations eligible; {args.global_samples:,} reproducible global samples, coordinate-local refinement, then strict three-seed MOCU validation.","selection_rule":"official gates: adaptive mean>=0.01 with positive LCB; non-myopic LCB>0; mean branch value>=0.01 with positive LCB; >=2 duration branches; dominance<=0.75; first duration differs in all seeds","obs_indices":obs.tolist(),"elapsed_seconds":time.time()-started}
    (diag/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); run.update(status="complete",n_passing=len(passing),elapsed_seconds=summary["elapsed_seconds"]); (out/"run_config.json").write_text(json.dumps(run,indent=2)+"\n")
    print(f"complete passing={len(passing)}/{len(final)} results={diag}",flush=True)

if __name__ == "__main__": main()
