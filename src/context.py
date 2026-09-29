"""Shared EIG/SIR context and method registry; no control-objective losses."""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from src.config import SBOEDConfig, repo_root
from src.control.posterior_ctrl import normalize_log_weights, posterior_control_decision
from src.observations.likelihood import vector_gaussian_loglik
from src.observations.noise import keyed_noise_vector
BELIEF_DIM = 33
GLOBAL_SEED = 77311

METHOD_ALIASES = {
    "dad": "DAD",
    "rl_sboed": "RL-sBOED",
    "rl-sboed": "RL-sBOED",
    "moe_sboed": "MoE-sBOED",
    "moe-sboed": "MoE-sBOED",
    "matched_dense": "MatchedDense",
    "matched-dense": "MatchedDense",
    "step_dad": "Step-DAD",
    "step-dad": "Step-DAD",
    "myopic": "Myopic",
    "fixed": "Fixed",
    "random": "Random",
}
# Default main-table methods.  Step-DAD and MatchedDense are ablation /
# modern-baseline methods selected explicitly (they cost extra online time or
# are only meaningful as MoE controls).
ALL_METHOD_KEYS = (
    "dad",
    "rl_sboed",
    "moe_sboed",
    "myopic",
    "fixed",
    "random",
)
EXTENDED_METHOD_KEYS = ALL_METHOD_KEYS + ("matched_dense", "step_dad")

# Methods that require an offline train step (shell scripts skip training.sh when
# the selected evaluate set contains only keys outside this tuple).
TRAINABLE_METHOD_KEYS = (
    "dad",
    "rl_sboed",
    "moe_sboed",
    "matched_dense",
)


@dataclass
class ExperimentContext:
    """Shared context for learned, hybrid, and baseline design methods."""

    system: str
    cfg: SBOEDConfig
    horizon: int
    n_actions: int
    n_obs: int
    n_sim: int
    obs_dim: int
    obs_indices: np.ndarray
    observation_mode: str
    sigma_y: float
    alpha: float
    margin: float
    u_grid: np.ndarray
    robust_rule: str
    snap_up: bool
    experiment_type: str
    # Method-visible centres only: (n_actions, n_support, obs_dim)
    centres_support: np.ndarray
    U_support: np.ndarray
    log_p0: np.ndarray
    M_support: np.ndarray
    K_support: np.ndarray
    particle_features: np.ndarray
    obs_mean: float
    obs_std: float
    test_systems: list[dict[str, Any]]
    train_systems: list[dict[str, Any]]
    validation_systems: list[dict[str, Any]]
    U_test: np.ndarray
    M_test: np.ndarray
    K_test: np.ndarray
    data_dir: Path
    out_dir: Path
    oracle_tolerance: float
    fixed_sequence: list[int]
    terminal_rule_hash: str
    config_hash: str
    control_safe_support: np.ndarray | None = None
    ocu_table_support: np.ndarray | None = None
    control_safe_test: np.ndarray | None = None
    ocu_table_test: np.ndarray | None = None
    continuous_duration_mode: bool = False
    reset_after_probe: bool = True


def resolve_n_obs(cfg: SBOEDConfig) -> int:
    """N_obs from YAML only (observation.N_obs or legacy top-level N_obs)."""
    obs = dict(cfg.raw.get("observation") or {})
    if "N_obs" in obs:
        return int(obs["N_obs"])
    if "N_obs" in cfg.raw:
        return int(cfg.raw["N_obs"])
    return 5


def resolve_oracle_tolerance(cfg: SBOEDConfig) -> float:
    oracle = dict(cfg.raw.get("oracle") or {})
    if "tolerance" in oracle:
        return float(oracle["tolerance"])
    if "oracle_tolerance" in cfg.raw:
        return float(cfg.raw["oracle_tolerance"])
    return 1e-4


def resolve_sigma_y(cfg: SBOEDConfig) -> float:
    obs = dict(cfg.raw.get("observation") or {})
    if "noise_sigma" in obs:
        return float(obs["noise_sigma"])
    return float(cfg.sigma_y)


def config_sha256(cfg: SBOEDConfig) -> str:
    data = cfg.config_path.read_bytes()
    return hashlib.sha256(data).hexdigest()[:16]


def normalize_method_key(method: str) -> str:
    raw = str(method).strip()
    key = raw.lower().replace("-", "_")
    compact = "".join(ch for ch in key if ch.isalnum())
    # Accept canonical keys, hyphen aliases, and display names (MatchedDense).
    for canon in EXTENDED_METHOD_KEYS:
        display = METHOD_ALIASES[canon]
        display_key = display.lower().replace("-", "_")
        display_compact = "".join(ch for ch in display_key if ch.isalnum())
        if key in (canon, display_key) or compact == display_compact:
            return canon
    if key in METHOD_ALIASES:
        # Non-canonical alias such as "moe-sboed" -> resolve via display.
        display = METHOD_ALIASES[key]
        return normalize_method_key(display)
    raise ValueError(
        f"Unknown method {method!r}. Allowed: {', '.join(EXTENDED_METHOD_KEYS)}"
    )


def method_display_name(method_key: str) -> str:
    return METHOD_ALIASES[normalize_method_key(method_key)]


def _dedupe_method_keys(keys: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in keys:
        canon = normalize_method_key(key)
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def methods_from_args(
    cfg: SBOEDConfig, method: str | None
) -> list[str]:
    """Return canonical method keys to run.

    ``method`` may be a single key/display name or a comma-separated list such
    as ``dad,random``.  When omitted, use ``experiment.methods`` from the config
    (with the legacy MoE auto-insert when the config leaves methods implicit).
    """
    if method is not None and str(method).strip() != "":
        raw = str(method).strip()
        if "," in raw:
            return _dedupe_method_keys(
                part.strip() for part in raw.split(",") if part.strip()
            )
        return [normalize_method_key(raw)]

    explicit = list(cfg.methods) if cfg.methods else []
    configured = explicit if explicit else list(ALL_METHOD_KEYS)
    keys = []
    for m in configured:
        try:
            keys.append(normalize_method_key(m))
        except ValueError:
            continue
    if not keys:
        keys = list(ALL_METHOD_KEYS)
    out = _dedupe_method_keys(keys)
    # Only auto-add MoE when the config did not declare experiment.methods
    # (Plan-2 configs omit MoE until DAD/RL beat Fixed/Myopic).
    if not explicit and "moe_sboed" not in out:
        out.insert(2 if "rl_sboed" in out else len(out), "moe_sboed")
    if not explicit and set(out) == {
        "dad",
        "moe_sboed",
        "myopic",
        "fixed",
        "random",
    }:
        out = list(ALL_METHOD_KEYS)
    return out


def training_method_keys(method_keys: list[str]) -> list[str]:
    """Offline trainers required for an evaluate set (preserves train order)."""
    normalized = _dedupe_method_keys(method_keys)
    selected = set(normalized)
    out = [key for key in TRAINABLE_METHOD_KEYS if key in selected]
    if "step_dad" in selected and "dad" not in out:
        out.insert(0, "dad")
    return out


def experiment_out_dir(
    cfg: SBOEDConfig,
    project_root: Path | None = None,
    *,
    experiment_type: str = "eig_based",
    exp_dir: Path | None = None,
    create_new: bool = False,
) -> Path:
    """Result folder: ``date_time_configname_experimentType_Tnum`` under experiments/."""
    from src.layout import resolve_result_dir

    return resolve_result_dir(
        cfg,
        experiment_type,
        exp_dir=exp_dir,
        project_root=project_root,
        create_new=create_new,
    )


def update_posterior_vector(
    ctx: ExperimentContext,
    log_w: np.ndarray,
    action: int,
    y_obs: np.ndarray,
) -> np.ndarray:
    centres = ctx.centres_support[int(action)]
    y = np.asarray(y_obs, dtype=np.float64).reshape(-1)
    return log_w + vector_gaussian_loglik(y, centres, ctx.sigma_y)


def observe_compressed(
    system_row: dict[str, Any],
    action: int,
    *,
    sigma_y: float,
    n_obs: int,
    global_seed: int,
    theta_id: int,
    rollout_id: int,
    step: int,
) -> np.ndarray:
    """Noisy method-visible observation via the shared observation interface."""
    forbidden = ("full_delta_f", "delta_f_full", "full_delta_f_bank", "max_rocof_bank")
    for key in forbidden:
        if key in system_row:
            raise RuntimeError(
                f"{key} must not enter method observation path (use obs_clean only)"
            )
    if "obs_clean" not in system_row and "delta_f_obs_clean" not in system_row:
        raise KeyError("system row missing obs_clean")
    clean_bank = system_row.get("obs_clean", system_row.get("delta_f_obs_clean"))
    clean = np.asarray(clean_bank[int(action)], dtype=np.float64).reshape(-1)
    dim = max(int(n_obs), 1)
    if clean.size != dim:
        raise ValueError(f"expected clean shape ({dim},), got {clean.shape}")
    z = keyed_noise_vector(
        global_seed=global_seed,
        theta_id=theta_id,
        rollout_id=rollout_id,
        step=step,
        action_id=int(action),
        n_obs=n_obs,
    )
    return clean + float(sigma_y) * z


def belief_summary(
    ctx: ExperimentContext,
    log_w: np.ndarray,
    observations: list[np.ndarray],
) -> np.ndarray:
    w = normalize_log_weights(log_w)
    feats = np.zeros(BELIEF_DIM, dtype=np.float32)
    feats[0] = float(len(observations)) / float(ctx.horizon)
    ess = float(1.0 / np.sum(w * w))
    feats[1] = ess / float(len(w))
    feats[2] = float(np.max(w))
    feats[3] = float(np.sum(w * ctx.M_support))
    feats[4] = float(
        np.sqrt(max(np.sum(w * (ctx.M_support - feats[3]) ** 2), 0.0))
    )
    feats[5] = float(np.sum(w * ctx.K_support))
    feats[6] = float(
        np.sqrt(max(np.sum(w * (ctx.K_support - feats[5]) ** 2), 0.0))
    )
    order = np.argsort(ctx.U_support, kind="mergesort")
    u_sorted = ctx.U_support[order]
    cdf = np.cumsum(w[order])
    for i, q in enumerate((0.05, 0.25, 0.50, 0.75, 0.95)):
        idx = int(np.searchsorted(cdf, q, side="left"))
        idx = min(max(idx, 0), u_sorted.size - 1)
        feats[7 + i] = float(u_sorted[idx])
    decision = posterior_control_decision(
        ctx.U_support,
        w,
        ctx.alpha,
        margin=ctx.margin,
        u_grid=ctx.u_grid,
        snap_up=bool(getattr(ctx, "snap_up", True)),
        robust_rule=getattr(ctx, "robust_rule", "quantile"),
    )
    feats[12] = float(decision.u_ctrl)
    for i, level in enumerate(ctx.u_grid[:16]):
        feats[13 + i] = float(np.sum(w[np.isclose(ctx.U_support, level)]))
    if observations:
        y = np.concatenate(
            [np.asarray(o, dtype=np.float64).reshape(-1) for o in observations]
        )
        feats[29] = float(y.mean())
        feats[30] = float(y.std() if y.size > 1 else 0.0)
        feats[31] = float(y.min())
        feats[32] = float(y.max())
    return feats


def build_context_from_config(cfg, *, project_root=None, ensure_bank=True,
                              smoke=False, out_dir=None, experiment_type="eig_based"):
    from src.domains.sir.context import is_sir_config, build_sir_context
    if not is_sir_config(cfg):
        raise ValueError("The equilibrium-bank experiment pipeline is retired; use run.sh for online non-reset experiments")
    return build_sir_context(cfg, project_root=project_root, ensure_bank=ensure_bank,
        smoke=smoke, out_dir=out_dir, experiment_type=experiment_type)
