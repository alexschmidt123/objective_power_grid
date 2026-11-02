# T3 training costs

Four-hour maximum per optimized/trainable method. Profiling is separate and its discarded weights are never reused.

|Method|Actual optimizer updates|Fresh trajectories|Charged seconds|Native schedule complete|Status|
|---|---:|---:|---:|---|---|
|random|0|0|0|True|NO_TRAINING_REQUIRED|
|fixed|50000|100000000|6822.325561644044|True|COMPUTE_LIMITED|
|dad|||||BLOCKED_READINESS|
|rl_sboed|155712|276600|14370.581867771922|False|COMPUTE_LIMITED|
|vsoed|286906|286926000|14371.236601116834|False|COMPUTE_LIMITED|
|scorebed|||||BLOCKED_READINESS|

Readiness measurements:

{
  "fixed": {
    "eligible": true,
    "unit_seconds": 0.1282238638959825,
    "projected_updates_4h": 111367,
    "minimum_projected_updates": 1000,
    "reason": "Native full-batch throughput gate; not convergence evidence",
    "scientifically_useful_1000_update_hours": 0.06895107330443959,
    "native_reference_updates": 50000
  },
  "dad": {
    "eligible": false,
    "unit_seconds": 18.6156983259134,
    "projected_updates_4h": 767,
    "minimum_projected_updates": 1000,
    "reason": "Native full-batch throughput gate; not convergence evidence",
    "scientifically_useful_1000_update_hours": 5.204360646087055,
    "native_reference_updates": 50000
  },
  "rl_sboed": {
    "eligible": true,
    "collection_seconds": 5.628783841151744,
    "optimization64_seconds": 0.6254477370530367,
    "warmup_collections": 334,
    "projected_warmup_seconds": 1880.0138029446825,
    "projected_postwarm_collections_4h": 1982,
    "projected_native_actor_updates": 126848,
    "unit_seconds": 6.254231578204781,
    "reason": "Native warm-up preserved; optimization cost measured separately with diagnostic replay, no warmed pilot checkpoint claimed"
  },
  "vsoed": {
    "eligible": true,
    "unit_seconds": 0.04701723181642592,
    "projected_actor_updates_4h": 303693,
    "reason": "Full native initial replay and actor/5critic/5posterior cycles exercised; projection omits future buffer-growth uncertainty"
  },
  "scorebed": {
    "eligible": false,
    "unit_seconds": 0.7251487020403147,
    "score_update_seconds": 0.7251487020403147,
    "policy_update_seconds": 0.3434766929131001,
    "initialization_seconds": 18.869335880037397,
    "projected_full_native_seconds": 37993.68790246127,
    "reason": "Required full50000-step score pretraining alone exceeds4h at measured throughput; policy stages and all10restarts add further cost. No shortened or one-restart primary substituted."
  }
}
