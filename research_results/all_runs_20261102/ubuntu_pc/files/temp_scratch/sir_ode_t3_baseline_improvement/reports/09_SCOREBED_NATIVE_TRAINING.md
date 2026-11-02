# 09 SCOREBED NATIVE TRAINING

{
  "status": "SCOREBED_NATIVE_BUDGET_INFEASIBLE",
  "projected_seconds": 37993.68790246127,
  "cap_seconds": 14400,
  "required_speedup": 2.638450548782033,
  "native_score_steps": 50000,
  "native_policy_steps": 500,
  "restarts": 10,
  "source": "Preserved native fullbatch compiled profile; no demonstrated exact acceleration sufficient to fit4h. No shortened variant substituted.",
  "profile": {
    "eligible": true,
    "unit_seconds": 0.7251487020403147,
    "score_update_seconds": 0.7251487020403147,
    "policy_update_seconds": 0.3434766929131001,
    "initialization_seconds": 18.869335880037397,
    "projected_full_native_seconds": 37993.68790246127,
    "reason": "Explicitly declared two-stage wall-time allocation with all ten restarts; incomplete native training remains COMPUTE_LIMITED"
  }
}
