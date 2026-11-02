COMPLETE matched7714update/15.428Mprimary control; NOT compute matched.

{
  "v5a": {
    "mean": 10.662698848876083,
    "MC_SE": 0.04366574485535465,
    "CI95": [
      10.577113988959589,
      10.748283708792577
    ]
  },
  "metadata": {
    "seed": 1001,
    "systems": 2000,
    "contrasts": 1000000,
    "paired_draws_sha256": "5545d4071aea78471408fae814302633f9ddead6da57942ffac7a28a72f07e0b",
    "seconds": 34.5705087040551,
    "forward_particle_stages": 60000060000,
    "device": "cuda:0",
    "mode": "hard_deployed_policy",
    "secondary_parameter_inference": false
  },
  "checkpoint": {
    "path": "/home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10312026_163000_moe_sboed_v5a_off_seed101/seed101/checkpoints/update7714_final.pt",
    "sha256": "43c848b136017bab0ee809f017b159c74d8dcc4e4adb9fbff11c278a162fb073",
    "update": 7714,
    "charged_seconds": 13123.04791652388,
    "eligible": true
  },
  "evaluation_seed": 1001,
  "training_seed": 101,
  "v5a_minus_v4": {
    "mean": -0.3197772767344179,
    "MC_SE": 0.038550250085423385,
    "CI95": [
      -0.39533576690184774,
      -0.2442187865669881
    ]
  },
  "v5a_minus_native_vsoed": {
    "mean": -1.8181480385715267,
    "MC_SE": 0.05463489524703485,
    "CI95": [
      -1.9252324332557151,
      -1.7110636438873383
    ]
  },
  "control": "OFF_matched7714_updates",
  "off": {
    "mean": 10.662698848876083,
    "MC_SE": 0.04366574485535465,
    "CI95": [
      10.577113988959589,
      10.748283708792577
    ]
  },
  "off_minus_on": {
    "mean": 0.896832920069621,
    "MC_SE": 0.05410661374888885,
    "CI95": [
      0.7907839571217988,
      1.002881883017443
    ]
  },
  "off_minus_v4": {
    "mean": -0.3197772767344179,
    "MC_SE": 0.038550250085423385,
    "CI95": [
      -0.39533576690184774,
      -0.2442187865669881
    ]
  },
  "off_minus_vsoed": {
    "mean": -1.8181480385715267,
    "MC_SE": 0.05463489524703485,
    "CI95": [
      -1.9252324332557151,
      -1.7110636438873383
    ]
  },
  "interpretation": "Fixed-policy development-evaluation uncertainty,NOT training-seed uncertainty"
}
No cross-seed superiority claim. Routing usage is not useful specialization evidence.
