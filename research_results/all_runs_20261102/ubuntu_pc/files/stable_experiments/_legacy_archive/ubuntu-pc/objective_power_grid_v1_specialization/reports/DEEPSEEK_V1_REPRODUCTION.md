# Frozen v1 reproduction

{
  "PASS": true,
  "source_commit": "13681d7091189a4e68056e6a2864dc58f9d5700b",
  "checkpoint_sha256": "f110766019e3628bb02c93815becd7e5795b3abc05626eabd91b7dbeb0381ee7",
  "architecture_source_sha256": {
    "src/research/deepseek_sboed.py": "85069fa88b27e62c82ab2fc326d36ac1aa573b01f5e4ceef2ffe0f4fd38e7573",
    "src/research/deepseek_sboed_experiment.py": "2ac005d655db1279c3c707d9ea2ef1daf5658bf643d39e7eb8d5fc71f9f41feb",
    "src/research/engine.py": "110c62b2dd2838903077a2aa967826f65ac05d29a0dd176687a55a6311e83714",
    "src/research/journal_validation.py": "9a69f6bc56c65b85d63ce1feaa9f85932f00eb9ed22cb8b580978f41920f2cff",
    "src/research/dad_location_policy.py": "40e24b2efa1fa2e5a3e334f95ca15632331e5d44c09421ab8d91afdc77f7a1d5",
    "src/research/benchmarks.py": "b23ec2406fde3efecf79f584be3bd145d6ea57c614c1c605be8e251e081e36ab"
  },
  "config_sha256": "b2c920b9830babf3cec75a6fc6c0c305e5ae1a1104744d3c9f368101f360d8b0",
  "EIG": 10.989065898584203,
  "MC_SE": 0.044794108684935864,
  "saved_EIG": 10.98906589858419,
  "maximum_per_trajectory_error": 6.10000938650046e-12,
  "maximum_manual_inference_error": 0.0,
  "metadata": {
    "seed": 1001,
    "systems": 2000,
    "contrasts": 1000000,
    "paired_draws_sha256": "5545d4071aea78471408fae814302633f9ddead6da57942ffac7a28a72f07e0b",
    "seconds": 34.694824415957555,
    "forward_particle_stages": 60000060000,
    "device": "cuda:0",
    "mode": "hard_deployed_policy",
    "secondary_parameter_inference": false
  },
  "wall_seconds": 35.2938752429327,
  "hardware": "NVIDIA GeForce RTX 5080",
  "original_checkpoint_unchanged": true,
  "optimization_updates": 0
}
