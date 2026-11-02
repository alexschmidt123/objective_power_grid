# Frozen reproducibility

{
  "source_hash": "97cfaedc12a9fc7cb7473bc163dd147f066f905d1bd6158bca77929f9f221373",
  "configuration_hash": "00a5d5f7a506a5be6870e5f38ae25037122e1fb1c9266ca19fa9c18e577e5057",
  "source_references": {
    "src/research/engine.py": "110c62b2dd2838903077a2aa967826f65ac05d29a0dd176687a55a6311e83714",
    "src/research/sir_t3_common.py": "0da5b014a2f990a5cbaccfd302e62685e14d8dd90f3149622fb216d8cc22a557",
    "src/research/sir_dad.py": "068aa553e0cfee3ffb02f96a30e3172229b5a0d3b62b7fd4725abe028ed7d184",
    "src/research/original_protocol.py": "0c075c4404110faf138ad03f8918fcc57f71f1fa3f4540be8330b27b8779d0db",
    "src/research/sir_t3_fused.py": "bdaebec6397f47f96ca233f1784b0f899de135b6fde80ec00a870369f1c9fc1b",
    "src/research/training.py": "ef3c5d9fe6d79d50247336038fff6075ed4ef87863e8637a25b2eec285bd6f11"
  },
  "CPU_checks": "PASS",
  "GPU_equivalence": "PENDING",
  "checkpoint_selection": "latest fully serialized within declared training budget; Fixed selected only by frozen development criterion"
}
Original methods and source unchanged. Native DAD before/after same4hmaximum and fresh seed101 initialization. Essential GPU validation is queued after original final evaluation.
