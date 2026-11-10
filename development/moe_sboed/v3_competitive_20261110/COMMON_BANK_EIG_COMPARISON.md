# V3 common-bank EIG comparison

INCOMPLETE — NO V3 MODEL OR NEW COMMON BANK.

No v3 result, uncertainty interval, dynamic-stage comparison or new baseline evaluation exists. These must not be replaced by historical values. Existing HEC-v2 reference results below are copied from its completed report for provenance only and do not constitute a v3 comparison.

```json
{
  "source_run": "moe_hec_v2_icml_20261109",
  "source_bank": "historical v2 bank,not a new v3 bank",
  "candidate": {
    "n": 2000,
    "mean": 12.929352605692825,
    "MCSE": 0.035015863472915895,
    "CI95": [
      12.860681195314418,
      12.998024016071232
    ],
    "p_two_sided": 0.0,
    "seconds": 44.221422001952305,
    "simulator_observations": 60000,
    "negative_density_calls": 60000000000,
    "positive_density_calls": 60000,
    "parameters": 189822,
    "unique_buffer_elements": 15266041
  },
  "stage": {
    "n": 2000,
    "mean": 12.923535256660337,
    "MCSE": 0.03362156106210024,
    "CI95": [
      12.857598284435047,
      12.989472228885626
    ],
    "p_two_sided": 0.0,
    "seconds": 44.30689471610822,
    "simulator_observations": 60000,
    "negative_density_calls": 60000000000,
    "positive_density_calls": 60000,
    "parameters": 189822,
    "unique_buffer_elements": 15266041
  },
  "continuation": {
    "n": 2000,
    "mean": 12.846723557084939,
    "MCSE": 0.03601504040364973,
    "CI95": [
      12.77609260944982,
      12.917354504720057
    ],
    "p_two_sided": 0.0,
    "seconds": 39.54245612304658,
    "simulator_observations": 60000,
    "negative_density_calls": 60000000000,
    "positive_density_calls": 60000,
    "parameters": 139297,
    "unique_buffer_elements": 15229473
  },
  "vsoed": {
    "n": 2000,
    "mean": 12.442719710104193,
    "MCSE": 0.0420862748225305,
    "CI95": [
      12.3601821524663,
      12.525257267742086
    ],
    "p_two_sided": 0.0,
    "seconds": 28.73318137298338,
    "simulator_observations": 60000,
    "negative_density_calls": 60000000000,
    "positive_density_calls": 60000,
    "parameters": 162306,
    "unique_buffer_elements": 0
  },
  "paired_v2_tests": {
    "stage": {
      "n": 2000,
      "mean": 0.005817349032490528,
      "MCSE": 0.025933992306260426,
      "CI95": [
        -0.04504313680962452,
        0.05667783487460558
      ],
      "p_two_sided": 0.8225361627979819,
      "Holm2_p": 0.8225361627979819
    },
    "continuation": {
      "n": 2000,
      "mean": 0.0826290486078883,
      "MCSE": 0.02592395670232853,
      "CI95": [
        0.031788244104676566,
        0.13346985311110005
      ],
      "p_two_sided": 0.0014579484765704371,
      "Holm2_p": 0.0029158969531408742
    }
  }
}
```

The historical bank used2000 trajectories/1M negatives. MCSE conditions on one trained lineage. Cross-bank pairing and compute-matched superiority over frozen vsOED are not claimed. No final-test data were accessed.
