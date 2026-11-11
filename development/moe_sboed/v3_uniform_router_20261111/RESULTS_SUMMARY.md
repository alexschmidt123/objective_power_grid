# Frozen V3:learned versus uniform routing

Same final actor;2000 full trajectories,T30,1M prior negatives,identical truth/noise/routing uniforms/action normals. Finite-contrast EIG lower bound,not exact EIG.

|Policy|EIG nats|MCSE|Evaluation seconds|
|---|---:|---:|---:|
|learned|12.594942|0.038961|47.51|
|uniform|12.588669|0.039664|47.70|

Paired learned-minus-uniform(ordinary95%CI;raw and supplementary Holm3 p):

```json
{
  "n": 2000,
  "mean": 0.00627263390278471,
  "MCSE": 0.017003011309909163,
  "CI95": [
    -0.02707284587481976,
    0.03961811368038918
  ],
  "p_two_sided": 0.7122315540733481,
  "Holm3_p": 0.7122315540733481
}
```

Full selection frequencies,per-stage counts and resource measurements are in RESULTS.json and resources.jsonl. Expert selection is sampled;coordinates are never averaged. Changed actions resimulate observations. No parameters or original outputs changed. Reused development bank and one trained lineage limit confirmatory claims;Holm3 augments the original two-primary family for this supplementary report without rewriting historical tests. Device utilization samples are not exact per-kernel attribution.

Aggregate selection and resource summary:

```json
{
  "selection": {
    "learned": {
      "counts": [
        57695,
        752,
        796,
        757
      ],
      "frequencies": [
        0.9615833333333333,
        0.012533333333333334,
        0.013266666666666666,
        0.012616666666666667
      ],
      "evaluation_seconds": 47.51437514391728,
      "CPU_seconds": 47.540385188,
      "GPU_peak_bytes": 248948736
    },
    "uniform": {
      "counts": [
        14898,
        15197,
        14964,
        14941
      ],
      "frequencies": [
        0.2483,
        0.25328333333333336,
        0.2494,
        0.24901666666666666
      ],
      "evaluation_seconds": 47.70009477902204,
      "CPU_seconds": 47.726768721000006,
      "GPU_peak_bytes": 248949248
    }
  },
  "resources": {
    "learned": {
      "CPU_percent_one_core": {
        "sample_mean": 100.05444848995784,
        "sample_max": 100.06345240613277
      },
      "RSS_bytes": {
        "sample_mean": 1448229487.3043478,
        "sample_max": 1448763392.0
      },
      "GPU_percent": {
        "sample_mean": 76.52173913043478,
        "sample_max": 100.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 923.0,
        "sample_max": 923.0
      },
      "GPU_power_W": {
        "sample_mean": 119.41304347826087,
        "sample_max": 122.4
      },
      "samples": 23
    },
    "uniform": {
      "CPU_percent_one_core": {
        "sample_mean": 100.05549353495114,
        "sample_max": 100.09714473916166
      },
      "RSS_bytes": {
        "sample_mean": 1449340928.0,
        "sample_max": 1449340928.0
      },
      "GPU_percent": {
        "sample_mean": 71.20833333333333,
        "sample_max": 100.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 923.0,
        "sample_max": 923.0
      },
      "GPU_power_W": {
        "sample_mean": 118.81958333333334,
        "sample_max": 121.71
      },
      "samples": 24
    }
  },
  "provenance": {
    "actor_sha256": "5683d61ad932752e65d9ab0d5d2052b4526c50dd291c5c1dc89bafe7e6fa5e35",
    "bank_sha256": "275cfb113bc97879d8f692e07e36b92d0ae0f81d6ab82a55a21cf39c3ad9b7a8",
    "same_frozen_actor": true,
    "training": false,
    "cases": 2000,
    "T": 30,
    "negative_contrasts": 1000000,
    "learned_reproduction_max_error": 0.0,
    "finite_contrast_lower_bound_not_exact_EIG": true,
    "reused_development_bank_not_final_test": true,
    "single_training_lineage": true
  }
}
```
