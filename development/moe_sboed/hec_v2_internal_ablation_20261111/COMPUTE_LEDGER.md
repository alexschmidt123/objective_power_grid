# Actual evaluation compute

```json
{
  "total_seconds": 283.3201981878374,
  "per_configuration": {
    "full": {
      "n": 2000,
      "mean": 12.929352605692825,
      "MCSE": 0.035015863472915895,
      "CI95": [
        12.860681195314418,
        12.998024016071232
      ],
      "p_two_sided": 0.0,
      "evaluation_seconds": 41.817537299823016,
      "process_CPU_seconds": 41.837665498,
      "GPU_peak_allocated_bytes": 189232640,
      "process_max_RSS_bytes": 1528086528,
      "simulator_episodes": 2000,
      "simulator_observations": 60000,
      "simulator_batch_calls": 3750,
      "negative_observation_density_terms": 60000000000,
      "positive_observation_density_terms": 60000,
      "model_queries": {
        "frozen_belief_rows": 58000,
        "frozen_SG_provisional_rows": 60000,
        "exact_signal_evaluations": 37440000,
        "analytic_parameter_jacobians": 3840000,
        "frozen_opportunity_critic_rows": 180000
      },
      "inference_ms_per_experiment": 3.0534217110835016
    },
    "expert1": {
      "n": 2000,
      "mean": 11.193216502555279,
      "MCSE": 0.05099812629930025,
      "CI95": [
        11.093201454737368,
        11.29323155037319
      ],
      "p_two_sided": 0.0,
      "evaluation_seconds": 41.64910415094346,
      "process_CPU_seconds": 41.673685408,
      "GPU_peak_allocated_bytes": 189240320,
      "process_max_RSS_bytes": 1528086528,
      "simulator_episodes": 2000,
      "simulator_observations": 60000,
      "simulator_batch_calls": 3750,
      "negative_observation_density_terms": 60000000000,
      "positive_observation_density_terms": 60000,
      "model_queries": {
        "frozen_belief_rows": 58000,
        "frozen_SG_provisional_rows": 60000,
        "exact_signal_evaluations": 37440000,
        "analytic_parameter_jacobians": 3840000,
        "frozen_opportunity_critic_rows": 180000
      },
      "inference_ms_per_experiment": 3.066709579434246
    },
    "expert2": {
      "n": 2000,
      "mean": 11.942015422282326,
      "MCSE": 0.047488697911909405,
      "CI95": [
        11.848882894930334,
        12.035147949634318
      ],
      "p_two_sided": 0.0,
      "evaluation_seconds": 41.64994462719187,
      "process_CPU_seconds": 41.67448832199999,
      "GPU_peak_allocated_bytes": 189240832,
      "process_max_RSS_bytes": 1528086528,
      "simulator_episodes": 2000,
      "simulator_observations": 60000,
      "simulator_batch_calls": 3750,
      "negative_observation_density_terms": 60000000000,
      "positive_observation_density_terms": 60000,
      "model_queries": {
        "frozen_belief_rows": 58000,
        "frozen_SG_provisional_rows": 60000,
        "exact_signal_evaluations": 37440000,
        "analytic_parameter_jacobians": 3840000,
        "frozen_opportunity_critic_rows": 180000
      },
      "inference_ms_per_experiment": 3.0497401603497565
    },
    "expert3": {
      "n": 2000,
      "mean": 11.421067197814137,
      "MCSE": 0.04921705808360627,
      "CI95": [
        11.324545094456855,
        11.51758930117142
      ],
      "p_two_sided": 0.0,
      "evaluation_seconds": 41.59941782290116,
      "process_CPU_seconds": 41.623441345,
      "GPU_peak_allocated_bytes": 189240832,
      "process_max_RSS_bytes": 1528086528,
      "simulator_episodes": 2000,
      "simulator_observations": 60000,
      "simulator_batch_calls": 3750,
      "negative_observation_density_terms": 60000000000,
      "positive_observation_density_terms": 60000,
      "model_queries": {
        "frozen_belief_rows": 58000,
        "frozen_SG_provisional_rows": 60000,
        "exact_signal_evaluations": 37440000,
        "analytic_parameter_jacobians": 3840000,
        "frozen_opportunity_critic_rows": 180000
      },
      "inference_ms_per_experiment": 3.048078170977533
    },
    "expert4": {
      "n": 2000,
      "mean": 12.427823126689205,
      "MCSE": 0.04191387826502605,
      "CI95": [
        12.345623664804894,
        12.510022588573516
      ],
      "p_two_sided": 0.0,
      "evaluation_seconds": 41.62774875503965,
      "process_CPU_seconds": 41.65142369700001,
      "GPU_peak_allocated_bytes": 189240832,
      "process_max_RSS_bytes": 1528086528,
      "simulator_episodes": 2000,
      "simulator_observations": 60000,
      "simulator_batch_calls": 3750,
      "negative_observation_density_terms": 60000000000,
      "positive_observation_density_terms": 60000,
      "model_queries": {
        "frozen_belief_rows": 58000,
        "frozen_SG_provisional_rows": 60000,
        "exact_signal_evaluations": 37440000,
        "analytic_parameter_jacobians": 3840000,
        "frozen_opportunity_critic_rows": 180000
      },
      "inference_ms_per_experiment": 3.042105510830879
    },
    "uniform": {
      "n": 2000,
      "mean": 12.937944097354702,
      "MCSE": 0.033806815639569776,
      "CI95": [
        12.871643812851715,
        13.004244381857688
      ],
      "p_two_sided": 0.0,
      "evaluation_seconds": 41.65495721204206,
      "process_CPU_seconds": 41.67629815699999,
      "GPU_peak_allocated_bytes": 189240832,
      "process_max_RSS_bytes": 1528086528,
      "simulator_episodes": 2000,
      "simulator_observations": 60000,
      "simulator_batch_calls": 3750,
      "negative_observation_density_terms": 60000000000,
      "positive_observation_density_terms": 60000,
      "model_queries": {
        "frozen_belief_rows": 58000,
        "frozen_SG_provisional_rows": 60000,
        "exact_signal_evaluations": 37440000,
        "analytic_parameter_jacobians": 3840000,
        "frozen_opportunity_critic_rows": 180000
      },
      "inference_ms_per_experiment": 3.0539116309955716
    },
    "conventional_moe": {
      "n": 2000,
      "mean": 12.120526774187969,
      "MCSE": 0.044393498239027766,
      "CI95": [
        12.03346440207016,
        12.207589146305777
      ],
      "p_two_sided": 0.0,
      "evaluation_seconds": 28.888526174006984,
      "process_CPU_seconds": 28.90383213300001,
      "GPU_peak_allocated_bytes": 189895168,
      "process_max_RSS_bytes": 1528086528,
      "simulator_episodes": 2000,
      "simulator_observations": 60000,
      "simulator_batch_calls": 3750,
      "negative_observation_density_terms": 60000000000,
      "positive_observation_density_terms": 60000,
      "model_queries": {},
      "inference_ms_per_experiment": 0.17023704014718533
    }
  },
  "sampled_resources": {
    "full": {
      "CPU_percent_one_core": {
        "sample_mean": 100.04771305788039,
        "sample_max": 100.0626519396534
      },
      "RSS_bytes": {
        "sample_mean": 1462765568.0,
        "sample_max": 1463205888.0
      },
      "GPU_percent": {
        "sample_mean": 81.0,
        "sample_max": 97.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 863.0,
        "sample_max": 863.0
      },
      "GPU_power_W": {
        "sample_mean": 120.8345,
        "sample_max": 123.64
      },
      "samples": 20
    },
    "expert1": {
      "CPU_percent_one_core": {
        "sample_mean": 100.05847573661343,
        "sample_max": 100.09602008974866
      },
      "RSS_bytes": {
        "sample_mean": 1475700638.4761906,
        "sample_max": 1475706880.0
      },
      "GPU_percent": {
        "sample_mean": 78.38095238095238,
        "sample_max": 97.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 863.0,
        "sample_max": 863.0
      },
      "GPU_power_W": {
        "sample_mean": 121.37571428571428,
        "sample_max": 124.46
      },
      "samples": 21
    },
    "expert2": {
      "CPU_percent_one_core": {
        "sample_mean": 100.05865291310262,
        "sample_max": 100.0638707156514
      },
      "RSS_bytes": {
        "sample_mean": 1476506965.3333333,
        "sample_max": 1476620288.0
      },
      "GPU_percent": {
        "sample_mean": 81.57142857142857,
        "sample_max": 99.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 863.0,
        "sample_max": 863.0
      },
      "GPU_power_W": {
        "sample_mean": 121.96523809523809,
        "sample_max": 124.94
      },
      "samples": 21
    },
    "expert3": {
      "CPU_percent_one_core": {
        "sample_mean": 100.05726703346465,
        "sample_max": 100.0864565631436
      },
      "RSS_bytes": {
        "sample_mean": 1477197824.0,
        "sample_max": 1477197824.0
      },
      "GPU_percent": {
        "sample_mean": 80.76190476190476,
        "sample_max": 99.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 863.0,
        "sample_max": 863.0
      },
      "GPU_power_W": {
        "sample_mean": 123.26857142857142,
        "sample_max": 127.25
      },
      "samples": 21
    },
    "expert4": {
      "CPU_percent_one_core": {
        "sample_mean": 100.05742117677384,
        "sample_max": 100.10217329095437
      },
      "RSS_bytes": {
        "sample_mean": 1477228446.4761906,
        "sample_max": 1477267456.0
      },
      "GPU_percent": {
        "sample_mean": 79.95238095238095,
        "sample_max": 96.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 863.0,
        "sample_max": 863.0
      },
      "GPU_power_W": {
        "sample_mean": 123.68190476190475,
        "sample_max": 127.27
      },
      "samples": 21
    },
    "uniform": {
      "CPU_percent_one_core": {
        "sample_mean": 100.05202411294428,
        "sample_max": 100.06283053350052
      },
      "RSS_bytes": {
        "sample_mean": 1477267456.0,
        "sample_max": 1477267456.0
      },
      "GPU_percent": {
        "sample_mean": 82.3,
        "sample_max": 96.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 863.0,
        "sample_max": 863.0
      },
      "GPU_power_W": {
        "sample_mean": 124.59599999999998,
        "sample_max": 126.54
      },
      "samples": 20
    },
    "conventional_moe": {
      "CPU_percent_one_core": {
        "sample_mean": 100.052070163931,
        "sample_max": 100.06093051352808
      },
      "RSS_bytes": {
        "sample_mean": 1478045696.0,
        "sample_max": 1478045696.0
      },
      "GPU_percent": {
        "sample_mean": 96.8,
        "sample_max": 100.0
      },
      "GPU_memory_MiB": {
        "sample_mean": 863.0,
        "sample_max": 863.0
      },
      "GPU_power_W": {
        "sample_mean": 162.12399999999997,
        "sample_max": 166.03
      },
      "samples": 15
    }
  }
}
```

Device telemetry sampled every2seconds;GPU memory samples and Torch allocated peaks are different measurements. CPU utilization is relative to one core;peak RAM uses Linux process high-water RSS. Per-policy wall/CPU clocks cover trajectory generation and contrast evaluation;single-case inference latency includes the shared frozen provider and all dense expert computations. No optimizer steps. No sparse-compute shortcut for disabled experts. Negative/positive observation-density terms are distinct from physical simulator observations and model-query counts.
