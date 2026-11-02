# Fresh final EIG

{
  "result": {
    "mean": 5.064614119752466,
    "MC_SE": 0.051068590098551886,
    "CI95_low": 4.964519683159304,
    "CI95_high": 5.164708556345628,
    "N": 2048,
    "metadata": {
      "seed": 2026103151,
      "N": 2048,
      "contrasts": [
        1000000
      ],
      "paired_cases_sha256": "207aaa1ac97e9c4fa83c207dc1db73b510858b8a1cca50586f879f903b002a07",
      "seconds": 1014.4892960509751,
      "terminal_full_horizon": true,
      "fixed_exact_deterministic_reuse": false,
      "MC_SE_scope": "conditional on training seed and contrast bank; contrast bias and training-seed variance excluded"
    }
  },
  "paired": {
    "dad-minus-random": {
      "mean": 0.8769086981105779,
      "MC_SE": 0.05179271471805843,
      "CI95_low": 0.7753949772631834,
      "CI95_high": 0.9784224189579724,
      "N": 2048
    },
    "dad-minus-rl_sboed": {
      "mean": -4.106505575806299,
      "MC_SE": 0.04597805530565261,
      "CI95_low": -4.196622564205378,
      "CI95_high": -4.01638858740722,
      "N": 2048
    },
    "dad-minus-vsoed": {
      "mean": -4.442297879764064,
      "MC_SE": 0.05453566488714635,
      "CI95_low": -4.549187782942871,
      "CI95_high": -4.335407976585257,
      "N": 2048
    }
  }
}