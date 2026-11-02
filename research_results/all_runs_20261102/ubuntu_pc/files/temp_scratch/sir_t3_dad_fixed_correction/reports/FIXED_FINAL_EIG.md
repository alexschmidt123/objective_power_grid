# Fresh final EIG

{
  "result": {
    "mean": 8.897182195794578,
    "MC_SE": 0.035513084797159804,
    "CI95_low": 8.827576549592145,
    "CI95_high": 8.96678784199701,
    "N": 2048,
    "metadata": {
      "seed": 2026103151,
      "N": 2048,
      "contrasts": [
        1000000
      ],
      "paired_cases_sha256": "207aaa1ac97e9c4fa83c207dc1db73b510858b8a1cca50586f879f903b002a07",
      "seconds": 5.619730008998886,
      "terminal_full_horizon": true,
      "fixed_exact_deterministic_reuse": true,
      "MC_SE_scope": "conditional on training seed and contrast bank; contrast bias and training-seed variance excluded"
    }
  },
  "paired": {
    "fixed-minus-random": {
      "mean": 4.709476774152689,
      "MC_SE": 0.06713464605205366,
      "CI95_low": 4.577892867890664,
      "CI95_high": 4.8410606804147145,
      "N": 2048
    },
    "fixed-minus-dad": {
      "mean": 3.8325680760421106,
      "MC_SE": 0.05628058950551264,
      "CI95_low": 3.722258120611306,
      "CI95_high": 3.9428780314729153,
      "N": 2048
    },
    "fixed-minus-rl_sboed": {
      "mean": -0.2739374997641877,
      "MC_SE": 0.02299509761071254,
      "CI95_low": -0.31900789108118427,
      "CI95_high": -0.2288671084471911,
      "N": 2048
    },
    "fixed-minus-vsoed": {
      "mean": -0.6097298037219528,
      "MC_SE": 0.024667174259173148,
      "CI95_low": -0.6580774652699322,
      "CI95_high": -0.5613821421739735,
      "N": 2048
    },
    "fixed-minus-original4h": {
      "mean": 5.010999383332983,
      "MC_SE": 0.057701951013428114,
      "CI95_low": 4.897903559346664,
      "CI95_high": 5.124095207319302,
      "N": 2048
    }
  }
}