# Exploratory five-seed V4 continuation replication

The seed101 pilot failed its predeclared expansion rule. Additional seeds were explicitly authorized after observing the routing effect. This is exploratory,not confirmatory. Original pilot results remain unchanged on their separate bank.

| Method |101|102|103|104|105|Mean ± training-seed SD|
|---|---:|---:|---:|---:|---:|---:|
|MoE-sBOED-V4|12.82525|12.77094|12.92833|12.89043|12.89163|12.86132 ± 0.06270|
|MoE-sBOED-V4-uniform|12.77675|12.73835|12.85208|12.79736|12.88428|12.80976 ± 0.05850|

| Seed | Paired learned-minus-uniform | Evaluation MCSE | Within-seed95% CI |
|---|---:|---:|---|
|101|+0.04851|0.03217|[-0.01458,+0.11159]|
|102|+0.03259|0.03207|[-0.03029,+0.09548]|
|103|+0.07625|0.02781|[+0.02170,+0.13080]|
|104|+0.09307|0.03072|[+0.03282,+0.15333]|
|105|+0.00734|0.02863|[-0.04880,+0.06349]|

Five-seed paired result:

```json
{
  "n_training_seeds": 5,
  "mean_delta": 0.051553643959554564,
  "SD_delta": 0.034122447408071584,
  "seed_SE": 0.015260022392621912,
  "CI95": [
    0.009185029482350733,
    0.09392225843675839
  ],
  "positive_seeds": 5,
  "interval": "Paired training-seed Student t,df4;conditional on one fixed bank"
}
```

Positive exploratory routing effect;not confirmatory and not proof of superiority over V3

Secondary comparison has previous collaborative checkpoints for seed101 ONLY on the same new bank:

```json
{
  "rows": [
    {
      "method": "Previous collaborative MoE-sBOED",
      "training_seed_coverage": [
        101
      ],
      "EIG": 12.842313942077817,
      "MCSE": 0.03647373808709798
    },
    {
      "method": "Previous collaborative MoE-sBOED-uniform",
      "training_seed_coverage": [
        101
      ],
      "EIG": 12.876493989884866,
      "MCSE": 0.03636666127977963
    }
  ],
  "paired_case_differences": {
    "Previous collaborative MoE-sBOED": {
      "n": 2000,
      "mean": -0.017059350216676733,
      "MCSE": 0.03724334727846716,
      "CI95": [
        -0.09009919362374802,
        0.055980493190394554
      ]
    },
    "Previous collaborative MoE-sBOED-uniform": {
      "n": 2000,
      "mean": -0.0512393980237268,
      "MCSE": 0.03682043312140523,
      "CI95": [
        -0.12344984273106585,
        0.02097104668361225
      ]
    }
  }
}
```

- Shared frozen V2 pretraining:independent continuation initialization/RNG only,not end-to-end independence.
- Expansion was explicitly authorized after original pilot failed its preregistered expansion criterion;all conclusions exploratory.
- MCSE conditions on the shared contrast bank;finite-contrast bias/contrast-bank uncertainty not covered.
- Same cases across seeds;10000 policy trajectories are not10000 training seeds. Seed-level inference uses n5.
- V3 secondary coverage is seed101 only on this bank;no five-seed V4-versus-V3 superiority claim.

ICML reporting:retain original pilot and this exploratory replication separately;report seed-level uncertainty and full compute/initialization lineage;do not equate a V4 routing gain with superior overall EIG. No further architectures or searches are scheduled.
