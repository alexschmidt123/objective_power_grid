# Independent RR-MoE EIG

Experiment status: COMPLETE.

2000 trajectories,T30,1M prior contrasts plus the positive once. New common bank; complete resimulation for every changed policy. Finite-contrast lower bound; MCSE conditional on one trained lineage and the common contrast bank.

|Policy|EIG nats|MCSE|
|---|---:|---:|
|full|12.942145|0.035031|
|control|12.913560|0.035766|
|hec_v2|12.947994|0.035326|
|hec_uniform|12.926559|0.034994|
|sg|12.742903|0.036750|
|conventional_moe|12.146780|0.043448|
|vsoed|12.496234|0.040492|
|uniform_deployment|12.961018|0.035293|
|remove1|12.946613|0.035388|
|remove2|12.943382|0.034951|
|remove3|12.973371|0.035367|
|remove4|12.952347|0.035085|
|dad|10.316674|0.048542|
|rl_sboed|11.160965|0.040036|
|scorebed|11.032075|0.045745|

Primary differences are full RR minus comparison. Ordinary95% t intervals are unadjusted; Holm3 p-values cover the primary family.

```json
{
  "control": {
    "n": 2000,
    "mean": 0.028584692188615392,
    "MCSE": 0.02217111425892815,
    "CI95": [
      -0.01489622002704533,
      0.07206560440427612
    ],
    "p_two_sided": 0.19745120691217657,
    "Holm3_p": 0.5923536207365298
  },
  "hec_v2": {
    "n": 2000,
    "mean": -0.005848805571804182,
    "MCSE": 0.024740585510255663,
    "CI95": [
      -0.054368839981093285,
      0.04267122883748493
    ],
    "p_two_sided": 0.8131424031420729,
    "Holm3_p": 1.0
  },
  "hec_uniform": {
    "n": 2000,
    "mean": 0.015586320684804396,
    "MCSE": 0.02725432222473424,
    "CI95": [
      -0.03786353205201925,
      0.06903617342162804
    ],
    "p_two_sided": 0.5674646185525213,
    "Holm3_p": 1.0
  }
}
```

Mechanistic contrasts: frozen learned actor versus uniform Top2 deployment and four individual adaptation removals. Full trajectories are resimulated; Holm5 applies.

```json
{
  "uniform_deployment": {
    "n": 2000,
    "mean": -0.018873190486645132,
    "MCSE": 0.02183618543907585,
    "CI95": [
      -0.061697256571596464,
      0.0239508755983062
    ],
    "p_two_sided": 0.38752239683774314,
    "Holm5_p": 1.0
  },
  "remove1": {
    "n": 2000,
    "mean": -0.004467966360657127,
    "MCSE": 0.013341994225706023,
    "CI95": [
      -0.030633637284307923,
      0.02169770456299367
    ],
    "p_two_sided": 0.7377507819069813,
    "Holm5_p": 1.0
  },
  "remove2": {
    "n": 2000,
    "mean": -0.0012373730631152807,
    "MCSE": 0.012153712286905698,
    "CI95": [
      -0.02507264317492479,
      0.022597897048694228
    ],
    "p_two_sided": 0.9189174627260144,
    "Holm5_p": 1.0
  },
  "remove3": {
    "n": 2000,
    "mean": -0.031226302202028853,
    "MCSE": 0.022952792242188016,
    "CI95": [
      -0.07624020330467572,
      0.01378759890061802
    ],
    "p_two_sided": 0.1738385190047666,
    "Holm5_p": 0.8691925950238331
  },
  "remove4": {
    "n": 2000,
    "mean": -0.010201808523304977,
    "MCSE": 0.02177340384880645,
    "CI95": [
      -0.05290275040333891,
      0.03249913335672896
    ],
    "p_two_sided": 0.6394463457795049,
    "Holm5_p": 1.0
  }
}
```

Frozen SG/conventional-MoE/vsOED/DAD/RL/ScoreBED contextual comparisons use Holm6 separately and are not training-budget matched. Historical quoted numbers from other banks were never paired with this bank.

```json
{
  "sg": {
    "n": 2000,
    "mean": 0.19924243015810425,
    "MCSE": 0.033696300010797764,
    "CI95": [
      0.1331588835374349,
      0.26532597677877356
    ],
    "p_two_sided": 3.943631778672654e-09,
    "Holm6_p": 3.943631778672654e-09
  },
  "conventional_moe": {
    "n": 2000,
    "mean": 0.7953653202874149,
    "MCSE": 0.050116672880486435,
    "CI95": [
      0.6970789360936255,
      0.8936517044812043
    ],
    "p_two_sided": 1.6381902567417217e-53,
    "Holm6_p": 4.914570770225165e-53
  },
  "vsoed": {
    "n": 2000,
    "mean": 0.4459114900537426,
    "MCSE": 0.04640665524911054,
    "CI95": [
      0.3549010122065838,
      0.5369219679009014
    ],
    "p_two_sided": 2.112233898736085e-21,
    "Holm6_p": 4.22446779747217e-21
  },
  "dad": {
    "n": 2000,
    "mean": 2.6254712652576684,
    "MCSE": 0.05322883234601604,
    "CI95": [
      2.5210814651082583,
      2.7298610654070785
    ],
    "p_two_sided": 0.0,
    "Holm6_p": 0
  },
  "rl_sboed": {
    "n": 2000,
    "mean": 1.7811804497869446,
    "MCSE": 0.04721554696769971,
    "CI95": [
      1.6885836127971103,
      1.8737772867767788
    ],
    "p_two_sided": 1.186391294407419e-235,
    "Holm6_p": 4.745565177629676e-235
  },
  "scorebed": {
    "n": 2000,
    "mean": 1.9100701329740664,
    "MCSE": 0.04989573284140683,
    "CI95": [
      1.8122170456516904,
      2.0079232202964423
    ],
    "p_two_sided": 5.4397558921058475e-241,
    "Holm6_p": 2.719877946052924e-240
  }
}
```
