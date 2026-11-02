# Optional straight-through DSelect-1 training

This is an opt-in training setting of **MoE-sBOED** (`moe`), not another
named method. Default relaxed training and historical checkpoint loading are
preserved. The setting is rejected for the three other family variants.

Use `moe_training_routing_mode: straight_through` in the MoE method settings;
the model-level argument is `training_routing_mode='straight_through'`.

## Forward and backward behavior

Let `p` be the existing relaxed DSelect-1 weights and `h` its native hard weights.
Training uses:

```python
w = h.detach() + (p - p.detach())
action_latent = sum(w[j] * expert[j](context[j]) for j in range(4))
```

The forward action is hard Top-1, including the native `logit >= 0` tie
convention. The backward routing derivative follows the relaxed weights.
All four proposals must be computed during training: sparse masking before
the weighted sum would remove unselected proposal contributions to the router
gradient. Only selected experts receive direct action gradients; complete
trajectory backpropagation also includes their earlier selected actions.
Hard inference retains the existing single-expert sparse dispatcher.

This is a **biased routing-gradient surrogate**, not a new EIG estimator.
Full-horizon sPCE is unchanged. Native load and selector-entropy penalties
continue to use relaxed probabilities, rather than hard usage frequencies.
Gamma, architecture, optimizer, clipping, first design and bounding are unchanged.

## Controlled development experiment

The authorized experiment trains seeds **103 and 105**, exactly 1,000,000 primary
episodes each, with final-budget checkpoint selection. Seed 103 tests the failed
training case; seed 105 checks whether a previously successful case is harmed.
The pair is selected development evidence, not a representative five-seed study.

```bash
bash run.sh --config configs/location_moe_st_development.yaml \
  --journal --variant moe --seed 103 --eval-seeds 1001 --eig-only
bash run.sh --config configs/location_moe_st_development.yaml \
  --journal --variant moe --seed 105 --eval-seeds 1001 --eig-only
```

Run sequentially on one GPU. These commands train from scratch and allocate
separate dated outputs; they do not replace old runs. The actual campaign uses
an isolated hashed source snapshot and exact command records under:

`experiments/10172026/10172026_moe_st_routing_dev_103_105/`

Checkpoint monitoring uses the common 512-case/L=1,000,000 development bank
(seed 771001), with paired hard/relaxed comparisons before and after 800k.
Final development comparison uses 2,000 paired trajectories, L=1,000,000,
seed 1001. Seed **8831001 remains reserved and unused**.

Report signed and absolute hard/relaxed gaps, original-versus-intervention
paired differences, expert utilization, gradient exposure, selector saturation,
actual Adam updates and physical designs. A late stable policy dominated by one
expert does not demonstrate beneficial expert specialization. Keep negative
results and numerical failures; do not change settings or retry favorable seeds.
If promising, recommend an unchanged five-seed replication before adoption.

## Correctness checks

```bash
python -m pytest tools/tests/test_st_dselect1.py tools/tests/test_moe_family.py \
  tools/tests/test_strategy_moe.py tools/tests/test_moe_architecture_controls.py -q
```

Checks cover native hard weights/ties, relaxed surrogate derivatives,
initialization, hard-forward rollouts, unchanged relaxed regularization,
unselected-proposal routing gradients, selected-expert gradients, sparse hard
inference, unsupported variants and legacy/new checkpoint compatibility.
The campaign also verifies historical seed-103/105 predictions and a real
batch-64/L=1024 forward/backward. Full-rollout equivalence tolerance is
`rtol=atol=1e-12`; dense/sparse batched arithmetic differs by about 1e-14.
