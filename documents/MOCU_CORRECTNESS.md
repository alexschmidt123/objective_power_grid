# MOCU correctness changes, 2026-09-06

IEEE9 and IEEE14 use the finite loss `C(u,U)=u+20 max(U-u,0)` and
`robust_rule: quantile`, `alpha: 0.05`, zero margin and zero event penalty.
The continuous Bayes action is the posterior 95th percentile. These studies
use a candidate-grid U bank, so the quantile is itself an available control.
The 95% posterior quantile is not a guarantee of 95% safety on unseen systems;
the physical simulator's held-out safety check remains separate.

Fixed calibration, Myopic fantasies, policy training, and evaluation dispatch
on the same rule and loss. CPU and Torch implementations are regression-tested.
Hard-maximum studies remain supported explicitly as `ibr_max`; changing the
rule invalidates old Fixed caches. Fixed cache fingerprints also include the
support, observations, grid, loss and search protocol. New caches are run-local.

`mean_mocu` means held-out realized operational regret, clustered by physical
system. `mean_posterior_mocu` records the model's posterior expectation separately.
New outputs identify `metric_schema: realized_operational_regret_v2`. Historical
posterior and realized results must not be pooled merely because both used the
name `mean_mocu`. Safety validity does not imply non-degenerate decisions.

Before a non-smoke training run, a deterministic random-design screen on the
validation systems checks finite losses and variation in terminal controls.
It writes `diagnostics/mocu_preflight.json`. A constant-control screen stops
training by default. This is not proof that every design is uninformative;
an intentional degeneracy study can explicitly set
`training.objective_based.require_decision_sensitivity: false`.
The screen does not use final test systems or require a learned method to win.

Bank writers lock a stable sibling lock file and generate in a private sibling
directory. Only successfully completed/validated generations are published.
Failed generations remain under `.BANK.building-*` for diagnosis. Replaced
directories remain under `.BANK.previous-*`, preserving open memory maps.
First publication is an atomic rename; replacement has a brief rename gap,
so an unsynchronized new reader might need to retry. No reader sees partially
written new arrays. Existing legacy banks are revalidated but explicitly lack
a verified historical physics fingerprint. New banks record a completion
manifest and reject configuration mismatches. The full-bank helper no longer
skips validation merely because filenames exist.

The control-extension reuse check compares physical settings and theta hashes;
changing a posterior decision rule alone does not require physical regeneration.
Run provenance includes content hashes of source/configuration files, including
untracked source files, even in a deployment without Git.

From the activated `mocu_optimized` environment:

```bash
python tools/test_mocu_correctness.py
python tools/check_mocu_alignment.py
python tools/mocu_preflight.py --config configs/ieee9_mocu.yaml
python tools/mocu_preflight.py --config configs/ieee14_mocu.yaml
```

The preflight tool skips expensive Fixed calibration and does not train a policy.
Its output is a diagnostic, not a comparison result. A bounded integration test
is available as `python tools/smoke_mocu_correctness.py --bank-root "$PWD"`;
it uses a trivial Fixed sequence and tiny training budget, and requires the CUDA
compiler on PATH for the oracle stage. It is not a publication experiment.

On the inspected production banks, the corrected T=3 validation screen found
four control levels for IEEE9 (0.38–0.41) and five for IEEE14 (0.37–0.41).
These establish decision sensitivity only, not learned-policy superiority or
held-out safety certification. Existing results/models have not been rewritten.

## Verification on labpc

- Eight targeted regression tests passed, including baseline loss consistency,
  realized oracle loss, decision preflight, and concurrent bank publication.
- Existing `tools/check_mocu_alignment.py` passed.
- Both production-size subset banks and the dense IEEE9 bank passed quality checks.
- IEEE9 integration smoke completed training and all six method evaluations,
  including the four-system oracle calculation. This used safety proxies and
  a tiny training budget; it does not certify publication safety or performance.
- Smoke output: `experiments/09062026_153610_correctness_smoke_Uctrl_T3_Nobs5_sigma0p005`.
- Pre-change source backup: `/tmp/objective_power_grid_before_mocu_correctness_20260906.tar.gz`.
- Pre-existing uncommitted diff: `/tmp/objective_power_grid_before_mocu_correctness_20260906.diff`.
