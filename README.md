# MoE-sBOED research project

The current target is **MoE-sBOED-v1**, the validated DeepSeekMoE-based direct experimental-design policy. It contains one shared expert, seven routed experts, top-three sparse routing and a common design head. Foundation: `13681d7091189a4e68056e6a2864dc58f9d5700b`.

Current comparisons contain six conventional baselines—Random, Optimized Fixed, Corrected DAD, Corrected RL-sBOED, Corrected vsOED and Corrected ScoreBED—and this single MoE target. Location v1 is structurally valid; useful specialization is unproven. SIR adaptation and its MoE result remain **PENDING** independent validation. No legacy checkpoint is substituted.

Read [project scope](docs/MOE_SBOED_PROJECT_SCOPE.md), [version registry](docs/MOE_SBOED_VERSION_REGISTRY.md), [architecture audit](reports/MOE_SBOED_ARCHITECTURE_AUDIT.md), [legacy exclusion inventory](reports/MOE_SBOED_LEGACY_EXCLUSIONS.md), and [active methods CSV](reports/MOE_SBOED_ACTIVE_METHODS.csv).

Safe registry inspection:

```bash
bash run.sh --list-current-methods
python -m src.research.project_scope inspect-archive /absolute/path/to/historical/checkpoint.pt
```

Current conventional configurations are `configs/current_location.yaml` and `configs/current_sir.yaml`. Current v1 has a separate verified native dispatch (`--moe-sboed-v1` with `configs/deepseek_sboed_location.yaml`). This cleanup starts no training or official evaluation. Use machine-local GPU locks and authorization for future experiments.

Historical PIMoE, residual/switch, RC/IR/FH and generic SIR MoE experiments are `LEGACY_EXCLUDED_FROM_CURRENT_PROJECT`. They are not renamed to v2/v3/v4. Unchanged original source/configuration/manuscripts are in [the checksum archive](historical/project_scope_20261026/ARCHIVE_INDEX.md); dated research records and experiment/checkpoint files remain preserved at their original locations. Current plotting and launchers do not include them.

Develop in isolated UbuntuPC worktrees; integrate, commit and push only through LabPC. Never edit active experiment source or dirty historical trees. Do not merge main automatically. Ordinary DAD, simulator, likelihood, baseline and evaluation utilities remain available.
