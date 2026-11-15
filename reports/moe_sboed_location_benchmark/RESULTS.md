# MoE-sBOED location-finding benchmark

Canonical proposed method: `moe-sboed`. Internal baselines: `moe-sboed-uniform`, `moe-sboed-e1`, `moe-sboed-e2`, `moe-sboed-e3`, `moe-sboed-e4`. E1-E4 inherit the corresponding learned checkpoint and require no independent training campaign. They remain separate inference baselines. Historical V2/V3 names are provenance, not separate proposed methods.

Six training families: moe-sboed, moe-sboed-uniform, vsoed, rl-sboed, scorebed and dad. Five target seeds:101-105. Existing learned checkpoints must not be classified as uniform. The user-identified five learned checkpoints remain subject to path/source/config/pretraining verification;their existence does not establish five uniform runs.

New fixed-uniform seed101 completed2000 native updates in8272.66s and evaluated at12.88328257 nats. It inherits the common pretrained V2 parent and is continuation evidence,not an independent end-to-end seed. No remaining training started automatically after this run. Old pending V2/router-only/encoder-only jobs are excluded.

The previous controller exited during report publication because CSV CRLF triggered Git whitespace checks. This did not affect training/checkpoints/EIG. Original reports were preserved,CSV line endings normalized with parsed-row equality verified,and the old controller will not be restarted.

The exact original evaluation seed is202611099206,verified by reproducing all saved truth/noise/contrast arrays. Do not replace it with1001. Primary bank/scorer/physics remain unchanged. Methods share truth/base noise,not resulting observations. No test-bank model selection.

MAIN_RESULTS.csv intentionally contains PENDING until checkpoint provenance and evaluation are verified. Historical >4h RL runs are not silently counted as four-hour-compliant. No superiority claim. No SIR or power-grid jobs are authorized until this benchmark is complete.
