# Completed frozen ablation validation

All10 variants completed on the exact2000-case/1M-contrast historical V2 bank. Source/checkpoint/bank hashes unchanged;zero training steps. Selector-initial state exactly matches the historical initial state digest. The bridge with original query cores reproduces original V2 actions on B2 at stages0,14,29 before using the fine-tuned query cores.

Variants:initial inherited router,argmax,critic-greedy,three-step commitment,whole-episode commitment,original V2 head/fusion bridge,and each of four heads immediately after distillation. Previous final V3,V2 and standalone score arrays are reused for paired comparisons,not regenerated on different cases. Common truth/noise/routing uniforms and the same full1M prior bank are used.

Existing10 comparison family uses Holm-adjusted p values. Additional bridge-versus-initial and bridge-versus-V2 comparisons are explicitly exploratory. The1M score is ceiling-limited and this is diagnostic reuse,not new acceptance evidence. Frozen bridge is an untrained assembly,not a replacement accepted model. All findings remain within expert-design and routing-quality categories;no unverified coding bug or performance fix is claimed.
