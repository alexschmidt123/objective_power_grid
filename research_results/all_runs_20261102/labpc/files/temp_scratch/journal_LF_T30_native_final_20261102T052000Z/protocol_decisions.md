# Prospective protocol decisions

Fixed/DAD schedules verified from https://github.com/ae-foster/dad/blob/main/README.md and location_finding.py; Adam betas .8/.998; schedule decay at update indices0,1000,... . Bounded common design map and source-prior initialization are explicit project adaptations; not bitwise author environment reproduction.

vsOED uses verified native **G-I**, not NF-I. The author README documents GMM incremental T30/n-incre30/discount.9. Local adapter explicitly use_NFs=False; introducing NF without validation is not warranted. 10001cycles plus20000 initial replay,1000fresh/cycle,5posterior/5critic, native learning settings retained. Native completion does not prove convergence.

ScoreBED authoritative pinned source cc8e996e3ee3237ab4c9bd3ef80e2e4b4aa9cee4 config/experiment/location_finding.yaml specifies **50000 score updates,20000 policy updates per restart,5 restarts**. It contradicts the proposed190000/10000/P1 setting. Use the verified pinned configuration including default random design mixture and batch1024. Native objectives/architectures/EMA remain unchanged. Independent diagnostic restart selection is charged in parent wall time; final common dev and sealed final banks never select restarts. Count simulator calls from executed ledger, not assumptions. Existing190k/P1 partial checkpoints are not reused.

RL: completed native2Mseed101 reused; native SAC/REDQ kernels/replay/critics verified in existing tests.2M is a prospective feasible project ceiling (~7h37m), not a paper convergence default. Replay/optimizer exact resume is unavailable; completed uninterrupted run reuse does not claim exact resume. Learning curve and remaining uncertainty disclosed.

No final-bank evaluation until the user freezes MoE. Existing four-hour/equal2M/partial runs remain separate unchanged evidence. This cohort is not equal-compute. Native-target completion and convergence are distinct.
