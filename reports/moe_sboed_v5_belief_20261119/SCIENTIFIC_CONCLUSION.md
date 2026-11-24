# New V5 primary canonical conclusion

- newV5_minus_uniform: +0.04341 nats;95% paired case CI [-0.01200,+0.09882].
- newV5_minus_previous_learned: -0.02417 nats;95% paired case CI [-0.08316,+0.03482].
- newV5_minus_previous_uniform: +0.02607 nats;95% paired case CI [-0.03216,+0.08429].
- newV5_minus_vsoed: +0.38922 nats;95% paired case CI [+0.29325,+0.48519].
- uniform_minus_V4uniform: -0.01734 nats;95% paired case CI [-0.07695,+0.04227].

Primary absolute-EIG improvement was not established. Stop this pilot;do not expand or tune against the canonical bank.

All numbers use the original canonical bank and frozen1M-contrast scorer. These are finite-contrast lower bounds with saturation;MCSE excludes contrast bias. This is one continuation seed,not end-to-end independent training or across-seed robustness.

Belief proper-score improvement is not EIG improvement. Importance-reference and critic-ranking diagnostics are finite/noisy;the critic objective was not changed. Actor/target belief refresh happens only between offline training blocks. Final deployment freezes all weights/buffers and passes the executable no-online-optimization audit. No test-time fitting,action search,MCMC or gradient optimization is used.
