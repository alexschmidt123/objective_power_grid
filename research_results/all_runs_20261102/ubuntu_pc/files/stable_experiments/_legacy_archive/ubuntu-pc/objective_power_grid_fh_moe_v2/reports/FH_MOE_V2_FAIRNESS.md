# FH-MoE-v2 fairness and limitations

The corrected protocol gives only primary LabPC seed101 a14400-second optimization allowance. It includes every trainable backbone, expert, router and joint optimization stage.
Ubuntu seed102 is matched workload: exactly the primary completed optimizer count with the same frozen source/configuration, batch, contrasts, precision, optimizer and update-based schedule; it has NO independent four-hour cutoff.
Matching updates does not remove different-GPU, different-PyTorch or floating-point effects.
The primary four-hour benchmark—not Ubuntu wall time—defines comparisons with historical LabPC methods.
No main run occurred, so no claim of matching four-hour end-to-end compute or of eliminating hardware effects is made.

Verified original v1 cost:29198updates,14398.167729s charged onRTX4090; B2000,L2000,H30.
FH-v1 costs:26335updates/14397.157810s onRTX4090,20575updates/14396.959129s onRTX5080. Those were two independent four-hour runs; the new matched-workload protocol is different.
Previously available parameter-matched dense control has8418parameters versus8434MoE and a different historical update count/throughput. Shared-only is smaller. These are useful references, not perfectly retrained causal controls.
Frozen mask/uniform interventions keep the trained checkpoint and are not equivalent to independently training smaller policies.

Historical canonical audit:
- vsOED:12.48084688744761±.04168005688526;14400s allowance,6805.638266s reported training,planned schedule completed early.
- RL-sBOED:11.13932941864758±.04128579391160;14398.285173s reported training under14400s allowance.
- ScoreBED:10.94043710161673±.04682270363223;13698.721751s reported training; clock components record both policy and score learning.
Their native algorithms have different work, trajectories and optimization components; equal optimizer counts must not be imposed across methods. The saved cost ledgers are included.
No v2 official comparison exists after the failed pilot.
Seed1001 is a historically reused benchmark, not untouched confirmation data.
Case-level paired SE requires identical truth/noise/order/contrast identity; the bounded pilot used its own common development bank and saved paired arrays.
No training-seed variance can be inferred from a single development pilot; the two earlier FH runs additionally used different GPUs and update counts.
Any future untouched confirmation bank must be reserved before seeing new official results.
