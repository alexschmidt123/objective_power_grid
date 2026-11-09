# Dynamic command and coordination rule

Commands c=tanh(C([S(h),a0/4,belief_summary])). Zero is the fixed control command.

- c1 tilts mixture-region weights by standardized component locations. It changes which posterior region the geometry expert summarizes.
- c2 controls a normalized spatial probe direction and a soft tilt over plausible hypotheses. The discrimination expert recomputes exact likelihood-derived features at a0 and a0±direction.
- c3 selects a normalized four-parameter direction and a spatial query offset. The sensitivity expert recomputes Fisher information at that query.
- c4 changes prospective spatial probe direction/radius and conditions the sequential expert's remaining-horizon calculation.

These operations occur before the expert networks; commands are not renamed fusion weights. Smooth tilts and bounded query offsets permit the original actor-value derivative to train commands. Known physical formulas are differentiable. No selection teacher, top-1 rule or utilization target is used.

Candidate alpha=softmax(10 tanh(G([S,z1,z2,z3,z4])/10)); control alpha uses only S. All positive feature paths remain available. The fusion head decides how command-conditioned features correct the provisional design.

History dependence is tested by replacing commands with calibration stage means or independent stage-matched donor commands, and by replacing coordination with stage means, uniform weights, or a calibration-only linear log-gate projection from history. The latter is an inference diagnostic, not an EIG-trained ordinary router. The matched control is the actual EIG-trained ordinary-router comparator.
