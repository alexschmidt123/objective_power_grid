# MoE-sBOED-v2 residual-context-readout candidate

One unified policy has the original complete-history sum encoder, one shared expert and seven independently parameterized routed experts, with learned top-three sparse selection. Shared output H(z+E_shared(z)) is unchanged. The only architectural correction relative to the previous direct-readout v2 is routed output sum_selected p_e D_e(z+E_e(z)), replacing sum_selected p_e D_e(E_e(z)). No new parameters, scale factor, teacher, role labels, online EIG search, auxiliary competence objective, or independent DAD policies are introduced. The common sigmoid feasible map stays [-4,4]^2.

Candidate total8658, activated6994 parameters/context; native-v1 total8434, activated6898. The separately trained dense width71 control has exactly8658 parameters; shared-only has5634. All arms share fresh native common-module initialization, data draws, Adam5e-5 betas(.8,.998), native.98 schedule every1000 after1, clipping5, B2000/L2000 and full30-stage terminal sPCE. Original native balance coefficient.001 is retained. No checkpoint or DAD pretraining is reused.

Autograd propagates through future simulated observations and complete adaptive histories. Top-k membership is discrete; derivatives are local pathwise derivatives within a fixed top-k branch, omitting membership-boundary terms, as in native-v1. Finite differences test locally stable full-horizon branches. Production inference is a single sparse history-to-design forward pass without online design optimization.

This is a disclosed development revision of v2, not a successful method version or a v3 rename. It remains in TEMP until correctness and scientific development gates support promotion. LabPC source is untouched.
