# MoE-sBOED-v1 and prior v2 diagnosis

Official historical v1 EIG 10.98906590 +/- 0.04479411 nats and vsOED 12.48084689 are reference figures from a different complete-training/evaluation protocol. They are not the native-v1 short-pilot scores below and are not used for candidate selection.

Recovered prior v2 pilot completed 3216 matched updates. Its four-arm aggregate training cost was 7197.076 seconds, not 7200 seconds per arm. Source revision 4b2bdfc; original checkpoint and source manifests verified. Held-out v2 EIG 9.71034799 +/- 0.08859369, native-v1 9.73445684 +/- 0.08838289; paired difference -0.02410885 +/- 0.08475476. Independent historical confirmation difference -0.08993965 +/- 0.08506299. No v1 improvement was established; do not restart or relabel that completed pilot.

New frozen checkpoint audit at N256/L65536 on seeds97402732/33 found paired v2-v1 differences -0.01307874 +/- 0.08922734 and -0.17333755 +/- 0.08838221. Masking and uniform redistribution hurt the frozen policy, demonstrating component dependence. None of the96 expert/stage/observable-context substitution effects passed within-split Holm. That is limited-power evidence, not proof that all experts are useless. There is no established conditional specialist competence.

Mechanistic limitation: existing v2 readouts decode E_e(z) whose down projection and separate readout both start small. The new single correction exposes z directly to each independent design readout via D_e(z+E_e(z)). This avoids making all routed design credit depend on the product of two small initial feature/readout factors. The unchanged terminal-information objective must decide whether this helps. Different gradients/weights alone will not count as success.

Recovered audit: /home/qiqi/Documents/objective_power_grid_temp/scratch_runs/10272026/10272026_moe_v2_recovered_audit. No official seed1001 or reserved confirmation seed8831001 was used.
