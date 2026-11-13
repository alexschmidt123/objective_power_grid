# Pretraining validation on LabPC

The actual frozen V2 runtime and actor checkpoint were hash-checked and restored on LabPC GPU. No optimizer steps were taken in this validation.

- Saved histories from eight episodes at stages 0, 14 and 29 were used (24 states).
- Freezing non-router actor weights preserved forward actions bitwise.
- Exactly five parameter tensors remain trainable: coordinator layer weights/biases and response_gate.weight, totaling 25,924 parameters.
- Existing actor optimizer membership and moments were retained.
- Backpropagating the unchanged native actor loss `-critic([history, actor(history)]).mean()` gave coordinator gradient norm 0.0116657712 and response-gate gradient norm 0.0074105357. All gradients were finite; every frozen actor parameter had no gradient.
- Learned versus uniform routing changed final actions, maximum absolute difference 0.3002973795 on these states.

Validation receipt: `scratch_runs/hec_v2_router_only_20261112_validation/VALIDATION.json` on LabPC. These checks establish that the controlled intervention works computationally; they do not establish an EIG improvement. The experiment itself asserts native optimizer counts, actual router updates and frozen-weight preservation before publishing results.
