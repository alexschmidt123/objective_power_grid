# Revised CER V3 run status

RUNNING on LabPC. Candidate process PID 2946889.

Source/settings frozen at commit d8a6a6c30e36e05c818c4e309fba8e456aab8b6d before scientific updates.
The validated complete-proposal policy was preserved byte-for-byte. The required
four-group gate is removed; 35 supported training-history examples define soft,
overlapping preparation distributions with uniform background exposure.

Preparation completed: 200 steps, 8,000,000 native critic
rows, no new simulator observations. Candidate then matched uniform-router
control each receive exactly 2000 native updates; no sweep or extension.

Observed candidate progress: 400/2000 updates at epoch 1794433885.549.
This is a progress snapshot, not a held-out EIG result. All four experts have
finite gradients. Exposure uses uniform selection initially, then transitions
to value routing with a minimum probability of 0.025 per expert.

After both arms freeze, the pipeline creates one new common bank for 2000 T30
trajectories and 1M prior contrasts. It evaluates the candidate, trained control,
uniform deployment, four individual experts, frozen HEC-v2 and compatible native
baselines. No main evaluation bank has been accessed during training.

LabPC scratch: TEMP/scratch_runs/cer_moe_v3_revised_20261111.
The MacBook completion watcher is active, with SSH access outside the restricted
sandbox. It will publish RESULTS.md/RESULTS.json and safely synchronize reports
through GitHub to UbuntuPC, or preserve work and record the exact blocker.
It does not restart training or run Ubuntu computation.

Higher EIG, useful specialist competencies and routing benefits remain unassessed.
