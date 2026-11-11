# One frozen V3 uniform-router evaluation

No training or parameter changes. Wait for the original V3 pipeline to finish successfully AND exit,then acquire the existing GPU lease. This prevents interference with training,evaluation and original artifact sealing. The waiting queue lives outside the V3 root;results will be added only under uniform_router_evaluation_20261111 after original sealing.

Use the successful final candidate actor and the existing independent common bank:2000 trajectories,T30,1M negatives. Freeze the model;resimulate learned and uniform policies using the same truth,observation noise,categorical uniforms and action-normal draws. Uniform means sample a complete expert with probability.25 at every step,not average physical designs. Changed actions cause new observations and new histories. The learned re-evaluation must match saved same-bank scores within1e-8 before comparative claims.

Report EIG lower-bound mean/MCSE,paired learned-minus-uniform difference and ordinary95%tCI. Report raw p and supplementary Holm3 p combined with the two historical primary contrasts;historical tests/artifacts remain untouched. One trained lineage;bank reused for this explicitly requested supplementary diagnostic,not a fresh final test. No multiple architecture search or subsequent training.

Report selection counts and proportions overall and by stage;paired evaluation wall/CPU time,2second GPU utilization/memory/power and process CPU/RAM samples. Bound evaluation to3600seconds,excluding waiting. Protect final actor,original bank/scores,original source and original artifact catalog hashes before/after. All outputs are new;no old file may be overwritten.

Duplicate safety:one queue lock,one launch receipt,one result directory. If upstream fails/disappears without a successful final checkpoint/bank,report BLOCKED;do not train or invent data. Source is committed/pushed on LabPC,fast-forwarded on Ubuntu and verified before queue launch. No Ubuntu computation.
