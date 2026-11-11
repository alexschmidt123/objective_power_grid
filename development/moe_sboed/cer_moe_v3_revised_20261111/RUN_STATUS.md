# Revised CER V3 run status

Validated single revised candidate and matched uniform-router control.
Scientific training launch is pending protocol lock; no performance claim.

LabPC scratch destination: TEMP/scratch_runs/cer_moe_v3_revised_20261111.
Native training: 2000 updates per arm, candidate then control, 10000 replay
episodes per update, gamma1 pure variational information. One independent
2000-trajectory/1M-contrast evaluation follows both frozen endpoints.

The MacBook completion watcher publishes aggregate RESULTS.md/RESULTS.json and
then safely synchronizes LabPC → GitHub → UbuntuPC. It does not restart training,
launch additional research, overwrite dirty work or run Ubuntu computation.
