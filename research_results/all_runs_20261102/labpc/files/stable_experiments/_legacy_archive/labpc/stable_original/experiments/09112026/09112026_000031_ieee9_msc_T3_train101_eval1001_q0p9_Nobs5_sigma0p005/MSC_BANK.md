# IEEE9 MSC bank for this campaign

This campaign uses posterior MSC with coverage 0.9. Coverage affects control selection, not the physical safety arrays.

- `banks/physical_subset/`: clean probe responses for the requested six durations crossed with nine probe buses (54 actions), extracted from the dense master bank. These responses were not resimulated.
- `banks/control/`: newly simulated control bank for this MSC campaign; 512 training-bank systems and 128 test systems, each evaluated at 27 control levels (17,280 control simulations).
- `u_optimal.npy`: model-specific minimum safe control on the candidate grid. `psi_star.npy` is the legacy equivalent field.
- `control_safe.npy`: safety of each candidate control for each model.
- `control_nadir.npy`, `control_rocof.npy`: physical frequency and RoCoF outputs.
- `theta_M.npy`, `theta_K.npy`: corresponding physical parameters.

The shared generator may print "MOCU control extension"; this is a legacy component name, not the training objective. This campaign trains and ranks by MSC. The control bank does not store a single universal posterior MSC: that decision is computed from each method's observation history at coverage 0.9.

The run configuration and provenance are in `manifest.json` and `source_snapshot/configs/ieee9_mocu.yaml`. Bank completion/provenance records are under each bank's `meta/` directory. Existing bank paths are preserved while the run is active.
