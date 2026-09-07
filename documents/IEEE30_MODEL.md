# IEEE30 reduced dynamic study model

This is an explicitly constructed dynamic extension of the **original IEEE CDF 30-bus network**, using MATPOWER `case_ieee30`, not MATPOWER `case30` (which relocates generators). Source: https://github.com/MATPOWER/matpower/blob/master/data/case_ieee30.m (retrieved 2026-09-07). The downloaded source is retained at `tools/reference_data/case_ieee30.m`; its SHA256 is in `configs/ieee30_mocu.yaml`.

The original case lists six online machine terminals: **1, 2, 5, 8, 11, 13**. Buses 1 and 2 have positive scheduled active generation; the other four have zero scheduled active generation and supply reactive power. We retain all six as rotating machines, including condenser terminals. A zero active dispatch is not a reason to remove rotating inertia.

Only these six retained machines have M and K coordinates, so N=6 and theta has 12 coordinates. The other 24 physical buses have no independent inertia or fast-response latent variables. Their injections are mapped through the algebraic Kron reduction. All 30 physical buses can be probed. Observation is at retained physical bus 1. Control bus 0 is reduced index 0, physical bus 1.

The 41 branch reactances define a lossless, unit-voltage network with nominal transformer taps. Resistance, shunts, voltage/reactive dynamics and AC operating-point angles are omitted, consistently with the project's IEEE9/14 reduced swing approximation. The equilibrium is zero angle with consistent zero deviation injections. This is not a reproduction of the original case's full AC transient dynamics.

The power-flow source does **not** supply dynamic M/K data. The following are synthetic, prespecified study assumptions:

- Total nominal 2H is 51 seconds on the study's common 100-MVA base, matched to IEEE14. Each of six machines has nominal 2H=8.5 seconds, M=8.5/(2*pi*60), with independent +/-30% uncertainty.
- Each terminal has an independent K in [0.05/6, 0.50/6], preserving the existing aggregate response range. K represents fast active frequency support in the existing reduced equation. In particular, positive K at a condenser terminal assumes supplementary fast active support; it is **not** a claim that an ordinary synchronous condenser has a turbine governor.
- Fixed D_i=0.1*M_nominal. The contingency and 95% quantile finite-loss rule match IEEE14. Baseline durations [0.3,0.6,1,1.5,2,2.5] are prespecified, not claimed optimal.

This is appropriate for a controlled method/scaling study, with the above assumptions disclosed. Claims about actual IEEE30 dynamic security require an independently justified dynamic data set. Moving from IEEE14 to IEEE30 changes the physical network from 14 to 30 buses, the state from five to six machines, and the six-duration catalog from 84 to 180 actions; it does not create 30 uncertain machines.

Validation checks six-dimensional priors, 30x6 injection mapping, identity at retained buses, conservation, connectivity, an independent full-network linear solve against the reduced solve, rejection of incorrect maps/N, and CUDA trajectories against adaptive CPU ODE integration.
