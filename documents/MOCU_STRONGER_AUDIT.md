# Stronger IEEE9 duration audit

This campaign preserves the finite-loss 95% quantile objective while checking
whether the earlier 32-candidate screen was limited by planning approximation
and validation uncertainty. It does not train DAD or change production probes.

The `prepare` stage first reevaluates the previous three finalists and the
baseline at T=3 on the old 64-system screening split, using three planning
budgets: current 32/16 fantasies with 12 first candidates; 32/16 with all 54
first candidates; and 64/32 with all 54 first candidates. The latter two also
use Fixed calibration=512 and eight restarts, versus 256/four in the first.
Compare the latter two to isolate the fantasy-budget effect. These remain
approximate planners and Fixed designs; unchanged results do not certify
convergence or global optimality.

It then screens 128 reproducibly sampled six-duration sets from all 281
options, including the original baseline and the earlier 32 candidates.
Screening remains cheap (8/4 fantasies, six first candidates) and uses old
validation only. Three finalists are selected by the smaller of the combined
and non-myopic gains, and the baseline is retained. All candidates and outcomes
are saved. This is not an exhaustive search of C(281,6).

After the finalists are frozen, `fresh` samples 256 new systems from the same
physical prior, seed 906202611. Exact latent rows are checked against existing
train/validation/test rows without using test observations for scoring.
Only the union of finalist probe actions is simulated, storing five sampled
observations rather than another complete dense trajectory bank. Two existing
systems are reproduced against the original dense bank before generation.
Infeasible or nonmonotone fresh control cases fail the campaign; they are not
discarded or resampled to improve reported safety.

The batched control oracle matches the existing scalar first-safe-bracket
algorithm, including its dense scan and local bisection at tolerance 1e-4.
Regression cases include monotone safety, safe-at-zero, a safe island and
infeasibility. Fresh generation additionally compares two physical systems
against the scalar oracle. The oracle is numerically refined, not an analytic
certificate of a globally minimal continuous safe control.

Each finalist is evaluated at T=2,3,4,5 on the same 256 new systems and noise
seeds 11001–11003, using the 64/32, all-first-actions planner and strengthened
Fixed calibration. The primary metric is realized finite loss relative to
the refined continuous-control oracle. Bank-grid regret, posterior risk,
physical safety at the chosen grid control, and posterior ESS are retained.
Safety is empirical under the specified simulator/prior/noise model; the
95% posterior decision rule does not certify 95% model-mismatch coverage.
Paired theta-cluster intervals and Bonferroni-adjusted intervals across all
candidate/horizon/contrast comparisons are reported. Noise repeats are not
counted as independent physical systems. Intervals remain conditional on the
posterior bank, calibration and planning randomness.

`tools/audit_ieee9_campaign.py` has stages `prepare`, `fresh`, `confirm`, and
`finalize`. Use `hprc/ieee9_stronger_audit_stage.slurm` with Slurm dependencies;
confirmation is an array with four slots (an unused slot exits successfully).
A frozen source snapshot and hashes accompany the campaign. The final report
is written only after all selected candidates and horizons are complete.
Existing SIR reevaluations use their separate frozen snapshot.

A four-fresh-system end-to-end smoke completed on labpc, including all four
stages and both physical reproduction/scalar-oracle checks. It verifies the
pipeline, not the scientific result of the production campaign.
