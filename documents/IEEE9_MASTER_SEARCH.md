# IEEE9 MOCU master-bank search

The current priority is IEEE9 MOCU. IEEE14 and IEEE30 master-bank work is deferred. This search uses the existing IEEE9 bank with 281 durations (0.20–3.00 s, step 0.01 s) and all nine injection buses: 2,529 individual designs. A candidate catalog consists of six durations crossed with all nine buses: 54 available designs. A T-step policy chooses T distinct designs from that catalog. This is not selection of six arbitrary individual bus-duration pairs.

The search space is C(281,6)=647,992,090,956 catalogs. A complete master bank makes any on-grid catalog available; it does not make exhaustive policy optimization over every catalog computationally practical. Results must be labeled best found, not globally optimal.

`tools/audit_ieee9_master_search.py` uses the corrected 95% posterior quantile and loss u+20*(U-u)_+-U. It does not use the legacy hard-maximum search script.

## Frozen search protocol

- Load full master observations and join the corrected control extension by exact theta-row identity; verify the current production subset's observations against the master. Preserve all 384 fit-support particles and use the first 64 off-support validation rows only for exploratory search. Do not score final test observations.
- Directly evaluate 1,024 global catalogs at T=3. Include the old 128 candidates and the production baseline. Ensure every one of the 281 durations appears in at least one directly evaluated catalog. Cheap screening uses inner/outer 8/4, six first candidates, all second actions, Fixed calibration 128 and noise seeds 101/202.
- Take up to eight parents by joint adaptive/nonmyopic gain and eight by low lookahead loss, plus baseline. Enumerate all one-duration substitutions across the full 281-duration grid. Directly score up to 256 new neighbors: half selected by response-diversity proxy, half uniformly sampled. The proxy proposes candidates only; it is not a MOCU score.
- Re-evaluate up to six candidates per ranking criterion, plus baseline, using all 54 first/second actions, inner/outer 32/16, Fixed calibration 512/eight restarts and three noise seeds. Select up to two by joint gain and two by low lookahead loss, plus unchanged baseline (at most five finalists). Check their sensitivity at 64/32; report this check without further selection.
- Freeze all finalists before generating 512 new physical validation systems, theta seed 907091701. Check no overlap with master training/test theta. Use the continuous-control oracle to 1e-4, verify two scalar-oracle cases and two reproduced master probe curves, and retain the original prior without resampling unsafe systems.
- Confirm at T=2,3,4,5, noise seeds 11001/11002/11003; full 54-action first/second search, inner/outer 64/32, Fixed calibration 512/eight restarts. Report realized operational regret and physical safety separately. Adjust paired theta-cluster bootstrap intervals across all finalist/horizon/contrast comparisons (up to 60).

Selection considers both absolute loss and space for adaptive/nonmyopic gains. It does not reward apparent branching alone. The empirical screen has been used in earlier development and supports exploratory ranking only. Fresh validation assesses the frozen candidates; choosing another winner after inspecting it would require another confirmation set. Approximate Fixed, rolling two-step planning, finite posterior support and the reduced physical model remain limitations.

Source snapshots and file hashes accompany each submission. Research code stays under `tools/`. The physical master bank is read, not regenerated for each catalog. Fresh confirmation simulates only the union of frozen finalist actions on new theta, avoiding unnecessary full-master replication.
