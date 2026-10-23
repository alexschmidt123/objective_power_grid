# Source audit

Parent source354021f. Historical branches unchanged. PIMoE online search is in src/research/pimoe.py:184-224, PIMoEPlanner.select:24candidate locations,3refinements,2shortlisted actions and2hypothetical observations, posterior updates and one-step future information max. This is online search, not an amortized actor.

CorrectedIC source src/research/pimoe_ic.py:18-45 preserves normalized predictive-region mixture; :56-72 trains mixture/conditional NLL plus independent information regression and resolved ranking. Region responsibilities are posterior-weighted hypotheses, not action-policy competence. No fixed expert task roles. Density experts have independent10-48-48-9tanh networks, router11-32-4,13800totalparams. Corrected log-domain conditional normalization avoids the original rare-expert ESS defect.

Original localDAD in src/research/dad_location_policy.py:11-69 uses3-256ReLU-16 summed history and16-2emitter; empty-history zeros map through emitter. It outputs a design directly. Local MoE-v2-family mechanism in src/research/moe_family.py:175-259 includes residual composition/frozen-base paths and straight-through option. This is a code-path comparison, not re-certification of the historical worker checkpoint. Historical relaxed-routing results remain relaxed. SA-MoE imports neither residual base nor IC density model.

NativevsOED deployment src/research/upstream_vsoed.py:95-110 encodes measured actions/observations and stage then calls actor(x), with no deployment experiment optimization.
