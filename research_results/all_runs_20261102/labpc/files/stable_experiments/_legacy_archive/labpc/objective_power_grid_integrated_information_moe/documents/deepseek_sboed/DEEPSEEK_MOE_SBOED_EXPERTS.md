# Expert evidence

Architecture: one384parameter always-active sharedfeatureFFN andseven independent384parameter routedfeatureFFNs. Each receives16shared historyfeatures and returns16hiddenfeatures. Router has112parameters; it predicts7affinities, not posteriorprobabilities or experimentEIG. Selectedscores retainfullsoftmaxvalues. Each selectedFFN receivespathwise sPCE and sharedencoder gradients; unselectedFFNs are not executed for thatcontext. No predeterminedroles areassigned.

Readinessonly: fullnativebatch2000/contrasts2000/T30 passed with finitegradient32.4718,11.1985GBpeakallocatedGPUmemory andall7expertsusedacrossthebatch. This is not trainingadequacy orspecialization evidence.

Planned savedmeasurements: selectionfrequencies,gateentropy/effectiveexperts,per-expertgradientexposure,hiddenfeaturecosine/variance, shared/routed-removal development shifts. Routingusage and unequalfunctionweights areinsufficient evidenceofusefulspecialization. Cheaptrainedcontrols at500updates may indicate earlyeffects but cannot support full-budgetcausalclaims.
