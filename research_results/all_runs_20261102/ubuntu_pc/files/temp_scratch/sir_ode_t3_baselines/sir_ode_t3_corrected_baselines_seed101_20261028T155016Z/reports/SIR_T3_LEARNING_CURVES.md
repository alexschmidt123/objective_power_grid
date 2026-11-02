# T3 learning curves

Fixed development cases:128 trajectories,65536 contrasts. Checkpoints are time-selected, never selected by favorable EIG. MC SE does not establish training-seed uncertainty.

|Method|Updates|Charged seconds|Development EIG|MC SE|
|---|---:|---:|---:|---:|
|fixed|12564|1800.044|1.699285|0.159702|
|fixed|25552|3600.027|1.970670|0.167500|
|fixed|50000|6822.319|2.542617|0.175816|
|rl_sboed|343|1801.828|6.426356|0.218903|
|rl_sboed|685|3601.433|8.879814|0.118003|
|rl_sboed|1382|7204.612|9.076067|0.105787|
|rl_sboed|2766|14370.575|9.096383|0.102239|
|vsoed|38366|1800.169|9.419448|0.104880|
|vsoed|76083|3600.197|9.409733|0.103450|
|vsoed|149437|7200.267|9.412480|0.104493|
|vsoed|286906|14370.490|9.412480|0.104493|
