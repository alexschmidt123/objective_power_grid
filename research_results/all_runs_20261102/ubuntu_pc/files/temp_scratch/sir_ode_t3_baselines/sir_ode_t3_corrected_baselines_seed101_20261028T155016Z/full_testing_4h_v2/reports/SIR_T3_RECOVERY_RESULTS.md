# SIR T=3 reviewed recovery

Final EIG uses2048paired trajectories and1,000,000contrasts, exact copied frozen prior/noise arrays. Development scores excluded. One training seed; MCSE excludes contrast bias and training-seed variance.

|Method|Training|Updates|Charged seconds|EIG|MCSE|Evaluation|
|---|---|---:|---:|---:|---:|---|
|random|NO_TRAINING_REQUIRED|0|0|4.221045908417774|0.06377721796735665|FINAL_CONFIRMATION_2048_L1000000|
|fixed|COMPLETE_WITHIN_CAP|112295|14370.046710351948|3.9125658588323944|0.0503851415922085|FINAL_CONFIRMATION_2048_L1000000|
|dad|COMPLETE_WITHIN_CAP|772|14382.777697022539|5.098577778499339|0.05028283637647344|FINAL_CONFIRMATION_2048_L1000000|
|rl_sboed|COMPLETE_WITHIN_CAP|155712|14370.581867771922|9.162122414281757|0.03298275499740154|FINAL_CONFIRMATION_2048_L1000000|
|vsoed|COMPLETE_WITHIN_CAP|286906|14371.236601116834|9.548669082626057|0.03246370109100792|FINAL_CONFIRMATION_2048_L1000000|
|scorebed|COMPUTE_LIMITED_NOT_TRAINED|0|0|||UNAVAILABLE|

ScoreBED remains COMPUTE_LIMITED_NOT_TRAINED: full50000score steps plus500policy steps for each of10restarts require approximately37993.69seconds(10.55h), beyond14400seconds. No shortened pretraining/restarts or longer run launched. DAD uses full nativebatch2000/contrasts2000 and latest serialized eligible checkpoint. Fixed/RL/vsOED records are preserved.
