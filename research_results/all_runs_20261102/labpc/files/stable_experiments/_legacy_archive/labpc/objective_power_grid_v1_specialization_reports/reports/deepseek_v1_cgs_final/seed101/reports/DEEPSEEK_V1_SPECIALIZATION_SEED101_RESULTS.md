# DeepSeekMoE-v1 contextual expert learning: seed101

Official EIG **11.04404166 +/- 0.04281140 nats**, seed1001,N2000,L1000000,H30.
Independent confirmation seed97763001: **11.04982475 +/- 0.04480642**.
All uncertainties are trajectory MC SE conditional on fixed contrast banks and this trained checkpoint.

| Reference | New minus reference | Paired MC SE | Approx95% CI |
|---|---:|---:|---|
|v1|+0.05497576|0.04543997|[-0.034086585569130384, 0.14403810159794309]|
|vsOED|-1.43680523|0.05665340|[-1.5478459040870414, -1.3257645576109809]|
|corrected_DAD|+0.76792879|0.05803476|[0.6541806692113336, 0.8816769206741916]|
|dense|-0.17435370|0.03982771|[-0.25241600260113123, -0.09629139713382051]|
|shared_only|-0.12693568|0.04851461|[-0.22202432512715825, -0.031847037733543124]|

Architecture is exactly the original v1 DeepSeekPolicy:8434 parameters,1shared+7routed SwiGLU experts,top3,unchanged encoder/head. The only learning change is context-protected,norm-preserving routed-expert gradients. Main initialization was fresh seed101. No pilot weights,teachers,IR/RC/SA/PIMoE architecture/training code or competitor policies were used.

Main updates26335; selected26335; charged seconds14397.157810 of14400.
Measured specialization subset1467.958009s INSIDE that allowance; no additional simulator forward calls are introduced by the two extra context backward passes.
Training trajectories/counts:{"fresh_complete_primary_trajectories": 52672000, "true_parameter_stages": 1580160000, "contrast_parameter_stages": 3160320000000, "scalar_conditional_likelihood_evaluations": 3161900160000, "policy_optimizer_updates": 26336, "shared_FFN_contexts": 1580100000, "routed_FFN_contexts": 4740300000, "specialization_context_backward_passes": 5266, "specialization_steps": 2633, "specialization_expert_adjustments": 7879, "specialization_step_microseconds": 1467958009}. Expert effective update counters:[26335, 26335, 26334, 26335, 26335, 26335, 26335].
No raw-SGD descent alignment check guarantees a finite Adam step improves EIG. No statistical guarantee of superiority is made.

Learned minus uniform-subset development routing:{"difference": 0.14470255312420638, "paired_MC_SE": 0.054029059945061735, "CI95": [0.038805595631885376, 0.2505995106165274], "N": 512}.
Full minus routed-masked:{"difference": 0.2408626011525236, "paired_MC_SE": 0.06430946510360083, "CI95": [0.11481604954946595, 0.3669091527555812], "N": 512}.
Development comparisons:{
  "full": {
    "EIG": 10.87145651219878,
    "MC_SE": 0.06693858978936085,
    "paired_intervention_minus_full": {
      "difference": 0.0,
      "paired_MC_SE": 0.0,
      "CI95": [
        0.0,
        0.0
      ],
      "N": 512
    },
    "replicates": 1,
    "metadata": [
      {
        "seed": 97663002,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
        "seconds": 3.492156221996993,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": false
  },
  "uniform_weights": {
    "EIG": 10.85271808663419,
    "MC_SE": 0.06706024885541954,
    "paired_intervention_minus_full": {
      "difference": -0.018738425564589857,
      "paired_MC_SE": 0.019602475092267542,
      "CI95": [
        -0.05715927674543424,
        0.019682425616254523
      ],
      "N": 512
    },
    "replicates": 1,
    "metadata": [
      {
        "seed": 97663002,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
        "seconds": 3.4723024500417523,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": true
  },
  "uniform_subset": {
    "EIG": 10.726753959074575,
    "MC_SE": 0.06793887529159494,
    "paired_intervention_minus_full": {
      "difference": -0.14470255312420638,
      "paired_MC_SE": 0.054029059945061735,
      "CI95": [
        -0.2505995106165274,
        -0.038805595631885376
      ],
      "N": 512
    },
    "replicates": 3,
    "metadata": [
      {
        "seed": 97663002,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
        "seconds": 3.5652446770109236,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      },
      {
        "seed": 97663002,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
        "seconds": 3.5613651720341295,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      },
      {
        "seed": 97663002,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
        "seconds": 3.561345048015937,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": true
  },
  "mask_routed": {
    "EIG": 10.630593911046258,
    "MC_SE": 0.07609758149466192,
    "paired_intervention_minus_full": {
      "difference": -0.2408626011525236,
      "paired_MC_SE": 0.06430946510360083,
      "CI95": [
        -0.3669091527555812,
        -0.11481604954946595
      ],
      "N": 512
    },
    "replicates": 1,
    "metadata": [
      {
        "seed": 97663002,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
        "seconds": 2.839269481948577,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": true
  },
  "mask_shared": {
    "EIG": 10.567437367514101,
    "MC_SE": 0.0799597935178553,
    "paired_intervention_minus_full": {
      "difference": -0.3040191446846787,
      "paired_MC_SE": 0.0780166793150318,
      "CI95": [
        -0.45693183614214106,
        -0.15110645322721641
      ],
      "N": 512
    },
    "replicates": 1,
    "metadata": [
      {
        "seed": 97663002,
        "systems": 512,
        "contrasts": 262144,
        "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
        "seconds": 3.4215537660056725,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": true
  },
  "frozen_v1": {
    "EIG": 10.89513355490786,
    "MC_SE": 0.06717459290048801,
    "new_minus_v1": {
      "difference": -0.023677042709079665,
      "paired_MC_SE": 0.07390957270036504,
      "CI95": [
        -0.16853980520179512,
        0.12118571978363579
      ],
      "N": 512
    },
    "metadata": {
      "seed": 97663002,
      "systems": 512,
      "contrasts": 262144,
      "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
      "seconds": 3.473086123005487,
      "forward_particle_stages": 4026547200,
      "device": "cuda:0",
      "mode": "hard_deployed_policy",
      "secondary_parameter_inference": false
    }
  },
  "dense": {
    "EIG": 10.91795330948066,
    "MC_SE": 0.06702999217049162,
    "new_minus_control": {
      "difference": -0.046496797281880844,
      "paired_MC_SE": 0.06844557303016627,
      "CI95": [
        -0.18065012042100675,
        0.08765652585724505
      ],
      "N": 512
    },
    "counts": {
      "total_parameters": 8418,
      "activated_parameters_per_context": 8418,
      "shared_experts": 1,
      "routed_experts": 0,
      "top_k": 0,
      "expert_feature_dimension": 16,
      "routed_FFN_width": 8,
      "shared_FFN_width": 66,
      "FFN_linear_FLOPs_per_context": 6336,
      "history_encoder_linear_FLOPs_per_observation": 9728,
      "router_linear_FLOPs_per_context": 0,
      "action_head_linear_FLOPs_per_context": 64,
      "FLOPs_definition": "2perlinear multiplyaccumulate; excludesnonlinearities,normalization,topk,dispatch,simulator,likelihood; not measuredGPUFLOPs"
    },
    "checkpoint_sha256": "608499782d0ac872089e3488761cc00bfe8d159fbecc4150c50377330e295d95",
    "metadata": {
      "seed": 97663002,
      "systems": 512,
      "contrasts": 262144,
      "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
      "seconds": 2.8100017639808357,
      "forward_particle_stages": 4026547200,
      "device": "cuda:0",
      "mode": "hard_deployed_policy",
      "secondary_parameter_inference": false
    },
    "scope": "Previously trained reference, original v1 architecture variant; historical throughput/training update differences remain limitations"
  },
  "shared_only": {
    "EIG": 10.908915545066312,
    "MC_SE": 0.0703546370517284,
    "new_minus_control": {
      "difference": -0.03745903286753202,
      "paired_MC_SE": 0.07674999006996872,
      "CI95": [
        -0.1878890134046707,
        0.11297094766960666
      ],
      "N": 512
    },
    "counts": {
      "total_parameters": 5634,
      "activated_parameters_per_context": 5634,
      "shared_experts": 1,
      "routed_experts": 0,
      "top_k": 0,
      "expert_feature_dimension": 16,
      "routed_FFN_width": 8,
      "shared_FFN_width": 8,
      "FFN_linear_FLOPs_per_context": 768,
      "history_encoder_linear_FLOPs_per_observation": 9728,
      "router_linear_FLOPs_per_context": 0,
      "action_head_linear_FLOPs_per_context": 64,
      "FLOPs_definition": "2perlinear multiplyaccumulate; excludesnonlinearities,normalization,topk,dispatch,simulator,likelihood; not measuredGPUFLOPs"
    },
    "checkpoint_sha256": "87af1654543c53c9429775f89e109ba54ff0442b1f599e8755140bd52c18f3ac",
    "metadata": {
      "seed": 97663002,
      "systems": 512,
      "contrasts": 262144,
      "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
      "seconds": 2.8088068939978257,
      "forward_particle_stages": 4026547200,
      "device": "cuda:0",
      "mode": "hard_deployed_policy",
      "secondary_parameter_inference": false
    },
    "scope": "Previously trained reference, original v1 architecture variant; historical throughput/training update differences remain limitations"
  }
}.
Frozen interventions retain trained weights and are not retrained controls. Dense/shared-only reference controls come from previous teacher-free training and are loaded through the original v1 architecture variants; their recorded training/throughput differences limit strict causal claims.

Useful specialization supported by all audited criteria:False.
Function changes:True; context reversals after correction:0; learned routing advantage:True; paired improvement over v1:False; improvement over dense:False; improvement over shared-only:False; routed contribution:True.
Different expert weights/features/utilization alone are insufficient. Full-horizon context comparisons use shared-prefix policies and independent joint-distribution histories. A generating theta is NOT treated as a fixed-history posterior. Context classifiers come from training only; exploratory contextual set selection is confirmed on disjoint development cases. Results conditional on small cohorts or incomplete combination coverage remain limited.

Online latency:{"mean_ms": 0.45929708615100634, "median_ms": 0.4614449862856418, "p95_ms": 0.474054177175276, "decisions": 300, "frozen_forward_only": true}. Deployment is unchanged frozen v1 inference with no online learning,MC search,teacher or numerical design optimization.
Checkpoint:/home/grads/g/g.lin/Documents/objective_power_grid_v1_specialization_integration/experiments/10262026/10262026_v1_cgs_seed101/models/full/selected.pt
SHA256:b680081479f1ca0227f250265642cea0452ce57e110d26220746d7966b50edfd
Foundation:13681d7091189a4e68056e6a2864dc58f9d5700b.
Source revision:042915bfe8312faa333b1b825b94766c00c68dbf
