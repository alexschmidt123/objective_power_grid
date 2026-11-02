# DeepSeekMoE-v1 contextual expert learning: seed102

Official EIG **10.83415593 +/- 0.04301131 nats**, seed1001,N2000,L1000000,H30.
Independent confirmation seed97763001: **10.81753690 +/- 0.04379193**.
All uncertainties are trajectory MC SE conditional on fixed contrast banks and this trained checkpoint.

| Reference | New minus reference | Paired MC SE | Approx95% CI |
|---|---:|---:|---|
|v1|-0.15490997|0.05095405|[-0.254779907169235, -0.055040037014392015]|
|vsOED|-1.64669096|0.05551418|[-1.755498756752677, -1.5378831651577858]|
|corrected_DAD|+0.55804306|0.05810096|[0.44416518661683685, 0.6719209430562486]|
|dense|-0.38423943|0.04903831|[-0.4803545207583976, -0.2881243391889765]|
|shared_only|-0.33682141|0.04816151|[-0.4312179789030756, -0.24242484417005844]|

Architecture is exactly the original v1 DeepSeekPolicy:8434 parameters,1shared+7routed SwiGLU experts,top3,unchanged encoder/head. The only learning change is context-protected,norm-preserving routed-expert gradients. Main initialization was fresh seed102. No pilot weights,teachers,IR/RC/SA/PIMoE architecture/training code or competitor policies were used.

Main updates20575; selected20575; charged seconds14396.959129 of14400.
Measured specialization subset1124.428173s INSIDE that allowance; no additional simulator forward calls are introduced by the two extra context backward passes.
Training trajectories/counts:{"fresh_complete_primary_trajectories": 41152000, "true_parameter_stages": 1234560000, "contrast_parameter_stages": 2469120000000, "scalar_conditional_likelihood_evaluations": 2470354560000, "policy_optimizer_updates": 20576, "shared_FFN_contexts": 1234500000, "routed_FFN_contexts": 3703500000, "specialization_context_backward_passes": 4114, "specialization_steps": 2057, "specialization_expert_adjustments": 6283, "specialization_step_microseconds": 1124428173}. Expert effective update counters:[20575, 20545, 20575, 20575, 20574, 20575, 20575].
No raw-SGD descent alignment check guarantees a finite Adam step improves EIG. No statistical guarantee of superiority is made.

Learned minus uniform-subset development routing:{"difference": -0.07167427291962492, "paired_MC_SE": 0.051039141182633196, "CI95": [-0.17171098963758596, 0.028362443798336148], "N": 512}.
Full minus routed-masked:{"difference": -0.0554651427289089, "paired_MC_SE": 0.0599491440481196, "CI95": [-0.1729654650632233, 0.06203517960540551], "N": 512}.
Development comparisons:{
  "full": {
    "EIG": 10.628096050669996,
    "MC_SE": 0.07229121345814836,
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
        "seconds": 3.3916814191034064,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": false
  },
  "uniform_weights": {
    "EIG": 10.657317862510453,
    "MC_SE": 0.07226465508087172,
    "paired_intervention_minus_full": {
      "difference": 0.029221811840458394,
      "paired_MC_SE": 0.03734237165828835,
      "CI95": [
        -0.04396923660978676,
        0.10241286029070355
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
        "seconds": 3.338131342898123,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": true
  },
  "uniform_subset": {
    "EIG": 10.699770323589618,
    "MC_SE": 0.06714295288895684,
    "paired_intervention_minus_full": {
      "difference": 0.07167427291962492,
      "paired_MC_SE": 0.051039141182633196,
      "CI95": [
        -0.028362443798336148,
        0.17171098963758596
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
        "seconds": 3.4758366260211915,
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
        "seconds": 3.4133244180120528,
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
        "seconds": 3.420374913024716,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": true
  },
  "mask_routed": {
    "EIG": 10.683561193398901,
    "MC_SE": 0.06979795232268418,
    "paired_intervention_minus_full": {
      "difference": 0.0554651427289089,
      "paired_MC_SE": 0.0599491440481196,
      "CI95": [
        -0.06203517960540551,
        0.1729654650632233
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
        "seconds": 2.6843946221051738,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": true
  },
  "mask_shared": {
    "EIG": 10.62074426688891,
    "MC_SE": 0.07331250342697063,
    "paired_intervention_minus_full": {
      "difference": -0.007351783781084496,
      "paired_MC_SE": 0.06759354171108461,
      "CI95": [
        -0.13983512553481034,
        0.12513155797264133
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
        "seconds": 3.282242990913801,
        "forward_particle_stages": 4026547200,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      }
    ],
    "frozen_intervention": true
  },
  "frozen_v1": {
    "EIG": 10.895133554907869,
    "MC_SE": 0.06717459290048738,
    "new_minus_v1": {
      "difference": -0.26703750423787254,
      "paired_MC_SE": 0.08153114971958841,
      "CI95": [
        -0.42683855768826584,
        -0.10723645078747926
      ],
      "N": 512
    },
    "metadata": {
      "seed": 97663002,
      "systems": 512,
      "contrasts": 262144,
      "paired_draws_sha256": "635f8b9f30bd65705da0e108891878855da55cb42a5a328cbdab7234da4b4381",
      "seconds": 3.3448471230221912,
      "forward_particle_stages": 4026547200,
      "device": "cuda:0",
      "mode": "hard_deployed_policy",
      "secondary_parameter_inference": false
    }
  },
  "dense": {
    "EIG": 10.917953309480678,
    "MC_SE": 0.06702999217049181,
    "new_minus_control": {
      "difference": -0.2898572588106835,
      "paired_MC_SE": 0.08761191184053038,
      "CI95": [
        -0.46157660601812306,
        -0.118137911603244
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
      "seconds": 2.651557345991023,
      "forward_particle_stages": 4026547200,
      "device": "cuda:0",
      "mode": "hard_deployed_policy",
      "secondary_parameter_inference": false
    },
    "scope": "Previously trained reference, original v1 architecture variant; historical throughput/training update differences remain limitations"
  },
  "shared_only": {
    "EIG": 10.908915545066314,
    "MC_SE": 0.07035463705172841,
    "new_minus_control": {
      "difference": -0.28081949439632015,
      "paired_MC_SE": 0.08445054939173463,
      "CI95": [
        -0.44634257120412,
        -0.11529641758852027
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
      "seconds": 2.6499546540435404,
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
Function changes:True; context reversals after correction:0; learned routing advantage:False; paired improvement over v1:False; improvement over dense:False; improvement over shared-only:False; routed contribution:False.
Different expert weights/features/utilization alone are insufficient. Full-horizon context comparisons use shared-prefix policies and independent joint-distribution histories. A generating theta is NOT treated as a fixed-history posterior. Context classifiers come from training only; exploratory contextual set selection is confirmed on disjoint development cases. Results conditional on small cohorts or incomplete combination coverage remain limited.

Online latency:{"mean_ms": 0.4594594876592358, "median_ms": 0.45986048644408584, "p95_ms": 0.477365538245067, "decisions": 300, "frozen_forward_only": true}. Deployment is unchanged frozen v1 inference with no online learning,MC search,teacher or numerical design optimization.
Checkpoint:/home/qiqi/Documents/objective_power_grid_v1_specialization_run/experiments/10262026/10262026_v1_cgs_seed102/models/full/selected.pt
SHA256:1256720d370f42ab7a069ffa87a20a565bcd3623d60480b377c9d02c9e79a65c
Foundation:13681d7091189a4e68056e6a2864dc58f9d5700b.
Source revision:042915bfe8312faa333b1b825b94766c00c68dbf
