# IR MoE code diagnosis

Authoritative repository: LabPC `~/Documents/objective_power_grid`; new isolated worktree `~/Documents/objective_power_grid_integrated_information_moe`, branch `research/integrated-information-moe`, base f4c8188. Original dirty worktree and all other worktrees untouched. Latest user instruction explicitly prioritizes IR and forbids running v3's four-hour training.

The initial audit on 2026-10-24 found RTX4090 idle, no training/evaluation Python jobs, and no compute processes in nvidia-smi. The existing host-wide advisory lock is reused; a registered lock queues us and unregistered GPU compute is retried without preemption. V2 and shared-only/routed-only/dense controls completed. V3's registry is FAILED_PRESERVED: it failed the device check before training. Its checkpoint is unavailable, not a result of zero.

`DeepSeekPolicy.forward` uses full-softmax scores, then deterministic topk indices. Autograd differentiates selected weights and the softmax denominator, including some unselected logits, but not changing top-k membership. It is inaccurate to say the old router has no information gradient: pathwise gate-weight credit exists. It lacks stochastic discrete-combination credit and evidence of useful membership specialization. V2 batches selected SwiGLU kernels; masking grad=None for unused experts preserves Adam skip semantics. V3 preserves this architecture and streams terminal sPCE; its original function returns no intermediate information.

Shared and selected routed FFNs receive pathwise gradients. Zero tensors for unselected stacked parameters are masked to None so stale optimizer momentum does not invent updates. Under stochastic PL training every finite-logit expert has positive inclusion probability; no capacity dropping or hard starvation rule is added. Actual probabilities and per-expert gradients must still be checked; positive theoretical probability alone does not guarantee useful finite-sample exposure.

A native GPU test found a genuine checkpoint replay device bug: default-device context is not reliably inherited by autograd recomputation, so replay_likelihood allocated CPU cumulative likelihoods while using CUDA parameters. IR fixes its chunk replay with an explicit `torch.device(parameters.device)` context; historical v3 source is untouched. The failed pre-optimization smoke, traceback and rerun are preserved separately. CPU-only tests could not expose this failure.

The integrated gradient test also showed that blindly detaching histories in every score term loses credit from earlier continuous designs to future routing probabilities. IR explicitly separates direct conditional routing scores (history detached) from the zero-value history-path correction. Exact expectation/finite differences verify the combined gradient. This is described in IR_MOE_MATHEMATICAL_METHOD.md; score coefficient is one, not tuned.

Existing v1 frozen diagnostics follow. These are reused historical diagnostics, not new IR evidence. Feature similarities/utilization do not by themselves demonstrate information specialization. The encoder uses ReLU and LayerNorm, not a saturating tanh history encoder. Gradient spikes, selected mass, feature contribution and action distributions are logged during IR training; preclip component norms are preserved.

```json
{
  "interventions": {
    "full": {
      "mean": 10.745049872026994,
      "MC_SE": 0.09625752354061946,
      "replicates": 1,
      "metadata": {
        "seed": 1031,
        "systems": 256,
        "contrasts": 262144,
        "paired_draws_sha256": "96ee5ca31ec293ee6b5ad725793132718f60d403f496dd87df0e06c69c35ddd5",
        "seconds": 1.9106783230090514,
        "forward_particle_stages": 2013273600,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      },
      "intervention": "Frozen unchangedweights; no retraining; uniform preserves original routed feature mass"
    },
    "shared_only": {
      "mean": 10.714972530454743,
      "MC_SE": 0.10220967322214927,
      "replicates": 1,
      "metadata": {
        "seed": 1031,
        "systems": 256,
        "contrasts": 262144,
        "paired_draws_sha256": "96ee5ca31ec293ee6b5ad725793132718f60d403f496dd87df0e06c69c35ddd5",
        "seconds": 1.4315425249951659,
        "forward_particle_stages": 2013273600,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      },
      "intervention": "Frozen unchangedweights; no retraining; uniform preserves original routed feature mass",
      "paired_difference_from_full": -0.030077341572253113,
      "paired_MC_SE": 0.09007779991728278
    },
    "routed_only": {
      "mean": 10.61666709856564,
      "MC_SE": 0.10842061298090148,
      "replicates": 1,
      "metadata": {
        "seed": 1031,
        "systems": 256,
        "contrasts": 262144,
        "paired_draws_sha256": "96ee5ca31ec293ee6b5ad725793132718f60d403f496dd87df0e06c69c35ddd5",
        "seconds": 1.7514393230085261,
        "forward_particle_stages": 2013273600,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      },
      "intervention": "Frozen unchangedweights; no retraining; uniform preserves original routed feature mass",
      "paired_difference_from_full": -0.12838277346135335,
      "paired_MC_SE": 0.1090821618151046
    },
    "uniform_weights_fixed_selection": {
      "mean": 10.743507364039571,
      "MC_SE": 0.0944343725677944,
      "replicates": 1,
      "metadata": {
        "seed": 1031,
        "systems": 256,
        "contrasts": 262144,
        "paired_draws_sha256": "96ee5ca31ec293ee6b5ad725793132718f60d403f496dd87df0e06c69c35ddd5",
        "seconds": 1.7731840400083456,
        "forward_particle_stages": 2013273600,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      },
      "intervention": "Frozen unchangedweights; no retraining; uniform preserves original routed feature mass",
      "paired_difference_from_full": -0.0015425079874245226,
      "paired_MC_SE": 0.02838745974984889
    },
    "uniform_subset_scale_preserved": {
      "mean": 10.707053527214029,
      "MC_SE": 0.087980063107359,
      "replicates": 3,
      "metadata": {
        "seed": 1031,
        "systems": 256,
        "contrasts": 262144,
        "paired_draws_sha256": "96ee5ca31ec293ee6b5ad725793132718f60d403f496dd87df0e06c69c35ddd5",
        "seconds": 1.8117235130048357,
        "forward_particle_stages": 2013273600,
        "device": "cuda:0",
        "mode": "hard_deployed_policy",
        "secondary_parameter_inference": false
      },
      "intervention": "Frozen unchangedweights; no retraining; uniform preserves original routed feature mass",
      "paired_difference_from_full": -0.037996344812966154,
      "paired_MC_SE": 0.06669559868468837
    }
  },
  "recorded_gradient_max": 723.3097560292559,
  "recorded_gradient_median": 32.62048546805278,
  "recorded_clip5_fraction": 1.0,
  "late_paired_improvement": {
    "15000": {
      "final_minus_checkpoint": 0.16834790264667676,
      "paired_MC_SE": 0.1189037094647746
    },
    "20000": {
      "final_minus_checkpoint": 0.2240029896371934,
      "paired_MC_SE": 0.1438398810094392
    },
    "25000": {
      "final_minus_checkpoint": 0.3094844901392864,
      "paired_MC_SE": 0.1667704606994425
    }
  }
}
```
