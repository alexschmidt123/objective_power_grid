# MoE-sBOED version registry

| Version | Status | Foundation / evidence |
|---|---|---|
| MoE-sBOED-v1 | REGISTERED; location structurally valid | `13681d7091189a4e68056e6a2864dc58f9d5700b` |
| MoE-sBOED-v2 | UNREGISTERED | No enhanced version approved by this registry |
| MoE-sBOED-v3 | UNREGISTERED | Historical FH-v3 is not this version |
| MoE-sBOED-v4 | UNREGISTERED | No training authorized |

SIR MoE-sBOED-v1: PENDING independent adaptation validation, source/configuration freeze and explicit registration. An isolated adapter has CPU checks from earlier work; that does not authorize checkpoint substitution or establish a completed scientific SIR result. No SIR MoE checkpoint is registered here.

Registered location checkpoint:
`labpc:/home/grads/g/g.lin/Documents/objective_power_grid_deepseekmoe_20261023/experiments/10232026/10232026_deepseekmoe_sboed/models/full/selected.pt`

* SHA256: `f110766019e3628bb02c93815becd7e5795b3abc05626eabd91b7dbeb0381ee7`
* Configuration: `configs/deepseek_sboed_location.yaml`
* Configuration SHA256: `b2c920b9830babf3cec75a6fc6c0c305e5ae1a1104744d3c9f368101f360d8b0`
* Horizon30; shared1/routed7/top3; parameters8434, activated6898 per context.
* Historical official EIG: 10.98906590 ± 0.04479411 nats (MC SE, not training-seed uncertainty).
* Serialization tag: `DeepSeekMoE-sBOED`, variant `full`, original source commit as above.

Registration of another checkpoint requires explicit validation and documentation, not merely matching tensors or editing checkpoint metadata. Future enhanced versions must document architecture, mathematical objective/gradient correctness, direct inference tests, system-specific physics, immutable configuration/checkpoint provenance, scientific intervention evidence and computational costs. Structural correctness and successful specialization must be reported separately. Historical RC/IR/FH results, shared-only/routed-only controls and gradient audits remain excluded exploratory records, never v2/v3/v4 registrations.
