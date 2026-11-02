# Four-expert architecture

src/research/moe_sboed_v4.py reuses the pinned v1 compact SwiGLU feature expert and raw-observation256ReLU->16 sum-history principle, followed by16-dimensional LayerNorm and known stage embedding. Domain scales/dimensions are adapters; no named physical parameters or true theta enter the general expert classes.

Exactly FOUR independently parameterized width16 feature experts, each with its own linear role-prediction head. NO always-active shared FFN expert. History encoding is shared, not a fifth expert. Router(history)->top2distinct experts; selected-softmax weights renormalized to one. Only selected rows execute in the policy forward pass. Unselected identity slots are zero; history plus four16-dimensional slots enter a nonlinear joint LayerNorm/32SiLU/action head, then the unchanged benchmark feasible map. No averaged physical action proposals, online EIG search, teacher, posterior numerical optimization, or deployment gradient update.

Offline auxiliary training evaluates all four role predictors on detached history embeddings. It is intentionally not sparse auxiliary computation and is charged. Deployment never executes prediction heads. Four role targets, not interchangeable names or weight diversity, define responsibilities. Predictions alone still do not establish causal competence.

Uniform control samples a uniform subset of two experts; BOTH learned and uniform-subset policies use the SAME selected-softmax normalization. This tests selection while preserving K and gate-weight semantics. Two independent fixed routing realizations are used for frozen interventions. Sparse membership is discrete; derivatives are exact conditional on the selected set, with no derivative for membership changes. This is a piecewise pathwise approximation, not an unbiased categorical routing estimator.

Dynamic-domain adapters include a known token-position channel to retain order; location retains the exact v1 sum-encoding input principle. This changes no physics and is not a fixed expert-stage assignment.
