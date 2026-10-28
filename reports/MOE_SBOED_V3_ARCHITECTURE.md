# Reusable architecture

History adapter: each encoded action/observation token plus normalized observation index -> Linear(256), ReLU, Linear(16), sum/sqrt(t), LayerNorm, and known stage embedding. All observed history is retained; the index preserves order for dynamical systems. Benchmark-specific action/observation normalization remains in the existing history_features contract.

Amortized uncertainty: Linear shared-vector -> four mixture weights, bounded means, full Cholesky factors; diagonal .05+softplus(raw), bounded off-diagonal .25 tanh(raw). Domain-transformed parameter dimension is configurable, never a named physical quantity in the core.

Exactly four independent process experts. Router input is history and all four operator views. Softmax scores -> top2 -> selected weights renormalized to one. Only selected expert rows execute. The encoder is not an expert. Unselected slots are zero-filled; four identity-preserving slots plus shared history enter LayerNorm -> Linear(32) -> SiLU -> joint action head. This is nonlinear fusion, not averaging proposals or large direct action corrections. Final weights start at standard deviation .01; expert scales are learnable and not permanently suppressed.

Feasibility is delegated unchanged to benchmark.feasible: location box, SIR increasing times/minimum gap, grid durations. No online numerical posterior fitting, EIG search, or true theta enters policy.forward(features,stage).

Controls: native v1 unchanged; dense receives the same uncertainty features and is matched to total parameters within one integer hidden-width increment; shared has the same history/density encoder and a trained no-expert head; noaux has identical four-expert architecture with density auxiliary disabled. Dense matching counts all parameters, not just active parameters. Native v1 keeps its own validated encoder and balance loss.
