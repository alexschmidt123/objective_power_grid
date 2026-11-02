# DeepSeekMoE-sBOED: compact paper-grounded adaptation

References: Dai etal.,ACL2024,Sections3.1-3.3,4.4-4.5,https://aclanthology.org/2024.acl-long.70.pdf ; Foster etal.,ICML2021,https://proceedings.mlr.press/v139/foster21a/foster21a.pdf . References were read before implementation. Paper mechanisms transferred: smaller FFNs with more possible routed combinations; one always-active shared FFN; sparse full-softmax top-k gates; expert-level frequency/probability balance; controlled shared/routed removal. No device-level balancing is used on oneGPU. Language-model results do not establish sBOED specialization.

## Exact implementation

Original correctedDAD history encoder retained verbatim: physical(design2,observation1)->Linear3-256,ReLU,Linear256-16, sum over observed history. Empty history iszero. New policy applies LayerNorm16 and a learned16-dimensional known-time vector(stage/30). No hidden true parameter or inferred posterior enters this policy. These normalization/time additions are declared architectural adaptations. No tanh history bottleneck or SA-MoE competence refresh is used.

One shared and seven routed featureFFNs, each16->8(gate/up)->16, bias-freeSwiGLU,384parameters each. Each produces16hiddenfeatures, NEVER designs. Full softmax s_i=softmax(z^Te_i); select largest3, use g_i=s_i ifselected, else0. Selected weights are NOT renormalized. u=z+FFN_shared(z)+sum_i g_iFFN_i(z). LayerNorm16->Linear16-2 produces logits; the unmodified benchmark sigmoid feasibility map returns[-4,4]^2 designs. All components train jointly from random seed101. There is no frozenDAD/residualaction model.

Total8434parameters;6898activated for a nonempty-history context. Fine-grained accounting corresponds to2width32 FFNs segmented into8width8FFNs; totalexpertlinearparameters3072, activated1536. Router/normalization/bias costs are separatelyincluded. The compact counts are adaptationchoices, not claimed paperrequirements.

Linear forwardFLOPs per history token9728; perdecision FFNs3072,router224,head64, plus historytoken costs. These exclude nonlinearities,LayerNorm,sorting,dispatch,simulator andlikelihood. They are arithmeticmodelcounts, not actualGPU-activeFLOPs. Measured latency/runtime complements them.

LB=alpha sum_i f_iP_i, f_i=(7/3)mean_context 1(i selected), P_i=mean_context s_i, alpha=.001. Shared expert is excluded. It is a mild collapsed-routing safeguard, not an equal-usage constraint or evidenceofspecialization. Main objective is the unchanged negativefull-sequenceDADsPCE plusLB. Autodiff reaches selectedFFNs andsoftmax scores; hard membership has no surrogategradient. This is piecewise differentiation; no theoremofunbiasedexpectedgradient through routingboundaries isclaimed.

## TikZ: offline
Requires positioning/arrows.meta.
\begin{tikzpicture}[node distance=13mm,every node/.style={draw,align=center}]
\node(h){Simulated measured history};\node(z)[below=of h]{DADsum encoder + normalization};\node(s)[below left=15mm and18mm of z]{SharedFFN\\alwaysactive};\node(r)[below right=15mm and18mm of z]{Top3of7routedFFNs};\node(u)[below=35mm of z]{Feature residualsum + oneactionhead};\node(sim)[below=of u]{True simulator and Gaussianlikelihood\\complete30stage rollout};\node(loss)[below=of sim]{Negativefull-horizon sPCE + mildbalance};
\draw[->](h)--(z);\draw[->](z)--(s);\draw[->](z)--(r);\draw[->](s)--(u);\draw[->](r)--(u);\draw[->](u)--(sim);\draw[->](sim)--(loss);
\draw[->](loss.west)--++(-35mm,0)|-node[pos=.25,left,draw=none]{Offline gradients}(z.west);
\end{tikzpicture}

## TikZ: direct deployment
\begin{tikzpicture}[node distance=13mm,every node/.style={draw,align=center}]
\node(h){Observedhistory only};\node(z)[below=of h]{Shared history encoder};\node(s)[below left=13mm and18mm of z]{SharedfeatureFFN};\node(r)[below right=13mm and18mm of z]{Neuralgate\\3routedfeatureFFNs};\node(o)[below=32mm of z]{Addhiddenfeatures + oneactionhead};\node(a)[below=of o]{Benchmarkfeasibilitymap\\next experiment};
\draw[->](h)--(z);\draw[->](z)--(s);\draw[->](z)--(r);\draw[->](s)--(o);\draw[->](r)--(o);\draw[->](o)--(a);
\end{tikzpicture}

No numerical belief update, candidate design enumeration, EIG evaluation, lookahead or optimization occurs in deployment. Top-k selects feature subnetworks, not designs.
