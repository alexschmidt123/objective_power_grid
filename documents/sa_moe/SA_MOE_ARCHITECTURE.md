# Teacher-free SA-MoE architecture

Measured history only: action2,observation1,observation-time index1. Shared token encoder4-64tanh-64tanh; masked sum divided bysqrt(historylength); concatenate stage then65-64tanh context. Four independent64-64tanh-64tanh-2linear action experts; each outputs4tanh(raw), bounded[-4,4]^2. Shared64-32tanh-4linear competence router.44716totalparameters. Single control shares encoder and uses64-159tanh-159tanh-2,44799parameters. No residuals, no pretrained model, no teacher or actor critic.

At deployment e=argmax g(z), action=pi_e(z). The router probabilities are decision competence probabilities, NEVER posterior hypothesis mass. Exactly one expert is evaluated per history. No simulator, EIG or counterfactual routine is imported by deployment module src/research/sa_moe.py. Shared hidden representation approximates a belief from observations; it is not an exact Bayesian posterior.

At training all expert outputs can enter the straight-through routing gradient, but the forward action is discrete. Unselected experts do not automatically receive pathwise expert gradients from that sample. Periodic forced-first-action conditional rollouts provide explicit gradients to every expert.

## Diagram A: offline
\begin{tikzpicture}[node distance=13mm, every node/.style={draw,align=center}]
\node (history) {Fresh simulated histories};
\node (encoder) [below=of history] {Shared history encoder};
\node (experts) [below=of encoder] {Four independent action experts};
\node (counter) [right=35mm of experts] {Offline conditional posterior\\full remaining-horizon rollouts};
\node (competence) [above=of counter] {Paired information lower bounds\\MC uncertainty to soft competence};
\node (router) [above=of competence] {Router cross entropy};
\draw[->] (history)--(encoder);\draw[->](encoder)--(experts);
\draw[->](experts)--(counter);\draw[->](counter)--(competence);\draw[->](competence)--(router);
\draw[->](counter.south) -- ++(0,-8mm) -| node[pos=.7,below,draw=none]{weighted full-sequence gradients}(experts.south);
\end{tikzpicture}

## Diagram B: deployment
\begin{tikzpicture}[node distance=12mm,every node/.style={draw,align=center}]
\node(h){Measured history};\node(z)[below=of h]{Shared encoder};\node(g)[below=of z]{Competence router\\argmax expert INDEX};\node(e)[below=of g]{Selected expert only};\node(a)[below=of e]{4tanh feasible experiment};
\draw[->](h)--(z);\draw[->](z)--(g);\draw[->](g)--(e);\draw[->](e)--(a);
\end{tikzpicture}

TikZ requires positioning and arrows.meta libraries. No predefined expert roles. A source-level and hook-based test rejects offline-evaluator/all-expert-search calls during deployment.
