# Frozen collaborative base with routed feature residuals

For observed prefix h, the entire verified HEC-v2 model runs with uniform
four-expert feature coordination. It returns a_base∈[-4,4]^2, shared128 history
features s, and the four real128-channel specialist packets E_k (first64 plus
response64). Every base parameter is converted to a frozen buffer on an isolated
copy, preserving exact numerical functions while excluding it from both Adam
and native Polyak averaging. All old checkpoints remain unchanged.

Four independent small adaptations A_k:256→32→16 receive [s,E_k]. They are
available at every experiment. No chronological roles or new expert labels are
assigned. The history router128→32→4 produces four bounded Plackett–Luce logits.
Two distinct adaptations are sampled at deployment.

Let z_k=A_k([s,E_k]) and t_k=SiLU(W_z z_k)⊙sigmoid(W_h s), with bias-free W_z.
For selected pair{i,j}, delta=0.5*tanh(W_out(t_i+t_j)/sqrt(2)), and
final design=clamp(a_base+delta,−4,4). W_out is bias-free and exactly zero at
initialization. Thus zero-initialized RR is bitwise identical to uniform HEC-v2.
No route-independent action correction can arise from history or bias alone:
zero selected specialist tokens imply delta=0 for every history. There is one
shared feature correction head, not an average of physical expert actions.

Only two new adaptations execute at deployment. Training enumerates all six
pairs for an exact categorical expectation. Frozen HEC still executes all its
original providers/pathways, so the whole policy is not claimed to have the cost
of two total experts. The original collaborative head is retained intact.

Frozen feature caching stores exactly the action/history/specialist functions
for each valid native replay prefix. It is reused by actor and target expectation
calculations because both frozen bases are identical; it never stores hidden
parameters or future observations as policy inputs. Native full-batch alignment
checks compare fresh versus cached values and poison unused future history.
