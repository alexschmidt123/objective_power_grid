"""Independent posterior-conditional full suffixes; no saved future observations.

Value = E[log q30(theta|new full history)-log qt(theta|prefix)].
The terminal variational score is native; SNIS corrects the conditional sampling
law using the exact likelihood. Its finite-sample bias is explicitly audited.
"""
from common import *

def gmm_log(theta, w, mu, sd):
    lp = -.5 * ((theta[:, :, None] - mu[:, None]) / sd[:, None]).square().sum(-1)
    lp = lp - sd.log().sum(-1)[:, None] - 2 * math.log(2 * math.pi)
    return torch.logsumexp(lp + w.clamp_min(1e-38).log()[:, None], -1)

def prefix_log_likelihood(x, theta):
    stages = x[:, :30].argmax(-1)
    ds = x[:, 30:88].reshape(-1, 29, 2)
    ys = x[:, 88:117]
    lp = theta.new_zeros(theta.shape[:2])
    for t in range(int(stages.max())):
        prediction = mean_signal(theta, ds[:, t:t+1])[:, 0]
        lp += -2 * (ys[:, t, None] - prediction).square() * (t < stages[:, None])
    return lp

@torch.no_grad()
def conditional_particles(belief, x, count, generator):
    w, mu, sd = belief.params(x)
    # Full-support .1prior+.9amortized-GMM importance proposal.
    k = torch.multinomial(w, count, replacement=True, generator=generator)
    mm = mu.gather(1, k[..., None].expand(-1, -1, 4))
    ss = sd.gather(1, k[..., None].expand(-1, -1, 4))
    th = mm + ss * torch.randn(mm.shape, generator=generator, device=x.device)
    use_prior = torch.rand((len(x), count), generator=generator, device=x.device) < .1
    prior_draw = torch.randn(th.shape, generator=generator, device=x.device)
    th = torch.where(use_prior[..., None], prior_draw, th)
    prior_lp = -.5 * th.square().sum(-1) - 2 * math.log(2 * math.pi)
    proposal_lp = torch.logaddexp(gmm_log(th, w, mu, sd) + math.log(.9), prior_lp + math.log(.1))
    log_weights = prior_lp + prefix_log_likelihood(x, th) - proposal_lp
    weights = log_weights.softmax(-1)
    ess = weights.square().sum(-1).reciprocal()
    return th, weights, ess

@torch.no_grad()
def terminal_log_q(belief, ds, ys, theta):
    net = belief.nets[29]
    inp = torch.cat([ds.reshape(len(ds), 60), ys.reshape(len(ds), 30)], -1)
    features = net.feature_net(inp)
    w = net.weight_net(features).exp()
    mu = net.mu_net(features).sigmoid().reshape(-1, 8, 4) * (net.mu_bounds[:, 1] - net.mu_bounds[:, 0]) + net.mu_bounds[:, 0]
    sd = net.sigma_net(features).sigmoid().reshape(-1, 8, 4) * net.max_sigmas + 1e-5
    return gmm_log(theta[:, None], w, mu, sd)[:, 0]

@torch.no_grad()
def suffix_values(model, x, actions, particles, weights, repeats, generator):
    # x is same-stage; random draws are shared across candidate actions only.
    stage = int(x[0, :30].argmax())
    assert bool((x[:, :30].argmax(-1) == stage).all())
    b, choices = actions.shape[:2]
    ids = torch.multinomial(weights, repeats, replacement=True, generator=generator)
    theta = particles.gather(1, ids[..., None].expand(-1, -1, 4))
    noise = torch.randn((b, repeats, 30-stage), generator=generator, device=x.device)
    theta = theta[:, None].expand(-1, choices, -1, -1).reshape(-1, 4)
    prefix = x[:, None, None].expand(-1, choices, repeats, -1).reshape(-1, 117).clone()
    ds = x.new_zeros(len(theta), 30, 2)
    ys = x.new_zeros(len(theta), 30, 1)
    ds[:, :stage] = prefix[:, 30:30+2*stage].reshape(-1, stage, 2)
    ys[:, :stage, 0] = prefix[:, 88:88+stage]
    initial_log_q = model.belief.log_prob(prefix, theta)
    for t in range(stage, 30):
        action = actions[:, :, None].expand(-1, -1, repeats, -1).reshape(-1, 2) if t == stage else model(prefix)
        y = mean_signal(theta[:, None], action[:, None])[:, 0, 0]
        y += .5 * noise[:, None, :, t-stage].expand(-1, choices, -1).reshape(-1)
        ds[:, t] = action; ys[:, t, 0] = y
        if t < 29: prefix = advance(prefix, action, y)
    result = (terminal_log_q(model.belief, ds, ys, theta) - initial_log_q).reshape(b, choices, repeats)
    return result

def main():
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    import gc
    assert not (O/'evidence/GROUP_SUPPORT.json').exists()
    original.setup(); leases = original.acquire()
    before = protected(); put(O/'evidence/PROTECTED_INPUTS_BEFORE.json', before)
    state = restore(); a = state['algorithm']; model = a.actor_net.eval().requires_grad_(False)
    rng = np.random.default_rng(CFG['signature_seed'])
    generator = torch.Generator(device='cuda').manual_seed(CFG['signature_seed'])
    rows, values, esses, proposals, stages = [], [], [], [], []
    tick = time.monotonic(); COST.clear()
    with torch.no_grad(), torch.device('cuda'):
        for stage in CFG['signature_stages']:
            ids = torch.tensor(rng.choice(len(a._runtime_buffer['params']), CFG['histories_per_stage'], replace=False), device='cuda')
            x = prefix_inputs(a._runtime_buffer['ds_hist'][ids], a._runtime_buffer['ys_hist'][ids], torch.full((len(ids),), stage, device='cuda'))
            # Eight fixed complete designs plus native HEC proposal. The candidate
            # palette is a probe, not an expert role or executed action mixture.
            anchors = x.new_tensor([[-1,-1],[-1,1],[1,-1],[1,1],[-2,0],[2,0],[0,-2],[0,2]])
            acts = torch.cat([anchors[None].expand(len(x),-1,-1), model(x)[:,None]], 1)
            estimates, ess_stage = [], []
            # Two independent posterior banks AND independent future simulations.
            for replica in range(2):
                parts = []; ee = []
                for lo in range(0, len(x), 8):
                    xx=x[lo:lo+8]; th,pw,ess=conditional_particles(model.belief,xx,CFG['posterior_particles'],generator)
                    parts.append(suffix_values(model,xx,acts[lo:lo+8],th,pw,CFG['suffix_replicates'],generator).cpu())
                    ee.append(ess.cpu())
                estimates.append(torch.cat(parts)); ess_stage.append(torch.cat(ee))
            vals=torch.stack(estimates,1).numpy()
            rows.append(x.cpu().numpy()); values.append(vals); esses.append(torch.stack(ess_stage,1).numpy()); proposals.append(acts.cpu().numpy()); stages.extend([stage]*len(x))
            print('SIGNATURE_STAGE',stage,'seconds',time.monotonic()-tick,'ESS_median',float(torch.stack(ess_stage).median()),flush=True)
    X=np.concatenate(rows); V=np.concatenate(values); ESS=np.concatenate(esses); A=np.concatenate(proposals); stages=np.array(stages)
    # Signature removes only the additive history baseline, never horizon length
    # or future information. Fit/calibration and validation are disjoint histories.
    signatures=V.mean((1,3)); signatures-=signatures.mean(1,keepdims=True)
    cal=np.concatenate([np.flatnonzero(stages==s)[:CFG['histories_per_stage']//2] for s in CFG['signature_stages']])
    val=np.setdiff1d(np.arange(len(X)),cal)
    scale=max(float(signatures[cal].std()),.1)
    km=KMeans(n_clusters=4,random_state=CFG['signature_seed']%2147483647,n_init=20).fit(signatures[cal]/scale)
    groups=km.predict(signatures/scale)
    centers=km.cluster_centers_*scale
    independent=[]
    for replica in range(2):
        ss=V[:,replica].mean(-1);ss-=ss.mean(-1,keepdims=True)
        independent.append(km.predict(ss/scale))
    occupancy=np.bincount(groups[val],minlength=4)/len(val)
    silhouette=float(silhouette_score(signatures[val]/scale,groups[val])) if len(set(groups[val]))==4 else -1.
    agreement=float(np.mean(independent[0][val]==independent[1][val]))
    preferred=centers.argmax(-1)
    within_stage={str(s):np.bincount(groups[val[stages[val]==s]],minlength=4).tolist() for s in CFG['signature_stages']}
    ess_fraction=float((ESS[val]>=CFG['minimum_posterior_ess']).mean())
    gate=dict(occupancy=bool(occupancy.min()>=.1),distinct_preferred_designs=len(set(preferred))==4,
              silhouette=silhouette>=.2,independent_MC_assignment_agreement=agreement>=.7,
              within_stage_variation=all(sum(v>=4 for v in c)>=2 for c in within_stage.values()),
              posterior_resolution=ess_fraction>=.9)
    result=dict(status='SUPPORTED' if all(gate.values()) else 'FOUR_GROUPS_NOT_SUPPORTED',criteria=gate,
                validation_occupancy=occupancy.tolist(),silhouette=silhouette,independent_MC_assignment_agreement=agreement,
                preferred_probe_indices=preferred.tolist(),within_stage_counts=within_stage,
                posterior_ess_adequate_fraction=ess_fraction,posterior_ess_quantiles=np.quantile(ESS,[0,.1,.5,.9,1]).tolist(),
                conditional_value='exact-likelihood SNIS posterior; full resimulated suffix; normalized native terminal variational log-density gain',
                limitations=['finite SNIS bias and missed modes remain possible','frozen variational terminal score is not exact conditional MI',
                             'fixed probe palette may miss optimal actions','suffix follows frozen HEC-v2, not an optimized continuation',
                             'groups must be reproducible before specialist preparation; kmeans alone does not establish four groups'],
                simulator_observations=sum(CFG['histories_per_stage']*9*2*CFG['suffix_replicates']*(30-s) for s in CFG['signature_stages']),
                exact_model_queries=dict(COST),seconds=time.monotonic()-tick,scientific_training_updates=0)
    np.savez_compressed(O/'evidence/signatures.npz',X=X,values=V,ess=ESS,proposals=A,stages=stages,groups=groups,calibration=cal,validation=val,centers=centers,scale=scale)
    assert before==protected(),'Original artifact changed'
    put(O/'evidence/GROUP_SUPPORT.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
