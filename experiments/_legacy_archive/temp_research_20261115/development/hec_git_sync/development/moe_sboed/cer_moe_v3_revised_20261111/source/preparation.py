"""Overlapping replay exposure; native information Q is the only design signal."""
from common import *
from integration import make_policy

def prepare(a):
    out=O/'preparation';out.mkdir()
    torch.manual_seed(CFG['preparation_seed'])
    policy=make_policy(a)
    with torch.no_grad():
        for head in policy.experts:
            head[-1].weight.add_(torch.randn_like(head[-1].weight)*CFG['initial_head_perturbation_SD'])
    # Independent first/second MC replicas from preserved training-data audit.
    audit=R/'cer_moe_sboed_v3_20261111_local_values/evidence/signatures.npz'
    z=np.load(audit);v=z['values'];d=v-v[:,:,8:9,:]
    chosen=d[:,0].mean(-1).argmax(-1)
    margin=d[np.arange(len(v)),1,chosen].mean(-1)-2*d[np.arange(len(v)),1,chosen].std(-1,ddof=1)/np.sqrt(v.shape[-1])
    qualified=np.flatnonzero((margin>0)&(z['ess'].min(-1)>=32)&(chosen!=8))
    rng=np.random.default_rng(CFG['preparation_seed']);device=next(policy.parameters()).device
    ids=torch.tensor(rng.choice(len(a._runtime_buffer['params']),CFG['preparation_pool'],replace=False),device=device)
    stages=torch.tensor(rng.integers(30,size=len(ids)),device=device)
    x=prefix_inputs(a._runtime_buffer['ds_hist'][ids],a._runtime_buffer['ys_hist'][ids],stages)
    with torch.no_grad():
        def summary(xx):
            w,mu,sd=policy.belief.params(xx);m=(w[...,None]*mu).sum(1)
            var=((w[...,None]*(mu.square()+sd.square())).sum(1)-m.square()).clamp_min(1e-6)
            return torch.cat([m,torch.log1p(var)],-1)
        features=summary(x);scale=features.std(0).clamp_min(.1)
        weights=x.new_ones(4,len(x));anchors=[]
        if len(qualified):
            anchors=rng.choice(qualified,size=4,replace=len(qualified)<4).tolist()
            centers=summary(torch.tensor(z['X'][anchors],device=device))
            dist=((features[None]-centers[:,None])/scale).square().mean(-1)
            kernels=torch.exp(-dist/(2*CFG['preparation_kernel_bandwidth']**2))
            weights=CFG['preparation_weight_floor']+(1-CFG['preparation_weight_floor'])*kernels
        weights/=weights.sum(-1,keepdim=True)
    policy.encoder.requires_grad_(False);a.critic_net.requires_grad_(False)
    lr=a.actor_optimizer.param_groups[0]['lr'];optimizer=torch.optim.Adam(policy.experts.parameters(),lr=lr)
    start=time.monotonic();generator=torch.Generator(device=device).manual_seed(CFG['preparation_seed'])
    for update in range(CFG['preparation_updates']):
        optimizer.zero_grad()
        for k,head in enumerate(policy.experts):
            ix=torch.multinomial(weights[k],CFG['preparation_batch'],replacement=True,generator=generator)
            for j in ix.split(CFG['actor_chunk']):
                xx=x[j];context=policy.context(xx)
                design=-4+8*head(context).sigmoid()
                loss=-a.critic_net(torch.cat([xx,design],-1)).mean()*len(j)/len(ix)/4
                loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.experts.parameters())
        optimizer.step()
    policy.encoder.requires_grad_(True);a.critic_net.requires_grad_(True)
    torch.save(dict(actor_state=policy.state_dict()),out/'actor.pt')
    receipt=dict(status='COMPLETE',updates=CFG['preparation_updates'],seconds=time.monotonic()-start,
                 supported_history_examples=len(qualified),anchor_training_rows=anchors,
                 no_groups_required=True,overlapping_weight_floor=CFG['preparation_weight_floor'],
                 effective_sample_size=(weights.sum(-1).square()/weights.square().sum(-1)).tolist(),
                 critic_rows=4*CFG['preparation_updates']*CFG['preparation_batch'],
                 new_simulator_observations=0,audit_sha256=sha(audit),actor_sha256=sha(out/'actor.pt'),
                 objective='native remaining variational Q on overlapping conditional replay distributions')
    put(out/'RESULT.json',receipt);print('PREPARATION',json.dumps(receipt),flush=True)
    return policy
