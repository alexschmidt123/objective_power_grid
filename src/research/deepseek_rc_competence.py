import copy,itertools,json,math,time
from pathlib import Path
from statistics import NormalDist
import numpy as np,torch
from src.research.engine import rollout,replay_likelihood,generator
from src.research.policies import history_features
from src.research.benchmarks import make_benchmark
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def paired(a,b):
    d=np.asarray(a)-np.asarray(b);se=float(d.std(ddof=1)/np.sqrt(len(d)))
    return dict(N=len(d),difference=float(d.mean()),paired_MC_SE=se,CI95=[float(d.mean()-1.96*se),float(d.mean()+1.96*se)])

def latent_for_set(model,f,stage,ids,uniform=False):
    z=model.encode_context(f,stage);scores=model.router(z).softmax(-1)
    weights=scores.gather(1,ids)
    if uniform:weights=weights.sum(-1,keepdim=True).expand_as(weights)/ids.shape[-1]
    return model.emitter(z+model.shared(z)+model.routed_feature_sum(z,weights,ids))

@torch.no_grad()
def posterior_reference(b,h,particles,seed):
    theta=b.sample((1,particles),generator(seed))
    lp,_,_=replay_likelihood(b,theta,h['actions'],h['observations'])
    w=lp[0].softmax(-1)
    return theta[0],w,float(1/w.square().sum())

@torch.no_grad()
def remaining_values(b,model,h,first_action,theta,contrasts,noise):
    """Conditional remaining-horizon sPCE; every future action uses original policy."""
    B=len(theta);L=contrasts.shape[1];parameters=torch.cat([theta[:,None],contrasts],1)
    state=b.initial((B,L+1));cumulative=theta.new_zeros(B,L+1)
    actions=[a.expand(B,-1) for a in h['actions']]
    observations=[y.expand(B,-1) for y in h['observations']]
    immediate=None
    for offset,t in enumerate(range(h['stage'],h['horizon'])):
        if offset==0:action=first_action.expand(B,-1)
        else:
            f=history_features(b,actions,observations,B,h['horizon'])
            action=b.feasible(model(f,t),actions,h['horizon'])
        mean,state=b.forward(parameters,state,action[:,None].expand(-1,L+1,-1))
        y=mean[:,0]+b.sigma*noise[:,offset]
        cumulative+=b.log_likelihood(y[:,None],mean)
        info=cumulative[:,0]-cumulative.logsumexp(-1)+math.log(L+1)
        if immediate is None:immediate=info.clone()
        actions.append(action);observations.append(y)
    return info,immediate

@torch.no_grad()
def feature_audit(model,cfg,D):
    b=make_benchmark(cfg['benchmark']);r=rollout(b,model,30,64,6242001,0);rows=[]
    for t,f in enumerate(r['features']):
        z=model.encode_context(f,t);scores=model.router(z).softmax(-1);rank=scores.topk(4,-1)
        ids=rank.indices[:,:3];weights=scores.gather(1,ids);shared=model.shared(z)
        routed=model.routed_feature_sum(z,weights,ids);all_features=torch.stack([e(z) for e in model.routed],1)
        normed=torch.nn.functional.normalize(all_features,dim=-1)
        full=b.feasible(model.emitter(z+shared+routed),r['actions'][:t],30)
        no_shared=b.feasible(model.emitter(z+routed),r['actions'][:t],30)
        no_routed=b.feasible(model.emitter(z+shared),r['actions'][:t],30)
        rows.append(dict(stage=t,selection_frequency=torch.nn.functional.one_hot(ids,7).sum(1).double().mean(0).cpu().tolist(),probability=scores.mean(0).cpu().tolist(),entropy=float(-(scores*scores.log()).sum(-1).mean()),top3_minus_top4_margin=float((rank.values[:,2]-rank.values[:,3]).mean()),shared_feature_L2=float(shared.norm(dim=-1).mean()),routed_feature_L2=float(routed.norm(dim=-1).mean()),context_L2=float(z.norm(dim=-1).mean()),shared_design_effect_RMS=float((full-no_shared).square().mean().sqrt()),routed_design_effect_RMS=float((full-no_routed).square().mean().sqrt()),feature_cosine=torch.einsum('bed,bfd->bef',normed,normed).mean(0).cpu().tolist(),action_boundary_fraction=float((full.abs()>3.95).double().mean())))
    write(D/'data/feature_audit.json',dict(seed=6242001,episodes=64,stages=rows,parameter_stage_propagations=b.calls))
    return rows

@torch.no_grad()
def competence(model,cfg,D,deadline):
    b=make_benchmark(cfg['benchmark'])
    r=rollout(b,model,30,2,6244001,0);sets=list(itertools.combinations(range(7),3))
    values=np.full((8,35,2,32),np.nan);immediate=values.copy();rows=[];case=0
    z_family=NormalDist().inv_cdf(1-.05/(2*8*34))
    for stage in [1,5,15,29]:
        for episode in range(2):
            if time.monotonic()>deadline:break
            h=dict(stage=stage,horizon=30,actions=[a[episode:episode+1] for a in r['actions'][:stage]],observations=[y[episode:episode+1] for y in r['observations'][:stage]])
            f=history_features(b,h['actions'],h['observations'],1,30)
            logits=model.router(model.encode_context(f,stage));selected=tuple(sorted(logits.topk(3,-1).indices[0].tolist()))
            post=[posterior_reference(b,h,262144,6245001+case*11+rep) for rep in range(2)]
            ess=[x[2] for x in post]
            candidate_actions=[]
            for ids in sets:
                index=torch.tensor([ids],device=f.device)
                action=b.feasible(latent_for_set(model,f,stage,index),h['actions'],30)
                candidate_actions.append(action)
            proposals=torch.cat(candidate_actions)
            row=dict(case=case,stage=stage,remaining=30-stage,episode=episode,selected=list(selected),selected_index=sets.index(selected),posterior_ESS=ess,posterior_particles_each=262144,proposed_actions=proposals.cpu().tolist(),action_dispersion_RMS=float((proposals-proposals.mean(0)).square().mean().sqrt()),status='LOW_ESS_WITHHELD')
            if min(ess)>=128:
                predictive=[]
                for theta,w,_ in post:
                    mean,_=b.forward(theta[None],None,candidate_actions[sets.index(selected)][:,None].expand(1,len(theta),2))
                    mu=(mean[0,:,0]*w).sum();predictive.append(float(((mean[0,:,0]-mu).square()*w).sum()))
                row['posterior_predictive_mean_variance']=predictive
                for rep,(theta,w,_) in enumerate(post):
                    g=generator(6246001+case*11+rep);L=65536;F=32
                    draw=theta[torch.multinomial(w,F*(L+1),True,generator=g)].reshape(F,L+1,4)
                    noise=torch.randn(F,30-stage,1,generator=g,dtype=torch.float64)
                    for j,action in enumerate(candidate_actions):
                        if time.monotonic()>deadline:break
                        v,one=remaining_values(b,model,h,action,draw[:,0],draw[:,1:],noise)
                        values[case,j,rep]=v.cpu().numpy();immediate[case,j,rep]=one.cpu().numpy()
                if np.isfinite(values[case]).all():
                    base=values[case,row['selected_index']]
                    delta=values[case]-base[None]
                    mean=delta.reshape(35,-1).mean(-1);se=delta.reshape(35,-1).std(-1,ddof=1)/8
                    replicated=(delta.mean(-1)>0).all(-1)
                    wins=(mean-z_family*se>0)&replicated
                    row.update(status='FINITE_IMPORTANCE_POSTERIOR_APPROXIMATION',mean_remaining_information=values[case].reshape(35,-1).mean(-1).tolist(),mean_immediate_information=immediate[case].reshape(35,-1).mean(-1).tolist(),paired_advantage_over_selected=mean.tolist(),paired_MC_SE=se.tolist(),familywise_95_z=z_family,reliable_alternative_wins=wins.tolist(),posterior_replicate_advantage=delta.mean(-1).tolist())
            rows.append(row);case+=1
            np.savez(D/'data/competence_values.npz',remaining=values,immediate=immediate)
            write(D/'data/competence.json',dict(sets=[list(s) for s in sets],histories=rows,fantasies_per_bank=32,posterior_banks=2,contrasts=65536,horizon=30,minimum_ESS=128,continuation='Original frozen deterministic policy, shared for all first-set interventions',uncertainty='Paired fantasy MC SE conditional on each finite importance posterior. Independent banks expose approximation sensitivity; ESS is not a convergence certificate.',familywise_95_z=z_family))
            print('COMPETENCE',case,'stage',stage,'ESS',ess,row['status'],flush=True)
    write(D/'costs/competence_counts.json',dict(parameter_stage_propagations=b.calls,histories=len(rows),maximum_histories=8,sets=35,optimizer_updates=0))

def plots(D):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    feature=json.loads((D/'data/feature_audit.json').read_text())['stages']
    fig,ax=plt.subplots(figsize=(10,4))
    image=ax.imshow(np.asarray([r['selection_frequency'] for r in feature]).T,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis')
    ax.set(xlabel='Experimental stage (zero based)',ylabel='Routed expert',title='Frozen deterministic expert utilization')
    fig.colorbar(image,ax=ax,label='Fraction of 64 histories selected');fig.tight_layout();fig.savefig(D/'figures/expert_utilization.png',dpi=180);plt.close(fig)
    data=json.loads((D/'data/competence.json').read_text());matrix=np.full((len(data['histories']),35),np.nan)
    for i,row in enumerate(data['histories']):
        if 'paired_advantage_over_selected' in row:matrix[i]=row['paired_advantage_over_selected']
    fig,ax=plt.subplots(figsize=(16,4));limit=max(.001,float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else .001)
    cmap=copy.copy(plt.get_cmap('coolwarm'));cmap.set_bad('#d9d9d9')
    image=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=-limit,vmax=limit)
    ax.set_xticks(range(35));ax.set_xticklabels([''.join(map(str,s)) for s in data['sets']],rotation=90,fontsize=7)
    ax.set_yticks(range(len(data['histories'])));ax.set_yticklabels(['t'+str(x['stage'])+'/h'+str(x['episode']) for x in data['histories']])
    ax.set(xlabel='Expert set (expert indices)',ylabel='Held-out history',title='Remaining-horizon information advantage over learned set; gray = unresolved posterior')
    fig.colorbar(image,ax=ax,label='Paired mean information difference (nats)');fig.tight_layout();fig.savefig(D/'figures/competence_matrix.png',dpi=180);plt.close(fig)