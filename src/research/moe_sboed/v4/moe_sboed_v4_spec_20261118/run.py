"""One matched seed101 V4-Spec pilot;unchanged native variational actor-critic."""
import argparse,copy,csv,fcntl,hashlib,importlib.util,io,json,math,os,subprocess,sys,time
from pathlib import Path
PROCESS_START=time.monotonic()
M=Path.home()/'Documents/objective_power_grid'
HERE=M/'src/research/moe_sboed_v4_spec_20261118'
O=M/'experiments/location_finding/moe_sboed_v4_spec_20261118'
R=M/'reports/moe_sboed_v4_spec_20261118'
C=M/'configs/moe_sboed_v4_spec_20261118.json'
SELF=M/'tools/moe_sboed_v4_spec_20261118.py'
spec=importlib.util.spec_from_file_location('spec_original_replication',M/'tools/moe_sboed_v4_replication_20261117.py')
v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
p=v.p;engine=p.engine
sys.path.insert(0,str(HERE))
from spec_policy import attach,SpecActor
from spec_regimes import scores,Regimes
import torch,numpy as np
from scipy.stats import t
NAMES={'learned':'MoE-sBOED-V4-Spec','uniform':'MoE-sBOED-V4-Spec-uniform',
       'original_learned':'Original V4 learned','original_uniform':'Original V4-uniform','vsoed':'vsOED'}
def read(path):return json.loads(Path(path).read_text())
def put(path,data):Path(path).parent.mkdir(parents=True,exist_ok=True);p.put(Path(path),data)
def cfg():return read(C)
def verify():
    v.verify_frozen()
    for path,value in cfg()['source_sha256'].items():assert p.sha(M/path)==value,path
def tree_digest(value):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous().numpy();h.update(str((a.dtype,a.shape)).encode());h.update(a.tobytes())
        elif isinstance(x,np.ndarray):h.update(str((x.dtype,x.shape)).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=repr):h.update(repr(key).encode());add(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x)).encode())
            for y in x:add(y)
        else:h.update(repr(x).encode())
    add(value);return h.hexdigest()
def identity(a,rng):
    return {key:tree_digest(value) for key,value in dict(actor=a.actor_net.state_dict(),target=a.actor_target_net.state_dict(),
      critic=a.critic_net.state_dict(),critic_target=a.critic_target_net.state_dict(),
      posterior={k:[net.state_dict() for net in nets] for k,nets in a.post_approx.poi_post_nets.items()},
      optimizers=[op.state_dict() for op,_ in engine.opts(a)],replay=a._runtime_buffer,rng=rng,
      counters=dict(update=int(a.update),exploration=a.design_noise_scale,actor_scheduler=a.actor_lr_scheduler.state_dict(),critic_scheduler=a.critic_lr_scheduler.state_dict())).items()}
def build(thresholds,device='cuda'):
    state=p.load_parent(device);a=state['algorithm'];actor=v.new_actor(a.actor_net,'uniform',101)
    digest=p.n.digest(actor.state_dict());expected=read(v.O/'SEED_SMOKE.json')['initializations']['101']['actor_digest'];assert digest==expected
    attach(actor,thresholds);a.actor_net=actor;a.actor_target_net=copy.deepcopy(actor);a.actor_target_net.configure('frozen');engine.optimizer(a)
    if device=='cuda':p.n.original.restore(state['rng'])
    engine.seed_all(101);return state,a,digest
def raw_bank():
    path=O/'development_validation_bank.npz';assert p.sha(path)==read(O/'BANK_IDENTITY.json')['sha256'];return np.load(path)
def initial_setup():
    assert not C.exists() and not O.exists() and not R.exists();O.mkdir(parents=True);R.mkdir(parents=True)
    source=[*HERE.glob('*.py'),SELF]
    config=dict(study='ONE_EXPLORATORY_MATCHED_V4_SPEC_SEED101_PILOT',seed=101,
      architecture='Original V4 unchanged:four256->128SiLU+nonaffineLN;644->64->4 router;weighted feature sum;128->32->2 head;117->16->2 weak branch;fixed.2 correction;bounds[-4,4]^2',
      parent_actor_sha256=p.n.ACTOR_SHA,parent_runtime_sha256=p.n.RUNTIME_SHA,initialization_seed=202611164101,
      template_seed=202611150101,training_rng_seed=101,initial_actor_digest=read(v.O/'SEED_SMOKE.json')['initializations']['101']['actor_digest'],
      warmup_updates=200,joint_updates=1800,actor_updates_per_method=2000,critic_updates_per_method=10000,posterior_updates_per_method=300000,
      episodes_per_method=2000000,observations_per_method=60000000,T=30,discount=1,use_PCE=False,
      hard_seconds_per_method_including_shared_warmup_and_evaluation=14400,training_soft_seconds=12600,
      warmup='One shared200-update uniform-forward specialization phase;original trainable shared components get normal native loss gradients;only expert parameter VJPs are masked',
      gradient_definition='z=f(c;stopgrad(theta))+m*(f(stopgrad(c);theta)-stopgrad(f(stopgrad(c);theta))).Forward values equal native z;shared gradients equal native;expert gradients are regime-selected native per-history contributions. This is explicit exposure,not a claim of unbiased full-batch expert gradients.',
      specialization=dict(calibration_seed=202611185201,replay_episodes=1024,all30_prefixes=True,threshold_quantile=.5,
        exposure='Hard0/1 expert-parameter mask,no normalization,no extra loss;overlap allowed',
        E1='log1p(trace(total posterior covariance)) >= training median',
        E2='normalized mixture entropy * log1p(max cross-probe between-component predicted-signal variance/sigma^2) >= training median;label-swapped identical predictions do not count as discrimination',
        E3='log1p(1/(lambda_min(Dscale posterior-weighted Fisher at posterior centroid Dscale)+1e-8)) >= training median',
        E4='remaining>=.5 and remaining*max_positive(frozen opportunity Q at5 centroid/cross probes minus centroid Q) >= median among eligible training prefixes',
        probes='Centroid plus(0,0),(1,0),(-1,0),(0,1),(0,-1),clipped to original bounds',
        labels='Regimes are exposure proxies,not labels of the optimal expert;no truth or future observations'),
      evaluation=dict(seed=202611185101,cases=2000,contrasts=1000000,T=30,sigma=.5,chunk=16384,scorer_sha256='64c67b6547c77f4823b487493ed0703efbd3f04d7308019e875d145505695f2f'),
      saturation=dict(near_ceiling_tolerances=[.001,.01,.05,.1,.5],material_trigger_fraction=.1,trigger_tolerance=.01,
        sensitivity_cases=64,nested_contrasts=[1000000,4000000],extension_seed=202611185102,reference_policies=['original_learned','original_uniform','vsoed'],
        scope='First64 cases of new fixed bank,original frozen policies only;estimator diagnosis,no tuning'),
      success='A:Spec-learned minus original-learned CI lower>0;B:difference of learned-uniform gaps CI lower>0 and Spec gap CI lower>0;uniform deterioration not accepted as gain',
      uniform_guard='Report Spec-uniform minus original-uniform and its paired CI;do not recommend expansion when the point estimate deteriorates. No arbitrary noninferiority margin introduced.',
      expansion='No automatic seeds102-105,no searches;recommend only if A/B and uniform guard pass;pilot is not seed robustness',
      shared_accounting='Warmup executed once;each logical method charged full shared preparation/warmup plus own joint/evaluation;physical unique episodes3.8M;logical episodes2M each',
      checkpoint_selection='Fixed terminal2000 only;never final-bank selection',source_sha256={str(x.relative_to(M)):p.sha(x) for x in source})
    put(C,config);put(R/'PROTOCOL.json',config)
    rng=np.random.default_rng(config['evaluation']['seed'])
    np.savez(O/'development_validation_bank.npz',truth=rng.standard_normal((2000,4)),noise=rng.standard_normal((2000,30)),contrasts=rng.standard_normal((1000000,4)))
    put(O/'BANK_IDENTITY.json',dict(sha256=p.sha(O/'development_validation_bank.npz'),**config['evaluation'],locked_before_pilot_training=True,architecture_selection_banks_unchanged=True))
    rows=[];cap=math.log(1000001)
    for item in read(M/'reports/moe_sboed_v4_allmethods_20261117/RESULTS.json')['records']:
        if item['method'] not in ['moe-sboed','moe-sboed-uniform','vsOED']:continue
        a=np.load(item['scores']);rows.append(dict(method=item['method'],seed=item['seed'],quantile_probabilities=[0,.05,.25,.5,.75,.9,.95,.99,1],quantiles=np.quantile(a,[0,.05,.25,.5,.75,.9,.95,.99,1]).tolist(),
          fractions_near_ceiling={str(tol):float(np.mean(a>=cap-tol)) for tol in config['saturation']['near_ceiling_tolerances']},scores_sha256=p.sha(item['scores'])))
    material=any(x['fractions_near_ceiling']['0.01']>.1 for x in rows)
    put(R/'CANONICAL_SATURATION.json',dict(ceiling=cap,rows=rows,material=material,higher_contrast_diagnostic_required=material,hyperparameters_selected_from_canonical_bank=False))
    print(json.dumps(dict(status='PREPARED',bank=p.sha(O/'development_validation_bank.npz'),saturation_material=material)),flush=True)

@torch.no_grad()
def calibrate():
    verify();p.setup();out=O/'calibration';out.mkdir(exist_ok=False);start=PROCESS_START
    with p.lease(),engine.Resources(out):
        state=p.load_parent('cuda');a=state['algorithm'];actor=v.new_actor(a.actor_net,'uniform',101).eval();buffer=a._runtime_buffer
        assert len(buffer['ds_hist'])==1000000
        rng=np.random.default_rng(cfg()['specialization']['calibration_seed']);ids=rng.choice(len(buffer['ds_hist']),1024,replace=False);np.save(out/'training_replay_indices.npy',ids)
        lib=p.n.load_library();values=[];remaining=[];histories=[]
        for first in range(0,len(ids),32):
            take=torch.as_tensor(ids[first:first+32],device=buffer['ds_hist'].device);ds=buffer['ds_hist'][take].cuda();ys=buffer['ys_hist'][take,:,0].cuda();xx=[]
            for stage in range(30):xx.append(lib.packed(ds,ys,stage))
            x=torch.cat(xx);c=actor.query(x);s,rem=scores(c,actor.experts[3].semantic.opportunity);values.append(s.cpu().numpy());remaining.append(rem.cpu().numpy());histories.append(x.cpu().numpy())
        values=np.concatenate(values);remaining=np.concatenate(remaining);thresholds=np.quantile(values,.5,axis=0);thresholds[3]=np.quantile(values[remaining>=.5,3],.5)
        masks=values>=thresholds;masks[:,3]&=remaining>=.5;freq=masks.mean(0)
        assert np.isfinite(thresholds).all() and np.all(freq>0) and np.all(freq<1),'Degenerate regime:stop,do not tune'
        np.savez_compressed(out/'training_only_prefixes.npz',X=np.concatenate(histories),scores=values,remaining=remaining)
        put(R/'REGIMES.json',dict(status='LOCKED_TRAINING_ONLY',thresholds=thresholds.tolist(),calibration_frequency=freq.tolist(),prefixes=len(values),episodes=1024,
          replay_source='Original frozen V2 parent training replay;only ds_hist and ys_hist accessed',prior_parameters_read=False,future_observations_in_selector=False,
          seconds=time.monotonic()-start,source_actor_digest=p.n.digest(actor.state_dict()),new_simulator_episodes=0,model_query_counts=dict(p.COST)))
    print(json.dumps(read(R/'REGIMES.json')),flush=True)

def native_baseline(device):
    path=Path(next(r for r in read(M/'reports/moe_sboed_location_benchmark/CHECKPOINT_INVENTORY.json') if r['method']=='vsOED' and r['seed']==101)['checkpoint'])
    from src.research.vendor.vsoed.utils import Net
    actor=Net([117,256,256,256,2],torch.nn.ReLU,torch.tensor([[-4.,4.]]*2,device=device,dtype=torch.float32),'actor').to(device)
    actor.load_state_dict(torch.load(path,map_location=device,weights_only=False)['actor_state']);actor.requires_grad_(False);actor.eval()
    return actor,path

def audit():
    verify();p.setup();out=O/'audit';out.mkdir(exist_ok=False);thresholds=read(R/'REGIMES.json')['thresholds'];start=PROCESS_START
    with p.lease(),engine.Resources(out):
        state,a,digest=build(thresholds);model=a.actor_net
        base_parent=p.load_parent('cuda');base=v.new_actor(base_parent['algorithm'].actor_net,'uniform',101);del base_parent
        base.load_state_dict(model.state_dict());base.configure('warmup');base.eval();model.eval()
        assert list(base.state_dict())==list(model.state_dict()) and sum(q.numel() for q in base.parameters())==sum(q.numel() for q in model.parameters())
        assert {name:type(m).__name__ for name,m in base.named_modules() if name}=={name:type(m).__name__ for name,m in model.named_modules() if name}
        assert not any(type(m).__name__ in {'SGActor','V2Actor','StudyMoE','SharedDecoder'} for m in model.modules())
        assert model.router[0].weight.shape==(64,644) and model.action_head[0].weight.shape==(32,128) and model.shared_small[0].weight.shape==(16,117)
        critic=copy.deepcopy(a.critic_net).requires_grad_(False)
        h=np.load(O/'calibration/training_only_prefixes.npz')['X'];x=torch.tensor(h[[0,32,160,448,640,928,1408,1888]],device='cuda')
        fixture=torch.tensor([[int((i+k)%3!=0) for k in range(4)] for i in range(len(x))],device='cuda',dtype=torch.bool)
        calls=[0,0,0,0];hooks=[]
        for k,e in enumerate(model.experts):
            def count(*unused,k=k):calls[k]+=1
            hooks.append(e.register_forward_hook(count))
        model.audit_masks=torch.ones((len(x),4),device='cuda',dtype=torch.bool)
        base.zero_grad();model.zero_grad();initial_base=base(x);initial_spec=model(x)
        torch.testing.assert_close(initial_spec,initial_base,atol=0,rtol=0)
        (-critic(torch.cat([x,initial_base],-1)).mean()).backward();(-critic(torch.cat([x,initial_spec],-1)).mean()).backward()
        for name,parameter in model.named_parameters():
            if parameter.requires_grad:torch.testing.assert_close(parameter.grad,dict(base.named_parameters())[name].grad,rtol=3e-5,atol=3e-6)
        for hook in hooks:hook.remove()
        assert all(value>0 for value in calls)
        model.audit_masks=fixture;base.zero_grad();model.zero_grad();act0=base(x);act=model(x);torch.testing.assert_close(act,act0,atol=0,rtol=0)
        q0=critic(torch.cat([x,act0],-1));q=critic(torch.cat([x,act],-1));torch.testing.assert_close(q,q0,atol=0,rtol=0)
        (-q0.mean()).backward();(-q.mean()).backward();shared_error=0.;expert_error=[]
        for name,parameter in model.named_parameters():
            if parameter.requires_grad and not name.startswith('experts.'):
                expected=dict(base.named_parameters())[name].grad;torch.testing.assert_close(parameter.grad,expected,rtol=2e-5,atol=2e-6);shared_error=max(shared_error,float((parameter.grad-expected).abs().max()))
        for k in range(4):
            named=[(n,q) for n,q in base.experts[k].named_parameters() if q.requires_grad]
            output=critic(torch.cat([x,base(x)],-1)).squeeze(-1)
            expected=torch.autograd.grad(-(output*fixture[:,k]).mean(),[q for _,q in named]);got=dict(model.experts[k].named_parameters());error=0.
            for (name,_),ref in zip(named,expected):torch.testing.assert_close(got[name].grad,ref,rtol=3e-5,atol=3e-6);error=max(error,float((got[name].grad-ref).abs().max()))
            expert_error.append(error)
        model.zero_grad();model.audit_masks=torch.zeros_like(fixture);(-critic(torch.cat([x,model(x)],-1)).mean()).backward()
        assert all(torch.count_nonzero(q.grad)==0 for e in model.experts for q in e.parameters() if q.requires_grad)
        model.audit_masks=None
        # Prefix handling must erase unavailable future design/observation slots.
        clean=x.clone();dirty=x.clone();stage=dirty[:,:30].argmax(-1)
        for i,t0 in enumerate(stage.tolist()):dirty[i,30+2*t0:88]=9000;dirty[i,88+t0:]=9000
        prefix=p.V4Actor.forward.__globals__['observed_prefix'];torch.testing.assert_close(prefix(clean),prefix(dirty),atol=0,rtol=0)
        with torch.no_grad():
            c=model.query(prefix(clean));d=model.query(prefix(dirty));ma,_=model.regimes(c,model.experts[3].semantic.opportunity);mb,_=model.regimes(d,model.experts[3].semantic.opportunity);assert torch.equal(ma,mb)
            assert torch.isfinite(act).all() and act.abs().max()<=4
            correction=.2*model.shared_small(x).tanh();assert correction.abs().max()<=.2
        # Small baseline adapter identity on training-only prefixes,not the new bank.
        reference,path=native_baseline('cuda');assert reference(x.float()).shape==(len(x),2)
        # Discarded actual native update:the real engine and checkpointed actor graph must pass.
        del base,model,a,state,critic,reference;torch.cuda.empty_cache();state,a,_=build(thresholds);test=out/'native';test.mkdir()
        receipt=engine.train_phase(a,test,'warmup',1,start,start+500)
        exposure=a.actor_net.exposure;assert int(exposure['total'])==300000,'Checkpoint recomputation must not double-count exposure'
        assert (exposure['selected']>0).all()
        logs=[json.loads(line) for line in (test/'TRAINING_METRICS.jsonl').read_text().splitlines()];grads=logs[-1]['gradient_group_norms'];assert all(grads[f'experts.{k}']>0 for k in range(4))
        assert receipt['counts']==dict(actor=1,critic=5,posterior=150)
        put(R/'GRADIENT_ARCHITECTURE_AUDIT.json',dict(status='PASS',initial_actor_digest=digest,architecture_state_keys_and_parameter_count_equal=True,
          forward_action_max_error=0,actor_scalar_loss_max_error=0,shared_gradient_max_error=shared_error,expert_gradient_max_errors=expert_error,
          allzero_exposure_yields_exact_zero_expert_gradients=True,allone_exposure_matches_native_gradients=True,all_four_executed=True,expert_execution_counts=calls,uniform_forward=True,no_future_information=True,
          native_update=receipt,native_gradient_norms=grads,actor_history_positions=int(exposure['total']),regime_exposure_counts=exposure['selected'].cpu().tolist(),
          discarded_new_simulator_episodes=1000,seconds=time.monotonic()-start,vsOED_checkpoint=str(path),vsOED_checkpoint_sha256=p.sha(path),
          objective_note='Native actor loss/critic/trajectory credit unchanged;expert gradient allocation is deliberately regime-conditioned,not an unbiased full-batch expert gradient.'))
    print(json.dumps(read(R/'GRADIENT_ARCHITECTURE_AUDIT.json')),flush=True)

def save_runtime(path,a):
    rng=p.n.original.rng();ident=identity(a,rng)
    with path.open('wb') as f:p.n.cloudpickle.dump(dict(algorithm=a,rng=rng,identity=ident),f,protocol=5)
    return ident
def load_runtime(path):
    old=torch.storage._load_from_bytes;torch.storage._load_from_bytes=lambda data:torch.load(io.BytesIO(data),map_location='cuda',weights_only=False)
    try:
        with path.open('rb') as f:state=p.n.cloudpickle.load(f)
    finally:torch.storage._load_from_bytes=old
    for op,_ in engine.opts(state['algorithm']):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
    assert identity(state['algorithm'],state['rng'])==state['identity'];return state

def warmup():
    verify();assert read(R/'GRADIENT_ARCHITECTURE_AUDIT.json')['status']=='PASS';p.setup();out=O/'shared';out.mkdir(exist_ok=False);start=PROCESS_START
    with p.lease(),engine.Resources(out):
        actor=None
        try:
            state,a,digest=build(read(R/'REGIMES.json')['thresholds']);actor=a.actor_net;p.COST.clear()
            receipt=engine.train_phase(a,out,'warmup',200,start,start+12600)
            assert int(actor.exposure['total'])==60000000
            assert torch.count_nonzero(actor.router[-1].weight)==0 and torch.count_nonzero(actor.router[-1].bias)==0
            ident=save_runtime(out/'runtime.pkl',a)
            put(out/'RESULTS.json',dict(status='COMPLETE',phase=receipt,initial_actor_digest=digest,identity=ident,runtime_sha256=p.sha(out/'runtime.pkl'),
              exposure={k:v.detach().cpu().tolist() for k,v in actor.exposure.items()},model_query_counts=dict(p.COST),seconds=time.monotonic()-start,
              physical_episodes=200000,physical_actor_updates=200,logical_credit_per_method='200 actor,1000 critic,30000 posterior,200000 episodes;shared physical execution once'))
        except Exception as error:
            if actor is not None:torch.save(dict(actor_state=actor.state_dict(),incomplete=True,error=repr(error)),out/'incomplete_actor.pt')
            put(out/'STATUS.json',dict(status='FAILED_OR_CAPPED',error=repr(error)));raise

@torch.no_grad()
def evaluate(actor,out,name,deadline):
    out.mkdir(exist_ok=False,parents=True);bank=raw_bank();lib=p.n.load_library();scorer=p.load_module('spec_eig_scorer',p.n.BASE/'source/frozen_evaluator.py').contrast_scores
    assert p.sha(p.n.BASE/'source/frozen_evaluator.py')==cfg()['evaluation']['scorer_sha256']
    negatives=torch.tensor(bank['contrasts'],device='cuda');values=[];allg=[];allm=[];allr=[];allcos=[];allweak=[];alleffects=[];latency=scoring=diagtime=0.;start=time.monotonic()
    # Same frozen belief/opportunity providers for posterior-regime diagnostics of every method.
    state=p.load_parent('cuda');provider=v.new_actor(state['algorithm'].actor_net,'uniform',101).eval().requires_grad_(False);del state
    selector=Regimes(read(R/'REGIMES.json')['thresholds']);actor.eval()
    for first in range(0,2000,16):
        engine.guard(deadline);truth=torch.tensor(bank['truth'][first:first+16],device='cuda');noise=torch.tensor(bank['noise'][first:first+16],device='cuda')
        ds=torch.zeros(len(truth),30,2,device='cuda');ys=torch.zeros(len(truth),30,device='cuda',dtype=torch.float64);gg=[];mm=[];rr=[];cc=[];ww=[];ee=[]
        for stage in range(30):
            x=lib.packed(ds,ys,stage);torch.cuda.synchronize();tick=time.monotonic()
            if isinstance(actor,p.V4Actor):action,info=actor(x,collect=True)
            else:action=actor(x.float());info=None
            torch.cuda.synchronize();latency+=time.monotonic()-tick;tick=time.monotonic()
            assert torch.isfinite(action).all() and action.abs().max()<=4.000001
            c=provider.query.inference(x);masks,_=selector(c,provider.experts[3].semantic.opportunity);mm.append(masks.cpu().numpy())
            if info is not None:
                gates=info['gates'];z=info['z'];assert z.shape[1:]==(4,128)
                if name.endswith('uniform'):assert torch.equal(gates,torch.full_like(gates,.25))
                gg.append(gates.cpu().numpy());rr.append(z.square().mean(-1).sqrt().cpu().numpy());ww.append(info['correction'].abs().cpu().numpy())
                cos=[torch.nn.functional.cosine_similarity(z[:,i],z[:,j],dim=-1) for i in range(4) for j in range(i+1,4)];cc.append(torch.stack(cos,-1).cpu().numpy())
                effects=[]
                for k in range(4):
                    changed=.9*gates;changed[:,k]+=.1;other,_=actor.decode(x,z,changed);effects.append((other-action).norm(dim=-1))
                ee.append(torch.stack(effects,-1).cpu().numpy())
            torch.cuda.synchronize();diagtime+=time.monotonic()-tick
            ds[:,stage]=action;ys[:,stage]=p.signal(truth.reshape(-1,2,2),action.double())+.5*noise[:,stage]
        torch.cuda.synchronize();tick=time.monotonic();bench=type('Benchmark',(),{'sigma':.5,'calls':0})();score,_=scorer(bench,dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=truth),negatives,16384,'cuda');torch.cuda.synchronize();scoring+=time.monotonic()-tick
        values.extend(score[:,-1].numpy());allm.append(np.stack(mm,1))
        if gg:allg.append(np.stack(gg,1));allr.append(np.stack(rr,1));allcos.append(np.stack(cc,1));allweak.append(np.stack(ww,1));alleffects.append(np.stack(ee,1))
        put(out/'STATUS.json',dict(status='EVALUATING',method=name,completed=first+len(truth),total=2000))
    values=np.asarray(values);np.save(out/'scores.npy',values);masks=np.concatenate(allm);data=dict(regimes=masks);diagnostics={}
    if allg:
        gates=np.concatenate(allg);data.update(gates=gates,expert_rms=np.concatenate(allr),expert_cosines=np.concatenate(allcos),weak_abs=np.concatenate(allweak),weight_action_effect=np.concatenate(alleffects))
        diagnostics=dict(mean_weights=gates.mean((0,1)).tolist(),stage_weights=gates.mean(0).tolist(),regime_frequency=masks.mean((0,1)).tolist(),
          weights_by_regime={str(k+1):gates[masks[:,:,k]].mean(0).tolist() if masks[:,:,k].any() else None for k in range(4)},
          expert_rms=data['expert_rms'].mean((0,1)).tolist(),expert_cosines=data['expert_cosines'].mean((0,1)).tolist(),
          weak_correction_abs_mean=float(data['weak_abs'].mean()),weak_correction_abs_max=float(data['weak_abs'].max()),weight_action_effect=data['weight_action_effect'].mean((0,1)).tolist())
    np.savez_compressed(out/'diagnostics.npz',**data)
    result=dict(status='COMPLETE',method=name,**engine.summaries(values),bank_sha256=p.sha(O/'development_validation_bank.npz'),
      seconds=time.monotonic()-start,policy_forward_batch16_seconds=latency,amortized_design_seconds=latency/60000,amortized_episode_seconds=latency/2000,
      scorer_seconds=scoring,diagnostic_seconds=diagtime,simulator_episodes=2000,simulator_observations=60000,diagnostics=diagnostics,
      timing_scope='CUDA-synchronized policy forward incl required belief/model queries;batch16 amortized,not batch1 latency;extra common regime diagnostics and scorer timed separately')
    put(out/'RESULTS.json',result);return result

def train_arm(arm):
    verify();p.setup();out=O/arm;out.mkdir(exist_ok=False);start=PROCESS_START;warm=read(O/'shared/RESULTS.json')
    preparation=sum(read(R/name).get('seconds',0) for name in ['REGIMES.json','GRADIENT_ARCHITECTURE_AUDIT.json'])
    credit=warm['seconds']+preparation;assert credit<12600
    with p.lease(),engine.Resources(out):
        actor=None
        try:
            assert p.sha(O/'shared/runtime.pkl')==warm['runtime_sha256'];state=load_runtime(O/'shared/runtime.pkl');a=state['algorithm'];assert state['identity']==warm['identity']
            put(out/'CLONED_WARMUP_IDENTITY.json',dict(identity=state['identity'],common_runtime_sha256=warm['runtime_sha256'],equal_to_shared=True))
            actor=a.actor_net;actor.arm=arm;a.actor_target_net.arm=arm;a.actor_target_net.configure('frozen');p.n.original.restore(state['rng']);actor.audit_masks=None
            p.COST.clear();phase=engine.train_phase(a,out,'joint',1800,start-credit,start+12600-credit)
            training_seconds=time.monotonic()-start;save_runtime(out/'runtime.pkl',a);actor.configure('frozen');actor.eval();modelqueries=dict(p.COST);del state,a;torch.cuda.empty_cache()
            ev=evaluate(actor,out/'evaluation',NAMES[arm],start+14100-credit)
            total=credit+time.monotonic()-start;assert total<=14400
            assert phase['counts']==dict(actor=1800,critic=9000,posterior=270000)
            put(out/'RESULTS.json',dict(status='COMPLETE',method=NAMES[arm],seed=101,phase=phase,shared_warmup=warm['phase'],
              shared_preparation_and_warmup_seconds_charged=credit,joint_training_seconds=training_seconds,logical_total_seconds=total,
              actor_updates=2000,critic_updates=10000,posterior_updates=300000,new_episodes_logical=2000000,unique_joint_episodes=1800000,
              checkpoint_sha256=p.sha(out/'joint_actor.pt'),model_query_counts=modelqueries,evaluation=ev))
        except Exception as error:
            if actor is not None:torch.save(dict(actor_state=actor.state_dict(),incomplete=True,error=repr(error)),out/'incomplete_actor.pt')
            put(out/'STATUS.json',dict(status='FAILED_OR_CAPPED',error=repr(error)));raise

def references():
    verify();p.setup();out=O/'references';out.mkdir(exist_ok=False)
    with p.lease(),engine.Resources(out),torch.no_grad():
        state=p.load_parent('cuda');parent=state['algorithm'].actor_net
        for arm in ['learned','uniform']:
            actor=v.new_actor(parent,arm,101);checkpoint=v.training_path(101,arm)/'joint_actor.pt';before=p.sha(checkpoint)
            actor.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)['actor_state']);actor.configure('frozen');actor.eval()
            result=evaluate(actor,out/('original_'+arm),NAMES['original_'+arm],time.monotonic()+2000);assert p.sha(checkpoint)==before
            put(out/('original_'+arm)/'CHECKPOINT.json',dict(path=str(checkpoint),sha256=before));del actor
        actor,path=native_baseline('cuda');before=p.sha(path);evaluate(actor,out/'vsoed',NAMES['vsoed'],time.monotonic()+2000);assert p.sha(path)==before
        put(out/'vsoed/CHECKPOINT.json',dict(path=str(path),sha256=before))

@torch.no_grad()
def saturation_sensitivity():
    verify();p.setup();out=O/'saturation';out.mkdir(exist_ok=False)
    with p.lease(),engine.Resources(out):
        bank=raw_bank();rng=np.random.default_rng(cfg()['saturation']['extension_seed']);negative=np.concatenate([bank['contrasts'],rng.standard_normal((3000000,4))])
        put(out/'BANK.json',dict(new_bank_sha256=p.sha(O/'development_validation_bank.npz'),extension_seed=cfg()['saturation']['extension_seed'],cases=64,contrasts=[1000000,4000000],purpose='Fixed estimator diagnosis only'))
        allrows=[];state=p.load_parent('cuda');parent=state['algorithm'].actor_net;lib=p.n.load_library();scorer=p.load_module('spec_saturation_scorer',p.n.BASE/'source/frozen_evaluator.py').contrast_scores
        for mode in ['original_learned','original_uniform','vsoed']:
            if mode=='vsoed':actor,path=native_baseline('cuda')
            else:
                arm=mode.split('_')[-1];actor=v.new_actor(parent,arm,101);actor.load_state_dict(torch.load(v.training_path(101,arm)/'joint_actor.pt',map_location='cuda',weights_only=True)['actor_state']);actor.configure('frozen')
            vv={1000000:[],4000000:[]}
            for first in range(0,64,16):
                truth=torch.tensor(bank['truth'][first:first+16],device='cuda');eps=torch.tensor(bank['noise'][first:first+16],device='cuda');ds=torch.zeros(len(truth),30,2,device='cuda');ys=torch.zeros(len(truth),30,device='cuda',dtype=torch.float64)
                for stage in range(30):
                    action=actor(lib.packed(ds,ys,stage));ds[:,stage]=action;ys[:,stage]=p.signal(truth.reshape(-1,2,2),action.double())+.5*eps[:,stage]
                for count in vv:
                    bench=type('Benchmark',(),{'sigma':.5,'calls':0})();value,_=scorer(bench,dict(actions=list(ds.unbind(1)),observations=list(ys[...,None].unbind(1)),truth=truth),torch.tensor(negative[:count],device='cuda'),16384,'cuda');vv[count].extend(value[:,-1].numpy())
            a=np.asarray(vv[1000000]);b=np.asarray(vv[4000000]);np.savez(out/(mode+'.npz'),L1M=a,L4M=b)
            allrows.append(dict(method=NAMES[mode],L1M=engine.summaries(a),L4M=engine.summaries(b),paired_4M_minus_1M=engine.summaries(b-a)))
        put(R/'SATURATION_SENSITIVITY.json',dict(status='COMPLETE',cases=64,rows=allrows,formal_pilot_protocol_unchanged=True,training_hyperparameters_selected=False))

def report():
    rows={};arrays={}
    for key in NAMES:
        folder=O/key/'evaluation' if key in ['learned','uniform'] else O/'references'/key
        path=folder/'RESULTS.json'
        if path.exists():d=read(path);rows[key]=d;arrays[key]=np.load(folder/'scores.npy');assert d['bank_sha256']==read(O/'BANK_IDENTITY.json')['sha256']
    comparisons={};decision=None
    if len(rows)==5:
        vectors=dict(learned_minus_uniform=arrays['learned']-arrays['uniform'],Spec_minus_original=arrays['learned']-arrays['original_learned'],
          Spec_minus_vsOED=arrays['learned']-arrays['vsoed'],original_routing_gap=arrays['original_learned']-arrays['original_uniform'],
          gap_improvement=(arrays['learned']-arrays['uniform'])-(arrays['original_learned']-arrays['original_uniform']),
          uniform_change=arrays['uniform']-arrays['original_uniform'])
        comparisons={key:engine.summaries(value) for key,value in vectors.items()}
        A=comparisons['Spec_minus_original']['CI95'][0]>0;B=comparisons['gap_improvement']['CI95'][0]>0 and comparisons['learned_minus_uniform']['CI95'][0]>0
        guard=comparisons['uniform_change']['mean']>=0
        decision=dict(improvement_A=A,improvement_B=B,uniform_not_deteriorated_point_estimate=guard,recommend_replication=A and B and guard,automatic_expansion=False,
          scope='One-seed development pilot,case uncertainty conditional on bank;not training-seed robustness or proof of optimal expert regimes')
    status='COMPLETE' if len(rows)==5 else 'PENDING'
    put(R/'RESULTS.json',dict(status=status,rows=rows,comparisons=comparisons,decision=decision,bank_identity=read(O/'BANK_IDENTITY.json'),
      caveats=['Full-horizon finite-contrast lower bound;contrast bias/saturation is separate from MCSE.','Shared frozen V2 parent;one continuation seed only.','Regime labels are exposure heuristics,not optimal-expert labels.','Uniform shares exact warmup and gets full matched joint training.']))
    text='# V4-Spec matched seed101 pilot\n\nStatus: '+status+'. Same NEW bank for all five policies;original canonical/pilot results unchanged.\n\n|Method|EIG nats|MCSE|\n|---|---:|---:|\n'
    for key in NAMES:
        if key in rows:text+=f"|{NAMES[key]}|{rows[key]['mean']:.5f}|{rows[key]['MCSE']:.5f}|\n"
        else:text+=f'|{NAMES[key]}|PENDING|PENDING|\n'
    text+='\n|Paired comparison|Difference|95% evaluation CI|\n|---|---:|---|\n'
    for key,d in comparisons.items():text+=f"|{key}|{d['mean']:+.5f}|[{d['CI95'][0]:+.5f},{d['CI95'][1]:+.5f}]|\n"
    text+='\nDecision: '+json.dumps(decision)+'. No automatic expansion.\n';(R/'RESULTS.md').write_text(text)

def controller():
    handle=(O/'.controller.lock').open('a+');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=['calibrate','audit','saturation','warmup','learned','uniform','references'];done=[]
    for name in jobs:
        put(O/'STATUS.json',dict(status='RUNNING',current=name,completed=done,pending=jobs[len(done)+1:],seed=101))
        command=[sys.executable,'-B','-u',str(SELF),'--task',name]
        seconds=14370 if name in ['warmup','learned','uniform'] else 3500
        while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            put(O/'STATUS.json',dict(status='WAITING_FOR_UNRELATED_GPU_JOB',current=name,completed=done));time.sleep(20)
        with (O/(name+'.log')).open('x') as log:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=30',str(seconds),*command],stdout=log,stderr=subprocess.STDOUT,cwd=M)
            put(O/'ACTIVE_PROCESS.json',dict(pid=child.pid,controller_pid=os.getpid(),task=name,seed=101));code=child.wait()
        if code:put(O/'STATUS.json',dict(status='FAILED_PRESERVED',current=name,returncode=code,completed=done));report();return
        done.append(name);report()
    put(O/'STATUS.json',dict(status='COMPLETE',completed=done,seed=101,no_expansion=True))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['prepare','calibrate','audit','saturation','warmup','learned','uniform','references','controller','report'],required=True);a=parser.parse_args()
    if a.task=='prepare':initial_setup()
    elif a.task=='calibrate':calibrate()
    elif a.task=='audit':audit()
    elif a.task=='saturation':saturation_sensitivity()
    elif a.task=='warmup':warmup()
    elif a.task in ['learned','uniform']:train_arm(a.task)
    elif a.task=='references':references()
    elif a.task=='controller':controller()
    else:report()
if __name__=='__main__':main()
