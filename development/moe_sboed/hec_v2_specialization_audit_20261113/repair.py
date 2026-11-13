"""Evidence-selected E4 adapter refresh; matched native V2 continuation."""
import argparse,gc,importlib.util,json,pathlib,sys,time,subprocess
import numpy as np
import torch
import audit
n=audit.native
AUDIT=audit.ROOT/'scratch_runs/hec_v2_specialization_audit_20261113'

def refresh(actor,arm,out,steps=300):
    assert arm in ['ordinary','E4_refresh']
    data=np.load(AUDIT/'RAW_COMPETENCY.npz');hist=np.load(AUDIT/'FRESH_HISTORIES.npz')
    count=256*3;x=torch.tensor(hist['X'][:count],device='cuda')
    target=torch.tensor(data['target3'][:count],device='cuda',dtype=torch.float32)
    scale=target.std(0,unbiased=False).clamp_min(.1)
    flags={key:p.requires_grad for key,p in actor.named_parameters()}
    with torch.no_grad():
        before=actor(x[:24]).clone();_,info=actor(x,collect=True)
        z=info['responses'][:,3,12:].cpu().numpy().astype(float)
    frozen={key:p.detach().clone() for key,p in actor.named_parameters() if not key.startswith(('adapter4.','response_readouts.1.'))}
    mu,sd,beta=audit.ridge_fit(z,data['target3'][:count])
    weight=beta[1:]/sd[:,None];bias=beta[0]-mu@weight
    head=actor.response_readouts[1]
    with torch.no_grad():
        head.weight.copy_(torch.tensor(weight.T,device='cuda',dtype=torch.float32));head.bias.copy_(torch.tensor(bias,device='cuda',dtype=torch.float32))
    actor.requires_grad_(False);head.requires_grad_(True)
    if arm=='E4_refresh':actor.adapter4.requires_grad_(True)
    params=[p for p in actor.parameters() if p.requires_grad];opt=torch.optim.Adam(params,lr=.001)
    rng=np.random.default_rng(202611132801)
    loss_values=[]
    for step in range(steps):
        idx=torch.tensor(rng.integers(count,size=256),device='cuda');opt.zero_grad(set_to_none=True)
        _,info=actor(x[idx],collect=True)
        prediction=head(info['responses'][:,3,12:]);loss=((prediction-target[idx])/scale).square().mean()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        opt.step();loss_values.append(float(loss.detach()))
    assert all(torch.equal(saved,dict(actor.named_parameters())[key]) for key,saved in frozen.items())
    with torch.no_grad():after=actor(x[:24])
    drift=float((after-before).abs().max())
    if arm=='ordinary':assert torch.equal(after,before)
    else:assert drift>0
    for key,p in actor.named_parameters():p.grad=None;p.requires_grad_(flags[key])
    n.put(out/f'{arm}_REFRESH.json',dict(steps=steps,loss_first=loss_values[0],loss_last=loss_values[-1],
        selected_modules=['response_readouts.1']+(['adapter4'] if arm=='E4_refresh' else []),
        other_actor_parameters_preserved=True,max_action_drift=drift,
        target='Unchanged v2_tasks E4 task,current-parent teacher,calibration episodes only',
        native_optimizer_moments_preserved=True,new_simulator_observations=0))
    return [key for key,p in actor.named_parameters() if p.requires_grad]

def competency(out,deadline):
    state=n.load_parent();a=state['algorithm'];parent=a.actor_net.eval().requires_grad_(False)
    data=np.load(AUDIT/'RAW_COMPETENCY.npz');hist=np.load(AUDIT/'FRESH_HISTORIES.npz')
    cal_x=torch.tensor(hist['X'][:768],device='cuda');cal_y=data['target3'][:768];scale=cal_y.std(0).clip(.1)
    lib=n.load_library();rng=np.random.default_rng(202611132802);cases=256
    theta=torch.tensor(rng.standard_normal((cases,4)),device='cuda');noise=torch.tensor(rng.standard_normal((cases,30)),device='cuda')
    ds,ys,_=lib.rollout(parent,theta,noise,deadline=deadline)
    test_x=torch.cat([lib.packed(ds,ys,t)[:,None] for t in [3,14,26]],1).reshape(-1,117)
    generator=torch.Generator(device='cuda').manual_seed(202611132803)
    labels=[];commands=[]
    with torch.no_grad(),torch.device('cuda'):
        cal_commands=parent(cal_x,collect=True)[1]['commands']
        for first in range(0,len(test_x),64):
            xx=test_x[first:first+64];_,info=parent(xx,collect=True)
            y,_=audit.labels(a,parent,parent,xx,info,generator,16)
            labels.append(y[3].cpu().numpy());commands.append(info['commands'])
    y=np.concatenate(labels);test_commands=torch.cat(commands)
    losses={};model=parent
    for arm in ['original','ordinary','E4_refresh']:
        if arm!='original':model.load_state_dict(torch.load(out/arm/'actor.pt',map_location='cuda',weights_only=True)['actor_state'])
        with torch.no_grad():
            _,ci=model(cal_x,command_override=cal_commands,collect=True)
            _,ti=model(test_x,command_override=test_commands,collect=True)
        cz=audit.representation(ci['slots'])[:,3].cpu().numpy().astype(float)
        tz=audit.representation(ti['slots'])[:,3].cpu().numpy().astype(float)
        probe=audit.ridge_fit(cz,cal_y);error=np.square((audit.ridge_predict(probe,tz)-y)/scale).mean(-1).reshape(cases,3).mean(1)
        losses[arm]=error;np.save(out/f'{arm}_fresh_competency_errors.npy',error)
    result=dict(cases=cases,teacher='frozen original V2',
        primary_ordinary_minus_refresh_error=audit.paired(losses['ordinary'],losses['E4_refresh']),
        original_minus_refresh_error=audit.paired(losses['original'],losses['E4_refresh']),
        errors={k:float(v.mean()) for k,v in losses.items()},targets_are_model_derived=True)
    n.put(out/'FRESH_COMPETENCY.json',result)
    del state,parent,model;gc.collect();torch.cuda.empty_cache()
    return result

def eig(out,cfg,deadline):
    lib=n.load_library();actor=lib.load_frozen('cuda');rng=np.random.default_rng(202611132804)
    cases=2048;theta=torch.tensor(rng.standard_normal((cases,4)),device='cuda');noise=torch.tensor(rng.standard_normal((cases,30)),device='cuda')
    array=np.random.default_rng(202611132805).standard_normal((16000000,4))
    bank=torch.tensor(array,device='cuda');del array
    scores={}
    for arm in ['original','ordinary','E4_refresh','E4_refresh_uniform']:
        if arm in ['ordinary','E4_refresh']:actor.load_state_dict(torch.load(out/arm/'actor.pt',map_location='cuda',weights_only=True)['actor_state'])
        values=[]
        for first in range(0,cases,16):
            th=theta[first:first+16];eps=noise[first:first+16]
            actions,ys,_=lib.rollout(actor,th,eps,'uniform' if arm.endswith('_uniform') else 'original',deadline=deadline)
            values.append(n.nested_scores(actions,ys,th,bank,[16000000],deadline=deadline)[16000000].cpu().numpy())
            n.put(out/'STATUS.json',dict(status='EVALUATING',arm=arm,completed=min(first+16,cases),total=cases))
        scores[arm]=np.concatenate(values);np.save(out/f'{arm}_scores.npy',scores[arm])
    result=dict(primary_refresh_minus_ordinary=audit.paired(scores['E4_refresh'],scores['ordinary']),
       secondary_refresh_minus_original=audit.paired(scores['E4_refresh'],scores['original']),
       secondary_refresh_minus_uniform=audit.paired(scores['E4_refresh'],scores['E4_refresh_uniform']),
       means={k:float(v.mean()) for k,v in scores.items()},cases=cases,contrasts=16000000)
    n.put(out/'EIG.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser().resolve();assert audit.ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(updates=2000,wall_seconds=14400,refresh_steps=300,refresh_batch=256,refresh_lr=.001,
       competency_cases=256,competency_seed=202611132802,teacher_seed=202611132803,
       evaluation_cases=2048,case_seed=202611132804,contrast_seed=202611132805,contrasts=16000000,
       matched='same parent,RNG,replay,native2000 updates,shared teacher dataset,300 refresh steps;control readout only has no policy effect;wall/FLOPs not exactly equal',
       native_objective_unchanged=True,checkpoint_selection='final only',source_commit=subprocess.check_output(['git','-C',str(audit.WORK),'rev-parse','HEAD'],text=True).strip())
    n.put(out/'CONFIG.json',cfg)
    files=[*pathlib.Path(__file__).parent.glob('*.py'),*sorted((audit.BASE/'source').glob('*.py')),
           *sorted((n.P/'source').rglob('*.py')),audit.PARENT/'train_router.py',n.P/'frozen_protocol.json',
           AUDIT/'RAW_COMPETENCY.npz',AUDIT/'FRESH_HISTORIES.npz',AUDIT/'RESULTS.json',AUDIT/'LABEL_NOISE.json',
           audit.BASE/'arms/candidate/actor.pt',audit.BASE/'arms/candidate/runtime.pkl']
    protected={str(f):n.sha(f) for f in files};n.put(out/'PRE_RUN_MANIFEST.json',protected)
    n.original.setup()
    while True:
        try:lease=n.original.acquire();break
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+cfg['wall_seconds']
    try:
        if args.smoke:
            for arm in ['ordinary','E4_refresh']:
                state=n.load_parent();refresh(state['algorithm'].actor_net,arm,out,steps=3);del state;gc.collect();torch.cuda.empty_cache()
            n.put(out/'STATUS.json',dict(status='VALIDATED',native_training_steps=0));return
        n.configure=lambda actor,arm:refresh(actor,arm,out)
        for arm in ['ordinary','E4_refresh']:
            status=n.train_arm(arm,out,cfg,start);gc.collect();torch.cuda.empty_cache()
            if status!='COMPLETE':n.put(out/'STATUS.json',dict(status=status,arm=arm));return
        comp=competency(out,deadline);value=eig(out,cfg,deadline)
        assert all(n.sha(f)==h for f,h in protected.items())
        accepted=comp['primary_ordinary_minus_refresh_error']['CI95'][0]>0 and value['primary_refresh_minus_ordinary']['CI95'][0]>0
        n.put(out/'RESULTS.json',dict(status='COMPLETE',accepted=accepted,competency=comp,EIG=value,seconds=time.monotonic()-start,
             limitations='One training lineage;finite prior-sPCE and model-derived competency;extra adapter-backprop compute differs from readout-only control,reported explicitly. Audit test histories informed method choice and are not final evaluation.'))
        n.put(out/'STATUS.json',dict(status='COMPLETE',accepted=accepted,seconds=time.monotonic()-start,historical_artifacts_preserved=True))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()

if __name__=='__main__':
    main()
