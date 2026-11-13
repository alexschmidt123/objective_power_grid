"""Reproduce only short E4 preparation, trace before/prepared/final competency."""
import argparse,gc,importlib.util,json,pathlib,sys,time
sys.dont_write_bytecode=True
import numpy as np
import torch
ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp';WORK=ROOT/'development/hec_git_sync'
SOURCE=WORK/'development/moe_sboed/hec_v2_specialization_audit_20261113'
sys.path.insert(0,str(SOURCE))
import repair
import audit
n=repair.n
PREVIOUS=ROOT/'scratch_runs/hec_v2_E4_refresh_20261113'

def reproduce(actor,arm,out):
    raw=np.load(repair.AUDIT/'RAW_COMPETENCY.npz');hist=np.load(repair.AUDIT/'FRESH_HISTORIES.npz')
    x=torch.tensor(hist['X'][:768],device='cuda');target=torch.tensor(raw['target3'][:768],device='cuda',dtype=torch.float32)
    scale=target.std(0,unbiased=False).clamp_min(.1);trace=[];step=0
    def measure():
        with torch.no_grad():
            _,info=actor(x,collect=True);prediction=actor.response_readouts[1](info['responses'][:,3,12:])
            loss=float(((prediction-target)/scale).square().mean())
        grads={}
        for name,p in actor.named_parameters():
            if name.startswith(('adapter4.','response_readouts.1.')) and p.grad is not None:
                assert torch.isfinite(p.grad).all();grads[name]=float(p.grad.norm())
        return dict(step=step,full_calibration_readout_MSE=loss,gradient_norms=grads)
    trace.append(dict(stage='before_readout_calibration',**measure()))
    # Observe the existing, unchanged preparation optimizer; no update is added.
    real_adam=torch.optim.Adam
    def observed_adam(*args,**kwargs):
        opt=real_adam(*args,**kwargs)
        trace.append(dict(stage='after_readout_calibration',**measure()))
        def observe(*unused):
            nonlocal step
            step+=1
            if step in [1,10,50,100,200,300]:trace.append(dict(stage='after_step',**measure()))
        opt.register_step_post_hook(observe)
        return opt
    torch.optim.Adam=observed_adam
    try:repair.refresh(actor,arm,out,steps=300)
    finally:torch.optim.Adam=real_adam
    assert step==300
    historical=json.loads((PREVIOUS/arm/'STARTED.json').read_text())
    digest=n.digest(actor.state_dict());assert digest==historical['initial_actor_digest'],'Prepared state does not reproduce historical digest'
    receipt=json.loads((out/f'{arm}_REFRESH.json').read_text());old=json.loads((PREVIOUS/f'{arm}_REFRESH.json').read_text())
    assert receipt==old,'Preparation receipt differs from original run'
    torch.save(dict(actor_state=actor.state_dict(),native_iteration=16001,reproduced=True),out/f'{arm}_prepared.pt')
    n.put(out/f'{arm}_PREPARATION_TRACE.json',dict(trace=trace,exact_historical_digest_match=True,digest=digest,
          exact_historical_receipt_match=True,preparation_steps=step,native_training_steps=0))
    return dict(digest=digest,trace=trace)

def relative_change(current,reference,prefix):
    keys=[k for k in reference if k.startswith(prefix)]
    numerator=sum(float((current[k].double()-reference[k].double()).square().sum()) for k in keys)
    denominator=sum(float(reference[k].double().square().sum()) for k in keys)
    return dict(relative_L2=(numerator/max(denominator,1e-30))**.5,
                max_abs=max(float((current[k]-reference[k]).abs().max()) for k in keys))

def summarize(errors):
    original=errors['original'];prepared=errors['E4_prepared'];final=errors['E4_final']
    gain=audit.paired(original,prepared);forget=audit.paired(final,prepared)
    if gain['CI95'][0]>0 and forget['CI95'][0]>0:decision='PREPARATION_GAIN_AND_LATER_DEGRADATION_SUPPORTED'
    elif gain['CI95'][0]<=0:decision='NO_DEMONSTRATED_PREPARATION_COMPETENCY_GAIN'
    else:decision='PREPARATION_GAIN_SUPPORTED_RETENTION_INCONCLUSIVE_OR_RETAINED'
    return dict(decision=decision,original_minus_prepared_error=gain,final_minus_prepared_error=forget,
        original_minus_final_error=audit.paired(original,final),
        ordinary_final_minus_E4_final_error=audit.paired(errors['ordinary_final'],final))

def tests():
    # Positive improvement means smaller error; detect only statistically supported forgetting.
    original=np.full(32,1.);prepared=np.full(32,.5);final=np.full(32,1.)
    fixture=dict(original=original,E4_prepared=prepared,E4_final=final,ordinary_final=original)
    assert summarize(fixture)['decision']=='PREPARATION_GAIN_AND_LATER_DEGRADATION_SUPPORTED'
    fixture['E4_prepared']=original
    assert summarize(fixture)['decision']=='NO_DEMONSTRATED_PREPARATION_COMPETENCY_GAIN'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    out=args.output.expanduser().resolve();assert ROOT/'scratch_runs' in out.parents and not out.exists();out.mkdir(parents=True)
    cfg=dict(preparation_steps_per_arm=300,native_training_steps=0,teacher_seed=202611132803,
        case_seed=202611132802,cases=256,stages=[3,14,26],teacher_replicates=16,
        evaluation='Replay previous independent competency bank,common parent commands and teacher,no checkpoint selection',
        exact_reproduction='Require prepared actor-state digest AND complete preparation receipt equal historical records',
        wall_seconds=900,source_commit=__import__('subprocess').check_output(['git','-C',str(WORK),'rev-parse','HEAD'],text=True).strip())
    n.put(out/'CONFIG.json',cfg)
    protected_files=[*sorted(pathlib.Path(__file__).parent.glob('*.py')),*sorted(SOURCE.glob('*.py')),
         *sorted((audit.BASE/'source').glob('*.py')),audit.PARENT/'train_router.py',
         audit.BASE/'arms/candidate/actor.pt',audit.BASE/'arms/candidate/runtime.pkl',
         repair.AUDIT/'RAW_COMPETENCY.npz',repair.AUDIT/'FRESH_HISTORIES.npz',
         *[p for p in PREVIOUS.rglob('*') if p.is_file() and p.suffix in ['.pt','.pkl','.json','.npy','.jsonl']]]
    manifest={str(f):n.sha(f) for f in protected_files};n.put(out/'PRE_RUN_MANIFEST.json',manifest)
    n.original.setup();tests()
    while True:
        try:lease=n.original.acquire();break
        except BlockingIOError:n.put(out/'STATUS.json',dict(status='WAITING_FOR_IDLE_GPU'));time.sleep(30)
    start=time.monotonic();deadline=start+cfg['wall_seconds']
    try:
        reproductions={}
        for arm in ['ordinary','E4_refresh']:
            state=n.load_parent();a=state['algorithm'];reproductions[arm]=reproduce(a.actor_net,arm,out)
            assert int(a.update)==16001
            del a,state;gc.collect();torch.cuda.empty_cache()
        state=n.load_parent();a=state['algorithm'];actor=a.actor_net.eval().requires_grad_(False)
        original_state={k:v.detach().clone() for k,v in actor.state_dict().items()}
        data=np.load(repair.AUDIT/'RAW_COMPETENCY.npz');hist=np.load(repair.AUDIT/'FRESH_HISTORIES.npz')
        cal_x=torch.tensor(hist['X'][:768],device='cuda');cal_y=data['target3'][:768];scale=cal_y.std(0).clip(.1)
        lib=n.load_library();rng=np.random.default_rng(cfg['case_seed']);cases=cfg['cases']
        theta=torch.tensor(rng.standard_normal((cases,4)),device='cuda');noise=torch.tensor(rng.standard_normal((cases,30)),device='cuda')
        ds,ys,_=lib.rollout(actor,theta,noise,deadline=deadline);del theta,noise
        test_x=torch.cat([lib.packed(ds,ys,t)[:,None] for t in cfg['stages']],1).reshape(-1,117)
        generator=torch.Generator(device='cuda').manual_seed(cfg['teacher_seed']);labels=[];commands=[]
        with torch.no_grad(),torch.device('cuda'):
            cal_commands=actor(cal_x,collect=True)[1]['commands']
            parent_action,pi=actor(test_x,collect=True)
            reference_features=pi['responses'][:,3,12:].detach().clone()
            for first in range(0,len(test_x),64):
                xx=test_x[first:first+64];_,info=actor(xx,collect=True)
                target,_=audit.labels(a,actor,actor,xx,info,generator,16)
                labels.append(target[3].cpu().numpy());commands.append(info['commands'])
        y=np.concatenate(labels);test_commands=torch.cat(commands)
        np.savez_compressed(out/'COMMON_EVALUATION.npz',X=test_x.cpu().numpy(),targets=y,commands=test_commands.cpu().numpy())
        paths=dict(original=None,ordinary_prepared=out/'ordinary_prepared.pt',E4_prepared=out/'E4_refresh_prepared.pt',
                   ordinary_final=PREVIOUS/'ordinary/actor.pt',E4_final=PREVIOUS/'E4_refresh/actor.pt')
        results={};errors={};full_errors={};head_errors={}
        for name,path in paths.items():
            if time.monotonic()>deadline:raise TimeoutError('Bounded trace budget')
            if path:actor.load_state_dict(torch.load(path,map_location='cuda',weights_only=True)['actor_state'])
            else:actor.load_state_dict(original_state)
            with torch.no_grad():
                _,ci=actor(cal_x,command_override=cal_commands,collect=True)
                action,ti=actor(test_x,command_override=test_commands,collect=True)
                native_action=actor(test_x)
                prediction=actor.response_readouts[1](ti['responses'][:,3,12:]).cpu().numpy()
            cz=audit.representation(ci['slots'])[:,3].cpu().numpy().astype(float)
            tz=audit.representation(ti['slots'])[:,3].cpu().numpy().astype(float)
            probe=audit.ridge_fit(cz,cal_y);error=np.square((audit.ridge_predict(probe,tz)-y)/scale).mean(-1).reshape(cases,3).mean(1)
            p52=audit.ridge_fit(ci['responses'][:,3,12:].cpu().numpy().astype(float),cal_y)
            e52=np.square((audit.ridge_predict(p52,ti['responses'][:,3,12:].cpu().numpy().astype(float))-y)/scale).mean(-1).reshape(cases,3).mean(1)
            he=np.square((prediction-y)/scale).mean(-1).reshape(cases,3).mean(1)
            errors[name]=error;head_errors[name]=he;full_errors[name]=e52
            change=(ti['responses'][:,3,12:]-reference_features)
            results[name]=dict(probe128_MSE=float(error.mean()),adapter52_probe_MSE=float(e52.mean()),existing_readout_MSE=float(he.mean()),
                adapter_relative_parameter_change=relative_change(actor.state_dict(),original_state,'adapter4.'),
                adapter_feature_RMS_change=float(change.square().mean().sqrt()),
                action_RMS_change_common_commands=float((action-parent_action).square().mean().sqrt()),
                action_max_change_common_commands=float((action-parent_action).abs().max()),
                action_RMS_change_native_commands=float((native_action-parent_action).square().mean().sqrt()))
            if name in ['original','ordinary_final','E4_final']:
                oldname={'original':'original','ordinary_final':'ordinary','E4_final':'E4_refresh'}[name]
                saved=np.load(PREVIOUS/f'{oldname}_fresh_competency_errors.npy')
                assert np.max(np.abs(error-saved))<1e-8,'Historical evaluation not reproduced'
            np.savez_compressed(out/f'{name}_ERRORS.npz',full_packet=error,adapter_only=e52,readout=he)
        assert np.array_equal(errors['original'],errors['ordinary_prepared'])
        result=dict(status='COMPLETE',stages=results,competency=summarize(errors),
           adapter_only_competency=summarize(full_errors),diagnostic_readout_competency=summarize(head_errors),
           historical_prepared_state_digests_match=True,historical_final_evaluation_reproduced=True,
           seconds=time.monotonic()-start,native_training_steps=0,preparation_steps=600,
           limitations='Diagnostic replay of previous heldout bank,not independent new policy acceptance;model-derived teacher,one lineage. Different minibatch endpoint losses are not evidence of learning. Exact matching ensures preparation replay fidelity. No new long training or EIG optimization.')
        assert all(n.sha(f)==h for f,h in manifest.items())
        n.put(out/'RESULTS.json',result);n.put(out/'STATUS.json',dict(status='COMPLETE',seconds=time.monotonic()-start,historical_artifacts_preserved=True,native_training_steps=0))
    except Exception as exc:n.put(out/'STATUS.json',dict(status='FAILED',error=str(exc)));raise
    finally:
        for f in lease:f.close()

if __name__=='__main__':main()
