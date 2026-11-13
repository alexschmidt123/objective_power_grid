"""Assess generalization of the saved failed V2 routing pilot; never retrain."""
import argparse,hashlib,json,pathlib,sys
sys.dont_write_bytecode=True
import torch

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=pathlib.Path)
    args=parser.parse_args()
    ROOT=pathlib.Path.home()/'Documents/objective_power_grid_temp'
    P=ROOT/'scratch_runs/hec_v2_counterfactual_routing_20261112_repaired_pilot'
    S=ROOT/'development/hec_git_sync/development/moe_sboed/hec_v2_counterfactual_routing_20261112'
    sys.path.insert(0,str(S))
    from routing import CounterfactualRouter,paired_stats
    paths=[P/'teacher_train.pt',P/'teacher_validation.pt',P/'router.pt']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p):sha(p) for p in paths}
    train=torch.load(paths[0],map_location='cpu',weights_only=True)
    val=torch.load(paths[1],map_location='cpu',weights_only=True)
    checkpoint=torch.load(paths[2],map_location='cpu',weights_only=True)
    router=CounterfactualRouter();router.load_state_dict(checkpoint['router_state']);router.eval()
    mean_target=(train['returns']-train['returns'][:,1:2]).mean(0)
    best_fixed=int(train['returns'].mean(0).argmax())
    out=dict(scope='existing single-intervention teacher labels; not new full-policy EIG or a new training run',
             best_fixed_option_selected_on_training=best_fixed,model='existing ridge value selector',datasets={})
    with torch.no_grad():
     for name,data in [('train',train),('validation',val)]:
      values=data['returns'];target=values-values[:,1:2];prediction=router(data['context'])
      choice=prediction.argmax(-1);chosen=values[torch.arange(len(choice)),choice]
      mse=float((prediction-target).square().mean());constant=float((target-mean_target).square().mean())
      out['datasets'][name]=dict(n=len(values),MSE=mse,training_mean_baseline_MSE=constant,
           R2_against_training_mean=1-mse/constant,mean_advantage_by_option=target.mean(0).tolist(),
           predicted_selected_advantage=float(prediction[torch.arange(len(choice)),choice].mean()),
           selected_minus_uniform=paired_stats(chosen-values[:,1]),
           selected_minus_original=paired_stats(chosen-values[:,0]),
           training_selected_fixed_minus_uniform=paired_stats(values[:,best_fixed]-values[:,1]),
           hindsight_max_minus_uniform=float((values.max(-1).values-values[:,1]).mean()),
           hindsight_caveat='observed latent/noise winner is optimistically biased; not deployable conditional oracle')
    assert all(sha(p)==before[str(p)] for p in paths)
    out.update(inputs_preserved=True,training_updates=0)
    if args.output is not None:
     O=args.output.expanduser().resolve()
     if O.exists():raise FileExistsError('Preserve prior analysis output')
     O.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
