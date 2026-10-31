"""One predeclared v2 residual-context-readout pilot; independent from LabPC."""
import copy,json,os,subprocess,time,traceback,math
from pathlib import Path
import numpy as np
import torch,yaml
from src.research.moe_v2_context import context_matched_models
from src.research.moe_v2_research import ROOT,write,sha,summary,source_manifest,validator,contexts,holm,STAGES
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
ORDER=['v1','v2','dense','shared']

def competence(model,out):
    thresholds={s:float(np.median(x)) for s,x in contexts(model,validator(97402754,512,0))[0].items()}
    write(out/'frozen_competence_contexts.json',thresholds);all_effects=[];all_interactions=[]
    for split,seed in [('specialization_development',97402755),('specialization_replication',97402756)]:
        v=validator(seed,256,65536);ctx,top2=contexts(model,v);baseline,_=v(model);effect_rows=[];interaction_rows=[]
        for stage in STAGES:
            scores={}
            for e in range(7):
                p=copy.deepcopy(model);p.substitution=(stage,e);scores[e],_=v(p);np.save(out/f'evaluations/{split}_stage{stage}_expert{e}.npy',scores[e])
                eligible=~(top2[stage]==e).any(1);bins=np.zeros(256,dtype=int) if stage==1 else (ctx[stage]>thresholds[stage]).astype(int)
                for c in (0,) if stage==1 else (0,1):
                    keep=eligible&(bins==c)
                    if keep.sum()<16:continue
                    row={'split':split,'stage':stage,'expert':e,'context':c,**summary(scores[e][keep]-baseline[keep])};row['p_approx_two_sided']=math.erfc(abs(row['mean']/max(row['MC_SE'],1e-300))/2**.5);effect_rows.append(row)
            if stage==1:continue
            for a in range(7):
                for b in range(a+1,7):
                    eligible=~((top2[stage]==a)|(top2[stage]==b)).any(1);lo=eligible&(ctx[stage]<=thresholds[stage]);hi=eligible&(ctx[stage]>thresholds[stage])
                    if min(lo.sum(),hi.sum())<16:continue
                    d=scores[a]-scores[b];l=summary(d[lo]);h=summary(d[hi]);delta=h['mean']-l['mean'];se=(h['MC_SE']**2+l['MC_SE']**2)**.5
                    interaction_rows.append({'split':split,'stage':stage,'expert_a':a,'expert_b':b,'low_context_difference':l,'high_context_difference':h,'interaction':delta,'MC_SE':se,'p_approx_two_sided':math.erfc(abs(delta/max(se,1e-300))/2**.5),'meaningful_point_rank_reversal':l['mean']*h['mean']<0 and min(abs(l['mean']),abs(h['mean']))>=.02})
        all_effects.extend(holm(effect_rows));all_interactions.extend(holm(interaction_rows))
    write(out/'specialization_results.json',{'thresholds':thresholds,'effects':all_effects,'context_by_expert_interactions':all_interactions,'criterion':'Holm-significant meaningful rank reversal must replicate on independent split; gradient/activation differences are not evidence','official_seed1001_used':False})
    return {'tested_effects':len(all_effects),'holm_significant_effects':sum(x['holm_p']<.05 for x in all_effects),'tested_interactions':len(all_interactions),'holm_significant_interactions':sum(x['holm_p']<.05 for x in all_interactions)}

def save_checkpoint(out,name,p,opt,scheduler,update,plan):
 target=out/'models'/f'{name}_latest.pt';temp=target.with_suffix('.partial')
 torch.save({'method':name,'candidate_revision':'v2_residual_context_readout','update':update,'state_dict':{k:v.detach().cpu() for k,v in p.state_dict().items()},'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'plan':plan},temp)
 temp.replace(target)

def reports(out,plan,state,audit):
 r=ROOT/'reports';r.mkdir(exist_ok=True)
 prior=audit['recovered_original_results'];new=audit['new_paired_evaluations']
 diagnosis=f'''# MoE-sBOED-v1 and prior v2 diagnosis

Official historical v1 EIG 10.98906590 +/- 0.04479411 nats and vsOED 12.48084689 are reference figures from a different complete-training/evaluation protocol. They are not the native-v1 short-pilot scores below and are not used for candidate selection.

Recovered prior v2 pilot completed 3216 matched updates. Its four-arm aggregate training cost was 7197.076 seconds, not 7200 seconds per arm. Source revision 4b2bdfc; original checkpoint and source manifests verified. Held-out v2 EIG {prior[0]['v2']['mean']:.8f} +/- {prior[0]['v2']['MC_SE']:.8f}, native-v1 {prior[0]['v1']['mean']:.8f} +/- {prior[0]['v1']['MC_SE']:.8f}; paired difference {prior[0]['v2_minus_v1']['mean']:.8f} +/- {prior[0]['v2_minus_v1']['MC_SE']:.8f}. Independent historical confirmation difference {prior[1]['v2_minus_v1']['mean']:.8f} +/- {prior[1]['v2_minus_v1']['MC_SE']:.8f}. No v1 improvement was established; do not restart or relabel that completed pilot.

New frozen checkpoint audit at N256/L65536 on seeds97402732/33 found paired v2-v1 differences {new[0]['v2_minus_v1']['mean']:.8f} +/- {new[0]['v2_minus_v1']['MC_SE']:.8f} and {new[1]['v2_minus_v1']['mean']:.8f} +/- {new[1]['v2_minus_v1']['MC_SE']:.8f}. Masking and uniform redistribution hurt the frozen policy, demonstrating component dependence. None of the96 expert/stage/observable-context substitution effects passed within-split Holm. That is limited-power evidence, not proof that all experts are useless. There is no established conditional specialist competence.

Mechanistic limitation: existing v2 readouts decode E_e(z) whose down projection and separate readout both start small. The new single correction exposes z directly to each independent design readout via D_e(z+E_e(z)). This avoids making all routed design credit depend on the product of two small initial feature/readout factors. The unchanged terminal-information objective must decide whether this helps. Different gradients/weights alone will not count as success.

Recovered audit: {plan['audit_from']}. No official seed1001 or reserved confirmation seed8831001 was used.
'''
 (r/'MOE_SBOED_V1_DIAGNOSIS.md').write_text(diagnosis)
 (r/'MOE_SBOED_V2_METHOD.md').write_text('''# MoE-sBOED-v2 residual-context-readout candidate

One unified policy has the original complete-history sum encoder, one shared expert and seven independently parameterized routed experts, with learned top-three sparse selection. Shared output H(z+E_shared(z)) is unchanged. The only architectural correction relative to the previous direct-readout v2 is routed output sum_selected p_e D_e(z+E_e(z)), replacing sum_selected p_e D_e(E_e(z)). No new parameters, scale factor, teacher, role labels, online EIG search, auxiliary competence objective, or independent DAD policies are introduced. The common sigmoid feasible map stays [-4,4]^2.

Candidate total8658, activated6994 parameters/context; native-v1 total8434, activated6898. The separately trained dense width71 control has exactly8658 parameters; shared-only has5634. All arms share fresh native common-module initialization, data draws, Adam5e-5 betas(.8,.998), native.98 schedule every1000 after1, clipping5, B2000/L2000 and full30-stage terminal sPCE. Original native balance coefficient.001 is retained. No checkpoint or DAD pretraining is reused.

Autograd propagates through future simulated observations and complete adaptive histories. Top-k membership is discrete; derivatives are local pathwise derivatives within a fixed top-k branch, omitting membership-boundary terms, as in native-v1. Finite differences test locally stable full-horizon branches. Production inference is a single sparse history-to-design forward pass without online design optimization.

This is a disclosed development revision of v2, not a successful method version or a v3 rename. It remains in TEMP until correctness and scientific development gates support promotion. LabPC source is untouched.
''')
 (r/'MOE_SBOED_V2_PILOT.md').write_text('# MoE-sBOED-v2 development pilot\n\n'+json.dumps({'protocol':plan,'current_state':state},indent=2)+'\n\nActual progress/status, charged per-arm cost and checkpoint hashes are recorded in the referenced scratch campaign. Pending EIG cells are unavailable, never zero. Selection is the latest common completed update, never the largest test EIG.\n')
 (r/'MOE_SBOED_V2_SPECIALIZATION.md').write_text('''# MoE-sBOED-v2 specialist usefulness

Existing pilot: masking and uniform-routing interventions show frozen routed/router dependence, but no native-v1 improvement and no Holm-significant conditional weakest-slot substitution effects in the independent audit. This does not establish useful expert specialization.

The new candidate will be assessed against matched native-v1, trained shared-only, and parameter-matched dense policies; routed-masked and gate-mass-matched uniform interventions remain separate frozen-policy estimands. Observable-context weakest-slot interventions use fixed fitting thresholds, paired truth/noise/contrast draws, Holm correction, and an independent replication split. Stage1 is a negative-control context. Weight differences, activation counts and gradient norms are descriptive only. No improvement or specialist competence is claimed before numerical evidence exists. Independent training-seed uncertainty is not measured by a single paired seed.
''')

def train(out,audit_from):
 audit=json.loads((Path(audit_from)/'audit_results.json').read_text());assert len(audit['new_paired_evaluations'])==2
 plan={'candidate':'MoE-sBOED-v2 residual-context design readout','source_base':'4b2bdfceca488f3b42be9d84c015dc8f9cf87a65','audit_from':str(audit_from),'training_seed':202,'arms':ORDER,'maximum_matched_updates':8192,'maximum_charged_training_seconds_per_arm':7200,'save_reserve_seconds':15,'batch_size':2000,'training_contrasts':2000,'horizon':30,'optimizer':{'name':'Adam','lr':5e-5,'betas':[.8,.998],'decay':.98,'decay_after_updates':[1,1001,2001,3001,4001,5001,6001,7001,8001],'clip':5},'balance_factor':.001,'curve_updates':[128,512,2048,4096,8192],'curve_seed':97402751,'curve_N':128,'curve_L':65536,'final_development_seed':97402752,'final_replication_seed':97402753,'final_N':512,'final_L':262144,'context_fit_seed':97402754,'substitution_N':256,'substitution_L':65536,'maximum_diagnostic_seconds':1800,'checkpoint_selection':'latest common completed update, never best EIG','harmful_stop':'At or beyond4096matched updates, stop only if candidate-v1 paired upper95CI<-.1','official_seed1001_used':False,'LabPC_source_modified':False,'no_full_official_training':True,'campaign':str(out)}
 for d in ('models','evaluations','logs','metadata'):(out/d).mkdir(exist_ok=True)
 write(out/'plan.json',plan);state={'state':'CORRECTNESS_GATE','pid':os.getpid(),'matched_updates':0};reports(out,plan,state,audit)
 env=dict(os.environ,OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
 gate_start=time.perf_counter()
 with (out/'logs/correctness.log').open('x') as log:
  gate=subprocess.run([str(Path.home()/'miniconda3/envs/mocu_optimized/bin/python'),'-m','pytest','-q','tools/tests/test_moe_v2_context.py','tools/tests/test_moe_v2_decision.py','tools/tests/test_deepseek_sboed.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert gate.returncode==0,'Candidate correctness gate failed; see logs/correctness.log'
 diagnostic=time.perf_counter()-gate_start
 # No expensive trainable initialization occurs before the charged interval.
 init=time.perf_counter();models=context_matched_models(202);models={k:p.to('cuda:0').train() for k,p in models.items()};opts={k:torch.optim.Adam(p.parameters(),lr=5e-5,betas=(.8,.998)) for k,p in models.items()};schedulers={k:torch.optim.lr_scheduler.ExponentialLR(opts[k],.98) for k in ORDER};raw=yaml.safe_load((ROOT/'configs/deepseek_sboed_location.yaml').read_text())['benchmark'];benches={k:make_benchmark(raw) for k in ORDER};torch.cuda.synchronize();initialization=time.perf_counter()-init
 charged={k:initialization/4 for k in ORDER};bound={k:.5 for k in ORDER};records=[];curves=[];completed=0;stop='NATIVE_PREDECLARED_PILOT_SCHEDULE_COMPLETE'
 write(out/'initialization_cost.json',{'joint_initialization_seconds':initialization,'allocation':'Equal quarter charge to each arm; native shared initialization generated once and copied only within this fresh paired initialization','charged_by_arm':charged})
 write(out/'status.json',{'state':'TRAINING','active_method':'v1','pid':os.getpid(),'matched_updates':0,'charged_training_seconds':charged})
 for update in range(1,8193):
  if any(charged[k]+1.5*bound[k]+15>=7200 for k in ORDER):stop='COMPUTE_LIMITED_DEVELOPMENT_PILOT';break
  order=ORDER[(update-1)%4:]+ORDER[:(update-1)%4]
  round_records=[]
  for name in order:
   tick=time.perf_counter();p=models[name];p.train();opts[name].zero_grad(set_to_none=True)
   with torch.device('cuda:0'):
    r=rollout(benches[name],p,30,2000,202+update*104729,2000);info=r['info'][:,-1].mean();loss=-info+p.balance_loss()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-horizon objective '+name)
    loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(p.parameters(),5,error_if_nonfinite=True));opts[name].step()
    if (update-1)%1000==0:schedulers[name].step()
    torch.cuda.synchronize()
   elapsed=time.perf_counter()-tick;charged[name]+=elapsed;bound[name]=max(bound[name],elapsed)
   if update==1 or update%100==0:
    def gnorm(module):
     values=[q.grad.detach().norm() for q in module.parameters() if q.grad is not None]
     return float(torch.stack(values).norm()) if values else 0.
    row={'arm':name,'update':update,'training_sPCE':float(info.detach()),'charged_training_seconds':charged[name],'native_update_seconds':elapsed,'preclip_gradient_norm':norm,'router_gradient_norm':None if p.router is None else gnorm(p.router),'expert_gradient_norms':[gnorm(e) for e in p.routed],'learning_rate':opts[name].param_groups[0]['lr'],'routing':p.routing_statistics()};round_records.append(row)
   del r,info,loss
  completed=update
  if update==1 or update%100==0 or update in plan['curve_updates']:
   for name,p in models.items():
    tick=time.perf_counter();save_checkpoint(out,name,p,opts[name],schedulers[name],update,plan);torch.cuda.synchronize();charged[name]+=time.perf_counter()-tick
   assert max(charged.values())<=7200,'Charged budget exceeded'
   records.extend(round_records);write(out/'training_progress.json',{'matched_updates':completed,'charged_training_seconds':charged,'initialization_seconds':initialization,'records':records,'new_trajectories_per_arm':completed*2000,'contrast_paths_per_arm':completed*2000*2000,'full_horizon':30})
   state={'state':'TRAINING','active_method':order[-1],'pid':os.getpid(),'matched_updates':completed,'charged_training_seconds':charged,'diagnostic_GPU_seconds':diagnostic,'checkpoint_selection':plan['checkpoint_selection']};write(out/'status.json',state)
   print('MATCHED_UPDATE',update,'CHARGED',charged,flush=True)
  if update in plan['curve_updates']:
   tick=time.perf_counter();v=validator(97402751,128,65536);values={}
   for name,p in models.items():
    values[name],meta=v(p.eval());np.save(out/f'evaluations/curve_{name}_{update}.npy',values[name]);curves.append({'method':name,'update':update,**summary(values[name]),'metadata':meta})
   diagnostic+=time.perf_counter()-tick;write(out/'development_curves.json',curves)
   if update>=4096 and summary(values['v2']-values['v1'])['CI95'][1]<-.1:stop='PREDECLARED_HARMFUL_REGRESSION_STOP';break
 for name,p in models.items():
  tick=time.perf_counter();save_checkpoint(out,name,p,opts[name],schedulers[name],completed,plan);torch.cuda.synchronize();charged[name]+=time.perf_counter()-tick
 assert max(charged.values())<=7200
 freeze={name:sha(out/f'models/{name}_latest.pt') for name in ORDER};write(out/'metadata/final_checkpoint_freeze.json',{'matched_updates':completed,'sha256':freeze,'selection':plan['checkpoint_selection'],'charged_training_seconds':charged})
 final=[];tick=time.perf_counter()
 for split,seed in [('development',97402752),('replication',97402753)]:
  v=validator(seed,512,262144);values={}
  for name,p in models.items():values[name],_=v(p.eval());np.save(out/f'evaluations/{split}_{name}.npy',values[name])
  for key in ('masked','uniform'):
   p=copy.deepcopy(models['v2']).eval();p.ablate_routed=key=='masked';p.routing_mode='uniform' if key=='uniform' else 'learned';values[key],_=v(p);np.save(out/f'evaluations/{split}_{key}.npy',values[key])
  final.append({'split':split,'seed':seed,'EIG':{k:summary(x) for k,x in values.items()},'v2_minus_v1':summary(values['v2']-values['v1']),'v2_minus_dense':summary(values['v2']-values['dense']),'v2_minus_shared':summary(values['v2']-values['shared']),'full_minus_mask':summary(values['v2']-values['masked']),'learned_minus_uniform':summary(values['v2']-values['uniform'])})
 diagnostic+=time.perf_counter()-tick
 tick=time.perf_counter();specialization=competence(models['v2'].eval(),out);diagnostic+=time.perf_counter()-tick
 # Deployment latency: forward-only, no simulator or optimizer calls.
 from tools.tests.test_deepseek_sboed import history
 latency={}
 with torch.inference_mode():
  f=history(20,batch=256).cuda()
  for name,p in models.items():
   for _ in range(10):p(f,20)
   torch.cuda.synchronize();tick=time.perf_counter()
   for _ in range(100):p(f,20)
   torch.cuda.synchronize();latency[name]={'batch_size':256,'seconds_per_batch':(time.perf_counter()-tick)/100}
 write(out/'inference_latency.json',latency)
 write(out/'final_development.json',{'results':final,'matched_updates':completed,'charged_training_seconds':charged,'diagnostic_GPU_seconds':diagnostic,'stop_reason':stop,'checkpoint_sha256':freeze,'official_benchmark':False,'automatic_promotion':False,'expert_competence':specialization})
 state={'state':'DEVELOPMENT_COMPLETE','pid':os.getpid(),'matched_updates':completed,'charged_training_seconds':charged,'stop_reason':stop,'diagnostic_GPU_seconds':diagnostic,'results':final};write(out/'status.json',state);reports(out,plan,state,audit)
 with (ROOT/'reports/MOE_SBOED_V2_SPECIALIZATION.md').open('a') as f:f.write('\n\nCompleted candidate audit:\n\n'+json.dumps(specialization,indent=2)+'\n\nSee '+str(out/'specialization_results.json')+' for effects, contexts, multiplicity adjustment, and independent replication.\n')
 return state
