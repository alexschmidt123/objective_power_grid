"""One predeclared, bounded router-independent discovery pilot. Never official data."""
import argparse,copy,hashlib,json,math,subprocess,time,traceback
from pathlib import Path
from contextlib import nullcontext
import numpy as np
import torch
from src.research.deepseek_sboed import DeepSeekPolicy
from src.research.deepseek_sboed_experiment import write,norm
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout
from src.research.journal_validation import CommonValidation
from src.research.fh_moe_v3_context import *

BASE='13681d7091189a4e68056e6a2864dc58f9d5700b'
RAW=dict(system=dict(name='location_finding'),observation=dict(noise_sigma=.5),location_finding=dict(design_bounds=[-4.,4.],random_design_distribution='truncated_standard_normal'))
STAGES=[1,10,20,29]
PLAN=dict(method='FH-MoE-sBOED-v3',foundation=BASE,training_seed=98102711,
    training_updates_per_arm=600,maximum_training_pair_seconds=1500,batch_size=2000,contrasts=2000,horizon=30,
    learning_rate=5e-5,adam_betas=[.8,.998],lr_decay=.98,lr_schedule='after update1 and every1000 thereafter, native v1',gradient_clip=5.,precision='float64',
    dataset_seeds=[98102721,98102722,98102723],dataset_batches_per_split=4,dataset_batch=2000,
    stages=STAGES,development_seed=98102731,development_N=512,development_L=262144,
    context_components=2,ridge=.01,temperature=.25,required_passed_stages=2,
    gate=dict(prediction='heldout MSE difference vs constant upper95<0',minimum_context_fraction=.1,minimum_split_cosine=.8,minimum_coherence=.1,minimum_mean_SNR=3,maximum_between_cosine=.8,minimum_assignment_kappa=.6,minimum_float32_assignment_agreement=.99),
    context_labels='No routed expert indices or exposure are used anywhere in discovery',
    pilot_sequence='shared-only reference/native paired prep, discovery, specialize ONLY if discovery passes',
    matched_native='same initialized full model, draws, batch, contrasts, completed preparation updates; reference is NOT claimed to be v3',
    future_specialization_gate='correctness; contexts at>=2stages; positive full-minus-mask and learned-minus-uniform estimates; matched native EIG difference lower95>-.1',
    primary_LabPC_seconds=14400,Ubuntu_replication='exact final primary update count, no independent cutoff',
    official_seed1001_used=False,confirmation_seed_reserved=98102791)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    for n in ['deepseek_sboed','deepseek_sboed_experiment','engine','dad_location_policy','benchmarks','journal_validation']:
        p=Path('src/research')/(n+'.py')
        assert p.read_bytes()==subprocess.check_output(['git','show',BASE+':'+str(p)])

def extract(model,seed,batch=2000):
    b=make_benchmark(RAW)
    with torch.device('cuda'),capture_contexts(model,STAGES) as states:
        r=rollout(b,model,30,batch,seed,2000)
        loss=-r['info'][:,-1].mean()
        gradients=torch.autograd.grad(loss,[states[t]['u'] for t in STAGES])
        # Features only depend on recorded observed actions and observations.
        with torch.no_grad():xs=[history_covariates(model,r['features'][t],t).detach().cpu() for t in STAGES]
    result={t:(x,normalized(g.detach().double()*batch).cpu(),(g.detach().double()*batch).cpu()) for t,x,g in zip(STAGES,xs,gradients)}
    return result,b.calls

class MixedPrecisionReference(torch.nn.Module):
    kind='deepseek_moe_sboed';outputs_design=False
    def __init__(self,m):super().__init__();self.policy=copy.deepcopy(m).float()
    def forward(self,f,t):return self.policy(f.float(),t).double()

def precision_probe(ref):
    start=time.perf_counter();data,calls=extract(ref,98102741,128)
    low=MixedPrecisionReference(ref);b=make_benchmark(RAW)
    with torch.device('cuda'),capture_contexts(low.policy,STAGES) as states:
        r=rollout(b,low,30,128,98102741,2000);loss=-r['info'][:,-1].mean()
        gradients=torch.autograd.grad(loss,[states[t]['u'] for t in STAGES])
    report={}
    for t,g in zip(STAGES,gradients):
        a=data[t][2];c=g.detach().cpu().double()*128
        cos=(normalized(a)*normalized(c)).sum(-1)
        report[t]=dict(mean_direction_cosine=float(cos.mean()),p05_direction_cosine=float(cos.quantile(.05)),relative_gradient_error=float((a-c).norm()/a.norm().clamp_min(1e-30)))
    return dict(stages=report,seconds=time.perf_counter()-start,forward_parameter_stages=calls+b.calls,
        scope='float32 reference network including history graph; simulator/likelihood remain canonical float64',
        PASS=all(v['mean_direction_cosine']>=.99 and v['relative_gradient_error']<.05 for v in report.values()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fh-moe-v3-discovery',action='store_true');parser.add_argument('--campaign',required=True);args=parser.parse_args();C=Path(args.campaign)
    C.mkdir(parents=True,exist_ok=True)
    assert not (C/'plan.json').exists(),'Never overwrite a pilot'
    write(C/'plan.json',PLAN);guard()
    sources=['run.sh','src/research/fh_moe_v3_context.py','src/research/fh_moe_v3_discovery.py','tools/tests/test_fh_moe_v3.py']
    hashes={p:sha(p) for p in sources};write(C/'source_freeze.json',hashes)
    from tools.journal_campaign import exclusive_device_lock
    lock=exclusive_device_lock(wait=False);started=time.perf_counter();torch.set_num_threads(2)
    ledger=dict(hardware=torch.cuda.get_device_name(0),training_updates_per_arm=0,training_forward_parameter_stages=[0,0],discovery_forward_parameter_stages=0,training_pair_seconds=0.,discovery_seconds=0.,evaluation_seconds=0.,inherited_weights=False,main_training_seconds=0.)
    try:
        torch.manual_seed(PLAN['training_seed']);ref=DeepSeekPolicy().cuda().train();native=copy.deepcopy(ref);ref.ablate_routed=True
        models=[ref,native];bench=[make_benchmark(RAW),make_benchmark(RAW)]
        opts=[torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.8,.998)) for m in models]
        sched=[torch.optim.lr_scheduler.ExponentialLR(o,.98) for o in opts]
        records=[];trainstart=time.perf_counter()
        for u in range(1,601):
            if time.perf_counter()-trainstart>PLAN['maximum_training_pair_seconds']:break
            row={'update':u}
            for k,(m,opt,b) in enumerate(zip(models,opts,bench)):
                opt.zero_grad(set_to_none=True)
                with torch.device('cuda'):
                    r=rollout(b,m,30,2000,98102800+u*104729,2000)
                    info=r['info'][:,-1].mean();loss=-info+m.balance_loss();loss.backward()
                    gradient=norm(m.parameters());assert math.isfinite(gradient)
                    torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
                    if (u-1)%1000==0:sched[k].step()
                    assert torch.stack(r['actions']).abs().max()<=4
                row[['shared_reference','native_control'][k]]=dict(sPCE=float(info.detach()),preclip_norm=gradient)
                del r,loss,info
            torch.cuda.synchronize();ledger['training_updates_per_arm']=u
            if u==1 or u%50==0:
                records.append(row);write(C/'training_progress.json',dict(records=records,seconds=time.perf_counter()-trainstart));print('PREPARATION',row,flush=True)
        ledger['training_pair_seconds']=time.perf_counter()-trainstart;ledger['training_forward_parameter_stages']=[b.calls for b in bench]
        for name,m in zip(['shared_reference','native_control'],models):
            m.eval();torch.save(dict(state_dict=m.state_dict(),ablate_routed=m.ablate_routed,update=ledger['training_updates_per_arm'],seed=PLAN['training_seed'],source_hashes=hashes),C/(name+'.pt'))
        del opts,sched;torch.cuda.empty_cache()
        start=time.perf_counter();splits=[]
        for split,seed in enumerate(PLAN['dataset_seeds']):
            blocks=[]
            for j in range(4):
                d,calls=extract(ref,seed+j*100003);blocks.append(d);ledger['discovery_forward_parameter_stages']+=calls
            combined={t:tuple(torch.cat([d[t][i] for d in blocks]) for i in range(3)) for t in STAGES}
            torch.save(combined,C/f'gradient_dataset_split{split}.pt');splits.append(combined)
            print('DISCOVERY_SPLIT',split,flush=True)
        results={};banks={}
        for t in STAGES:
            result,bank=discovery_metrics(*[(s[t][0],s[t][1]) for s in splits]);results[t]=result
            if result['PASS']:banks[t]=bank
        precision=precision_probe(ref);ledger['discovery_forward_parameter_stages']+=precision['forward_parameter_stages'];ledger['discovery_seconds']=time.perf_counter()-start
        passed=len(banks)>=2 and precision['PASS'] and ledger['training_updates_per_arm']==600
        torch.save(banks,C/'validated_contexts.pt')
        write(C/'context_gate.json',dict(PASS=passed,stages=results,precision=precision,passed_stages=list(banks),no_router_labels=True))
        # Reference-vs-native is descriptive; NOT a trained v3 specialization result.
        start=time.perf_counter();v=CommonValidation(RAW,30,98102731,512,262144,batch=16,chunk=16384,device='cuda:0')
        scores={};metas={}
        for name,m in zip(['shared_reference','native_control'],models):
            scores[name],metas[name]=v(m);np.save(C/(name+'_development_scores.npy'),scores[name])
        delta=torch.as_tensor(scores['shared_reference']-scores['native_control'])
        write(C/'development.json',dict(results={n:dict(EIG=float(s.mean()),MC_SE=float(s.std(ddof=1)/len(s)**.5),metadata=metas[n]) for n,s in scores.items()},shared_reference_minus_native=confidence(delta),v3_specialization_result=False))
        ledger['evaluation_seconds']=time.perf_counter()-start
        ledger.update(total_task_wall_seconds=time.perf_counter()-started,training_trajectories_per_arm=ledger['training_updates_per_arm']*2000,
            training_likelihood_evaluations_per_arm=ledger['training_updates_per_arm']*2000*2001*30,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),GPU_kernel_active_seconds=None,CPU_process_seconds=time.process_time())
        assert hashes=={p:sha(p) for p in sources},'Running source changed'
        write(C/'ledger.json',ledger);write(C/'status.json',dict(status='CONTEXT_GATE_PASSED_SPECIALIZATION_PENDING' if passed else 'FAILED_CONTEXT_DISCOVERY_NO_MAIN_TRAINING',context_PASS=passed))
        print('CONTEXT_GATE',passed,json.dumps(results),flush=True)
    except Exception as e:
        write(C/'failure.json',dict(error=str(e),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-started,ledger=ledger));raise
    finally:lock.close()
if __name__=='__main__':
    raise SystemExit('LEGACY_EXCLUDED_FROM_CURRENT_PROJECT: archived exploratory entry point')
if False:main()
