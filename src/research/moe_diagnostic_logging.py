"""RNG-neutral observation of unchanged development training; no optimizer edits."""
import io,json,math,time,os,hashlib
from pathlib import Path
import numpy as np
import torch


def quantiles(x):
    a=np.asarray(x,dtype=float)
    return np.quantile(a,[0,.01,.1,.5,.9,.99,1]).tolist()


class DevelopmentTrace:
    def __init__(self,policy,benchmark,horizon,settings,directory,observer):
        self.p=policy;self.b=benchmark;self.h=horizon;self.s=settings
        self.out=Path(directory)/'late_diagnostics';self.out.mkdir(exist_ok=False)
        self.observer=observer;self.seconds=0.;self.calls=[];self.handle=None
        self.start=int(settings.get('diagnostic_start_episodes',750000))
        self.targets=sorted(set(range(750000,1000001,12500))|{832000,883200})
        if self.start==0:self.targets=[64,128,192,256]
        if 'diagnostic_checkpoint_episodes' in settings:self.targets=sorted(set(map(int,settings['diagnostic_checkpoint_episodes'])))
        self.next=0;self.saved=[]

    def begin(self,count,n):
        self.calls=[];self.active=count+n>=self.start
        if not self.active:return
        def capture(module,inputs,output):
            if output.requires_grad:output.retain_grad()
            self.calls.append((inputs[0].detach().clone(),output))
        self.handle=self.p.router.bits.register_forward_hook(capture)

    def end_rollout(self):
        if self.handle is not None:self.handle.remove();self.handle=None

    def before_update(self,result):
        if not self.active:return
        self.weights={name:v.detach().clone() for name,v in self.p.named_parameters()}
        self.sample_stages=[j for j in [0,9,19,28] if j<len(self.calls)]
        self.context=torch.cat([self.calls[j][0][:4] for j in self.sample_stages])
        self.features=torch.cat([result['features'][j+1][:4].detach() for j in self.sample_stages])
        with torch.no_grad():
            self.old_logits=self.p.router.bits(self.context).clone()
            self.old_proposals=torch.stack([e(self.context) for e in self.p.experts],1)

    def after_update(self,row,result,groups,optimizer,scheduler):
        if not self.active:return
        started=time.perf_counter()
        logits=np.stack([v.detach().cpu().numpy() for _,v in self.calls])
        gamma=self.p.router.gamma;saturated=np.abs(logits)>=gamma/2
        lg=np.stack([v.grad.detach().cpu().numpy() if v.grad is not None else np.zeros_like(v.detach().cpu().numpy()) for _,v in self.calls])
        delta={}
        for group,parameters in groups.items():
            ids={id(v) for v in parameters};norm=reference=0.
            for name,p in self.p.named_parameters():
                if id(p) in ids:
                    norm+=float((p.detach()-self.weights[name]).square().sum())
                    reference+=float(self.weights[name].square().sum())
            delta[group]=dict(update_l2=math.sqrt(norm),relative_update_l2=math.sqrt(norm)/max(math.sqrt(reference),1e-15))
        with torch.no_grad():
            direct=self.p.router.bits(self.context)
            context_new=self.p.context(self.features).mean(1)
            combined=self.p.router.bits(context_new)
            expert_only=torch.stack([e(self.context) for e in self.p.experts],1)
            all_new=torch.stack([e(context_new) for e in self.p.experts],1)
            feasible=lambda z:-4+8*z.sigmoid()
            old=feasible(self.old_proposals)
            expert_change=(feasible(expert_only)-old).norm(dim=-1)
            combined_change=(feasible(all_new)-old).norm(dim=-1)
        rec=dict(update=row['update'],primary_episodes=row['primary_episodes'],gamma=gamma,
            logit_quantiles=quantiles(logits),saturated_bit_fraction=float(saturated.mean()),
            fully_saturated_selector_fraction=float(saturated.all(-1).mean()),
            logit_quantiles_by_stage=[quantiles(x) for x in logits],
            saturated_bit_fraction_by_stage=saturated.mean((1,2)).tolist(),
            logit_gradient_abs_quantiles=quantiles(abs(lg)),
            component_gradient_before=row['gradient_before'],component_gradient_after=row['gradient_after'],
            actual_parameter_updates=delta,training_spce=row['training_eig'],
            training_expert_usage=row.get('hard_choice_frequency'),
            training_actions_mean=torch.stack(result['actions']).detach().mean((0,1)).cpu().tolist(),
            router_weight_only_bit_flip_fraction=float(((direct>=0)!=(self.old_logits>=0)).double().mean()),
            router_plus_encoder_bit_flip_fraction=float(((combined>=0)!=(self.old_logits>=0)).double().mean()),
            expert_weight_only_action_change=float(expert_change.mean()),
            expert_plus_encoder_action_change=float(combined_change.mean()),
            local_change_probe='first4 cases at routed stages2,11,21,30; fixed observed histories, not rollout EIG')
        if row['update']%100==0 or row['gradient_before']['router']>1e7:
            rec.update(raw_training_logits_first8=logits[:,:8].tolist(),
                physical_training_actions_first4=torch.stack(result['actions']).detach()[:,:4].cpu().tolist(),
                training_observations_first4=torch.stack(result['observations']).detach()[:,:4].cpu().tolist())
        with (self.out/'training_trace.jsonl').open('a') as f:f.write(json.dumps(rec,allow_nan=False)+'\n')
        self.seconds+=time.perf_counter()-started
        if self.next<len(self.targets) and row['primary_episodes']>=self.targets[self.next]:
            self.capture_checkpoint(row,result,optimizer,scheduler)
            while self.next<len(self.targets) and row['primary_episodes']>=self.targets[self.next]:self.next+=1

    def capture_checkpoint(self,row,result,optimizer,scheduler):
        started=time.perf_counter();episodes=row['primary_episodes']
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            checkpoint=self.p.checkpoint();checkpoint['diagnostic_episodes']=episodes
            checkpoint.update(diagnostic_optimizer_state=optimizer.state_dict(),diagnostic_scheduler_state=scheduler.state_dict(),
                diagnostic_update=row['update'],diagnostic_rng_cpu=cpu_rng,diagnostic_rng_cuda=cuda_rng)
            path=self.out/f'checkpoint_{episodes}.pt';temp=path.with_suffix('.tmp');torch.save(checkpoint,temp);os.replace(temp,path)
            record=dict(primary_episodes=episodes,gamma=self.p.router.gamma,checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if self.observer is not None:
                scores,cost=self.observer.validator(self.p)
                np.save(self.out/f'validation_scores_{episodes}.npy',scores)
                record.update(validation_eig=float(scores.mean()),mc_se=float(scores.std(ddof=1)/len(scores)**.5),validation=cost)
                with torch.device('cpu'),torch.no_grad():
                    buffer=io.BytesIO();torch.save(self.p,buffer);buffer.seek(0)
                    p=torch.load(buffer,map_location='cpu',weights_only=False).eval();p.set_routing('learned_switch',False)
                    from src.research.engine import rollout
                    from src.research.benchmarks import make_benchmark
                    validator=self.observer.validator;b=make_benchmark(validator.raw)
                    hard_routes=[];hard_logits=[]
                    def route_hook(module,inputs,output):
                        route=module.last_route
                        if route is None:return
                        w=route['weights'].detach()
                        data=dict(selected_experts=w.argmax(-1).tolist(),utilization=torch.bincount(w.argmax(-1),minlength=4).double().div(len(w)).tolist())
                        if 'strategy' in route:
                            data['strategy']={k:v.detach().tolist() for k,v in route['strategy'].items()}
                        hard_routes.append(data)
                    h1=p.register_forward_hook(route_hook)
                    h2=p.router.bits.register_forward_hook(lambda module,inputs,output:hard_logits.append(output.detach().tolist()))
                    try:rr=rollout(b,p,self.h,32,validator.seed,0,truth=validator.truth[:32],noise=validator.noise[:32])
                    finally:h1.remove();h2.remove()
                    record['probe_physical_actions']=torch.stack(rr['actions']).tolist()
                    record['probe_observations']=torch.stack(rr['observations']).tolist()
                    record['probe_hard_routing']=hard_routes;record['probe_hard_logits']=hard_logits
            self.saved.append(record)
            (self.out/'checkpoints.json').write_text(json.dumps(self.saved,indent=2,allow_nan=False)+'\n')
            print('LATE_DIAGNOSTIC_CHECKPOINT',episodes,record.get('validation_eig'),flush=True)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            elapsed=time.perf_counter()-started;self.seconds+=elapsed
            if self.observer is not None:self.observer.validation_seconds+=elapsed
