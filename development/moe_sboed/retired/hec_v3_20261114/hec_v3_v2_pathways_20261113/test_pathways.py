"""Actual V2 numerical parity and standalone/native MoE tests."""
import argparse,copy,json,pathlib,time
import torch
from common import ROOT,n,audit
from pathway_policy import PathwayMoE,prefix
from native_adapter import build_native_class
from engine import Logger,all_optimizers

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=pathlib.Path,required=True);args=p.parse_args()
    out=args.output.expanduser();assert not out.exists();out.mkdir(parents=True)
    n.original.setup();lease=n.original.acquire()
    try:
        state=n.load_parent();a=state['algorithm'];parent=a.actor_net.eval()
        import numpy as np
        data=np.load(audit.BASE/'evaluation/diagnostic_candidate_histories.npz')
        x=torch.tensor(data['X'][:2,[0,14,29]].copy(),device='cuda').reshape(-1,117)
        model=PathwayMoE(parent).cuda();result={}
        with torch.no_grad():
            _,reference=parent(x,collect=True);c=model.context(x);proposals,g,legacy,responses=model.components(x)
            torch.testing.assert_close(g,reference['gates'],atol=1e-6,rtol=1e-5)
            for k in range(4):
                first,response,z=model.experts[k].packet(c)
                torch.testing.assert_close(first,reference['slots'][:,k,:64],atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(response,reference['responses'][:,k],atol=1e-6,rtol=1e-5)
                torch.testing.assert_close(z,reference['legacy_features'][:,k],atol=1e-6,rtol=1e-5)
                w=x.new_zeros(4);w[k]=1;old=parent(x,routing_override=w)
                torch.testing.assert_close(proposals[:,k],old,atol=2e-6,rtol=2e-5)
                single=model.standalone(k).eval()
                torch.testing.assert_close(single(x),proposals[:,k],atol=1e-6,rtol=1e-5)
                result[f'E{k+1}']=dict(packet_parity=True,onehot_design_parity=True,standalone=True,
                    has_only_own_query_modules=not any(hasattr(single.expert,key) for key in ['region','discriminate','sensitivity','sequential'] if key!=['region','discriminate','sensitivity','sequential'][k]))
            u=x.new_tensor([.01,.26,.51,.76,.02,.99]);action,info=model(x,mode='uniform',uniforms=u,collect=True)
            assert info['selected'].tolist()==[0,1,2,3,0,3]
            assert torch.equal(action,proposals[torch.arange(len(x),device='cuda'),info['selected']])
            poison=x.clone();stage=x[:,:30].argmax(-1);mask=torch.arange(29,device='cuda')[None]>=stage[:,None]
            pd=poison[:,30:88].reshape(-1,29,2);pd[mask]=99;poison[:,88:][mask]=99
            assert torch.equal(prefix(poison),prefix(x))
        loss=model.native_actor_loss(a.critic_net,x);loss.backward()
        assert all(p.grad is None for p in model.experts.parameters())
        router_grads={key:float(p.grad.norm()) for key,p in model.named_parameters() if p.requires_grad and p.grad is not None}
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert sum(router_grads.values())>0
        # Complete expert training must have a valid action-gradient path.
        single=model.standalone(3);single.expert.requires_grad_(True)
        (-a.critic_net(torch.cat([x,single(x)],-1)).mean()).backward()
        assert single.expert.head[0].weight.grad is not None
        assert single.expert.adapter4[0].weight.grad is not None
        cls,patch=build_native_class(type(a));a.__class__=cls;a.actor_net=model;a.actor_target_net=copy.deepcopy(model)
        lr=a.actor_optimizer.param_groups[0]['lr'];model.zero_grad(set_to_none=True)
        a.actor_optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=lr)
        a.actor_lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(a.actor_optimizer,gamma=.9999)
        logger=Logger(a,'validation',out,dict(updates=1,wall_seconds=300),time.monotonic());a.dowel=logger;a.post_approx.dowel=logger;a.save_path=None
        for optimizer,key in all_optimizers(a):
            def count(*unused,key=key):logger.counts[key]+=1
            optimizer.register_step_post_hook(count)
        before=n.digest({k:v for k,v in model.state_dict().items() if k.startswith(('context.','experts.'))})
        profile=json.loads((n.P/'frozen_protocol.json').read_text())['profile'];profile['discount']=1.
        n.original.train_native(a,logger,profile,target=16002)
        assert logger.counts==dict(actor=1,critic=5,posterior=150)
        assert n.digest({k:v for k,v in model.state_dict().items() if k.startswith(('context.','experts.'))})==before
        for op,_ in all_optimizers(a):op._optimizer_step_pre_hooks.clear();op._optimizer_step_post_hooks.clear()
        with (out/'VALIDATION_RUNTIME.pkl').open('wb') as f:n.cloudpickle.dump(dict(algorithm=a,rng=n.original.rng()),f,protocol=5)
        final=dict(status='PASS',experts=result,router_weights_initial_V2_parity=True,no_soft_action_average=True,
                   no_future_history_leak=True,router_gradients=router_grads,native_counts=logger.counts,
                   frozen_context_and_experts_preserved=True,native_patch=patch,
                   total_parameters=sum(p.numel() for p in model.parameters()),
                   router_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        n.put(out/'VALIDATION.json',final);print(json.dumps(final),flush=True)
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
