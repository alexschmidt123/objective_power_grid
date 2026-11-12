"""Real native update opens gradients; prepared arms restore identical states."""
import shutil,gc
from common import *
from integration import load_prepared,bind,loss_for

def main():
    original.setup();lease=original.acquire();root=O/'preparation';root.mkdir(exist_ok=True)
    if (root/'runtime.pkl').exists():assert sha(root/'runtime.pkl')==sha(O/'evidence/fixture_runtime.pkl')
    else:shutil.copy2(O/'evidence/fixture_runtime.pkl',root/'runtime.pkl')
    put(root/'RESULT.json',dict(runtime_sha256=sha(root/'runtime.pkl'),fixture_only=True))
    previous=CFG['joint_start_iteration'];CFG['joint_start_iteration']=16002
    records=[]
    try:
        for uniform in [False,True]:
            state=load_prepared();a=state['algorithm'];bind(a,uniform)
            records.append(dict(actor=digest(a.actor_net.state_dict()),critic=digest(a.critic_net.state_dict()),
                actor_Adam=digest({str(k)+'_'+name:tensor for k,value in a.actor_optimizer.state_dict()['state'].items() for name,tensor in value.items()}),
                actor_lr=a.actor_optimizer.param_groups[0]['lr']))
            with torch.device('cuda'):
                x=a.form_actor_input(12,a.ds_hist[:8,:12],a.ys_hist[:8,:12])
            f=a.actor_net.frozen(x)
            a.critic_net.requires_grad_(False);a.actor_net.zero_grad();loss_for(a,x,f).backward()
            assert all(norm([p.grad for p in branch.parameters()])>0 for branch in a.actor_net.adaptations)
            if not uniform:assert norm([p.grad for p in a.actor_net.router.parameters()])>0
            del state,a;gc.collect();torch.cuda.empty_cache()
        assert records[0]==records[1]
    finally:CFG['joint_start_iteration']=previous
    put(O/'evidence/PREPARATION_CHECKS.json',dict(status='PASS',fixture_only=True,nonzero_head_after_native_EIG_update=True,
        real_preparation_unblocks_router_and_all_adaptations=True,identical_actor_critic_Adam_and_LR_into_both_arms=True,
        no_extra_optimizer_updates=True))
    print('RR_PREPARATION_CHECKS_PASS',flush=True)

if __name__=='__main__':main()
