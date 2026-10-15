import tempfile
from pathlib import Path
import torch
from torch import nn
from src.research.matched_sum_single import MatchedSumSingle193
from src.research.sparse_moe import SparseMoESBOED,stabilize_gradients
from src.research.moe_architecture_controls import train_control,component_groups
from src.research.benchmarks import LocationFinding
from src.research.engine import rollout


def benchmark():
    return LocationFinding({'location_finding':{'design_bounds':[-4.,4.]},
        'observation':{'noise_sigma':.5,'sampling':'normal_log_signal'},
        'research':{'paper_protocol':{'corrected_methods':True}}})


def pair(hidden=64):
    torch.set_num_threads(1)
    b=benchmark()
    torch.manual_seed(101)
    moe=SparseMoESBOED(2,1,30,hidden,first_action=[0.,0.],history_encoder_revision=2,action_scale=b.action_scale,obs_scale=b.obs_scale)
    torch.manual_seed(101)
    single=MatchedSumSingle193(30,hidden,b.action_scale,b.obs_scale)
    return moe,single


def features():
    g=torch.Generator().manual_seed(771)
    f=torch.zeros(4,121,dtype=torch.float64);f[:,0]=.5
    slots=f[:,1:].reshape(4,30,4)
    slots[:,:15,:3]=torch.randn(4,15,3,generator=g,dtype=torch.float64);slots[:,:15,3]=1
    return f


def test_exact_context_and_scratch_initialization():
    moe,p=pair()
    torch.testing.assert_close(moe.first_design,p.first_design,rtol=0,atol=0)
    for a,b in zip(moe.sum_encoder.parameters(),p.sum_encoder.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    for a,b in zip(moe.experts[0].parameters(),p.head.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    f=features();torch.testing.assert_close(moe.context(f),p.context(f),rtol=0,atol=0)
    torch.testing.assert_close(p(f,15),moe.experts[0](moe.context(f)[:,0]),rtol=0,atol=0)
    torch.testing.assert_close(p(f,0),moe(f,0),rtol=0,atol=0)
    assert p.head[0].in_features==193 and p.head[0].out_features==64
    assert isinstance(p.head[1],nn.Tanh) and isinstance(p.head[3],nn.Tanh)
    assert p.head[2].out_features==64 and p.head[4].out_features==2
    assert not hasattr(p,'router') and not hasattr(p,'experts')
    assert sum(v.numel() for v in p.parameters())==21844


def test_permutation_mask_stage_and_checkpoint():
    _,p=pair();f=features();permuted=f.clone()
    slots=permuted[:,1:].reshape(4,30,4);slots[:,:15]=f[:,1:].reshape(4,30,4)[:,list(reversed(range(15)))]
    torch.testing.assert_close(p(f,15),p(permuted,15),rtol=1e-12,atol=1e-12)
    masked=f.clone();masked[:,1:].reshape(4,30,4)[:,15:,:3]=999
    torch.testing.assert_close(p(f,15),p(masked,15),rtol=0,atol=0)
    torch.testing.assert_close(p.context(f)[:,0,-1],f[:,0],rtol=0,atol=0)
    assert torch.count_nonzero(p.context(f)[:,:,16:-1])==0
    restored=MatchedSumSingle193(30,action_scale=p.action_scale,obs_scale=p.obs_scale);restored.load_state_dict(p.state_dict())
    torch.testing.assert_close(p(f,15),restored(f,15),rtol=0,atol=0)


def test_bounded_full_horizon_gradients():
    _,p=pair();r=rollout(benchmark(),p,30,4,41001,8)
    r['info'][:,-1].mean().backward()
    assert torch.isfinite(r['info']).all()
    assert all(bool(((a>=-4)&(a<=4)).all()) for a in r['actions'])
    groups=component_groups(p);assert set(groups)=={'encoder','head','first_design'}
    for group in groups.values():assert any(v.grad is not None and v.grad.abs().sum()>0 for v in group)


class ExpertZeroReference(SparseMoESBOED):
    def forward(self,f,stage):
        if stage==0:return self.first_design[None].expand(len(f),-1)
        return self.experts[0](self.context(f)[:,0])


def test_two_updates_match_existing_expert_zero_reference():
    # Independent reference verifies the unchanged Adam schedule, rollout RNG,
    # sPCE gradient and component clipping against the existing MoE expert path.
    torch.set_num_threads(1);b=benchmark();torch.manual_seed(101)
    ref=ExpertZeroReference(2,1,30,8,first_action=[0.,0.],history_encoder_revision=2,action_scale=b.action_scale,obs_scale=b.obs_scale)
    s=dict(hidden=8,updates=2,batch_size=4,contrasts=8,learning_rate=5e-5,
           spce_adam_beta1=.8,spce_adam_beta2=.998,spce_lr_decay=.98,
           spce_lr_schedule_every=1000,validate_every=1,moe_gradient_clip_mode='component')
    opt=torch.optim.Adam(ref.parameters(),lr=5e-5,betas=(.8,.998))
    scheduler=torch.optim.lr_scheduler.ExponentialLR(opt,.98)
    for step in range(2):
        r=rollout(b,ref,30,4,101+(step+1)*104729,8)
        opt.zero_grad();(-r['info'][:,-1].mean()).backward()
        stabilize_gradients(ref,component_groups(ref),'component',5.)
        opt.step()
        if step%1000==0:scheduler.step()
    with tempfile.TemporaryDirectory() as directory:
        trained,record=train_control(b,30,s,101,Path(directory),'sum_single_193',trajectory_budget=8)
        assert record['generated_training_trajectories']==8
        assert record['validation_history']==[]
        torch.testing.assert_close(trained.first_design,ref.first_design,rtol=0,atol=1e-12)
        for x,y in zip(trained.sum_encoder.parameters(),ref.sum_encoder.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=1e-12)
        for x,y in zip(trained.head.parameters(),ref.experts[0].parameters()):torch.testing.assert_close(x,y,rtol=0,atol=1e-12)
