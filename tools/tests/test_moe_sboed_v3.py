import copy,io,math,inspect,shutil
from pathlib import Path
import numpy as np
import pytest,torch,yaml
from src.research.moe_sboed_v3 import ProcessMoE,ConditionalMixture,process_views,dense_match
from src.research.benchmarks import make_benchmark
from src.research.engine import rollout,replay_likelihood

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
def raw():return yaml.safe_load((ROOT/'configs/deepseek_sboed_location.yaml').read_text())['benchmark']
def history(batch=4,stage=10):
    f=torch.randn(batch,121,dtype=torch.float64);f[:,0]=stage/30
    f[:,1:].reshape(batch,30,4)[:,stage:]=0
    return f

def test_conditional_density_matches_independent_mvn():
    torch.manual_seed(73);q=ConditionalMixture(4).double();d=q(torch.randn(7,16,dtype=torch.float64));w,mu,L=d;t=torch.randn(7,4,dtype=torch.float64)
    reference=torch.distributions.MultivariateNormal(mu,scale_tril=L).log_prob(t[:,None])
    torch.testing.assert_close(q.log_prob(d,t),torch.logsumexp(w.log()+reference,-1))

def test_total_covariance_and_component_permutation():
    torch.manual_seed(44);d=ConditionalMixture(4).double()(torch.randn(5,16,dtype=torch.float64))
    a=process_views(d);b=process_views(tuple(x[:,[2,0,3,1]] for x in d))
    for x,y in zip(a,b):torch.testing.assert_close(x,y)
    assert all(torch.isfinite(x).all() for x in a)
    assert not torch.allclose(a[0],a[1]) and not torch.allclose(a[2],a[3])

def test_sparse_exactly_two_and_gradient_paths():
    torch.manual_seed(19);m=ProcessMoE();f=history(32);calls=[0]*4
    hooks=[e.register_forward_hook(lambda a,b,c,i=i:calls.__setitem__(i,calls[i]+len(c))) for i,e in enumerate(m.experts)]
    v=m(f,10);v.square().sum().backward();ids=m.last_dispatch['indices']
    assert sum(calls)==64 and ids.shape==(32,2) and (ids[:,0]!=ids[:,1]).all()
    assert m.router.weight.grad.abs().sum()>0
    for e in ids.unique():assert sum(p.grad.abs().sum() for p in m.experts[e].parameters() if p.grad is not None)>0
    for h in hooks:h.remove()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

def test_router_local_gradient_finite_difference():
    torch.manual_seed(19);m=ProcessMoE().eval();f=history(3);v=m(f,10).square().sum();v.backward()
    p=m.router.weight;i=tuple(torch.unravel_index(p.grad.abs().argmax(),p.shape));g=float(p.grad[i]);old=float(p[i]);vals=[]
    with torch.no_grad():
        for x in [-1e-5,1e-5]:p[i]=old+x;vals.append(float(m(f,10).square().sum()))
        p[i]=old
    assert abs(g-(vals[1]-vals[0])/2e-5)<1e-7

def test_full_spce_and_all_history_paths():
    torch.manual_seed(72);m=ProcessMoE();b=make_benchmark(raw());r=rollout(b,m,30,3,97402861,16)
    bank=torch.cat((r['truth'][:,None],r['particles']),1)
    ll,_,_=replay_likelihood(b,bank,r['actions'],r['observations'])
    direct=ll[:,0]-torch.logsumexp(ll,-1)+math.log(17)
    torch.testing.assert_close(direct,r['info'][:,-1]);(-direct.mean()).backward()
    assert m.adapter.token[0].weight.grad.abs().sum()>0 and m.uncertainty.head.weight.grad.abs().sum()>0
    assert all(a.abs().max()<=4 for a in r['actions'])
    # A terminal decision depends differentiably on the first observed action
    # and observation, through all retained history slots.
    f=r['features'][-1].detach().requires_grad_();v=m(f,29).square().sum();g=torch.autograd.grad(v,f)[0]
    assert g[:,1:4].abs().sum()>0 and g[:,1:].reshape(3,30,4)[:,:29,:3].abs().sum()>0

def test_terminal_objective_gradient_finite_difference():
    torch.manual_seed(22);m=ProcessMoE();b=make_benchmark(raw())
    def objective():return -rollout(b,m,30,2,97402862,8)['info'][:,-1].mean()
    loss=objective();loss.backward();p=m.fusion[-1].bias;g=float(p.grad[0]);old=float(p[0]);vals=[]
    with torch.no_grad():
        for x in [-1e-5,1e-5]:p[0]=old+x;vals.append(float(objective()))
        p[0]=old
    assert abs(g-(vals[1]-vals[0])/2e-5)<1e-3*max(1,abs(g))

def test_auxiliary_is_proper_and_does_not_backpropagate_into_actions():
    torch.manual_seed(31);m=ProcessMoE();f=[history(stage=i).requires_grad_() for i in range(3)];target=torch.randn(4,4,dtype=torch.float64,requires_grad=True)
    loss=m.auxiliary_on(f,target);loss.backward()
    assert all(x.grad is None for x in f) and target.grad is None
    assert m.uncertainty.head.weight.grad.abs().sum()>0 and m.adapter.token[0].weight.grad.abs().sum()>0
    assert m.router.weight.grad is None

def test_conditional_density_learns_bimodal_fixture():
    # Independent synthetic fit check; empirical physical-history calibration
    # remains required after pilot training, not inferred from this fixture.
    torch.manual_seed(8);q=ConditionalMixture(1).double();x=torch.randn(256,16,dtype=torch.float64);y=torch.where(torch.rand(256,1)>.5,2.,-2.).double()+.15*torch.randn(256,1)
    opt=torch.optim.Adam(q.parameters(),lr=.025);first=float(-q.log_prob(q(x),y).mean())
    for _ in range(150):opt.zero_grad();loss=-q.log_prob(q(x),y).mean();loss.backward();opt.step()
    assert float(loss)<first*.65

def test_checkpoint_reproducibility_and_deterministic_deployment():
    torch.manual_seed(31);m=ProcessMoE().eval();f=history();a=m(f,10);stream=io.BytesIO();torch.save(m,stream);stream.seek(0);q=torch.load(stream,weights_only=False)
    torch.testing.assert_close(a,q(f,10),rtol=0,atol=0);torch.testing.assert_close(a,q(f,10),rtol=0,atol=0)
    for mode in ['uniform','random']:
        q.routing_mode=mode;assert torch.isfinite(q(f,10)).all() and q.last_dispatch['indices'].shape[-1]==2
    q.mask_routed=True;assert torch.isfinite(q(f,10)).all()

@pytest.mark.parametrize('name,path,horizon',[('location','configs/deepseek_sboed_location.yaml',30),('sir','configs/sir_seed101_fourhour.yaml',5),('ieee9','configs/ieee9_research_eig.yaml',3),('ieee14','configs/ieee14_research_eig.yaml',3)])
def test_domain_interfaces(name,path,horizon):
    if name.startswith('ieee') and shutil.which('nvcc') is None:pytest.skip('Unchanged grid simulator requires unavailable nvcc; adapter tested separately')
    cfg=yaml.safe_load((ROOT/path).read_text());cfg=cfg.get('benchmark',cfg);b=make_benchmark(cfg)
    m=ProcessMoE(b.action_dim,b.obs_dim,b.param_dim,horizon)
    r=rollout(b,m,horizon,1,97402863,1)
    assert len(r['actions'])==horizon and torch.isfinite(r['info']).all()
    assert all(x.shape==(1,b.action_dim) for x in r['actions'])
    (-r['info'][:,-1].mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

@pytest.mark.parametrize('name',['ieee9','ieee14'])
def test_grid_adapter_dimensions_without_simulator(name):
    cfg=yaml.safe_load((ROOT/f'configs/{name}_research_eig.yaml').read_text())
    d=len(cfg['swing_equation']['M_lower_nodes'])+len(cfg['swing_equation']['K_lower_nodes'])
    m=ProcessMoE(1,1,d,3);f=torch.zeros(2,1+3*3,dtype=torch.float64)
    for stage in range(3):
        f[:,0]=stage/3;assert m(f,stage).shape==(2,1)
    m(f,2).square().sum().backward();assert m.fusion[-1].weight.grad is not None

def test_controls_and_parameter_matching():
    m=ProcessMoE();d=dense_match(m.counts()['total_parameters']);s=ProcessMoE(variant='shared');n=ProcessMoE(variant='noaux')
    assert abs(d.counts()['total_parameters']/m.counts()['total_parameters']-1)<.02
    assert len(s.experts)==0 and len(n.experts)==4
    assert all(torch.isfinite(x(history(),10)).all() for x in [m,d,s,n])

def test_observed_only_no_online_optimizer():
    assert list(inspect.signature(ProcessMoE.forward).parameters)==['self','features','stage']
    source=inspect.getsource(__import__('src.research.moe_sboed_v3',fromlist=['ProcessMoE']))
    assert 'torch.optim' not in source and 'make_benchmark' not in source
    m=ProcessMoE().eval();f=history();a=m(f,10);f2=f.clone();f2[:,1:].reshape(4,30,4)[:,10:]=100
    torch.testing.assert_close(a,m(f2,10))
