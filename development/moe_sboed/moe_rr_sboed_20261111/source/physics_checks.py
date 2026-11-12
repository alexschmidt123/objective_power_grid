"""Native likelihood, posterior prefix, and undiscounted reward identities."""
from common import *
from normalized_reward import normalized_gmm_forward

def check(a):
    with torch.no_grad(),torch.device('cuda'):
        theta=a.params[:7];ds=a.ds_hist[:7];ys=a.ys_hist[:7]
        t=12;design=ds[:,t];obs=ys[:,t]
        mean=mean_signal(theta[:,None,:4],design[:,None])[:,0,0]
        literal=-2*(obs[:,0]-mean).square()-math.log(.5*math.sqrt(2*math.pi))
        native=a.loglikeli(t,obs,theta,design,theta.new_zeros(len(theta),0)).reshape(-1)
        torch.testing.assert_close(literal,native,rtol=2e-5,atol=2e-5)
        for t in [0,3,12,24,29]:
            inp=a.post_approx.form_input(t,ds,ys)
            expected=torch.cat([ds[:,:t+1].reshape(len(ds),-1),ys[:,:t+1].reshape(len(ys),-1)],-1)
            torch.testing.assert_close(inp,expected,rtol=0,atol=0)
            dd=ds.clone();yy=ys.clone();dd[:,t+1:]=999;yy[:,t+1:]=-999
            torch.testing.assert_close(inp,a.post_approx.form_input(t,dd,yy),rtol=0,atol=0)
        inp=torch.cat([a.ds_hist.reshape(len(a.params),60),a.ys_hist.reshape(len(a.params),30)],-1)
        q=normalized_gmm_forward(a.post_approx.poi_post_nets[29][0],inp,a.params)
        prior=-.5*a.params[:,:4].square().sum(-1)-2*math.log(2*math.pi)
        torch.testing.assert_close(a.rewards_hist.sum(-1),q-prior,rtol=2e-5,atol=2e-4)
        assert a.discount==1 and not a.use_PCE
    return dict(explicit_likelihood_identity=True,posterior_prefix_future_mutation=True,native_undiscounted_variational_reward_telescopes=True,no_sPCE_training=True)

if __name__=='__main__':
    import io
    original.setup();lease=original.acquire()
    with (O/'evidence/fixture_runtime.pkl').open('rb') as f:state=cloudpickle.load(f)
    result=check(state['algorithm']);put(O/'evidence/PHYSICS_CHECKS.json',dict(status='PASS',**result));print('RR_PHYSICS_CHECKS_PASS',flush=True)
