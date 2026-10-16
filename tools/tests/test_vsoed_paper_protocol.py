import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import torch
from src.research.benchmarks import LocationFinding
from src.research.engine import generator,rollout,logmeanexp
from src.research.location_evaluation import contrast_scores
from src.research.policies import make_policy,history_features
from src.research.runner import DEFAULTS
from src.research.training import train_spce
from src.research.upstream_vsoed import UpstreamActor
from src.research.vendor.vsoed.models import SOURCE
from src.research.vendor.vsoed import PGvsOED


def benchmark():
    return LocationFinding(dict(location_finding={'design_bounds':[-4,4]},
                                research={'paper_protocol':{'case':'1a'}}))


class PaperProtocolTests(unittest.TestCase):
    def test_model_matches_upstream(self):
        b=benchmark(); rng=generator(93)
        theta=b.sample((20,),rng)
        action=b.feasible(b.random_latent(20,rng),[],30)
        mean,_=b.propagate(theta,b.initial((20,)),action)
        expected,sigma=SOURCE().deterministic(theta,action)
        torch.testing.assert_close(mean,expected,rtol=1e-12,atol=1e-12)
        self.assertEqual(float(sigma),b.sigma)
        self.assertTrue(torch.all((action>=-4)&(action<=4)))

    def test_chunked_evaluation_matches_direct(self):
        b=benchmark(); rng=generator(94)
        bank=b.sample((37,),rng)
        p=make_policy(b,5,'dad',8)
        r=rollout(b,p,5,3,51,37,prior_particles=bank[None].expand(3,-1,-1))
        upper=r['log_likelihood'][:,0]-logmeanexp(r['log_likelihood'][:,1:])
        for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
            for chunk in [1,11,37]:
                info,u=contrast_scores(b,r,bank,chunk,device)
                torch.testing.assert_close(info,r['info'],rtol=1e-11,atol=1e-11)
                torch.testing.assert_close(u,upper,rtol=1e-11,atol=1e-11)

    def test_upstream_actor_history_adapter(self):
        b=benchmark(); source=SOURCE()
        a=PGvsOED(30,4,2,1,source,source,[(-4,4)]*2)
        adapter=UpstreamActor(a.actor_net,30)
        rng=generator(95)
        actions=[b.feasible(b.random_latent(3,rng),[],30) for _ in range(29)]
        observations=[torch.randn(3,1,generator=rng,dtype=torch.float64) for _ in range(29)]
        for stage in [0,1,15,29]:
            features=history_features(b,actions[:stage],observations[:stage],3,30)
            ds=torch.stack(actions[:stage],1).float() if stage else torch.empty(3,0,2)
            ys=torch.stack(observations[:stage],1).float() if stage else torch.empty(3,0,1)
            expected=a.actor_net(a.form_actor_input(stage,ds,ys))
            torch.testing.assert_close(adapter(features,stage).float(),expected,rtol=1e-6,atol=1e-6)

    def test_budget_accounts_for_contrasts_and_partial_batch(self):
        cfg=DEFAULTS|dict(updates=10,batch_size=4,contrasts=8,validation_systems=2,
                          validation_contrasts=8,validate_every=1,hidden=8)
        with tempfile.TemporaryDirectory() as d:
            _,r=train_spce(benchmark(),'dad',3,cfg,97,Path(d),
                          final_checkpoint=True,trajectory_budget=45)
        self.assertEqual(r['actual_updates'],2)
        self.assertEqual(r['generated_training_trajectories'],5)
        self.assertEqual(r['contrast_training_trajectories'],40)
        self.assertEqual(r['training_forward_trajectories'],45)

    def test_pinned_source_is_unmodified(self):
        p=Path(__file__).resolve().parents[2]/'src/research/vendor/vsoed'
        source=json.loads((p/'PROVENANCE.json').read_text())
        for file,digest in source['unmodified_sha256'].items():
            self.assertEqual(hashlib.sha256((p/file).read_bytes()).hexdigest(),digest)

    def test_dad_pooling_handles_history_lengths(self):
        b=benchmark(); policy=make_policy(b,30,'dad',64)
        rng=generator(96)
        actions=[b.feasible(b.random_latent(3,rng),[],30) for _ in range(29)]
        observations=[torch.randn(3,1,generator=rng,dtype=torch.float64) for _ in range(29)]
        for stage in [0,1,15,29]:
            f=history_features(b,actions[:stage],observations[:stage],3,30)
            if stage:
                raw=torch.stack([torch.cat((a,y),-1) for a,y in zip(actions[:stage],observations[:stage])],1)
                expected=policy.emitter(policy.encoder(raw).sum(1))
            else:
                expected=policy.emitter(torch.zeros(3,16,dtype=torch.float64))
            torch.testing.assert_close(policy(f,stage),expected)


if __name__=='__main__':unittest.main()
