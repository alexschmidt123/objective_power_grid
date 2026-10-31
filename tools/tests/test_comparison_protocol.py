import copy
import tempfile
import unittest
from pathlib import Path
import torch,yaml
from src.research.comparison_protocol import validate_protocol
from src.research.runner import DEFAULTS
from src.research.benchmarks import make_benchmark
from src.research.policies import make_policy,history_features,parameter_count
from src.research.engine import rollout
from src.research.training import train_spce

class Tests(unittest.TestCase):
    def cfg(self):
        raw=yaml.safe_load((Path(__file__).resolve().parents[2]/'configs/location_finding_eig.yaml').read_text())
        return raw,DEFAULTS|raw['research']['settings']
    def validate(self,raw,s):
        return validate_protocol(raw,s,30,['dad','moe_sboed','vsoed'],101,[1001,1002,1003,1004,1005],True,False)
    def test_frozen_config_and_budget(self):
        raw,s=self.cfg();self.assertTrue(self.validate(raw,s)['enforced'])
        p=raw['research']['upstream_vsoed'];self.assertEqual(p['initial_replay']+p['updates']*p['new_trajectories'],10000000)
        for key,value in [('contrasts',128),('batch_size',32),('training_bound',2),('moe_revision',2),('spce_curriculum',2)]:
            with self.assertRaises(ValueError):self.validate(raw,s|{key:value})
        for key,value in [('discount',.9),('updates',10001)]:
            r=copy.deepcopy(raw);r['research']['upstream_vsoed'][key]=value
            with self.assertRaises(ValueError):self.validate(r,s)
    def test_seed_separation(self):
        raw,s=self.cfg()
        with self.assertRaises(ValueError):validate_protocol(raw,s,30,['dad'],101,[82937191],True,False)
        with self.assertRaises(ValueError):validate_protocol(raw,s,30,['dad'],48563549,raw['research']['journal_protocol']['evaluation_seeds'],True,False)
        self.assertFalse(validate_protocol(raw,s,30,['dad'],48563549,[63777278],True,True)['enforced'])
    def test_pilot_seeds(self):
        raw,s=self.cfg()
        r=validate_protocol(raw,s,30,['dad'],101,[1001],pilot=True)
        self.assertTrue(r['enforced']);self.assertEqual(r['study'],'pilot')
        self.assertFalse(r['untouched_confirmation_data'])
        with self.assertRaises(ValueError):validate_protocol(raw,s,30,['dad'],102,[1001],pilot=True)
    def test_shared_initial_design_and_gradients(self):
        raw,s=self.cfg();b=make_benchmark(raw);f=history_features(b,[],[],3,30)
        for kind in ['dad','moe_sboed','uniform_moe','matched_dense']:
            p=make_policy(b,30,kind,64,s)
            torch.testing.assert_close(b.feasible(p(f,0),[],30),torch.full((3,2),.01,dtype=torch.float64))
            value=rollout(b,p,30,4,801,31)['info'][:,-1].mean();value.backward()
            self.assertGreater(float(p.first_design.grad.norm()),1e-10)
        m=make_policy(b,30,'moe_sboed',64,s);d=make_policy(b,30,'matched_dense',64,s)
        self.assertLess(abs(parameter_count(m)-parameter_count(d))/parameter_count(m),.02)
    def test_shared_sPCE_budget_counter(self):
        raw,s=self.cfg();b=make_benchmark(raw);s=s|dict(updates=10,contrasts=8,validate_every=1)
        for method in ['dad','moe_sboed','matched_dense','uniform_moe']:
            with tempfile.TemporaryDirectory() as d:
                _,r=train_spce(b,method,3,s,802,Path(d),final_checkpoint=True,trajectory_budget=45)
            self.assertEqual(r['training_forward_trajectories'],45)
            self.assertEqual(r['generated_training_trajectories'],5)
            self.assertEqual(r['contrast_training_trajectories'],40)
            self.assertEqual(r['initial_learning_rate'],.00005)

if __name__=='__main__':unittest.main()
