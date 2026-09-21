"""Opt-in reuse: provenance, identity, source compatibility and immutable checkpoints."""
import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from src.checkpoint_evaluation import (inherit_settings, read_source, validate_reuse,
    load_policies, verify_source_code)
from src.objectives.eig.continuous_eig import DurationPolicy
from src.objectives.eig.continuous_redq import REDQPolicy

class CheckpointEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'run'
        (self.source / 'models').mkdir(parents=True)
        (self.source / 'source_snapshot/configs').mkdir(parents=True)
        (self.source / 'source_snapshot/configs/ieee9_eig.yaml').write_text('test')
        self.settings = dict(config='configs/ieee9_eig.yaml', objective='eig', methods='dad,rl_sboed,step_dad,fixed',
            seed=101, T=3, updates=2, window=3.5, N_obs=0, observation_kind='endpoint_rocof')
        self.config = dict(settings=self.settings, protocol='test', duration_order='strictly_increasing',
            physical_config={'bus_map': {'0': 1}}, observation_kind='endpoint_rocof')
        self.done = dict(objective='eig', methods=self.settings['methods'].split(','), evaluation_seeds=[1001], records=8)
        for name, value in [('run_config.json', self.config), ('completion.json', self.done), ('summary.json', [])]:
            (self.source / name).write_text(json.dumps(value))
        self.args = SimpleNamespace(**self.settings, estimate_only=False, preflight_max_hours=0.)
        for method in ('dad', 'fixed', 'rl_sboed'):
            policy = REDQPolicy(3, 1) if method=='rl_sboed' else DurationPolicy(3, 1, fixed=method=='fixed')
            torch.save(dict(state_dict=policy.state_dict(), method=method, horizon=3, n_obs=1,
                protocol='test', settings=self.settings), self.source/'models'/(method+'.pth'))
            (self.source/'models'/(method+'_training_diagnostics.json')).write_text('{"completed_updates":2}')
        self.reuse = self.source, self.config, self.done

    def test_explicit_opt_in_only(self):
        p=argparse.ArgumentParser()
        p.add_argument('--evaluate-from', default=None)
        with patch('sys.argv', ['run']):
            self.assertIsNone(inherit_settings(p))

    def test_source_settings_inherited_and_new_seed_required(self):
        p=argparse.ArgumentParser()
        p.add_argument('--evaluate-from')
        for name in ('config','seed','T','window'):p.add_argument('--'+name)
        p.add_argument('--eval-seeds')
        with patch('sys.argv',['run','--evaluate-from',str(self.source),'--eval-seeds','1002']):
            inherit_settings(p);args=p.parse_args()
            self.assertEqual(args.seed,101);self.assertEqual(args.window,3.5)
            self.assertIn('source_snapshot',args.config)
        with patch('sys.argv',['run','--evaluate-from',str(self.source)]):
            with self.assertRaises(SystemExit):inherit_settings(p)

    def test_settings_and_duplicate_seeds_rejected(self):
        for key,value in [('seed',202),('T',4),('window',3.),('observation_kind','max_absolute_rocof')]:
            a=copy.copy(self.args);setattr(a,key,value)
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'settings mismatch'):
                validate_reuse(self.reuse,a,self.root,[1002])
        with self.assertRaisesRegex(ValueError,'additional evaluation'):
            validate_reuse(self.reuse,self.args,self.root,[1001])

    def test_subset_and_larger_test_set_allowed(self):
        a=copy.copy(self.args);a.methods='dad,fixed';a.eval_systems=512
        with patch('src.checkpoint_evaluation.verify_source_code'):
            validate_reuse(self.reuse,a,self.root,[2001])
        a.methods='moe_sboed'
        with self.assertRaisesRegex(ValueError,'lack matching'):
            validate_reuse(self.reuse,a,self.root,[2001])

    def test_same_seed_expanded_evaluation_allowed(self):
        config=copy.deepcopy(self.config);config['settings']['eval_systems']=128
        a=copy.copy(self.args);a.eval_systems=512
        with patch('src.checkpoint_evaluation.verify_source_code'):
            validate_reuse((self.source,config,self.done),a,self.root,[1001,1002,1003])
        a.eval_systems=128
        with self.assertRaisesRegex(ValueError,'additional evaluation'):
            validate_reuse((self.source,config,self.done),a,self.root,[1001])

    def test_deferred_stepdad_same_test_seed(self):
        config=copy.deepcopy(self.config);done=copy.deepcopy(self.done)
        config['settings']['methods']='dad,fixed';done['methods']=['dad','fixed']
        a=copy.copy(self.args);a.methods='step_dad'
        with patch('src.checkpoint_evaluation.verify_source_code'):
            validate_reuse((self.source,config,done),a,self.root,[1001])

    def test_failed_source_rejected(self):
        (self.source/'failure.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'failed'):read_source(self.source)

    def test_load_preserves_weights_and_no_source_writes(self):
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()}
        meta=copy.deepcopy(self.config);meta['physical_config']={'bus_map':{0:1}}
        policies,prov=load_policies(self.reuse,self.args,SimpleNamespace(n_obs=1),meta,DurationPolicy,REDQPolicy)
        self.assertEqual(set(policies),{'dad','fixed','rl_sboed'})
        self.assertEqual(len(prov['checkpoints']),3)
        for name,policy in policies.items():
            saved=torch.load(self.source/'models'/(name+'.pth'),weights_only=True)
            for key,value in policy.state_dict().items():self.assertTrue(torch.equal(value,saved['state_dict'][key]))
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_method_selection_in_resolved_config(self):
        config=copy.deepcopy(self.config)
        config['physical_config']['experiment']={'methods':['dad','fixed'],'step_number':3}
        meta=copy.deepcopy(config);meta['physical_config']['experiment']['methods']=['dad']
        a=copy.copy(self.args);a.methods='dad'
        policies,_=load_policies((self.source,config,self.done),a,SimpleNamespace(n_obs=1),meta,DurationPolicy,REDQPolicy)
        self.assertEqual(set(policies),{'dad'})
        meta['physical_config']['experiment']['step_number']=4
        with self.assertRaisesRegex(ValueError,'experiment mismatch'):
            load_policies((self.source,config,self.done),a,SimpleNamespace(n_obs=1),meta,DurationPolicy,REDQPolicy)

    def test_checkpoint_seed_and_physics_mismatch_rejected(self):
        meta=copy.deepcopy(self.config);meta['physical_config']['bus_map']['0']=2
        with self.assertRaisesRegex(ValueError,'experiment mismatch'):
            load_policies(self.reuse,self.args,SimpleNamespace(n_obs=1),meta,DurationPolicy,REDQPolicy)
        path=self.source/'models/fixed.pth';blob=torch.load(path,weights_only=True)
        blob['settings']['seed']=202;torch.save(blob,path)
        with self.assertRaisesRegex(ValueError,'saved settings mismatch'):
            load_policies(self.reuse,self.args,SimpleNamespace(n_obs=1),self.config,DurationPolicy,REDQPolicy)

    def test_cli_only_change_allowed_but_science_change_rejected(self):
        rel='src/objectives/eig/continuous_eig.py'
        old=self.source/'source_snapshot'/rel;old.parent.mkdir(parents=True)
        original='def train(): return 1\ndef main(): return 1\n'
        old.write_text(original)
        current=self.root/rel;current.parent.mkdir(parents=True)
        current.write_text('def train(): return 1\ndef main(): return 2\n')
        config={'source_hashes':{rel:hashlib.sha256(original.encode()).hexdigest()}}
        verify_source_code(self.root,self.source,config)
        current.write_text('def train(): return 2\ndef main(): return 2\n')
        with self.assertRaisesRegex(ValueError,'implementation mismatch'):
            verify_source_code(self.root,self.source,config)

if __name__=='__main__':unittest.main()
