"""MSC decision, policy-objective isolation, and six-method integration checks."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from src.config import SBOEDConfig, load_config, resolve_training_block
from src.objectives.msc import posterior_msc, validate_msc_support
from src.objectives.mocu.context import (posterior_objective_batch, posterior_mocu_batch,
    _score_fixed_subset, expected_mocu_after_action_vector, vector_gaussian_loglik,
    posterior_objective, ExperimentContext)
from src.objectives.mocu.train import (_posterior_mocu_gpu, train_policy, sample_trajectory,
    load_trained_policy, evaluate_policy)
from src.objectives.mocu.evaluate import rank_msc, summarize_rows
from src.control.posterior_ctrl import normalize_log_weights
from src.layout import make_experiment_dir_name, parse_result_dir_name

class MSCTests(unittest.TestCase):
    def test_discrete_constraint_against_enumeration(self):
        rng=np.random.default_rng(42)
        required=np.array([.12,.17,.35,.42])
        grid=np.array([0,.1,.2,.3,.4,.5])
        for q in [.8,.85,.9,.95,.99]:
            for _ in range(10):
                w=rng.dirichlet(np.ones(4))
                expected=min(u for u in grid if w[required<=u].sum()>=q)
                self.assertEqual(posterior_msc(required,w,coverage=q,grid=grid),expected)
    def test_not_mocu_or_oracle(self):
        common=dict(alpha=.1,margin=0,u_grid=np.array([.2,.5]),undercontrol_penalty=10,
                    violation_penalty=0,robust_rule='quantile')
        req=np.array([.2,.5]);w=np.array([[.95,.05],[0,1.]])
        msc,_=posterior_objective_batch(req,w,objective='msc',**common)
        mocu,_=posterior_mocu_batch(req,w,**common)
        np.testing.assert_allclose(msc,[.2,.5])
        self.assertLess(msc[0],msc[1]);self.assertGreater(mocu[0],mocu[1])
    def test_cpu_torch_parity_and_penalty_independence(self):
        req=np.array([.2,.3,.5]);grid=np.array([0,.2,.3,.5])
        logw=np.random.default_rng(41).normal(size=(16,3))
        w=np.array([normalize_log_weights(x) for x in logw])
        for q in [.8,.85,.9,.95,.99]:
            expected=np.array([posterior_msc(req,x,coverage=q,grid=grid) for x in w])
            for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
                for penalty in [0,20,1000]:
                    scores,controls,_=_posterior_mocu_gpu(torch.tensor(logw,device=device),
                        torch.tensor(req,device=device),torch.tensor(grid,device=device),
                        alpha=1-q,margin=0,undercontrol_penalty=penalty,violation_penalty=99,
                        robust_rule='quantile',objective='msc')
                    np.testing.assert_allclose(scores.cpu(),expected)
                    np.testing.assert_allclose(controls.cpu(),expected)
    def test_infeasibility_and_nonmonotone_bank_rejected(self):
        for req in [[.8],[np.inf],[np.nan],[-.1]]:
            with self.assertRaises(ValueError): validate_msc_support(req,[0,.5])
        with self.assertRaises(ValueError): validate_msc_support([.2],[0,.2,.5],[[0,1,0]])
        validate_msc_support([.2],[0,.2,.5],[[0,1,1]])
    def test_domain_and_independent_config(self):
        for name in ['ieee9','ieee14','ieee30']:
            c=load_config(Path('configs')/f'{name}_mocu.yaml');c.validate_msc()
            c.raw['training']['msc_based']['updates']=7
            self.assertEqual(c.training_for('msc_based')['updates'],7)
            self.assertNotEqual(c.training_for('objective_based')['updates'],7)
        c=load_config('configs/sir_ode_eig.yaml')
        with self.assertRaises(ValueError): c.validate_msc()
        with self.assertRaises(ValueError): resolve_training_block({'objective_based':{'updates':1}},'msc_based')
    def test_distinct_run_identity(self):
        for et in ['msc_based','objective_based','eig_based']:
            name=make_experiment_dir_name('ieee9_mocu',et,3,n_obs=5,noise_sigma=.005)
            self.assertEqual(parse_result_dir_name(name)['experiment_type'],et)
    def test_fixed_and_myopic_use_msc(self):
        req=np.array([.2,.5]);prior=np.log([.5,.5])
        centres=np.array([[[0.],[3.]],[[0.],[0.]]])
        kw=dict(alpha=.1,margin=0,u_grid=req,undercontrol_penalty=10,violation_penalty=0,
                robust_rule='quantile',objective='msc')
        idx=np.array([0,1]);noise=np.zeros((2,1))
        for action in [0,1]:
            weights=[normalize_log_weights(prior+vector_gaussian_loglik(y,centres[action],1))
                     for y in centres[action,idx]]
            expected=np.mean([posterior_msc(req,w,coverage=.9,grid=req) for w in weights])
            got=expected_mocu_after_action_vector(action,prior,centres=centres,U=req,sigma_y=1,
                                                  idx=idx,noise=noise,**kw)
            self.assertAlmostEqual(got,expected)
        eps=np.random.default_rng(17).normal(0,1,size=(1,2,2,1))
        expected=np.mean([posterior_msc(req,normalize_log_weights(prior+vector_gaussian_loglik(
            centres[0,t]+eps[0,t,0],centres[0],1)),coverage=.9,grid=req) for t in [0,1]])
        got=_score_fixed_subset([0],centres_by_theta=centres.transpose(1,0,2),U_support=req,
                               log_p0=prior,sigma_y=1,seed=17,**kw)
        self.assertAlmostEqual(got,expected)
    def test_validation_scores_msc_and_keeps_mocu_diagnostic(self):
        trajectory=dict(actions=[0],terminal_u_ctrl=.3,terminal_objective=.3,terminal_posterior_mocu=.02)
        ctx=SimpleNamespace(undercontrol_penalty=10,violation_penalty=0)
        with patch('src.objectives.mocu.train.sample_trajectory',return_value=trajectory):
            val=evaluate_policy(ctx,None,[{'u_req':.4}],n_rollouts=1,global_seed=1,
                                reward_mode='dad_terminal',device='cpu')
        self.assertAlmostEqual(val['mean_objective'],.3)
        self.assertAlmostEqual(val['mean_posterior_mocu'],.02)
    def test_ranking_uses_msc_not_mocu(self):
        rows=[dict(method='A',n=10,mean_msc=.3,mean_posterior_mocu=.01),
              dict(method='B',n=10,mean_msc=.2,mean_posterior_mocu=.02)]
        self.assertEqual([x['method'] for x in rank_msc(rows)],['B','A'])
    def test_summary_separates_oracle_and_msc(self):
        rows=[dict(method='DAD',theta_id=0,u_ctrl=.4,control_gap=.1,ocu=.1,
                   posterior_mocu=.02,method_safe=1,u_ctrl_opt=.3,sequence='0 1')]
        summary=summarize_rows(rows,'DAD')
        self.assertAlmostEqual(summary['mean_msc'],.4)
        self.assertAlmostEqual(summary['mean_oracle_msc'],.3)
        self.assertAlmostEqual(summary['mean_posterior_mocu'],.02)

    def test_oracle_only_computes_evaluated_systems_and_preserves_ids(self):
        from src.objectives.mocu.evaluate import attach_oracle
        with tempfile.TemporaryDirectory() as temp:
            ctx=SimpleNamespace(out_dir=Path(temp),test_systems=[{}]*8,
                M_test=np.arange(8.)[:,None],K_test=np.ones((8,1)),oracle_tolerance=.0001,
                undercontrol_penalty=20,violation_penalty=0)
            rows=[dict(method='Fixed',theta_id=i,u_ctrl=.5) for i in [7,2,7]]
            with patch('src.objectives.mocu.evaluate.control_engine_for',return_value=(None,None)), \
                 patch('src.objectives.mocu.evaluate.load_or_compute_oracle_cache',return_value=[
                     dict(theta_id=0,u_ctrl_opt=.2),dict(theta_id=1,u_ctrl_opt=.4)]) as oracle:
                enriched,errors=attach_oracle(ctx,rows,skip_cuda_safety=True)
            self.assertEqual(len(oracle.call_args.args[1]),2)
            self.assertEqual([x['M'] for x in oracle.call_args.args[1]],[[2.],[7.]])
            self.assertEqual([x['u_ctrl_opt'] for x in enriched],[.4,.2,.4])
            self.assertFalse(errors)
            self.assertTrue(all(x['safety_evaluation']=='oracle_threshold_proxy' for x in enriched))

    def test_six_methods_toy_integration_and_checkpoint_isolation(self):
        """Bounded synthetic unit fixture, not an IEEE experiment or physical safety validation."""
        from src.objectives.mocu.evaluate import evaluate_fixed, evaluate_random, evaluate_myopic
        from src.objectives.mocu.step_dad import evaluate_step_dad, StepDADConfig
        from src.objectives.mocu.context import _greedy_fixed_sequence
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            cfg=load_config('configs/ieee9_mocu.yaml')
            cfg.raw['training']['device']='cpu'
            req=np.array([.2,.3,.4,.5])
            centres=np.array([[[-2],[-1],[1],[2]],[[1],[-1],[1],[-1]],[[0],[0],[0],[0]]],float)
            systems=[dict(obs_clean=centres[:,i,:],u_req=float(req[i]),theta_id=i) for i in range(4)]
            seq,_=_greedy_fixed_sequence(centres_by_theta=centres.transpose(1,0,2),U_support=req,
                log_p0=np.log([.25]*4),sigma_y=.5,alpha=.05,margin=0,u_grid=req,
                undercontrol_penalty=20,violation_penalty=0,horizon=2,robust_rule='quantile',objective='msc')
            ctx=ExperimentContext(system='ieee9',cfg=cfg,horizon=2,n_actions=3,n_obs=1,n_sim=1,
                obs_dim=1,obs_indices=np.array([0]),observation_mode='sampled_delta_f',sigma_y=.5,
                alpha=.05,margin=0,u_grid=req,robust_rule='quantile',snap_up=True,experiment_type='msc_based',
                centres_support=centres,U_support=req,log_p0=np.log([.25]*4),M_support=req,K_support=req,
                particle_features=np.c_[req,req,req].astype(np.float32),obs_mean=0,obs_std=1,
                test_systems=systems,train_systems=systems,validation_systems=systems,U_test=req,
                M_test=req[:,None],K_test=req[:,None],data_dir=Path(temp),out_dir=Path(temp),
                oracle_tolerance=.0001,fixed_sequence=seq,terminal_rule_hash='toy-rule',config_hash='toy',
                undercontrol_penalty=20,violation_penalty=0)
            for method in ['DAD','RL-sBOED']:
                with patch('src.objectives.mocu.train._write_loss_curve'):
                    result=train_policy(ctx,method=method,seed=101,smoke=True)
                self.assertEqual(result['objective'],'msc')
                policy=load_trained_policy(ctx,method)
                for mode in ['dad_terminal','rl_sboed_stepwise']:
                    traj=sample_trajectory(ctx,policy,systems[0],theta_id=0,rollout_id=0,global_seed=1001,
                        reward_mode=mode,device=torch.device('cpu'),deterministic=True)
                    self.assertAlmostEqual(traj['terminal_objective'],traj['terminal_u_ctrl'])
                    expected_return = -traj['u_path'][-1] if mode == 'dad_terminal' else traj['u_path'][0]-traj['u_path'][-1]
                    self.assertAlmostEqual(sum(traj['rewards']),expected_return)
                # A checkpoint trained for MSC must not load as a MOCU policy.
                ctx.experiment_type='objective_based'
                with self.assertRaises(ValueError):load_trained_policy(ctx,method)
                ctx.experiment_type='msc_based'
            for key in ['random','fixed','myopic']:
                rows={'fixed':evaluate_fixed,'random':evaluate_random,'myopic':evaluate_myopic}[key](ctx,2,eval_seed=1001)
                self.assertEqual(len(rows),2)
            rows,meta=evaluate_step_dad(ctx,2,eval_seed=1001,
                config=StepDADConfig(refinement_steps=1,fantasy_rollouts=2),device_name='cpu')
            self.assertEqual(len(rows),2)
            self.assertIn('msc',meta['objective'].lower())

if __name__=='__main__': unittest.main()
