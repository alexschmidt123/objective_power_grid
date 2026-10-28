from src.research.sir_scorebed_compat import sir_score_config

def test_only_upstream_sir_dimension_compatibility_setting_changes():
 cfg={'score_network':{'_target_':'policy_learning.score_nets.ConservativeScoreNetwork','base_score_network':{'model_dim':256,'num_heads':8}},'score':{'batch_size':64,'num_iters':10828},'policy':{'grad_est_type':'score'}}
 actual=sir_score_config(cfg)
 assert 'potential_hidden_dim' not in cfg['score_network']
 assert actual['score_network'].pop('potential_hidden_dim')==256
 assert actual==cfg
