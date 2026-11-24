"""Pinned upstream conservative-score compatibility for scalar SIR data/designs."""
import copy

def sir_score_config(config):
    """Use the upstream documented nondegenerate potential head, not a new loss.

    Source-location has three output channels; SIR has only two. Upstream
    ConservativeScoreNetwork documents the two-channel LayerNorm degeneracy
    and supplies potential_hidden_dim for this exact dimensional case. Keep
    its established model_dim=256 for this native compatibility projection.
    """
    cfg=copy.deepcopy(config)
    score=cfg['score_network']
    if not score['_target_'].endswith('ConservativeScoreNetwork'):
        raise ValueError('This compatibility rule applies only to the pinned conservative score')
    width=int(score['base_score_network']['model_dim'])
    if width!=256:raise ValueError('Preserve the established ScoreBED Transformer width256')
    score['potential_hidden_dim']=width
    return cfg
