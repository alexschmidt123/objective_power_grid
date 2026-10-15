"""Single-policy control with the corrected sum-MoE's exact193-input interface."""
from src.research.sparse_moe import SparseMoESBOED


class MatchedSumSingle193(SparseMoESBOED):
    """Fresh shared sum encoder plus one64/64tanh head, with no router.

    Construct the fresh reference first so a common training seed gives the
    identical untrained encoder, first action and expert0 initialization. No
    trained checkpoint is loaded. Extra experts and router are discarded before
    optimization; the context and first-action code remain inherited unchanged.
    """
    architecture='matched_sum193_single_policy_v1'

    def __init__(self,horizon,hidden=64,action_scale=1.,obs_scale=3.):
        super().__init__(2,1,horizon,hidden,k=1,dense=False,first_action=[0.,0.],
                         history_encoder_revision=2,action_scale=action_scale,obs_scale=obs_scale)
        self.head=self.experts[0]
        del self.experts
        del self.router
        self.dense=True
        self.control_method='sum_single_193'
        self.routing_override=None
        self.architecture='matched_sum193_single_policy_v1'

    def forward(self,features,stage):
        self.last_route=None
        if stage==0:return self.first_design[None].expand(len(features),-1)
        return self.head(self.context(features)[:,0])
