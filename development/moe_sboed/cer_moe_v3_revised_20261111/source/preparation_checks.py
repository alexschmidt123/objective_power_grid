"""One disposable preparation step; never a scientific initialization."""
from common import *
from preparation import prepare

def main():
    original.setup();lease=original.acquire();state=restore();a=state['algorithm']
    original_settings=CFG.copy()
    try:
        CFG.update(preparation_updates=1,preparation_pool=64,preparation_batch=32)
        policy=prepare(a)
        receipt=json.loads((O/'preparation/RESULT.json').read_text())
        assert receipt['critic_rows']==128 and receipt['no_groups_required'] and receipt['new_simulator_observations']==0
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        assert policy.encoder[0].weight.requires_grad
        put(O/'evidence/PREPARATION_CHECKS.json',dict(status='PASS',fixture_only=True,overlapping_sampling=True,
            supported_history_examples=receipt['supported_history_examples'],no_four_group_requirement=True,
            one_disposable_step=True,critic_rows=128,new_simulator_observations=0))
    finally:CFG.clear();CFG.update(original_settings)
    print('CER_PREPARATION_CHECKS_PASS',flush=True)

if __name__=='__main__':main()
