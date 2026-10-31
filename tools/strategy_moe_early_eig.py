"""Same paired common EIG, overlapped with existing posterior diagnostics."""
import os,time
from pathlib import Path
from tools.report_strategy_moe import *

def main():
    while not (RUN/'models/strategy_moe_sboed_training.json').exists() or not (RUN/'eval/contrast_bank_seed1001.npy').exists():
        if (RUN/'exit_code').exists() and (RUN/'exit_code').read_text().strip()!='0':raise RuntimeError('Pilot failed')
        time.sleep(20)
    torch.set_num_threads(2)
    cfg=read(RUN/'run_config.json');assert read(RUN/'models/strategy_moe_sboed_training.json')['generated_training_trajectories']==1000000
    rows=read(OLD/'eval/moe_sboed_seed1001.json');assert len(rows)==2000
    assert digest(OLD/'eval/contrast_bank_seed1001.npy')==digest(RUN/'eval/contrast_bank_seed1001.npy')
    bank=torch.from_numpy(np.load(RUN/'eval/contrast_bank_seed1001.npy'));assert bank.shape==(1000000,4)
    b=make_benchmark(cfg['benchmark']);saved=torch.load(RUN/'models/strategy_moe_sboed.pt',weights_only=False,map_location='cpu')
    results={}
    for label,mode,soft in [('early_learned_hard','learned',False),('always_reroute','always_reroute',False),('always_continue','always_continue',False),('learned_relaxed','learned',True)]:
        result,scores=evaluate_variant(b,saved,rows,bank,label,mode,soft)
        results[label]=result
        if label=='early_learned_hard':
            result['paired_minus_original_sum_moe']=paired(scores,[r['spce'] for r in rows])
            result['status']='Full 2000-case common EIG complete; native posterior/behavior report still pending'
            write(C/'early_eig.json',result)
        write(C/'early_mode_eig.json',results)
    print({k:x['eig'] for k,x in results.items()},flush=True)
if __name__=='__main__':main()
