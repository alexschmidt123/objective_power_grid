"""Independent teacher replicate: estimate target MC noise before training."""
import argparse,json,pathlib,time
import numpy as np
import torch
import audit

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',action='store_true');args=p.parse_args()
    if not args.run:p.error('Explicit --run required')
    run=audit.ROOT/'scratch_runs/hec_v2_specialization_audit_20261113';dest=run/'LABEL_NOISE.json'
    assert not dest.exists();cfg=json.loads((run/'CONFIG.json').read_text());assert not cfg['smoke']
    audit.native.original.setup();lease=audit.native.original.acquire();start=time.monotonic()
    try:
        state=audit.native.load_parent();a=state['algorithm'];actor=a.actor_net.eval().requires_grad_(False)
        before=audit.native.digest(actor.state_dict());data=np.load(run/'RAW_COMPETENCY.npz')
        x=torch.tensor(np.load(run/'FRESH_HISTORIES.npz')['X'],device='cuda')
        generator=torch.Generator(device='cuda').manual_seed(cfg['seed']+2)
        second=[[] for _ in range(4)];calls=0
        for first in range(0,len(x),64):
            assert time.monotonic()-start<1800
            with torch.no_grad(),torch.device('cuda'):
                xx=x[first:first+64];_,info=actor(xx,collect=True)
                values,receipt=audit.labels(a,actor,actor,xx,info,generator,cfg['teacher_replicates'])
                calls+=receipt['predictive_observations']
                for k in range(4):second[k].append(values[k].cpu().numpy())
        result={};fit=3*cfg['episodes'][0];test=3*sum(cfg['episodes'][:2]);nt=cfg['episodes'][2]
        for k in range(4):
            one=data[f'target{k}'];two=np.concatenate(second[k]);scale=one[:fit].std(0).clip(.1)
            noise=.5*np.square((one-two)/scale).mean(-1)
            residual=data['losses'][k,k]
            statistic=audit.paired(audit.grouped_mean(residual[test:],nt),audit.grouped_mean(noise[test:],nt))
            result[f'E{k+1}']=dict(normalized_teacher_noise=float(noise[test:].mean()),
                observed_probe_MSE=float(residual[test:].mean()),
                probe_residual_minus_estimated_noise=statistic,
                note='Paired independent16-replicate target estimates; noise floor estimate assumes independent unbiased MC labels for this fixed approximate-model teacher. Not a physical-truth accuracy test.')
        assert audit.native.digest(actor.state_dict())==before
        audit.put(dest,dict(status='COMPLETE',experts=result,seconds=time.monotonic()-start,
                           teacher_predictive_observations=calls,optimizer_steps=0))
        print(dest.read_text(),flush=True)
    finally:
        for f in lease:f.close()
if __name__=='__main__':main()
