"""Batched evaluation-only oracle, matching the scalar first-safe-bracket search."""
import numpy as np


def batch_control_oracle(engine, M, K, spec, tolerance=1e-4):
    M=np.asarray(M,dtype=float); K=np.asarray(K,dtype=float)
    candidates=np.asarray(spec.u_candidates,dtype=float)
    grid=np.unique(np.r_[candidates,np.linspace(0,candidates.max(),max(33,2*len(candidates)))])
    n=len(M)
    def safe(rows, controls):
        r,f=engine.simulate_metrics_batch(M[rows],K[rows],controls,batch_size=512)
        if not (np.isfinite(r).all() and np.isfinite(f).all()):
            raise ValueError('Nonfinite physical oracle metrics')
        return (r<=spec.rocof_limit_hz_s)&(f>=spec.delta_f_nadir_hz)
    table=safe(np.repeat(np.arange(n),len(grid)),np.tile(grid,n)).reshape(n,-1)
    feasible=table.any(1)
    first=table.argmax(1)
    lower=grid[np.maximum(first-1,0)].copy(); upper=grid[first].copy()
    upper[~feasible]=candidates.max()
    active=feasible&(first>0)
    while np.any(active & (upper-lower>tolerance)):
        rows=np.flatnonzero(active & (upper-lower>tolerance))
        mid=(lower[rows]+upper[rows])/2
        ok=safe(rows,mid)
        upper[rows[ok]]=mid[ok]; lower[rows[~ok]]=mid[~ok]
    monotonic=~np.any(table[:,:-1]&~table[:,1:],axis=1)
    monotonic[~feasible]=False
    return {'u_opt':upper,'feasible':feasible,'monotonic':monotonic,'coarse_grid':grid,
            'coarse_safe':table,'bracket_lower':lower,'bracket_upper':upper}
