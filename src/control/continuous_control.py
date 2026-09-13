"""Continuous terminal control from the actual post-probe particle states.

The scan grid brackets numerical roots; it is NOT the control action space.
Results are continuous bracketed minima to declared tolerance, conditional on
sampled monotonicity checks, not proofs of global feasibility between samples.
"""
import numpy as np
from src.control.u_req import ControlSpec


def select_control(grid, safe, weights, coverage):
    """Select a numerical search bracket, never the final continuous action."""
    grid, safe, weights=np.asarray(grid,float),np.asarray(safe,bool),np.asarray(weights,float)
    if grid.ndim!=1 or not len(grid) or not np.isfinite(grid).all() or np.any(grid<0) or np.any(np.diff(grid)<=0):
        raise ValueError('Invalid increasing numerical scan')
    if safe.shape!=(len(weights),len(grid)) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Invalid safety table or posterior')
    if not 0<coverage<1:raise ValueError('Coverage must be in (0,1)')
    mass=weights/weights.sum()@safe
    indices=np.flatnonzero(mass>=coverage-1e-12)
    if not len(indices):raise ValueError('Posterior control infeasible within allowed bounds')
    i=int(indices[0])
    return dict(u_ctrl=float(grid[i]),posterior_safe_mass=float(mass[i]),control_index=i)


def continuous_decision(requirements,weights,coverage,objective):
    requirements,weights=np.asarray(requirements,float),np.asarray(weights,float)
    if requirements.shape!=weights.shape or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
        raise ValueError('Invalid posterior weights')
    if np.isnan(requirements).any() or np.any(requirements<0) or (objective=='mocu' and not np.isfinite(requirements).all()) or not 0<coverage<1:
        raise ValueError('Continuous requirement model infeasible or invalid coverage')
    weights=weights/weights.sum()
    order=np.argsort(requirements,kind='stable')
    k=min(int(np.searchsorted(np.cumsum(weights[order]),coverage,side='left')),len(order)-1)
    u=float(requirements[order[k]])
    if not np.isfinite(u):raise ValueError('Posterior control infeasible within continuous bounds')
    row={'u_ctrl':u,'msc':u,'posterior_ess':float(1/np.sum(weights**2)),
         'posterior_safe_mass':float(weights[requirements<=u].sum())}
    if objective=='mocu':
        row['posterior_mocu']=float(weights@(u+np.maximum(requirements-u,0)/(1-coverage)-requirements))
    elif objective!='msc':raise ValueError('Unknown control objective')
    return row


class CarriedStateControl:
    def __init__(self,cfg,simulator):
        from src.control.cuda_control import CudaControlEngine
        raw=dict(cfg.raw['control'])
        self.bounds=tuple(map(float,raw['u_bounds']))
        self.tolerance=float(raw.get('solver_tolerance',1e-5))
        points=int(raw.get('solver_scan_points',33))
        if len(self.bounds)!=2 or not 0<=self.bounds[0]<self.bounds[1] or not self.tolerance>0 or points<3:
            raise ValueError('Invalid continuous control bounds or solver settings')
        self.grid=np.linspace(*self.bounds,points)
        # ControlSpec is shared physical infrastructure. Its candidate field is
        # supplied only as a numerical bracket grid, never exposed as actions.
        from types import SimpleNamespace
        self.spec=ControlSpec.from_cfg(SimpleNamespace(raw={**cfg.raw,'control':{**raw,'u_candidates':self.grid.tolist()}}))
        self.coverage=float(raw['posterior_coverage'])
        self.engine=CudaControlEngine(simulator,self.spec)
        if self.spec.profile.t_start!=0 or self.spec.profile.shape!='step' or self.spec.profile.duration<self.spec.T_obs_sec:
            raise ValueError('Require immediate step control throughout the safety window')
        if not np.isclose(self.spec.T_obs_sec/self.spec.ode_dt,round(self.spec.T_obs_sec/self.spec.ode_dt)):
            raise ValueError('Control window must contain an integer number of integration steps')

    def metrics(self,theta,states,controls):
        theta,states=np.asarray(theta,float),np.asarray(states,float)
        controls=np.asarray(controls,float)
        if np.any(controls<self.bounds[0]) or np.any(controls>self.bounds[1]):raise ValueError('Control outside continuous bounds')
        n=self.engine.N
        r,f=self.engine.simulate_metrics_batch(theta[:,:n],theta[:,n:],controls,initial_states=states,batch_size=32768)
        return r,f,(r<=self.spec.rocof_limit_hz_s)&(f>=self.spec.delta_f_nadir_hz)

    def safety_table(self,theta,states):
        count,levels=len(theta),len(self.grid)
        _,_,safe=self.metrics(np.repeat(theta,levels,axis=0),np.repeat(states,levels,axis=0),np.tile(self.grid,count))
        return safe.reshape(count,levels)

    def requirements(self,theta,states,*,allow_infeasible=False):
        theta,states=np.asarray(theta,float),np.asarray(states,float)
        safe=self.safety_table(theta,states)
        if np.any(np.diff(safe.astype(int),axis=1)<0):
            raise ValueError('Non-monotone sampled safety: scalar continuous requirement solver unsupported')
        feasible=safe.any(axis=1)
        if not feasible.all() and not allow_infeasible:
            raise ValueError('Continuous control infeasible for posterior support; do not drop or resample particles')
        first=safe.argmax(axis=1)
        lo=self.grid[np.maximum(first-1,0)].copy()
        hi=self.grid[first].copy()
        active=feasible&(first>0)
        while np.any(active & (hi-lo>self.tolerance)):
            idx=np.flatnonzero(active & (hi-lo>self.tolerance))
            mid=(lo[idx]+hi[idx])/2
            _,_,ok=self.metrics(theta[idx],states[idx],mid)
            hi[idx[ok]]=mid[ok];lo[idx[~ok]]=mid[~ok]
        hi[~feasible]=np.inf
        return hi

    def decision(self,theta,states,weights,objective='msc'):
        req=self.requirements(theta,states,allow_infeasible=objective=='msc')
        row=continuous_decision(req,weights,self.coverage,objective)
        # Independent direct safety calculation at selected continuous action.
        _,_,safe=self.metrics(theta,states,np.full(len(theta),row['u_ctrl']))
        mass=float(np.average(safe,weights=weights))
        if mass<self.coverage-1e-12:raise ValueError('Selected continuous control failed posterior safety verification')
        row['posterior_safe_mass']=mass
        row['control_solver_tolerance']=self.tolerance
        return row

    def oracle(self,theta,state):
        req=self.requirements(np.asarray(theta)[None],np.asarray(state)[None],allow_infeasible=True)[0]
        return {'oracle_msc':float(req) if np.isfinite(req) else None,
                'oracle_feasible':bool(np.isfinite(req)),
                'oracle_kind':'continuous_bracketed_minimum_at_actual_terminal_state',
                'oracle_tolerance':self.tolerance}
