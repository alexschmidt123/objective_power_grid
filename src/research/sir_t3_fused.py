"""T3 isolated exact zero-padding elimination: each CUDA block stops at its own maximum subdivision count. Float64 RK4 operations on every nonzero step are unchanged."""
"""Fused CUDA execution of the existing discrete adaptive RK4 calculation.

The backward recomputes the exact first-order sensitivities of each discrete
RK4 operation, holding the existing detached subdivision counts constant.
This is not a continuous-adjoint or surrogate-gradient approximation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rhs(s,i,b,g,pop:tl.constexpr):
    incidence=b*s*i/pop
    recovery=g*i
    return -incidence,incidence-recovery,recovery


@triton.jit
def _rhs_dual(s,i,ds,di,b,g,db,dg,pop:tl.constexpr):
    a,c,e=_rhs(s,i,b,g,pop)
    # Follow the original left-to-right multiplication order.
    dincidence=((db*s+b*ds)*i+(b*s)*di)/pop
    drecovery=dg*i+g*di
    return a,c,e,-dincidence,dincidence-drecovery,drecovery


@triton.jit
def _forward(theta,initial,duration,counts,out,n,maximum,
             pop:tl.constexpr,BLOCK:tl.constexpr):
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=x<n
    b=tl.load(theta+2*x,valid,other=0.)
    g=tl.load(theta+2*x+1,valid,other=0.)
    s=tl.load(initial+3*x,valid,other=0.)
    i=tl.load(initial+3*x+1,valid,other=0.)
    r=tl.load(initial+3*x+2,valid,other=0.)
    count=tl.load(counts+x,valid,other=1)
    dt=tl.load(duration+x,valid,other=0.)/count
    for k in range(tl.max(count)):
        step=tl.where(k<count,dt,0.)
        a1,a2,a3=_rhs(s,i,b,g,pop)
        b1,b2,b3=_rhs(s+step*a1/2,i+step*a2/2,b,g,pop)
        c1,c2,c3=_rhs(s+step*b1/2,i+step*b2/2,b,g,pop)
        d1,d2,d3=_rhs(s+step*c1,i+step*c2,b,g,pop)
        s=s+step*(a1+2*b1+2*c1+d1)/6
        i=i+step*(a2+2*b2+2*c2+d2)/6
        r=r+step*(a3+2*b3+2*c3+d3)/6
    tl.store(out+3*x,s,valid)
    tl.store(out+3*x+1,i,valid)
    tl.store(out+3*x+2,r,valid)


@triton.jit
def _backward(theta,initial,duration,counts,cotangent,grad_theta,grad_initial,
              grad_duration,n,maximum,pop:tl.constexpr,BLOCK:tl.constexpr,DIRS:tl.constexpr):
    x=(tl.program_id(0)*BLOCK+tl.arange(0,BLOCK))[:,None]
    lane=tl.arange(0,DIRS)[None,:]
    if DIRS==8:
        j=lane
    elif DIRS==4:
        j=tl.where(lane<3,lane,5)
    else:
        j=lane+5
    valid=x<n
    b=tl.load(theta+2*x,valid,other=0.)
    g=tl.load(theta+2*x+1,valid,other=0.)
    s=tl.load(initial+3*x,valid,other=0.)
    i=tl.load(initial+3*x+1,valid,other=0.)
    count=tl.load(counts+x,valid,other=1)
    dt=tl.load(duration+x,valid,other=0.)/count
    ds=tl.cast(j==0,tl.float64)+tl.zeros((BLOCK,DIRS),tl.float64)
    di=tl.cast(j==1,tl.float64)+tl.zeros((BLOCK,DIRS),tl.float64)
    dr=tl.cast(j==2,tl.float64)+tl.zeros((BLOCK,DIRS),tl.float64)
    db=tl.cast(j==3,tl.float64)
    dg=tl.cast(j==4,tl.float64)
    ddt=tl.cast(j==5,tl.float64)/count
    for k in range(tl.max(count)):
        step=tl.where(k<count,dt,0.)
        dstep=tl.where(k<count,ddt,0.)
        a1,a2,a3,da1,da2,da3=_rhs_dual(s,i,ds,di,b,g,db,dg,pop)
        b1,b2,b3,db1,db2,db3=_rhs_dual(
            s+step*a1/2,i+step*a2/2,
            ds+(dstep*a1+step*da1)/2,di+(dstep*a2+step*da2)/2,
            b,g,db,dg,pop)
        c1,c2,c3,dc1,dc2,dc3=_rhs_dual(
            s+step*b1/2,i+step*b2/2,
            ds+(dstep*b1+step*db1)/2,di+(dstep*b2+step*db2)/2,
            b,g,db,dg,pop)
        d1,d2,d3,dd1,dd2,dd3=_rhs_dual(
            s+step*c1,i+step*c2,
            ds+dstep*c1+step*dc1,di+dstep*c2+step*dc2,
            b,g,db,dg,pop)
        ds=ds+(dstep*(a1+2*b1+2*c1+d1)+step*(da1+2*db1+2*dc1+dd1))/6
        di=di+(dstep*(a2+2*b2+2*c2+d2)+step*(da2+2*db2+2*dc2+dd2))/6
        dr=dr+(dstep*(a3+2*b3+2*c3+d3)+step*(da3+2*db3+2*dc3+dd3))/6
        s=s+step*(a1+2*b1+2*c1+d1)/6
        i=i+step*(a2+2*b2+2*c2+d2)/6
    gs=tl.load(cotangent+3*x,valid,other=0.)
    gi=tl.load(cotangent+3*x+1,valid,other=0.)
    gr=tl.load(cotangent+3*x+2,valid,other=0.)
    value=gs*ds+gi*di+gr*dr
    tl.store(grad_initial+3*x+j,value,valid&(j<3))
    tl.store(grad_theta+2*x+(j-3),value,valid&(j>=3)&(j<5))
    tl.store(grad_duration+x+tl.zeros((BLOCK,DIRS),tl.int32),value,valid&(j==5))


class _DiscreteRK4(torch.autograd.Function):
    @staticmethod
    def forward(ctx,theta,initial,duration,counts,population,maximum):
        if not theta.is_cuda or theta.dtype!=torch.float64:
            raise ValueError('Fused SIR requires CUDA float64 tensors')
        shape=initial.shape
        th=theta.contiguous().reshape(-1,2)
        st=initial.contiguous().reshape(-1,3)
        du=duration.contiguous().reshape(-1)
        co=counts.contiguous().reshape(-1)
        out=torch.empty_like(st)
        _forward[(triton.cdiv(len(du),128),)](th,st,du,co,out,len(du),maximum,
            pop=population,BLOCK=128,enable_fp_fusion=True)
        ctx.save_for_backward(th,st,du,co)
        ctx.population,ctx.maximum=population,maximum
        ctx.shapes=theta.shape,shape,duration.shape
        return out.reshape(shape)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx,grad):
        th,st,du,co=ctx.saved_tensors
        gt=torch.empty_like(th);gs=torch.empty_like(st);gd=torch.empty_like(du)
        dirs=8 if ctx.needs_input_grad[0] else 4 if ctx.needs_input_grad[1] else 1
        block=128//dirs
        _backward[(triton.cdiv(len(du),block),)](th,st,du,co,grad.contiguous(),gt,gs,gd,
            len(du),ctx.maximum,pop=ctx.population,BLOCK=block,DIRS=dirs,enable_fp_fusion=True)
        return (gt.reshape(ctx.shapes[0]) if ctx.needs_input_grad[0] else None,
                gs.reshape(ctx.shapes[1]) if ctx.needs_input_grad[1] else None,
                gd.reshape(ctx.shapes[2]) if ctx.needs_input_grad[2] else None,None,None,None)


def fused_rk4(theta,initial,duration,counts,population,maximum):
    return _DiscreteRK4.apply(theta,initial,duration,counts,population,maximum)
